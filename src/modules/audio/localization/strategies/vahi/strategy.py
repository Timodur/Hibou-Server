from src.modules.audio.localization.analyzer import AudioAnalyzer
from src.modules.audio.localization.data import AudioBuffer, InferenceResult, MicInfo
from scipy import signal as sig
from typing import override
from sklearn.linear_model import RANSACRegressor, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.base import BaseEstimator, RegressorMixin
from collections import deque

import pyroomacoustics as pra
import numpy as np
import math


nfft = 512
hop = nfft // 2
HISTORY_LEN = 10        # how many past frames to fit the trend over
TREND_THRESHOLD = 0.05  # min slope to extrapolate vs. fall back to circular mean


# ── Extrapolation helpers (unchanged from power strategy) ────────────────────

class _PolyRidge(BaseEstimator, RegressorMixin):
    def __init__(self, degree: int = 2):
        self.degree = degree

    def fit(self, X, y, sample_weight=None):
        self._poly = PolynomialFeatures(degree=self.degree, include_bias=False)
        self._ridge = Ridge(alpha=1e-3)
        self._ridge.fit(self._poly.fit_transform(X), y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        return self._ridge.predict(self._poly.transform(X))

    def score(self, X, y, sample_weight=None):
        residuals = y - self.predict(X)
        if sample_weight is not None:
            return -float(np.average(residuals ** 2, weights=sample_weight))
        return -float(np.mean(residuals ** 2))


def _make_ransac(k: int, degree: int, residual_threshold: float) -> RANSACRegressor:
    return RANSACRegressor(
        estimator=_PolyRidge(degree=degree),
        min_samples=max(degree + 1, int(0.5 * k)),
        residual_threshold=residual_threshold,
        max_trials=500,
        random_state=42,
    )


def _weighted_circular_mean(angles_deg: np.ndarray, weights: np.ndarray) -> float:
    rad = np.radians(angles_deg)
    cx = np.sum(weights * np.cos(rad))
    cy = np.sum(weights * np.sin(rad))
    return float(np.degrees(np.arctan2(cy, cx)))


def extrapolate_angle(angles_deg: np.ndarray, degree: int = 2) -> float:
    """
    Extrapolate the next angle from a history of per-frame angles.
    No magnitude weights here — NormMUSIC angles are already high-quality
    estimates, so uniform weighting is appropriate.
    Falls back to circular mean when no meaningful trend is detected.
    """
    k = len(angles_deg)
    w = np.ones(k)
    baseline = _weighted_circular_mean(angles_deg, w)

    t = np.arange(k).reshape(-1, 1)
    t_next = np.array([[k]])
    rad = np.radians(angles_deg)
    cx, cy = np.cos(rad), np.sin(rad)

    ransac_x = _make_ransac(k, degree, residual_threshold=0.15)
    ransac_y = _make_ransac(k, degree, residual_threshold=0.15)

    try:
        ransac_x.fit(t, cx)
        ransac_y.fit(t, cy)
    except ValueError:
        return baseline

    slope_x = float(np.abs(ransac_x.predict([[k - 1]]) - ransac_x.predict([[0]])) / k)
    slope_y = float(np.abs(ransac_y.predict([[k - 1]]) - ransac_y.predict([[0]])) / k)

    if max(slope_x, slope_y) < TREND_THRESHOLD:
        return baseline

    return float(np.degrees(np.arctan2(ransac_y.predict(t_next)[0],
                                        ransac_x.predict(t_next)[0])))


# ── STFT helper ───────────────────────────────────────────────────────────────

def compute_stfts(
    signals: np.ndarray,
    nfft: int,
    hop: int,
    fs: int,
    num_mics: int,
) -> np.ndarray:
    f, t, Zxx = sig.stft(
        signals[0], fs=fs, nperseg=nfft, noverlap=nfft - hop, return_onesided=True
    )
    X = np.zeros((num_mics, *Zxx.shape), dtype=complex)
    X[0] = Zxx
    for i in range(1, num_mics):
        _, _, X[i] = sig.stft(
            signals[i], fs=fs, nperseg=nfft, noverlap=nfft - hop, return_onesided=True
        )
    return X


# ── Analyzer ──────────────────────────────────────────────────────────────────

class Analyzer(AudioAnalyzer):
    def __init__(self, sample_rate: int, mic_infos: list[MicInfo]):
        super().__init__(sample_rate)

        self.mic_angles = np.array([mic.orientation for mic in mic_infos])
        if math.nan in self.mic_angles:
            raise ValueError(
                f"Invalid data has been provided as mic orientation: {mic_infos}"
            )

        self.mic_count = len(mic_infos)
        self.mic_positions = np.array([
            [mic.xpos for mic in mic_infos],
            [mic.ypos for mic in mic_infos],
            np.zeros(len(mic_infos)),
        ])

        self.room = pra.AnechoicRoom(fs=sample_rate, temperature=20)
        self.room.add_microphone_array(
            pra.MicrophoneArray(self.mic_positions, sample_rate)
        )

        self.audio_buffers = {}
        self.inference_results = {}

        # Per-source angle history for extrapolation.
        # Keyed by source index (0, 1, …); each holds a fixed-length deque
        # of past NormMUSIC azimuths.
        self._angle_history: dict[int, deque[float]] = {}

    @override
    def push_buffer(self, buffer: AudioBuffer):
        self.audio_buffers[buffer.channel] = buffer.data

    @override
    def push_inference(self, inference: InferenceResult):
        self.inference_results[inference.channel] = inference.drone

    @override
    def get_angle(self):
        if (
            len(self.audio_buffers) != self.mic_count
            or len(self.inference_results) != self.mic_count
        ):
            return math.nan

        audios   = np.array([self.audio_buffers[i]       for i in range(self.mic_count)])
        inferred = np.array([self.inference_results[i]   for i in range(self.mic_count)])

        self.audio_buffers = {}
        self.inference_results = {}

        return self._guess(audios, inferred)

    def _guess(self, audios: np.ndarray, inferred: np.ndarray):
        n_detected = int(np.sum(inferred))
        if n_detected == 0:
            self._angle_history.clear()   # no source → reset history
            return np.array([])

        X = compute_stfts(audios, nfft, hop, self.sample_rate, self.mic_count)
        frame_window = 10
        mid = X.shape[2] // 2
        frame_start = max(0, mid - frame_window // 2)
        frame_end   = min(X.shape[2], mid + frame_window // 2)
        frame_start, frame_end = 0, X.shape[2]

        doa = pra.doa.NormMUSIC(
            self.mic_positions,
            self.sample_rate,
            nfft,
            n_src=n_detected,
            num_iter=5,
        )
        doa.locate_sources(X[:, :, frame_start:frame_end])

        raw_azimuths = np.rad2deg(doa.azimuth_recon)   # shape: (n_detected,)

        # Accumulate history and extrapolate for each detected source.
        # Sources are matched by index — stable as long as n_detected is constant.
        # If the number of sources changes, old histories are dropped.
        active_keys = set(range(n_detected))
        stale_keys  = set(self._angle_history) - active_keys
        for k in stale_keys:
            del self._angle_history[k]

        smoothed = np.empty(n_detected)
        for i, az in enumerate(raw_azimuths):
            if i not in self._angle_history:
                self._angle_history[i] = deque(maxlen=HISTORY_LEN)
            self._angle_history[i].append(float(az))

            history = np.array(self._angle_history[i])
            if len(history) < 3:
                # Not enough history yet — return the raw NormMUSIC estimate
                smoothed[i] = az
            else:
                smoothed[i] = extrapolate_angle(history, degree=min(2, len(history) - 1))

        return smoothed, inferred
