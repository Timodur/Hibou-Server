from src.modules.audio.localization.analyzer import AudioAnalyzer
from src.modules.audio.localization.data import AudioBuffer, InferenceResult, MicInfo
from src.settings import SETTINGS

from typing import override
import numpy as np
import math
import csv

from sklearn.linear_model import RANSACRegressor, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.base import BaseEstimator, RegressorMixin

from itertools import zip_longest


class _PolyRidge(BaseEstimator, RegressorMixin):
    def __init__(self, degree: int = 2):
        self.degree = degree

    def fit(self, X, y, sample_weight=None):
        self._poly = PolynomialFeatures(degree=self.degree, include_bias=False)
        self._ridge = Ridge(alpha=1e-3)
        Xp = self._poly.fit_transform(X)
        self._ridge.fit(Xp, y, sample_weight=sample_weight)
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


def _circular_std(angles_deg: np.ndarray, weights: np.ndarray) -> float:
    rad = np.radians(angles_deg)
    w = weights / weights.sum()
    R = np.sqrt(np.sum(w * np.cos(rad))**2 + np.sum(w * np.sin(rad))**2)
    R = np.clip(R, 0.0, 1.0)
    return float(np.degrees(np.sqrt(-2 * np.log(R + 1e-12))))


def extrapolate_angle(
        angles_deg: np.ndarray,
        weights: np.ndarray | None = None,
        degree: int = 2,
        trend_threshold: float = 0.05,
) -> float:
    k = len(angles_deg)
    w = weights / (weights.max() + 1e-12) if weights is not None else np.ones(k)

    baseline = _weighted_circular_mean(angles_deg, w)

    t = np.arange(k).reshape(-1, 1)
    t_next = np.array([[k]])
    rad = np.radians(angles_deg)
    cx = np.cos(rad)
    cy = np.sin(rad)

    threshold = 0.15
    ransac_x = _make_ransac(k, degree, threshold)
    ransac_y = _make_ransac(k, degree, threshold)

    try:
        ransac_x.fit(t, cx, sample_weight=w)
        ransac_y.fit(t, cy, sample_weight=w)
    except ValueError:
        return baseline

    slope_x = float(np.abs(ransac_x.predict([[k - 1]]) - ransac_x.predict([[0]])) / k)
    slope_y = float(np.abs(ransac_y.predict([[k - 1]]) - ransac_y.predict([[0]])) / k)

    if max(slope_x, slope_y) < trend_threshold:
        return baseline

    cx_next = ransac_x.predict(t_next)[0]
    cy_next = ransac_y.predict(t_next)[0]
    return float(np.degrees(np.arctan2(cy_next, cx_next)))


def extract_data(data: list[tuple[float, float]]) -> tuple[np.ndarray, np.ndarray]:
    angles = np.array([d[0] for d in data])
    mags = np.array([d[1] for d in data])
    return angles, mags


def to_carthesian(angle: float):
    return math.cos(math.radians(angle)), math.sin(math.radians(angle))


class Analyzer(AudioAnalyzer):
    TRESHOLD = 0.2
    PARTING = 50

    def __init__(self, sample_rate: int, mic_infos: list[MicInfo]):
        super().__init__(sample_rate)

        self.mic_infos = mic_infos
        self.audio_buffers = {}
        self.inference_results = {}

        self.vectors = np.array([to_carthesian(m.orientation) for m in mic_infos])

        self._angle_est = None
        self._velocity_est = 0.0
        self._last_time = None

        if SETTINGS.AUDIO_STRATEGY_REPORT:
            self.data = {"mag": [], "angle": [], "measured": [], "filtered": [], "circ_std": []}

    @override
    def push_buffer(self, buffer: AudioBuffer):
        self.audio_buffers[buffer.channel] = buffer.data

    @override
    def push_inference(self, inference: InferenceResult):
        self.inference_results[inference.channel] = inference.drone

    def _alpha_beta_filter(self, measured_angle: float) -> float:
        """
        Lightweight Kalman-like filter for angle tracking.
        Handles wrap-around and fast motion.
        """

        # tuning parameters (you WILL want to tweak these)
        alpha = 0.65  # trust in measurement
        beta = 0.25  # trust in velocity update
        dt = 1.0  # assume constant time step (adjust if needed)

        # first measurement initialization
        if self._angle_est is None:
            self._angle_est = measured_angle
            self._velocity_est = 0.0
            return measured_angle

        # unwrap angle difference to [-180, 180]
        delta = (measured_angle - self._angle_est + 180) % 360 - 180

        predicted_angle = self._angle_est + self._velocity_est * dt

        # measurement residual
        residual = delta

        self._angle_est = predicted_angle + alpha * residual
        self._velocity_est = self._velocity_est + (beta * residual) / dt

        # wrap back to [-180, 180] or [0, 360]
        self._angle_est = (self._angle_est + 180) % 360 - 180

        return self._angle_est

    @override
    def get_angle(self):
        data = self._compute()
        angles, mags = extract_data(data)

        measured_angle = _weighted_circular_mean(angles, mags)
        filtered_angle = self._alpha_beta_filter(measured_angle)

        self.audio_buffers = {}
        self.inference_results = {}

        if SETTINGS.AUDIO_STRATEGY_REPORT:
            cstd = _circular_std(angles, mags / (mags.max() + 1e-12))
            print(
                f"  slices : {np.round(angles, 1).tolist()}\n"
                f"  mags   : {np.round(mags, 4).tolist()}\n"
                f"  measured: {measured_angle:.1f}°\n"
                f"  filtered: {filtered_angle:.1f}°\n"
                f"  circ_std: {cstd:.1f}°\n"
            )

            self.data["measured"].append(measured_angle)
            self.data["filtered"].append(filtered_angle)
            self.data["circ_std"].append(cstd)

            with open("stronger.csv", 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(self.data.keys())
                columns = [np.asarray(val).tolist() for val in self.data.values()]
                writer.writerows(zip_longest(*columns, fillvalue=''))

        return filtered_angle

    def _compute(self):
        data = []

        parts = len(self.audio_buffers[0]) // self.PARTING
        for i in range(self.PARTING):
            # RMS per microphone — correct measure of audio power.
            # np.mean() averages to ~0 for AC audio signals, making all
            # magnitudes collapse and arctan2 output random angles.
            rms = np.array([
                np.sqrt(np.mean(self.audio_buffers[j][i*parts:(i+1)*parts] ** 2))
                for j in range(len(self.mic_infos))
            ])

            avgs = rms / np.sum(rms)
            weighted = self.vectors * avgs[:, np.newaxis]

            x = np.sum(weighted[:, 0])
            y = np.sum(weighted[:, 1])

            mag = np.sqrt(x**2 + y**2)
            angle = np.arctan2(y, x)

            _angle = math.degrees(angle)

            if SETTINGS.AUDIO_STRATEGY_REPORT:
                self.data["angle"].append(_angle)
                self.data["mag"].append(mag)

            data.append((_angle, mag))

        return data
