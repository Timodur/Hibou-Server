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


def _wrap(angle_deg: float) -> float:
    """Wrap angle to [-180, 180]."""
    return (angle_deg + 180.0) % 360.0 - 180.0


def _wrap_arr(arr: np.ndarray) -> np.ndarray:
    return (arr + 180.0) % 360.0 - 180.0


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


# ---------------------------------------------------------------------------
# Particle Filter for wrapped angle tracking
# ---------------------------------------------------------------------------

class ParticleFilter:
    """
    Particle filter for tracking a wrapped angle (degrees).

    State: angle in degrees, wrapped to [-180, 180].
    Process model: random walk with given process noise standard deviation.
    Measurement model: Gaussian on the wrapped angular difference, with
    measurement noise standard deviation that can be fixed or dynamically
    provided per update.

    Parameters
    ----------
    num_particles : int
        Number of particles used for the filter.
    process_std : float
        Standard deviation of the process noise (degrees per step).
    meas_std : float
        Default measurement noise standard deviation (degrees). Can be
        overridden in update().
    """

    def __init__(self, num_particles: int = 500, process_std: float = 2.0, meas_std: float = 20.0):
        self.num_particles = num_particles
        self.process_std = process_std
        self.default_meas_std = meas_std

        self.particles: np.ndarray | None = None
        self.weights: np.ndarray | None = None
        self._initialized = False

    @staticmethod
    def _wrap(angle: float) -> float:
        """Wrap angle to [-180, 180]."""
        return (angle + 180.0) % 360.0 - 180.0

    def initialize(self, first_angle: float | None = None) -> None:
        """
        Initialize the particle set.

        If first_angle is provided, particles are sampled from a Gaussian
        around that angle with standard deviation default_meas_std.
        Otherwise, particles are uniformly distributed over [-180, 180].
        """
        if first_angle is not None:
            # Sample around the first measurement
            self.particles = np.random.normal(first_angle, self.default_meas_std, self.num_particles)
        else:
            self.particles = np.random.uniform(-180.0, 180.0, self.num_particles)
        self.particles = self._wrap(self.particles)
        self.weights = np.ones(self.num_particles) / self.num_particles
        self._initialized = True

    def predict(self) -> None:
        """Apply process model (random walk) to all particles."""
        noise = np.random.normal(0.0, self.process_std, self.num_particles)
        self.particles += noise
        self.particles = self._wrap(self.particles)

    def update(self, measured_angle: float, sigma_meas: float | None = None) -> None:
        """
        Update the filter with a new measurement.

        Parameters
        ----------
        measured_angle : float
            Observed angle in degrees, will be wrapped.
        sigma_meas : float, optional
            Measurement noise standard deviation (degrees). If None,
            self.default_meas_std is used.
        """
        if not self._initialized:
            self.initialize(measured_angle)
            return

        if sigma_meas is None:
            sigma_meas = self.default_meas_std

        # Wrapped angular difference
        diff = self._wrap(measured_angle - self.particles)
        # Gaussian likelihood
        log_weights = -0.5 * (diff / sigma_meas) ** 2
        # Numerical stability: subtract max
        log_weights -= np.max(log_weights)
        self.weights = np.exp(log_weights)
        self.weights += 1e-300  # avoid zeros
        self.weights /= np.sum(self.weights)

        # Resample
        self._resample()

    def _resample(self) -> None:
        """Systematic resampling."""
        indices = self._systematic_resample(self.weights)
        self.particles = self.particles[indices]
        self.weights = np.ones(self.num_particles) / self.num_particles

    @staticmethod
    def _systematic_resample(weights: np.ndarray) -> np.ndarray:
        N = len(weights)
        cumulative = np.cumsum(weights)
        # Sample N equally spaced positions with a random offset
        positions = (np.arange(N) + np.random.uniform(0, 1)) / N
        indices = np.zeros(N, dtype=int)
        i = j = 0
        while i < N:
            if positions[i] < cumulative[j]:
                indices[i] = j
                i += 1
            else:
                j += 1
        return indices

    def estimate(self) -> float:
        """
        Compute the circular mean of the current particle set.

        Returns the filtered angle in degrees, wrapped to [-180, 180].
        """
        if not self._initialized:
            return 0.0
        # Particles are equally weighted after resampling
        sin_sum = np.sum(np.sin(np.radians(self.particles)))
        cos_sum = np.sum(np.cos(np.radians(self.particles)))
        angle = np.degrees(np.arctan2(sin_sum, cos_sum))
        return self._wrap(angle)


# ---------------------------------------------------------------------------
# Audio Analyzer using Particle Filter
# ---------------------------------------------------------------------------

class Analyzer(AudioAnalyzer):
    TRESHOLD = 0.2
    PARTING = 50

    def __init__(self, sample_rate: int, mic_infos: list[MicInfo]):
        super().__init__(sample_rate)

        self.mic_infos = mic_infos
        self.audio_buffers = {}
        self.inference_results = {}

        self.vectors = np.array([to_carthesian(m.orientation) for m in mic_infos])

        # Particle filter replaces the UKF
        self._pf = ParticleFilter(
            num_particles=500,
            process_std=2.0,    # expected bearing drift per step (deg)
            meas_std=20.0,      # default measurement noise, will be overwritten by circ_std
        )

        if SETTINGS.AUDIO_STRATEGY_REPORT:
            self.data = {"mag": [], "angle": [], "measured": [], "filtered": [], "circ_std": []}

    @override
    def push_buffer(self, buffer: AudioBuffer):
        self.audio_buffers[buffer.channel] = buffer.data

    @override
    def push_inference(self, inference: InferenceResult):
        self.inference_results[inference.channel] = inference.drone

    @override
    def get_angle(self):
        data = self._compute()
        angles, mags = extract_data(data)

        measured_angle = _weighted_circular_mean(angles, mags)
        cstd = _circular_std(angles, mags / (mags.max() + 1e-12))

        # Update particle filter with measurement and use circular std as measurement noise
        self._pf.update(measured_angle, sigma_meas=cstd)
        filtered_angle = self._pf.estimate()

        self.audio_buffers = {}
        self.inference_results = {}

        if SETTINGS.AUDIO_STRATEGY_REPORT:
            print(
                f"  slices   : {np.round(angles, 1).tolist()}\n"
                f"  mags     : {np.round(mags, 4).tolist()}\n"
                f"  measured : {measured_angle:.1f}°\n"
                f"  filtered : {filtered_angle:.1f}°\n"
                f"  circ_std : {cstd:.1f}°\n"
            )

            self.data["measured"].append(measured_angle)
            self.data["filtered"].append(filtered_angle)
            self.data["circ_std"].append(cstd)

            with open("temanu.csv", 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(self.data.keys())
                columns = [np.asarray(val).tolist() for val in self.data.values()]
                writer.writerows(zip_longest(*columns, fillvalue=''))

        return filtered_angle

    def _compute(self):
        data = []

        parts = len(self.audio_buffers[0]) // self.PARTING
        for i in range(self.PARTING):
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
