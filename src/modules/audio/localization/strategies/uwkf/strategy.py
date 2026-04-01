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
# Merwe Scaled Sigma-Point UKF — state: [angle_deg]  (scalar, wrapped)
#
# Rationale for scalar state:
#   Angular velocity is not directly observable from a single noisy angle
#   measurement per step. Keeping a velocity state causes the filter to
#   interpret measurement noise as motion, leading to runaway estimates.
#   A scalar UKF with well-tuned Q/R provides the correct amount of
#   smoothing without the instability.
# ---------------------------------------------------------------------------

class _MerweSigmaPoints:
    """
    Merwe scaled sigma-point transform.
    n      : state dimension
    alpha  : spread of sigma points around mean (1e-3 .. 1.0)
    beta   : prior knowledge of distribution (2 is optimal for Gaussian)
    kappa  : secondary scaling (0 or 3-n are common choices)
    """

    def __init__(self, n: int, alpha: float = 0.3, beta: float = 2.0, kappa: float = 0.0):
        self.n = n
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        lam = alpha ** 2 * (n + kappa) - n
        self.lam = lam

        # weights for mean
        self.Wm = np.full(2 * n + 1, 0.5 / (n + lam))
        self.Wm[0] = lam / (n + lam)

        # weights for covariance
        self.Wc = self.Wm.copy()
        self.Wc[0] += (1.0 - alpha ** 2 + beta)

    def sigma_points(self, x: np.ndarray, P: np.ndarray) -> np.ndarray:
        n = self.n
        lam = self.lam
        U = np.linalg.cholesky((n + lam) * P).T
        sigmas = np.empty((2 * n + 1, n))
        sigmas[0] = x
        for i in range(n):
            sigmas[i + 1]     = x + U[i]
            sigmas[n + i + 1] = x - U[i]
        return sigmas


class AngleUKF:
    """
    Scalar Unscented Kalman Filter for tracking a wrapped angle (degrees).

    State:  x = [angle_deg]
    Measurement: z = [angle_deg]

    All angular quantities are wrapped to [-180, 180].

    Parameters
    ----------
    q_angle : process noise std (deg) — expected angle change per step.
              Lower = smoother but more lag on real motion.
    r_angle : measurement noise std (deg) — distrust of each raw reading.
              Higher = smoother but more lag.
    """

    def __init__(
        self,
        q_angle: float = 2.0,
        r_angle: float = 20.0,
        alpha: float = 0.3,
        beta: float = 2.0,
        kappa: float = 0.0,
    ):
        n = 1
        self._sp = _MerweSigmaPoints(n, alpha, beta, kappa)

        self.Q = np.array([[q_angle ** 2]])
        self.R = np.array([[r_angle ** 2]])

        # State & covariance — uninitialised until first measurement
        self.x: np.ndarray | None = None
        self.P: np.ndarray = np.array([[180.0 ** 2]])

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _f(x: np.ndarray) -> np.ndarray:
        """Process model: angle is constant between steps (random-walk)."""
        return np.array([_wrap(x[0])])

    @staticmethod
    def _h(x: np.ndarray) -> np.ndarray:
        """Observation model: we directly observe the angle."""
        return np.array([_wrap(x[0])])

    @staticmethod
    def _circular_mean(sigmas: np.ndarray, Wm: np.ndarray) -> np.ndarray:
        sin_sum = np.dot(Wm, np.sin(np.radians(sigmas[:, 0])))
        cos_sum = np.dot(Wm, np.cos(np.radians(sigmas[:, 0])))
        return np.array([np.degrees(np.arctan2(sin_sum, cos_sum))])

    @staticmethod
    def _wrap_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.array([_wrap(a[0] - b[0])])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, measured_angle_deg: float) -> float:
        """
        Feed one new angle measurement.
        Returns the filtered angle estimate in degrees, wrapped to [-180, 180].
        """
        z = np.array([_wrap(measured_angle_deg)])

        if self.x is None:
            self.x = z.copy()
            return float(self.x[0])

        sp = self._sp

        # ── PREDICT ────────────────────────────────────────────────────
        sigmas = sp.sigma_points(self.x, self.P)
        sigmas_f = np.array([self._f(s) for s in sigmas])

        x_pred = self._circular_mean(sigmas_f, sp.Wm)

        P_pred = self.Q.copy()
        for i, s in enumerate(sigmas_f):
            dx = self._wrap_diff(s, x_pred)
            P_pred += sp.Wc[i] * np.outer(dx, dx)

        # ── UPDATE ─────────────────────────────────────────────────────
        sigmas_h = np.array([self._h(s) for s in sigmas_f])

        z_pred_sin = np.dot(sp.Wm, np.sin(np.radians(sigmas_h[:, 0])))
        z_pred_cos = np.dot(sp.Wm, np.cos(np.radians(sigmas_h[:, 0])))
        z_pred = np.array([np.degrees(np.arctan2(z_pred_sin, z_pred_cos))])

        Pzz = self.R.copy()
        for i, sh in enumerate(sigmas_h):
            dz = self._wrap_diff(sh, z_pred)
            Pzz += sp.Wc[i] * np.outer(dz, dz)

        Pxz = np.zeros((1, 1))
        for i, (sf, sh) in enumerate(zip(sigmas_f, sigmas_h)):
            dx = self._wrap_diff(sf, x_pred)
            dz = self._wrap_diff(sh, z_pred)
            Pxz += sp.Wc[i] * np.outer(dx, dz)

        K = Pxz @ np.linalg.inv(Pzz)

        # State update
        innovation = self._wrap_diff(z, z_pred)
        self.x = x_pred + (K @ innovation).flatten()
        self.x[0] = _wrap(self.x[0])

        # Covariance update — Joseph form for numerical stability
        IKH = np.eye(1) - K @ (Pxz.T @ np.linalg.inv(P_pred))
        self.P = IKH @ P_pred @ IKH.T + K @ self.R @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        self.P += np.eye(1) * 1e-6

        # Eigenvalue clamp fallback
        eigvals = np.linalg.eigvalsh(self.P)
        if eigvals.min() <= 0.0:
            vals, vecs = np.linalg.eigh(self.P)
            vals = np.maximum(vals, 1e-6)
            self.P = vecs @ np.diag(vals) @ vecs.T

        return float(self.x[0])


class Analyzer(AudioAnalyzer):
    TRESHOLD = 0.2
    PARTING = 50

    def __init__(self, sample_rate: int, mic_infos: list[MicInfo]):
        super().__init__(sample_rate)

        self.mic_infos = mic_infos
        self.audio_buffers = {}
        self.inference_results = {}

        self.vectors = np.array([to_carthesian(m.orientation) for m in mic_infos])

        # UKF replaces the old alpha-beta filter.
        # Tune q_angle, q_vel, r_angle to your drone dynamics:
        #   • Increase r_angle  → smoother but more lag
        #   • Increase q_angle  → tracks fast moves but noisier
        self._ukf = AngleUKF(
            q_angle=2.0,   # expected bearing drift per step (deg)
            r_angle=20.0,  # measurement noise — ~circ_std of 3-4° but trust less
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
        filtered_angle = self._ukf.update(measured_angle)

        self.audio_buffers = {}
        self.inference_results = {}

        if SETTINGS.AUDIO_STRATEGY_REPORT:
            cstd = _circular_std(angles, mags / (mags.max() + 1e-12))
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

            with open("uwkf.csv", 'w', newline='') as file:
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
