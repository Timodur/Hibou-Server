from src.modules.audio.localization.analyzer import AudioAnalyzer
from src.modules.audio.localization.data import AudioBuffer, InferenceResult, MicInfo
from src.arguments import args

from typing import override
import numpy as np
import math
import csv
from scipy.stats import circmean

from sklearn.linear_model import RANSACRegressor, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.base import BaseEstimator, RegressorMixin

from itertools import zip_longest
import numpy as np
from pyroomacoustics.experimental import tdoa
from pyroomacoustics.doa.srp import SRP
import pyroomacoustics as pra

import time
from scipy.signal import stft
from nara_wpe.wpe import wpe

import time
import numpy as np
from scipy.signal import stft
from pyroomacoustics.doa.srp import SRP
import pyroomacoustics as pra



class Analyzer(AudioAnalyzer):
    def __init__(self, sample_rate: int, mic_infos: list[MicInfo]):
        super().__init__(sample_rate)

        self.mic_infos = mic_infos

        self.mic_angles = np.array([mic.orientation for mic in mic_infos])
        if math.nan in self.mic_angles:
            raise ValueError(
                f"Invalid data has been provided as mic orientation:{mic_infos}"
            )

        self.mic_count = len(mic_infos)
        self.mic_positions = np.array(
            [
                [mic.xpos for mic in mic_infos],
                [mic.ypos for mic in mic_infos],
                np.zeros(len(mic_infos)),
            ]
        )
        print(f"mic_positions:\n{self.mic_positions}")

        self.room = pra.AnechoicRoom(fs=sample_rate, temperature=20)
        self.room.add_microphone_array(
            pra.MicrophoneArray(self.mic_positions, sample_rate)
        )

        self.audio_buffers = {}
        self.inference_results = {}
        self.angle_history = []

        self.data = {"filtered": [], "measured": []}

        self.nfft = 256
        self.doa = SRP(self.mic_positions, sample_rate, self.nfft, c=343, num_src=1, max_four=360)

    @override
    def push_buffer(self, buffer: AudioBuffer):
        self.audio_buffers[buffer.channel] = buffer.data

    @override
    def push_inference(self, inference: InferenceResult):
        self.inference_results[inference.channel] = inference.drone

    def _resolve_ambiguity(self, raw_angle: float) -> float:
        """Use mic orientations and energies to pick the correct half-plane."""
        # Compute energy per mic
        energies = np.array([
            np.var(self.audio_buffers[i]) for i in range(len(self.mic_infos))
        ])

        # Find the mic with highest energy
        dominant_mic = np.argmax(energies)
        dominant_orientation = self.mic_angles[dominant_mic]  # in degrees

        # The mirror candidate
        mirror_angle = (raw_angle + 180) % 360

        # Pick the candidate closest to the dominant mic's orientation
        diff_raw = abs((raw_angle - dominant_orientation + 180) % 360 - 180)
        diff_mirror = abs((mirror_angle - dominant_orientation + 180) % 360 - 180)

        return raw_angle if diff_raw < diff_mirror else mirror_angle

    @override
    def get_angle(self) -> float:
        for i in range(len(self.mic_infos)):
            buf = self.audio_buffers[i]
            print(f"  mic {i}: min={buf.min():.4f} max={buf.max():.4f} mean={buf.mean():.4f}")

        # --- Checks ---
        if len(self.audio_buffers) != len(self.mic_infos):
            raise ValueError(
                f"Buffer count ({len(self.audio_buffers)}) != "
                f"mic count ({len(self.mic_infos)})"
            )

        nfft = 256
        hop = nfft // 2
        n_mics = len(self.mic_infos)

        # --- STFT in the same order as microphones ---
        X_complex = []
        for i, mic in enumerate(self.mic_infos):
            # Assumes that the key in audio_buffers is the channel index (0,1,2,...)
            x = self.audio_buffers[i]
            x = x.astype(np.float64)
            _, _, Zxx = stft(x, fs=self.sample_rate, nperseg=nfft,
                             noverlap=nfft - hop, boundary=None, padded=False)
            # Zxx shape: (freq_bins, frames) -> transpose to (frames, freq_bins)
            X_complex.append(Zxx)

        X_complex = np.array(X_complex)  # (n_mics, frames, freq_bins)
        print(f"STFT shape: {X_complex.shape}")

        # --- Verify microphone positions ---
        print(f"mic_positions shape: {self.mic_positions.shape}")  # should be (3, n_mics)
        mic_positions_for_srp = self.mic_positions.T  # (n_mics, 3)
        print(f"mic_positions.T shape: {mic_positions_for_srp.shape}")

        print(f"SRP: M = {self.doa.M}")  # should equal n_mics

        # Pass the data directly – the first dimension must be n_mics
        self.doa.locate_sources(
            X_complex,
            freq_bins=np.arange(20, 70)
        )

        print(f"grid values min/max: {self.doa.grid.values.min():.6f} / {self.doa.grid.values.max():.6f}")
        print(f"grid values sample: {self.doa.grid.values[:5]}")
        print(f"azimuth_recon: {self.doa.azimuth_recon}")
        print(f"colatitude_recon: {self.doa.colatitude_recon}")

        az = self.doa.azimuth_recon  # shape: (num_src,), in radians
        el = self.doa.colatitude_recon  # shape: (num_src,), in radians (colatitude, not elevation)

        if len(az) == 0:
            return np.nan

        raw_angle = np.degrees(az[0])
        resolved = self._resolve_ambiguity(raw_angle)

        self.angle_history.append(np.radians(resolved))
        smoothed = np.degrees(circmean(list(self.angle_history)))
        print(f"  Raw: {raw_angle:.1f}°  Resolved: {resolved:.1f}°  Smoothed: {smoothed:.1f}°")

        # --- Clean up ---
        self.audio_buffers = {}
        self.inference_results = {}

        return smoothed
