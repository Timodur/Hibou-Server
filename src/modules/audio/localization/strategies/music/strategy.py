from src.modules.audio.localization.analyzer import AudioAnalyzer
from src.modules.audio.localization.data import AudioBuffer, InferenceResult, MicInfo
from scipy import signal as sig
from typing import override

import pyroomacoustics as pra
import numpy as np
import math


mic_radius = 0.2
nfft = 512  # FFT size
hop = nfft // 2  # Hop length (50% overlap)
frame_window = 10


# Temporary values used for convenience. To remove later.
num_mics = 4
angles = np.linspace(0, 2 * np.pi, num_mics, endpoint=False)

mic_positions = np.array(
    [
        mic_radius * np.cos(angles),
        mic_radius * np.sin(angles),
        np.zeros(num_mics),  # z=0 is OK in AnechoicRoom
    ]
)


def compute_stfts(signals, nfft, hop, fs, num_mics):
    # Compute STFT for first channel to get dimensions
    f, t, Zxx = sig.stft(
        signals[0],
        fs=fs,  # We only need the STFT, not actual frequencies
        nperseg=nfft,
        noverlap=nfft - hop,
        return_onesided=True,
    )

    n_freq, n_frames = Zxx.shape

    # Preallocate array for all channels
    X = np.zeros((num_mics, n_freq, n_frames), dtype=complex)
    X[0] = Zxx

    # Compute STFT for remaining channels
    for i in range(1, num_mics):
        _, _, X[i] = sig.stft(
            signals[i], fs=fs, nperseg=nfft, noverlap=nfft - hop, return_onesided=True
        )

    return X


class Analyzer(AudioAnalyzer):
    def __init__(self, sample_rate: int, mic_infos: list[MicInfo]):
        super().__init__(sample_rate, mic_infos)

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

        self.room = pra.AnechoicRoom(fs=sample_rate, temperature=20)
        self.room.add_microphone_array(
            pra.MicrophoneArray(self.mic_positions, sample_rate)
        )

        self.audio_buffers = {}
        self.inference_results = {}

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

        return self._guess(
            np.array([self.audio_buffers[i] for i in range(self.mic_count)]),
            np.array([self.inference_results[i] for i in range(self.mic_count)]),
        )

    def _guess(self, audios, inferred):
        n_detected = np.sum(inferred)
        if n_detected == 0:
            return np.array([])

        X = compute_stfts(audios, nfft, hop, self.sample_rate, self.mic_count)
        mid_frame = X.shape[2] // 2
        frame_start = max(0, mid_frame - frame_window // 2)
        frame_end = min(X.shape[2], mid_frame + frame_window // 2)

        doa = pra.doa.NormMUSIC(
            self.mic_positions, self.sample_rate, nfft, n_src=n_detected, num_iter=5
        )  # num_iter can help stability

        # Keep 3D shape: (n_mics, n_freq, n_snapshots)
        doa.locate_sources(X[:, :, frame_start:frame_end])

        estimated_azimuths = np.rad2deg(doa.azimuth_recon)  # Convert to degrees
        return estimated_azimuths, inferred
