from __future__ import annotations

from typing import override

import numpy as np

from src.modules.audio.localization.analyzer import AudioAnalyzer
from src.modules.audio.localization.data import AudioBuffer, InferenceResult, MicInfo


def _last_consecutive_true_run(flags: list[bool]) -> tuple[int, int] | None:
    """
    Max contiguous subsequences where flags[i] is True.
    Returns (start, end) inclusive indices of the last such run, or None if none.
    """
    n = len(flags)
    runs: list[tuple[int, int]] = []
    i = 0
    while i < n:
        if not flags[i]:
            i += 1
            continue
        start = i
        while i < n and flags[i]:
            i += 1
        runs.append((start, i - 1))
    return runs[-1] if runs else None


def _rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


class Analyzer(AudioAnalyzer):
    """
    Localize using energy on microphones: among the last group of side-by-side
    positive drone inferences, pick the channel with the strongest signal (RMS).
    """

    def __init__(self, sample_rate: int, mic_infos: list[MicInfo]):
        super().__init__(sample_rate)
        self.mic_infos = mic_infos
        self.audio_buffers: dict[int, np.ndarray] = {}
        self.inference_results: dict[int, bool] = {}

    @override
    def push_buffer(self, buffer: AudioBuffer):
        self.audio_buffers[buffer.channel] = buffer.data

    @override
    def push_inference(self, inference: InferenceResult):
        self.inference_results[inference.channel] = inference.drone

    @override
    def get_angle(self) -> float:
        n = len(self.mic_infos)
        flags = [self.inference_results.get(i, False) for i in range(n)]

        run = _last_consecutive_true_run(flags)
        try:
            if run is None:
                angle = 0.0
            else:
                lo, hi = run
                energies: list[tuple[int, float]] = []
                for ch in range(lo, hi + 1):
                    buf = self.audio_buffers.get(ch)
                    if buf is None:
                        continue
                    energies.append((ch, _rms(buf)))
                if not energies:
                    angle = 0.0
                else:
                    best_ch = max(energies, key=lambda t: t[1])[0]
                    angle = float(self.mic_infos[best_ch].orientation)
        finally:
            self.audio_buffers = {}
            self.inference_results = {}

        return angle
