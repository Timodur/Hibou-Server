from collections import deque

import numpy as np

from src.arguments import args
from src.modules.audio.detection.ai import ModelProxy
from src.modules.audio.localization.data import AudioBuffer, InferenceResult, MicInfo
from src.modules.audio.localization.strategies.stronger.strategy import Analyzer
from src.modules.audio.streaming import GstChannel
from src.modules.audio.streaming.play import play_sample
from src.settings import SETTINGS
from src.helpers.ipc.base_ipc import get_ipc_handler
from src.helpers.system_status import SystemStatusUpdater


class AudioDispatcher:
    """
    Class responsible for dispatching audio data to the appropriate processing modules, such as inference and localization.
    """

    def __init__(self):
        self.audio_queue = deque(maxlen=20)
        # True or False
        self.predictions_queue = deque(maxlen=20)
        # Confidence of the prediction, between 0 and 1
        self.probabilities_queue = deque(maxlen=20)
        self.model = ModelProxy(args.audio_model)
        self.ipc = get_ipc_handler()

        # [TODO]: For now it is hard-coded, but either it must be taken frow the web client, or a conf file somehow.
        mic_radius = 0.2
        num_mics = 3
        angles = np.linspace(0, 2 * np.pi, num_mics, endpoint=False)  # [0°, 120°, 240°]
        angles_deg = np.degrees(angles)

        mic_positions = np.array([
            mic_radius * np.cos(angles),  # x coords: [ 0.2,  -0.1,  -0.1]
            mic_radius * np.sin(angles),  # y coords: [ 0.0,   0.173, -0.173]
        ])

        mic_infos = [
            MicInfo(i,
                    mic_positions[0][i],
                    mic_positions[1][i],
                    angles_deg[i])
            for i in range(num_mics)
        ]

        self.analyzer = Analyzer(SETTINGS.AUDIO_REC_HZ, mic_infos)

        self.system_status_updater = SystemStatusUpdater(
            system_name="preamplifier",
        )


    def process(self, audio_samples: list[GstChannel]):
        self.audio_queue.append(audio_samples)
        
        # Update system status when receiving audio data
        self.system_status_updater.update()

        if SETTINGS.AUDIO_PLAYBACK:  # Only for debug purposes
            play_sample(audio_samples[0], 0)

        res, prb = self.model.infer(audio_samples)
        self.predictions_queue.append(res)
        self.probabilities_queue.append(prb)

        self.ipc.publish(SETTINGS.IPC_ACOUSTIC_DETECTION_TOPIC, "drone" if any(res) else "other")

        # if any(res):
        #     print("Drone detected")
        # else:
        #     print("No drone detected")

        i = 0
        for audio, pts in audio_samples:
            self.analyzer.push_buffer(
                AudioBuffer(timestamp=pts, channel=i, data=np.array(audio))
            )
            i += 1
        i = 0
        for pred in res:
            self.analyzer.push_inference(
                InferenceResult(
                    timestamp=audio_samples[i][1], channel=i, confidence=0, drone=pred
                )
            )
            i += 1

        angle = self.analyzer.get_angle()
        self.ipc.publish(SETTINGS.IPC_ACOUSTIC_ANGLE_TOPIC, f"{float(angle)}")

    def get_last_channels(self) -> list[GstChannel] | None:
        try:
            return self.audio_queue.pop()
        except IndexError:
            return None

    def is_empty(self) -> bool:
        return len(self.audio_queue) == 0
