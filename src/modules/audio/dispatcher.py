from collections import deque
from pathlib import Path
import numpy as np

from src.arguments import args
from src.modules.audio.detection.ai import ModelProxy
from src.modules.audio.localization.data import AudioBuffer, InferenceResult, MicInfo
from src.modules.audio.localization.strategies.energy.strategy import Analyzer
from src.modules.audio.streaming import GstChannel
from src.modules.audio.streaming.play import play_sample
from src.settings import SETTINGS
from src.helpers.ipc.base_ipc import get_ipc_handler
from src.helpers.json import read_json
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

        mic_array = read_json(Path("./mic_information.json"))["array"]
        mic_infos = [
            MicInfo.from_dict(info)
            for info in mic_array
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

        serialized_preds = str([int(pred) for pred in res])[1:-1]
        self.ipc.publish(SETTINGS.IPC_ACOUSTIC_DETECTION_TOPIC, serialized_preds)

        if any(res):
            print("Drone detected")
        else:
            print("No drone detected")

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
