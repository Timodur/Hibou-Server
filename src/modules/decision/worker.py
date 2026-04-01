from src.helpers.ipc.base_ipc import get_ipc_handler
from src.helpers.decorators import SingletonMeta
from src.logger import CustomLogger
from src.settings import SETTINGS
import datetime
import time
import os
import numpy as np

logger = CustomLogger("decision").get_logger()
from src.helpers.system_status import SystemStatusUpdater


class DecisionWorker:
    def __init__(self, dt: datetime.datetime):
        logger.info(f"Started Decision Worker | PID: {os.getpid()}")

        self.ipc = get_ipc_handler()
        self.ipc.subscribe(SETTINGS.IPC_ACOUSTIC_DETECTION_TOPIC, self._update_audio_inf)
        self.ipc.subscribe(SETTINGS.IPC_ACOUSTIC_ANGLE_TOPIC, self._update_audio_angle)

        self.angle = 0.0
        self.inf = []

        self._reset = True

        self.system_status_updater = SystemStatusUpdater(
            system_name="worker:decision",
        )

        try:
            self.run()
        except KeyboardInterrupt:
            logger.critical("Stopping Audio Worker...")
        finally:
            SingletonMeta.clear()

    def _update_audio_inf(self, topic, values):
        self.inf = values

    def _update_audio_angle(self, topic, angle):
        self.angle = angle

    def run(self):
        logger.info("Running decision worker...")
        while True:
            time.sleep(0.01)
            if np.any(self.inf) or self.angle != 0.0:
                self._reset = True
                print(self.inf)
                print(self.angle)
                # [TODO] Add logic here
            elif self._reset:
                self._reset = False
                print("Waiting for angle update...")

            self.system_status_updater.update()
