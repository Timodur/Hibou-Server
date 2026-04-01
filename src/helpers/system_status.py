from src.helpers.ipc.base_ipc import get_ipc_handler
from src.settings import SETTINGS
import time

class SystemStatusUpdater:
    """
    Class to update the system status every {interval} seconds.
    """
    def __init__(self, system_name: str, interval: int = 2):
        self.ipc = get_ipc_handler()
        self.system_name = system_name
        self.interval = interval
        self.last_update = time.time()

    def update(self, status: str = "active"):
        if time.time() - self.last_update > self.interval:
            self.last_update = time.time()
            self.ipc.publish(SETTINGS.IPC_SYSTEM_STATUS_TOPIC, f"{self.system_name}:{status}")