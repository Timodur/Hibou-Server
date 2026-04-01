from src.modules.vision.streaming.video_streaming_publisher import VideoStreamingPublisher
from src.modules.vision.streaming.video_source import VideoSource
from .detection_recorder import DetectionRecording
from ultralytics.engine.results import Results
from .models.yolo_model import YOLOModel
from collections import deque
from pathlib import Path
import threading
import time

from src.logger import CustomLogger

logger = CustomLogger("vision").get_logger()
import cv2


class DroneDetection:
    """Handles drone detection using a YOLOv8 model with optional background threading."""

    def __init__(
        self,
        model_type: str = "yolo",
        model_path: Path = "yolov8n.pt",
        enable: bool = True,
        enable_recording: bool = False,
        save_fp: Path = Path(),
    ):
        self.enable = enable
        if not self.enable:
            logger.warning("Drone detection disabled.")
            self.model = None
            self.channels = None
        else:
            self.model = YOLOModel(model_path)
            self.channels = self.model.model.yaml.get("channels")
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._stream: VideoSource | None = None
        self._frame_interval = None
        self.fps = None
        self.enable_recording = enable_recording
        self.recording = DetectionRecording(save_fp)

        self.results_queue = deque(maxlen=1)

        self.ipc_publisher = VideoStreamingPublisher()

        logger.info(f"DroneDetection initialized with model: {model_type}")

    def _run_detection(self, display: bool = True):
        """Internal method running detection loop in a thread."""
        if self._stream is None or not self._stream.is_opened():
            logger.error("Invalid stream")
            return

        logger.info("Detection loop started")

        next_frame_time = time.time()
        while not self._stop_event.is_set():
            next_frame_time += self._frame_interval

            ret, frame = self._stream.get_frame()
            if not ret:
                time.sleep(0.005)
                continue

            if self.channels == 1:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = self.model.track(frame)

            if any(len(result.boxes) > 0 for result in results):
                self.results_queue.append(results)

            if self.enable_recording or display:
                annotated_frame = results[0].plot()
                self.ipc_publisher.publish(frame, annotated_frame)

                if self.enable_recording:
                    self.recording.update_frame(annotated_frame)

                if display:
                    cv2.imshow("Tracking", annotated_frame)
                    cv2.waitKey(1)

            sleep_time = next_frame_time - time.time()

            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                # we're behind schedule -> reset timing
                next_frame_time = time.time()

        logger.info("Detection loop ended")
        cv2.destroyAllWindows()

    def get_last_results(self) -> list[Results] | None:
        """Retrieves the first available result."""
        try:
            return self.results_queue.pop()
        except IndexError:
            return None

    def is_empty(self) -> bool:
        """Tells if the results list is empty."""
        return len(self.results_queue) == 0

    def start(self, stream: VideoSource, display: bool = True):
        """Start detection in a background thread."""
        if not self.model or not self.enable:
            logger.warning("Model not loaded or detection disabled.")
            return
        if self._thread and self._thread.is_alive():
            logger.warning("Detection already running.")
            return

        self._stream = stream

        self.fps = self._stream.get_fps()
        self._frame_interval = (
            1.0 / self.fps if self.fps > 0 else 0.03
        )  # fallback ~30 FPS

        if self.enable_recording:
            self.recording.start_recording()

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_detection, args=(display,), daemon=True
        )
        self._thread.start()

    def stop(self):
        """Stop the detection thread."""
        if not self.enable:
            return
        self._stop_event.set()
        self.recording.stop_recording()
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None

    def is_running(self) -> bool:
        """Check if detection thread is active."""
        return self._thread is not None and self._thread.is_alive()

    def __del__(self):
        """Ensure resources are cleaned up on destruction."""
        if not self.enable:
            return
        self.stop()
        cv2.destroyAllWindows()
