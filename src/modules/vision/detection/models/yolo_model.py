from ultralytics import YOLO
from pathlib import Path

import torch

from src.logger import CustomLogger

logger = CustomLogger("vision").get_logger()


class YOLOModel:
    """General YOLO model wrapper for v8, v11, etc."""

    def __init__(self, model_path: Path):
        try:
            self.model = YOLO(model_path)
            logger.info(f"Loaded YOLO model: {model_path}")

            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(device)
            logger.info(f"YOLO Using device: {device}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise

    def track(self, frame):
        """Run tracking/detection on a frame."""
        return self.model.track(
            frame,
            persist=True,
            conf=0.55,
            iou=0.3,
            verbose=False,
        )

    def predict(self, frame):
        return self.model.predict(frame, conf=0.3, iou=0.5, verbose=False)
