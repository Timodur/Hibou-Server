from dataclasses import dataclass
from dotenv import load_dotenv

import logging
import shutil
import os


current_file_path = os.path.abspath(__file__)
script_dir = os.path.dirname(current_file_path)
project_root = os.path.abspath(os.path.join(script_dir, ".."))

# Paths
source_file = os.path.join(project_root, ".env.example")
target_file = os.path.join(project_root, ".env")

# Copy .env if it does not exist
if not os.path.exists(target_file) and os.path.exists(source_file):
    shutil.copy2(source_file, target_file)
    logging.info(f"Copied {source_file} → {target_file}")

if not load_dotenv():
    raise FileNotFoundError("Failed to load .env file.")


@dataclass
class Settings:
    AUDIO_ANGLE_COVERAGE: int
    AUDIO_CHUNK_DURATION: int
    AUDIO_STREAM_LATENCY: int
    AUDIO_REC_HZ: int
    AUDIO_VOLUME: float

    REC_AUDIO_ENABLE: bool
    REC_VIDEO_ENABLE: bool
    REC_VIDEO_ON_DETECTION: bool
    REC_SAVE_FP: str

    DEVICES_CONFIG_PATH: str
    STATIONARY: bool

    LOG_PATH: str
    LOG_CONF_PATH: str
    LOG_LEVEL: str

    PTZ_USERNAME: str
    PTZ_PASSWORD: str
    PTZ_HOST: str
    PTZ_VIDEO_CHANNEL: int
    PTZ_RTSP_PORT: int
    PTZ_START_AZIMUTH: int
    PTZ_END_AZIMUTH: int

    INFER_FROM_FOLDER: str
    AI_NUM_PROC: int
    AI_DEVICE: str
    AI_CV_MODEL: str
    AI_CV_MODEL_TYPE: str
    AI_CV_ENABLE: bool
    AI_MODELS_FOLDER: str

    IPC_PROXY_XSUB_PORT: int
    IPC_PROXY_XPUB_PORT: int
    IPC_VIDEO_STREAMING_RAW_PORT: int
    IPC_VIDEO_STREAMING_ANNOTATED_PORT: int
    IPC_ACOUSTIC_ANGLE_TOPIC: str
    IPC_ACOUSTIC_DETECTION_TOPIC: str
    IPC_VISION_DECISION_TOPIC: str
    IPC_VISION_ANGLE_TOPIC: str
    IPC_VISION_DETECTION_TOPIC: str

    AUDIO_PLAYBACK: bool = False  # Only for debug purposes
    AUDIO_ENERGY_SPECTRUM: bool = False  # Only for debug purposes
    AUDIO_STFT_SPECTRUM: bool = False  # Only for debug purposes
    AUDIO_RADAR: bool = False  # Only for debug purposes
    CV_VIDEO_PLAYBACK: bool = False  # Only for debug purposes

def parse_list(value: str):
    """Split a comma-separated string and strip whitespace."""
    return [v.strip() for v in value.split(",") if v.strip()]


def parse_bool(value: str) -> bool:
    """Parse a boolean from string (True/False, yes/no)."""
    return str(value).strip().lower() in ("true", "1", "yes")


try:
    if Settings.CV_VIDEO_PLAYBACK and (
        Settings.AUDIO_RADAR or Settings.AUDIO_ENERGY_SPECTRUM
    ):
        logging.warning(
            "Both CV video and audio visualization are enabled. Disabling CV video."
        )
        Settings.CV_VIDEO_PLAYBACK = False

    SETTINGS = Settings(
        REC_AUDIO_ENABLE=parse_bool(os.getenv("REC_AUDIO_ENABLE")),
        REC_VIDEO_ENABLE=parse_bool(os.getenv("REC_VIDEO_ENABLE")),
        REC_VIDEO_ON_DETECTION=parse_bool(os.getenv("REC_VIDEO_ENABLE")),
        REC_SAVE_FP=os.getenv("REC_SAVE_FP"),
        AUDIO_CHUNK_DURATION=int(os.getenv("AUDIO_CHUNK_DURATION")) * 10**6,  # ns
        AUDIO_REC_HZ=int(os.getenv("AUDIO_REC_HZ")),
        AUDIO_STREAM_LATENCY=int(os.getenv("AUDIO_STREAM_LATENCY")),
        DEVICES_CONFIG_PATH=os.getenv("DEVICES_CONFIG_PATH"),
        STATIONARY=parse_bool(os.getenv("STATIONARY")),
        AI_DEVICE=os.getenv("AI_DEVICE"),
        LOG_PATH=os.getenv("LOG_PATH"),
        LOG_CONF_PATH=os.getenv("LOG_CONF_PATH"),
        LOG_LEVEL=os.getenv("LOG_LEVEL"),
        INFER_FROM_FOLDER=os.getenv("INFER_FROM_FOLDER"),
        AUDIO_VOLUME=float(os.getenv("AUDIO_VOLUME")),
        PTZ_USERNAME=os.getenv("PTZ_USERNAME"),
        PTZ_PASSWORD=os.getenv("PTZ_PASSWORD"),
        PTZ_HOST=os.getenv("PTZ_HOST"),
        PTZ_VIDEO_CHANNEL=int(os.getenv("PTZ_VIDEO_CHANNEL")),
        PTZ_RTSP_PORT=int(os.getenv("PTZ_RTSP_PORT")),
        AUDIO_ANGLE_COVERAGE=int(os.getenv("AUDIO_ANGLE_COVERAGE")),
        PTZ_START_AZIMUTH=int(os.getenv("PTZ_START_AZIMUTH")),
        PTZ_END_AZIMUTH=int(os.getenv("PTZ_END_AZIMUTH")),
        AI_NUM_PROC=int(os.getenv("AI_NUM_PROC")),
        AI_CV_MODEL=os.getenv("AI_CV_MODEL"),
        AI_CV_MODEL_TYPE=os.getenv("AI_CV_MODEL_TYPE"),
        AI_MODELS_FOLDER=os.getenv("AI_MODELS_FOLDER"),
        AI_CV_ENABLE=parse_bool(os.getenv("AI_CV_ENABLE")),
        IPC_ACOUSTIC_ANGLE_TOPIC=os.getenv("IPC_ACOUSTIC_ANGLE_TOPIC"),
        IPC_ACOUSTIC_DETECTION_TOPIC=os.getenv("IPC_ACOUSTIC_DETECTION_TOPIC"),
        IPC_VISION_DECISION_TOPIC=os.getenv("IPC_VISION_DECISION_TOPIC"),
        IPC_VISION_DETECTION_TOPIC=os.getenv("IPC_VISION_DETECTION_TOPIC"),
        IPC_VISION_ANGLE_TOPIC=os.getenv("IPC_VISION_ANGLE_TOPIC"),
        IPC_PROXY_XSUB_PORT=int(os.getenv("IPC_PROXY_XSUB_PORT")),
        IPC_PROXY_XPUB_PORT=int(os.getenv("IPC_PROXY_XPUB_PORT")),
        IPC_VIDEO_STREAMING_RAW_PORT=int(os.getenv("IPC_VIDEO_STREAMING_RAW_PORT")),
        IPC_VIDEO_STREAMING_ANNOTATED_PORT=int(os.getenv("IPC_VIDEO_STREAMING_ANNOTATED_PORT")),
    )


except TypeError as e:
    raise ValueError(f"Invalid value in .env: {e}. Please check the .env file.")
