import threading
import queue
import zmq
from src.settings import SETTINGS
from src.logger import CustomLogger

logger = CustomLogger("vision").get_logger()

"""
Video Streaming Server
This file implements a server that streams video from frames Queue to a rtp client.
"""

class VideoStreamingPublisher:
    def __init__(self):
        self.ctx = zmq.Context.instance()
        self.pub_socket_raw = self.ctx.socket(zmq.PUB)
        self.pub_socket_raw.bind(f"tcp://*:{SETTINGS.IPC_VIDEO_STREAMING_RAW_PORT}")
        self.pub_socket_annotated = self.ctx.socket(zmq.PUB)
        self.pub_socket_annotated.bind(f"tcp://*:{SETTINGS.IPC_VIDEO_STREAMING_ANNOTATED_PORT}")
        logger.info(f"Video Streaming Publisher started on port {SETTINGS.IPC_VIDEO_STREAMING_RAW_PORT} and {SETTINGS.IPC_VIDEO_STREAMING_ANNOTATED_PORT}")

    def publish(self, raw_frame, annotated_frame):
        print("Publishing video streaming", raw_frame.shape, annotated_frame.shape)
        self.pub_socket_raw.send_pyobj(raw_frame)
        self.pub_socket_annotated.send_pyobj(annotated_frame)

    def close(self):
        self.pub_socket_raw.close()
        self.pub_socket_annotated.close()
        self.ctx.term()