import logging
import threading
import time
from typing import Callable

import zmq
from src.helpers.ipc.base_ipc import BaseIPC
from src.settings import SETTINGS

_PUB_POST_CONNECT_DELAY_S = 0.15

from src.helpers.decorators import singleton

@singleton
class ZmqHandler(BaseIPC):
    # Publisher binds here
    XSUB_PORT = SETTINGS.IPC_PROXY_XSUB_PORT
    # Subscriber connects there
    XPUB_PORT = SETTINGS.IPC_PROXY_XPUB_PORT

    def __init__(self):
        self.context = zmq.Context()
        self.stop_event = threading.Event()
        self.sub_threads = []
        self._pub_socket_local = threading.local()
        self.listeners: dict[str, list[Callable]] = {}

        threading.Thread(target=self._listen, daemon=True).start()
    
    def lifespan(self):
        """
        The proxy will forward messages from publishers to subscribers. Publishers connect to the XSUB socket, and subscribers connect to the XPUB socket.
        """
        def proxy():
            xsub = self.context.socket(zmq.XSUB)
            xpub = self.context.socket(zmq.XPUB)
            xsub.bind(f"tcp://*:{self.XSUB_PORT}")
            xpub.bind(f"tcp://*:{self.XPUB_PORT}")
            logging.info("ZMQ Proxy started - XSUB on port %d, XPUB on port %d", self.XSUB_PORT, self.XPUB_PORT)
            zmq.proxy(xsub, xpub)

        threading.Thread(target=proxy, daemon=True).start()



    def _get_pub_socket(self) -> zmq.Socket:
        """
        Return (or lazily create) a PUB socket for the calling thread.
        ZMQ sockets must not be shared across threads.
        """
        if not hasattr(self._pub_socket_local, "socket"):
            sock = self.context.socket(zmq.PUB)
            sock.connect(f"tcp://127.0.0.1:{self.XSUB_PORT}")
            time.sleep(_PUB_POST_CONNECT_DELAY_S)
            self._pub_socket_local.socket = sock
            logging.debug(f"New PUB socket created for thread {threading.current_thread().name}")
        return self._pub_socket_local.socket

    def publish(self, topic: str, message: str):
        """
        Any thread can call this — each gets its own PUB socket transparently.
        """
        sock = self._get_pub_socket()
        sock.send_string(f"{topic} {message}")

    def _listen(self):
        sub_socket = self.context.socket(zmq.SUB)
        sub_socket.connect(f"tcp://127.0.0.1:{self.XPUB_PORT}")
        sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        sub_socket.setsockopt(zmq.RCVTIMEO, 250)
        try:
            while not self.stop_event.is_set():
                try:
                    raw = sub_socket.recv()
                except zmq.Again:
                    continue
                parts = raw.split(maxsplit=1)
                if len(parts) != 2:
                    logging.warning("Malformed IPC message: %r", raw)
                    continue
                topic_b, message_b = parts
                for callback in self.listeners.get(topic_b.decode(), []):
                    callback(topic_b.decode(), message_b.decode())
        finally:
            sub_socket.close()

    def subscribe(self, topic: str, callback: callable):
        self.listeners.setdefault(topic, []).append(callback)
    
    def close(self):
        """Clean up ZMQ sockets and context, shutdown threads."""
        self.stop_event.set()

        for thread in self.sub_threads:
            thread.join(timeout=2.0)

