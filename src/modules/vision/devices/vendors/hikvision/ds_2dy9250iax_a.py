from src.helpers.ipc.base_ipc import get_ipc_handler
from src.modules.vision.devices.vendors.base_vendor import BaseVendor, PTZAxisRange
from src.modules.vision.streaming.rtsp_stream import RtspSource
from src.helpers.decorators import Range
from src.helpers.math import map_range
from hikvisionapi import Client

import threading
import time

from src.logger import CustomLogger
from src.settings import SETTINGS
from src.helpers.system_status import SystemStatusUpdater

logger = CustomLogger("vision").get_logger()


class DS2DY9250IAXA(BaseVendor):
    """
    Singleton PTZ camera controller.
    Provides an interface to control a PTZ (Pan-Tilt-Zoom) camera, ensuring only one
    instance of the class exists across the entire program.
    Datasheet: https://www.hikvision.com/content/dam/hikvision/products/S000000001/S000000002/S000000011/S000000013/OFR000059/M000005882/Data_Sheet/Datasheet-of-DS-2DY9250IAX-A-D_20190816.pdf
    """

    _instance = None
    _lock = threading.Lock()  # For thread-safe singleton creation

    _PAN_RANGE = PTZAxisRange(logical=BaseVendor.PAN_RANGE, hardware=Range(1, 3600))
    _TILT_RANGE = PTZAxisRange(logical=BaseVendor.TILT_RANGE, hardware=Range(-900, 400))
    _ZOOM_RANGE = PTZAxisRange(logical=BaseVendor.ZOOM_RANGE, hardware=Range(10, 67))
    _SPEED_RANGE = PTZAxisRange(
        logical=BaseVendor.SPEED_RANGE, hardware=Range(-100, 100)
    )

    # Camera motors channel
    CHANNEL_ID = 1

    # XML content type
    XML_CONTENT_TYPE = "application/xml"

    def __new__(cls, *args, **kwargs):
        """Ensure only one PTZ instance is created."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        name: str,
        host: str,
        username: str,
        password: str,
        start_azimuth: int = None,
        end_azimuth: int = None,
        rtsp_port: int = 554,
        video_channel: int = 1,
    ):
        # Prevent reinitialization if already initialized
        if hasattr(self, "_initialized") and self._initialized:
            return

        if not host or not username or not password:
            logger.warning(
                "No username or password provided for PTZ connection. Skipping initialization."
            )
            self._initialized = False
            return

        self._initialized = True  # Flag so __init__ runs only once

        self._name = name
        self._host = host
        self._username = username
        self._password = password
        self._client = None
        self._start_azimuth = start_azimuth
        self._end_azimuth = end_azimuth

        self._current_pan = 0
        self._current_tilt = 0
        self._current_zoom = 1

        self._current_elevation = 0
        self._current_azimuth = 0
        self._current_zoom_hw = 1

        self._current_pan_speed = 0  # Store logical speed
        self._current_tilt_speed = 0

        self._last_angle_update_time = 0

        self._status = None

        self.rtsp_url = f"rtsp://{username}:{password}@{host}:{rtsp_port}/Streaming/Channels/10{video_channel}/"
        self.rtsp_stream = RtspSource(self.rtsp_url, self._name)
        self.rtsp_stream.start()

        if not self.rtsp_stream.is_opened():
            logger.error("Cannot open RTSP stream. Check the URL or credentials.")
            logger.error(
                f"RTSP URL: rtsp://{username}:XXX@{host}:{rtsp_port}/Streaming/Channels/10{video_channel}/"
            )
        else:
            logger.info("RTSP stream opened")

        try:
            self._client = Client(
                f"http://{self._host}", self._username, self._password
            )
            logger.info(f"Connected to PTZ camera at {self._host}")
        except Exception as e:
            logger.error(f"Failed to connect to PTZ camera at {self._host}: {e}")

        self.system_status_updater = SystemStatusUpdater(
            system_name="ptz_camera",
        )
        threading.Thread(target=self._update_status_loop, daemon=True).start()

    @staticmethod
    def _build_absolute_position_xml(
        elevation: float, azimuth: float, zoom: float
    ) -> str:
        """Build XML command for absolute positioning."""
        return f"""
        <PTZData>
            <AbsoluteHigh>
                <elevation>{elevation}</elevation>
                <azimuth>{azimuth}</azimuth>
                <absoluteZoom>{zoom}</absoluteZoom>
            </AbsoluteHigh>
        </PTZData>
        """.strip()

    @staticmethod
    def _build_continuous_movement_xml(pan: int, tilt: int) -> str:
        """Build XML command for continuous movement."""
        return f"<PTZData><pan>{pan}</pan><tilt>{tilt}</tilt></PTZData>"

    @staticmethod
    def _build_3d_position(start_x: int, start_y: int, end_x: int, end_y: int) -> str:
        return f"""
        <position3D>
            <StartPoint>
                <positionX>{start_x}</positionX>
                <positionY>{start_y}</positionY>
            </StartPoint>
            <EndPoint>
                <positionX>{end_x}</positionX>
                <positionY>{end_y}</positionY>
            </EndPoint>
        </position3D>
        """.strip()

    def _calculate_pan_tilt(self, pan_speed: int, tilt_speed: int) -> tuple[int, int]:
        """Calculate pan and tilt values based on axis and direction."""
        pan_speed = map_range(
            pan_speed,
            self._SPEED_RANGE.logical.min,
            self._SPEED_RANGE.logical.max,
            self._SPEED_RANGE.hardware.min,
            self._SPEED_RANGE.hardware.max,
        )

        tilt_speed = map_range(
            tilt_speed,
            self._SPEED_RANGE.logical.min,
            self._SPEED_RANGE.logical.max,
            self._SPEED_RANGE.hardware.min,
            self._SPEED_RANGE.hardware.max,
        )

        if abs(pan_speed) == 10:
            pan_speed *= 1.5  # 15 is the minimum speed to turn camera
        if abs(tilt_speed) == 10:
            tilt_speed *= 1.5
        return int(pan_speed), int(tilt_speed)

    def _convert_pan_to_azimuth(self, pan: float) -> int:
        return int(
            map_range(
                pan,
                self._PAN_RANGE.logical.min,
                self._PAN_RANGE.logical.max,
                self._PAN_RANGE.hardware.min,
                self._PAN_RANGE.hardware.max,
            )
        )

    def _azimuth_to_pan(self, elevation: int) -> float:
        return map_range(
            elevation,
            self._PAN_RANGE.hardware.min,
            self._PAN_RANGE.hardware.max,
            self._PAN_RANGE.logical.min,
            self._PAN_RANGE.logical.max,
        )

    def _convert_tilt_to_elevation(self, tilt: float) -> int:
        return int(
            map_range(
                tilt,
                self._TILT_RANGE.logical.min,
                self._TILT_RANGE.logical.max,
                self._TILT_RANGE.hardware.min,
                self._TILT_RANGE.hardware.max,
            )
        )

    def _elevation_to_tilt(self, azimuth: int) -> float:
        return map_range(
            azimuth,
            self._TILT_RANGE.hardware.min,
            self._TILT_RANGE.hardware.max,
            self._TILT_RANGE.logical.min,
            self._TILT_RANGE.logical.max,
        )

    def _convert_zoom_to_hw_zoom(self, zoom: int) -> int:
        return int(
            map_range(
                zoom,
                self._ZOOM_RANGE.logical.min,
                self._ZOOM_RANGE.logical.max,
                self._ZOOM_RANGE.hardware.min,
                self._ZOOM_RANGE.hardware.max,
            )
        )

    def _hw_zoom_to_zoom(self, zoom: int) -> int:
        return int(
            map_range(
                zoom,
                self._ZOOM_RANGE.hardware.min,
                self._ZOOM_RANGE.hardware.max,
                self._ZOOM_RANGE.logical.min,
                self._ZOOM_RANGE.logical.max,
            )
        )

    def _convert_logical_to_hardware(self, pan, tilt, zoom):
        return (
            self._convert_pan_to_azimuth(pan),
            self._convert_tilt_to_elevation(tilt),
            self._convert_zoom_to_hw_zoom(zoom),
        )

    def _convert_hardware_to_logical(self, elevation, azimuth, zoom):
        return (
            self._azimuth_to_pan(elevation),
            self._elevation_to_tilt(azimuth),
            self._hw_zoom_to_zoom(zoom),
        )

    def _set_absolute_ptz_position(
        self,
        pan: float | None = None,
        tilt: float | None = None,
        zoom: int | None = None,
    ) -> bool:
        """
        Command the camera to move to an absolute pan/tilt/zoom position.

        This method issues an absolute PTZ movement request to the camera using
        logical pan, tilt, and zoom coordinates. Any axis set to ``None`` will
        retain its current value. Before sending the command, the method enforces
        both an angular tolerance threshold and a minimum time interval between
        updates to avoid redundant or excessive PTZ commands.

        Logical PTZ values are converted to hardware-specific coordinates prior
        to transmission. On successful execution, both logical and hardware PTZ
        state caches are updated to reflect the new target position.

        The command will not be sent if:
        - The controller is not initialized
        - The requested movement is within the configured angle tolerance
        - The minimum time interval since the last update has not elapsed

        Parameters
        ----------
        pan : float | None, optional
            Target pan angle in logical degrees. If ``None``, the current pan
            position is used.
        tilt : float | None, optional
            Target tilt angle in logical degrees. If ``None``, the current tilt
            position is used.
        zoom : int | None, optional
            Target zoom level in logical units. If ``None``, the current zoom
            level is used.

        Returns
        -------
        bool
            ``True`` if the PTZ command was successfully sent and the internal state
            was updated, ``False`` otherwise.
        """

        if not self.is_initialized():
            return False

        if pan is None:
            pan = self._current_pan

        if tilt is None:
            tilt = self._current_tilt

        if zoom is None:
            zoom = self._current_zoom

        delta_pan = abs(pan - self._current_pan)
        delta_tilt = abs(tilt - self._current_tilt)
        delta_zoom = abs(zoom - self._current_zoom)

        # Check tolerance and minimal time interval
        movement_small = (
            delta_pan < self.ANGLE_TOLERANCE
            and delta_tilt < self.ANGLE_TOLERANCE
            and delta_zoom == 0
        )

        now = time.time()
        dt = now - self._last_angle_update_time

        if movement_small or dt < BaseVendor.RATE_LIMIT_INTERVAL:
            return False

        self._last_angle_update_time = now

        azimuth, elevation, zoom_hw = self._convert_logical_to_hardware(pan, tilt, zoom)

        xml_command = self._build_absolute_position_xml(elevation, azimuth, zoom_hw)

        try:
            self._client.PTZCtrl.channels[self.CHANNEL_ID].absolute(
                method="put",
                data=xml_command,
                headers={"Content-Type": self.XML_CONTENT_TYPE},
            )

            self._current_pan = pan
            self._current_tilt = tilt
            self._current_zoom = zoom

            self._current_azimuth = azimuth
            self._current_elevation = elevation
            self._current_zoom_hw = zoom_hw

            return True

        except Exception as e:
            logger.error(f"Error sending PTZ absolute command: {e}")
            return False

    def _set_relative_ptz_position(
        self,
        pan: float | None = None,
        tilt: float | None = None,
        zoom: int | None = None,
    ) -> bool:
        """
        Move the camera by a relative pan, tilt, and zoom offset.

        Parameters
        ----------
        pan : float | None
            Pan offset in logical degrees. If ``None``, pan is unchanged.
        tilt : float | None
            Tilt offset in logical degrees. If ``None``, the tilt is unchanged.
        zoom : int | None
            Zoom offset in logical units. If ``None``, the zoom is unchanged.
        """
        if pan is not None:
            pan = self._current_pan + pan
        else:
            pan = self._current_pan

        if tilt is not None:
            tilt = self._current_tilt + tilt
        else:
            tilt = self._current_tilt

        if zoom is not None:
            zoom = self._current_zoom + zoom
        else:
            zoom = self._current_zoom

        return self._set_absolute_ptz_position(pan, tilt, zoom)

    def _send_continuous_ptz_command(self, pan: int, tilt: int) -> bool:
        """
        Sends a continuous pan-tilt-zoom (PTZ) command to the PTZ client.

        This method constructs an XML payload for the PTZ command, specifying the pan
        and tilt values, and attempts to send the command via the PTZ client. If the
        client is not initialized or an error occurs during transmission, the method
        logs the error and returns False.

        Args:
            pan: The pan (horizontal) value to send in the PTZ command.
            tilt: The tilt (vertical) value to send in the PTZ command.

        Returns:
            bool: True if the command was sent successfully, otherwise False.
        """
        if not self.is_initialized():
            return False

        xml_command = self._build_continuous_movement_xml(pan, tilt)

        try:
            self._client.PTZCtrl.channels[self.CHANNEL_ID].continuous(
                method="put",
                data=xml_command,
                headers={"Content-Type": self.XML_CONTENT_TYPE},
            )

            return True

        except Exception as e:
            logger.error(f"Error sending PTZ continuous command: {e}")
            return False

    def _start_continuous(self, pan_speed: int, tilt_speed: int) -> bool:
        """
        Start or update a continuous Pan-Tilt (PT) movement.

        This method initiates or updates a continuous pan and tilt movement
        using the specified speeds. If the camera is not initialized, the
        movement is not started and the method returns False.

        If the requested pan and tilt speeds are already active, no new
        movement command is issued and the method returns True.

        Args:
            pan_speed (int): Horizontal (pan) movement speed. The sign
                determines the direction.
            tilt_speed (int): Vertical (tilt) movement speed. The sign
                determines the direction.

        Returns:
            bool: True if the movement is active or successfully started;
            False otherwise.
        """

        if not self.is_initialized():
            return False

        if (
            pan_speed == self._current_pan_speed
            and tilt_speed == self._current_tilt_speed
        ):
            return True

        pan, tilt = self._calculate_pan_tilt(pan_speed, tilt_speed)

        success = self._send_continuous_ptz_command(pan, tilt)

        self._current_pan_speed = pan_speed
        self._current_tilt_speed = tilt_speed

        if not success:
            logger.debug("Failed to start continuous PTZ movement.")
        return success

    def _update_status(self) -> None:
        """Update internal status from PTZ camera."""
        if not self.is_initialized():
            return

        try:
            status = self._client.PTZCtrl.channels[self.CHANNEL_ID].status(method="get")
            absolute_high = status["PTZStatus"]["AbsoluteHigh"]

            self._current_azimuth = int(absolute_high["azimuth"])
            self._current_elevation = int(absolute_high["elevation"])
            self._current_zoom_hw = int(absolute_high["absoluteZoom"])

            self._current_pan, self._current_tilt, self._current_zoom = (
                self._convert_hardware_to_logical(
                    self._current_azimuth,
                    self._current_elevation,
                    self._current_zoom_hw,
                )
            )
            self._status = status
        except Exception as e:
            logger.error(f"Failed to get PTZ status: {e}")

    def stop_continuous(self) -> None:
        """
        Stops any ongoing continuous PTZ (Pan-Tilt-Zoom) commands.

        This method sends a stop signal to terminate any currently running
        continuous PTZ commands. It sets the pan and tilt velocity to zero, ensuring
        that the movement of the camera stops immediately.

        Raises:
            Exception: If there is an issue while sending the stop command.
        """
        self._send_continuous_ptz_command(0, 0)
        self._current_pan_speed = 0
        self._current_tilt_speed = 0
        self._update_status()

    def is_initialized(self) -> bool:
        return self._initialized and self._client is not None

    def get_azimuth(self) -> int:
        """
        Gets the azimuth value from the PTZ (Pan-Tilt-Zoom) status.

        The method retrieves the PTZ status and extracts the azimuth value from the
        AbsoluteHigh component. The azimuth represents the horizontal angle of the
        camera's orientation in the PTZ system.

        Returns:
            int: The azimuth value as an integer.
        """
        return self._current_azimuth

    def get_elevation(self) -> int:
        """
        Gets the elevation value from the PTZ (Pan-Tilt-Zoom) status.

        The method retrieves PTZ status data and extracts the 'elevation' value
        from the absolute high-position information.

        Returns:
            int: The elevation value extracted from the PTZ status.

        """
        return self._current_elevation

    @classmethod
    def get_instance(cls) -> "DS2DY9250IAXA":
        """Return the singleton PTZ instance, if already created."""
        if cls._instance is None:
            raise RuntimeError("PTZ has not been initialized yet. Call PTZ(...) first.")
        return cls._instance

    def get_status(self, force_update: bool = False) -> dict:
        """
        Retrieve current PTZ status and parse useful values.
        """
        if force_update:
            self._update_status()
        return self._status

    def set_3d_position(self, start_x, start_y, end_x, end_y) -> bool:
        if not self.is_initialized():
            return False

        xml_command = self._build_3d_position(start_x, start_y, end_x, end_y)

        try:
            self._client.PTZCtrl.channels[self.CHANNEL_ID].position3D(
                method="put",
                data=xml_command,
                headers={"Content-Type": self.XML_CONTENT_TYPE},
            )

            return True

        except Exception as e:
            logger.error(f"Error sending PTZ continuous command: {e}")
            return False

    def get_zoom(self) -> int:
        """
        Gets the zoom level of the PTZ (Pan-Tilt-Zoom) camera.

        This method retrieves the absolute zoom level from the camera's current
        status data. The zoom level represents the current zoom factor of the
        camera, as an integer value.

        Returns:
            int: The current absolute zoom level of the camera.
        """
        return self._current_zoom

    def get_speed(self) -> tuple[int, int]:
        """
        Return the current pan and tilt speed.
        """
        return self._current_pan_speed, self._current_tilt_speed

    def get_video_stream(self):
        if not self._initialized:
            return None
        return self.rtsp_stream

    def release_stream(self):
        """Safely release the RTSP stream."""
        if hasattr(self, "rtsp_stream") and self.rtsp_stream is not None:
            self.rtsp_stream.stop()
            self.rtsp_stream = None
            logger.info("RTSP stream released.")

    def _update_status_loop(self):
        """
        Continuously update PTZ status in a separate thread.
        Send on the IPC status informations
        """
        ipc = get_ipc_handler()
        while True:
            self._update_status()
            if status := self.get_status():
                if "PTZStatus" in status and "AbsoluteHigh" in status["PTZStatus"]:
                    azimuth = status["PTZStatus"]["AbsoluteHigh"]["azimuth"]
                    elevation = status["PTZStatus"]["AbsoluteHigh"]["elevation"]
                    ipc.publish(SETTINGS.IPC_VISION_ANGLE_TOPIC, f"{azimuth},{elevation}")
                    self.system_status_updater.update()
            time.sleep(0.25)