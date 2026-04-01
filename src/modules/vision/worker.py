import time
import datetime
from pathlib import Path

from src.logger import CustomLogger

logger = CustomLogger("vision").get_logger()
import os
from src.modules.vision.detection import DroneDetection
from src.modules.vision.devices.ptz_controller import PTZController
from src.modules.vision.devices.vendors.hikvision.ds_2dy9250iax_a import DS2DY9250IAXA
from src.modules.vision.tracking.ibvs_tracker import IBVSTracker
from src.settings import SETTINGS
from src.helpers.system_status import SystemStatusUpdater


class VisionWorker:
    def __init__(self, dt: datetime.datetime):
        logger.info(f"Started Vision Worker | PID: {os.getpid()}")
        self.recs_folder_name = os.path.join(
            SETTINGS.REC_SAVE_FP,
            f"{dt.strftime('%Y-%m-%d_%H:%M:%S')}",
        )

        Path(self.recs_folder_name).mkdir(parents=True, exist_ok=True)

        self.drone_detector = DroneDetection(
            enable=SETTINGS.AI_CV_ENABLE,
            model_type=SETTINGS.AI_CV_MODEL_TYPE,
            model_path=Path("assets/computer_vision_models/", SETTINGS.AI_CV_MODEL),
            enable_recording=SETTINGS.REC_VIDEO_ENABLE,
            save_fp=Path(self.recs_folder_name, "main_camera_box.avi"),
        )

        PTZController(
            "main_camera",
            DS2DY9250IAXA,
            host=SETTINGS.PTZ_HOST,
            username=SETTINGS.PTZ_USERNAME,
            password=SETTINGS.PTZ_PASSWORD,
            start_azimuth=SETTINGS.PTZ_START_AZIMUTH,
            end_azimuth=SETTINGS.PTZ_END_AZIMUTH,
            rtsp_port=SETTINGS.PTZ_RTSP_PORT,
            video_channel=SETTINGS.PTZ_VIDEO_CHANNEL,
        )
        self.stream = PTZController("main_camera").get_video_stream()
        self.tracker = IBVSTracker()

        self.system_status_updater = SystemStatusUpdater(
            system_name="worker:vision",
        )

        self.start_time = time.time()

        try:
            self.run()
        except KeyboardInterrupt:
            logger.critical("\nStopping Vision Worker...")
        finally:
            self.stream.stop_recording()
            self.drone_detector.stop()
            PTZController.remove()

    def run(self):
        PTZController("main_camera").set_absolute_ptz_position(
            pan=180,
            tilt=10,
            zoom=1,
        )

        if SETTINGS.REC_VIDEO_ENABLE:
            self.stream.start_recording(self.recs_folder_name)

        self.drone_detector.start(self.stream, display=SETTINGS.CV_VIDEO_PLAYBACK)
        while True:
            time.sleep(0.01)

            # Drone detection logic
            results = self.drone_detector.get_last_results()
            best_box = None
            best_conf = 0.0

            if results is not None:
                for result in results:
                    if result is None:
                        continue

                    boxes = result.boxes
                    for box, cls_id, conf in zip(boxes.xyxyn, boxes.cls, boxes.conf):
                        class_id = int(cls_id.item())

                        # Only drone class (assuming 0 = drone)
                        if class_id != 0:
                            continue

                        confidence = float(conf.item())
                        if confidence > best_conf:
                            best_conf = confidence
                            best_box = box

            if time.time() - self.start_time > 5:
                controls = self.tracker.update(best_box)

                if controls is not None:
                    pan_vel, tilt_vel, zoom_vel = controls

                    if pan_vel == 0 and tilt_vel == 0:
                        current_pan_vel, current_tilt_vel = PTZController(
                            "main_camera"
                        ).get_speed()
                        if current_pan_vel != 0 or current_tilt_vel != 0:
                            PTZController("main_camera").stop_continuous()
                    else:
                        PTZController("main_camera").start_continuous(
                            pan_speed=-pan_vel,
                            tilt_speed=tilt_vel,
                            clamp=True,
                        )

            self.system_status_updater.update()
