import datetime
import time
import os
from collections import deque

from src.arguments import args
from src.helpers.decorators import SingletonMeta
from src.logger import CustomLogger
from src.modules.audio.localization.energy import compute_energy
from src.modules.audio.streaming.debug.channel_spectrogram import (
    ChannelTimeSpectrogram,
    StftSpectrogram,
)
from src.modules.audio.streaming.debug.radar import RadarPlot

logger = CustomLogger("audio").get_logger()
from src.modules.audio.devices.audio_device_controller import ADCControllerManager
from src.modules.audio.dispatcher import AudioDispatcher
from src.modules.audio.streaming import GstChannel
from src.modules.audio.streaming.play import play_sample
from src.modules.audio.streaming.sources.file_source import FileAudioSource
from src.modules.audio.streaming.sources.rtp_source import RTPAudioSource
from src.settings import SETTINGS
from src.helpers.system_status import SystemStatusUpdater


class AudioWorker:
    """
    Main class responsible for managing audio devices, streaming audio data and inferencing
    """

    def __init__(self, dt: datetime.datetime):
        logger.info(f"Started Audio Worker | PID: {os.getpid()}")

        self.initial_dt = dt
        # In charge of managing audio devices, including discovery and control
        self.controller_manager = ADCControllerManager()
        self._load_devices()
        # Get the source, either from a folder or from the network, based on command-line arguments
        self.source = self._get_source()

        self.system_status_updater = SystemStatusUpdater(
            system_name="worker:audio",
        )

        try:
            self.run()
        except KeyboardInterrupt:
            logger.critical("Stopping Audio Worker...")
        finally:
            self.source.stop()
            SingletonMeta.clear()

    def _load_devices(self):
        """
        Load devices from configuration file or auto-discover them on the network.
        """
        if SETTINGS.DEVICES_CONFIG_PATH:
            self.controller_manager.load_devices_from_files(
                SETTINGS.DEVICES_CONFIG_PATH
            )
        else:
            self.controller_manager.auto_discover()

        logger.info(f"{len(self.controller_manager.adc_devices)} devices loaded")
        logger.debug(f"Devices: {self.controller_manager.adc_devices}")
        for dev in self.controller_manager.adc_devices:
            if not dev.is_online():
                logger.warning(f"{dev.name} is offline")

    def _get_source(self):
        """
        Return the audio source based on the command-line arguments.
        """
        # Folder to save recordings
        recs_folder_name = os.path.join(
            SETTINGS.REC_SAVE_FP,
            f"{self.initial_dt.strftime('%Y-%m-%d_%H:%M:%S')}",
        )

        if args.infer_from_folder:
            return FileAudioSource(
                folder_path=args.infer_from_folder,
                channel_prefix=args.channel_prefix,
                channels_count=args.channel_count,
                save_fp=recs_folder_name,
                enable_recording_saves=SETTINGS.REC_AUDIO_ENABLE,
                record_duration=SETTINGS.AUDIO_CHUNK_DURATION,
            )
        else:
            return RTPAudioSource(
                devices=self.controller_manager.adc_devices,
                enable_recording_saves=SETTINGS.REC_AUDIO_ENABLE,
                save_fp=recs_folder_name,
                record_duration=int(SETTINGS.AUDIO_CHUNK_DURATION),
                rec_hz=int(SETTINGS.AUDIO_REC_HZ),
                stream_latency=int(SETTINGS.AUDIO_STREAM_LATENCY),
                channel_prefix=args.channel_prefix,
            )

    def run(self):
        audio = AudioDispatcher()
        self.source.set_callback(audio.process)
        self.source.start()

        nb_channels = sum([x.nb_channels for x in self.controller_manager.adc_devices])
        if nb_channels == 0:
            raise Exception("No ADC devices found! 0 channels available. Exiting.")
        frame_duration_s = SETTINGS.AUDIO_CHUNK_DURATION / 1000

        # Only for debug purposes
        energy_spectrum_plot = (
            ChannelTimeSpectrogram(nb_channels, frame_duration_s)
            if SETTINGS.AUDIO_ENERGY_SPECTRUM
            else None
        )
        # Only for debug purposes
        stft_spectrum_plot = (
            StftSpectrogram(nb_channels, frame_duration_s)
            if SETTINGS.AUDIO_STFT_SPECTRUM
            else None
        )
        # Only for debug purposes
        radar_plot = RadarPlot() if SETTINGS.AUDIO_RADAR else None
        phi_angle = 0

        while True:
            time.sleep(0.01)

            if not audio.is_empty():
                channels = audio.get_last_channels()

                #  Only for debug purposes
                energies = [compute_energy(ch) for ch in channels]
                if SETTINGS.AUDIO_RADAR and radar_plot is not None:
                    radar_plot.set_input(phi_angle, max(energies))

                # Only for debug purposes
                if SETTINGS.AUDIO_ENERGY_SPECTRUM and energy_spectrum_plot is not None:
                    energy_spectrum_plot.set_input(energies)

                # Only for debug purposes
                if SETTINGS.AUDIO_STFT_SPECTRUM and stft_spectrum_plot is not None:
                    stft_spectrum_plot.set_input([channel[0] for channel in channels])


            if stft_spectrum_plot:
                stft_spectrum_plot.update()
            if energy_spectrum_plot:
                energy_spectrum_plot.update()
            if radar_plot:
                radar_plot.update()
                phi_angle = (phi_angle + 5) % 360

            self.system_status_updater.update()