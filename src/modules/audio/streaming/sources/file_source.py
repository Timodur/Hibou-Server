from src.modules.audio.streaming.sources.gstreamer_source import GstreamerSource
from src.settings import SETTINGS
import os
import math


def get_wav_dir_bounds(directory: str):
    # Get all numeric .wav filenames
    numeric_wav_files = []
    for f in os.listdir(directory):
        name, ext = os.path.splitext(f)
        if ext.lower() == ".wav" and name.isdigit():
            numeric_wav_files.append(int(name))

    if numeric_wav_files:
        return min(numeric_wav_files), max(numeric_wav_files)
    else:
        return -math.inf, math.inf


class FileAudioSource(GstreamerSource):

    def __init__(
        self,
        folder_path: str,
        channel_prefix: str,
        channels_count: int,
        enable_recording_saves: bool,
        save_fp: str,
        record_duration: int,
    ):
        """
        Initialize a FileAudioSource to read multichannel audio from WAV files
        using GStreamer pipelines, optionally in real-time playback and with
        recording/saving of processed streams.

        Args:
            folder_path (str): Path to the root folder containing per-channel
                subfolders with WAV files. Each channel folder should be named
                as "{channel_prefix}{channel_index}".

            channel_prefix (str): Prefix used for channel subfolder names
                (e.g., "ch" for folders like "ch0", "ch1", etc.).

            channels_count (int): Number of audio channels to read.

            enable_recording_saves (bool): If True, the pipeline will save
                processed audio to disk using splitmuxsink.

            save_fp (str): Root folder path where recordings will be saved if
                enable_recording_saves is True. Subfolders for each channel
                will be created automatically.

            record_duration (int): Duration of each recording segment in
                nanoseconds for splitmuxsink (used only if
                enable_recording_saves=True).

        Raises:
            FileNotFoundError: If any channel folder or WAV file is missing.
            ValueError: If WAV files have inconsistent sample rates, channel
                counts, or lengths.

        Notes:
            - This class builds a GStreamer pipeline per file per channel.
            - For real-time playback, an identity element with sync=true is
              used.
            - The pipelines mimic the structure of live UDP streams but read
              from disk instead.
        """

        self._folder_path = folder_path
        self._channel_prefix = channel_prefix
        self._channels_count = channels_count
        self._thread = None
        self._continue = False
        self._audio_paths = []
        self._enable_recording_saves = enable_recording_saves
        self._range = (0, -1)

        self._setup()

        pipeline_strings = []
        rec_hz = SETTINGS.AUDIO_REC_HZ

        for ch, directory in enumerate(self._audio_paths):
            """
            It is required to have many filesrc elements, because a multifilesrc is not designed to have multiple EOS
            within, it only glues raw files together, before feeding the next element. There is almost no doc about
            streamsynchronizer too, or how gst-play with --gapless works. Due to this, our only option to have a gapless
            stream, is to have as many sources as files we have, and then concatenate them. concat will handle gap and
            time (pts, dts) fixing itself, and then forward the correctly ordered buffers to the next element.
            """

            gst_pipeline_str = " ".join(
                [
                    f'filesrc location="{directory}/{i}.wav" ! wavparse ! c. '
                    for i in range(self._range[0], self._range[1] + 1)
                ]
            )

            gst_pipeline_str += (
                f"concat name=c ! "
                f"audioconvert ! audioresample ! "
                f"audio/x-raw, format=(string)F32LE, rate=(int){rec_hz}, channels=(int)1 ! "  # All files should contain only one channel.
                f"identity sync=true ! "  # Throttle to real time
                f"tee name=t "
                f"t. ! appsink name=appsink_{ch} async=false "
            )

            if self._enable_recording_saves:
                os.makedirs(f"{save_fp}/{channel_prefix}{ch}", exist_ok=True)

                gst_pipeline_str += (
                    f't. ! splitmuxsink location="{save_fp}/{channel_prefix}{ch}/%d.wav" '
                    f"muxer=wavenc max-size-time={record_duration}"
                )

            pipeline_strings.append(gst_pipeline_str)

        # Our audios are F32LE, so each "element" is of size 4.
        super().__init__(pipeline_strings, int((rec_hz * record_duration / 1e9) * 4))

    def _setup(self):
        """
        Collect file paths for all channels (auto-detect files inside each channel folder).
        Also determines the bounds of the audio files.
        """
        upper_bounds = []
        lower_bounds = []

        self._audio_paths = []
        for ch in range(self._channels_count):
            channel_folder = os.path.join(
                self._folder_path, f"{self._channel_prefix}{ch}"
            )
            if not os.path.isdir(channel_folder):
                raise FileNotFoundError(f"Channel folder missing: {channel_folder}")

            self._audio_paths.append(channel_folder)
            bounds = get_wav_dir_bounds(channel_folder)
            lower_bounds.append(bounds[0])
            upper_bounds.append(bounds[1])

        lower = min(upper_bounds)
        upper = max(lower_bounds)

        upper = -1 if upper == math.inf else upper
        lower = 0 if lower == -math.inf else lower

        self._range = (upper, lower)
