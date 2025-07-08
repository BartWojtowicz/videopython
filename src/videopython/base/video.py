from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Literal, get_args

import numpy as np
from soundpython import Audio

from videopython.base.utils import generate_random_name

ALLOWED_VIDEO_FORMATS = Literal["mp4", "avi", "mov", "mkv", "webm"]


class VideoMetadataError(Exception):
    """Raised when there's an error getting video metadata"""

    pass


@dataclass
class VideoMetadata:
    """Class to store video metadata."""

    height: int
    width: int
    fps: float
    frame_count: int
    total_seconds: float

    def __str__(self) -> str:
        return f"{self.width}x{self.height} @ {self.fps}fps, {self.total_seconds} seconds"

    def __repr__(self) -> str:
        return self.__str__()

    def get_frame_shape(self) -> np.ndarray:
        """Returns frame shape."""
        return np.array((self.height, self.width, 3))

    def get_video_shape(self) -> np.ndarray:
        """Returns video shape."""
        return np.array((self.frame_count, self.height, self.width, 3))

    @staticmethod
    def _run_ffprobe(video_path: str | Path) -> dict:
        """Run ffprobe and return parsed JSON output."""
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate,nb_frames",
            "-show_entries",
            "format=duration",
            "-print_format",
            "json",
            str(video_path),
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return json.loads(result.stdout)
        except subprocess.CalledProcessError as e:
            raise VideoMetadataError(f"FFprobe error: {e.stderr}")
        except json.JSONDecodeError as e:
            raise VideoMetadataError(f"Error parsing FFprobe output: {e}")

    @classmethod
    def from_path(cls, video_path: str | Path) -> VideoMetadata:
        """Creates VideoMetadata object from video file using ffprobe."""
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        probe_data = cls._run_ffprobe(video_path)

        try:
            stream_info = probe_data["streams"][0]

            width = int(stream_info["width"])
            height = int(stream_info["height"])

            try:
                fps_fraction = Fraction(stream_info["r_frame_rate"])
                fps = float(fps_fraction)
            except (ValueError, ZeroDivisionError):
                raise VideoMetadataError(f"Invalid frame rate: {stream_info['r_frame_rate']}")

            if "nb_frames" in stream_info and stream_info["nb_frames"].isdigit():
                frame_count = int(stream_info["nb_frames"])
            else:
                duration = float(probe_data["format"]["duration"])
                frame_count = int(round(duration * fps))

            total_seconds = round(frame_count / fps, 2)

            return cls(height=height, width=width, fps=fps, frame_count=frame_count, total_seconds=total_seconds)

        except KeyError as e:
            raise VideoMetadataError(f"Missing required metadata field: {e}")
        except Exception as e:
            raise VideoMetadataError(f"Error extracting video metadata: {e}")

    @classmethod
    def from_video(cls, video: Video) -> VideoMetadata:
        """Creates VideoMetadata object from Video instance."""
        frame_count, height, width, _ = video.frames.shape
        total_seconds = round(frame_count / video.fps, 2)

        return cls(height=height, width=width, fps=video.fps, frame_count=frame_count, total_seconds=total_seconds)

    def can_be_merged_with(self, other_format: VideoMetadata) -> bool:
        """Check if videos can be merged."""
        return (
            self.height == other_format.height
            and self.width == other_format.width
            and round(self.fps) == round(other_format.fps)
        )

    def can_be_downsampled_to(self, target_format: VideoMetadata) -> bool:
        """Checks if video can be downsampled to target_format."""
        return (
            self.height >= target_format.height
            and self.width >= target_format.width
            and round(self.fps) >= round(target_format.fps)
            and self.total_seconds >= target_format.total_seconds
        )


class Video:
    def __init__(self, frames: np.ndarray, fps: int | float, audio: Audio | None = None):
        self.frames = frames
        self.fps = fps
        if audio:
            self.audio = audio
        else:
            self.audio = Audio.create_silent(
                duration_seconds=round(self.total_seconds, 2), stereo=True, sample_rate=44100
            )

    @classmethod
    def from_path(
        cls, path: str, read_batch_size: int = 100, start_second: float | None = None, end_second: float | None = None
    ) -> Video:
        try:
            # Get video metadata using VideoMetadata.from_path
            metadata = VideoMetadata.from_path(path)

            width = metadata.width
            height = metadata.height
            fps = metadata.fps
            total_duration = metadata.total_seconds

            # Validate time bounds
            if start_second is not None and start_second < 0:
                raise ValueError("start_second must be non-negative")
            if end_second is not None and end_second > total_duration:
                raise ValueError(f"end_second ({end_second}) exceeds video duration ({total_duration})")
            if start_second is not None and end_second is not None and start_second >= end_second:
                raise ValueError("start_second must be less than end_second")

            # Build FFmpeg command with improved segment handling
            ffmpeg_cmd = ["ffmpeg"]

            # Add seek option BEFORE input for more efficient seeking
            if start_second is not None:
                ffmpeg_cmd.extend(["-ss", str(start_second)])

            ffmpeg_cmd.extend(["-i", path])

            # Add duration AFTER input for more precise timing
            if end_second is not None and start_second is not None:
                duration = end_second - start_second
                ffmpeg_cmd.extend(["-t", str(duration)])
            elif end_second is not None:
                ffmpeg_cmd.extend(["-t", str(end_second)])

            # Output format settings - removed problematic -vsync 0
            ffmpeg_cmd.extend(
                [
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "-vcodec",
                    "rawvideo",
                    "-avoid_negative_ts",
                    "make_zero",  # Handle timing issues
                    "-y",
                    "pipe:1",
                ]
            )

            # Start FFmpeg process with stderr redirected to avoid deadlock
            process = subprocess.Popen(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,  # Redirect stderr to avoid deadlock
                bufsize=10**8,  # Use large buffer for efficient I/O
            )

            # Calculate frame size in bytes
            frame_size = width * height * 3  # 3 bytes per pixel for RGB

            # Estimate frame count for pre-allocation
            if start_second is not None and end_second is not None:
                estimated_duration = end_second - start_second
            elif end_second is not None:
                estimated_duration = end_second
            elif start_second is not None:
                estimated_duration = total_duration - start_second
            else:
                estimated_duration = total_duration

            # Add 10% buffer to handle frame rate variations and rounding
            estimated_frames = int(estimated_duration * fps * 1.1) + 10

            # Pre-allocate numpy array
            frames = np.empty((estimated_frames, height, width, 3), dtype=np.uint8)
            frames_read = 0

            try:
                while frames_read < estimated_frames:
                    # Calculate remaining frames to read
                    remaining_frames = estimated_frames - frames_read
                    batch_size = min(read_batch_size, remaining_frames)

                    # Read batch of data
                    batch_data = process.stdout.read(frame_size * batch_size)  # type: ignore

                    if not batch_data:
                        break

                    # Convert to numpy array
                    batch_frames = np.frombuffer(batch_data, dtype=np.uint8)

                    # Calculate how many complete frames we got
                    complete_frames = len(batch_frames) // (height * width * 3)

                    if complete_frames == 0:
                        break

                    # Only keep complete frames
                    complete_data = batch_frames[: complete_frames * height * width * 3]
                    batch_frames_array = complete_data.reshape(complete_frames, height, width, 3)

                    # Check if we have room in pre-allocated array
                    if frames_read + complete_frames > estimated_frames:
                        # Need to expand array - this should be rare with our buffer
                        new_size = max(estimated_frames * 2, frames_read + complete_frames + 100)
                        new_frames = np.empty((new_size, height, width, 3), dtype=np.uint8)
                        new_frames[:frames_read] = frames[:frames_read]
                        frames = new_frames
                        estimated_frames = new_size

                    # Store batch in pre-allocated array
                    end_idx = frames_read + complete_frames
                    frames[frames_read:end_idx] = batch_frames_array
                    frames_read += complete_frames

            finally:
                # Ensure process is properly terminated
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()

                # Clean up pipes
                if process.stdout:
                    process.stdout.close()

            # Check if FFmpeg had an error (non-zero return code)
            if process.returncode not in (0, None) and frames_read == 0:
                raise ValueError(f"FFmpeg failed to process video (return code: {process.returncode})")

            if frames_read == 0:
                raise ValueError("No frames were read from the video")

            # Trim the pre-allocated array to actual frames read
            frames = frames[:frames_read]  # type: ignore

            # Load audio for the specified segment
            try:
                audio = Audio.from_file(path)
                # Slice audio to match the video segment
                if start_second is not None or end_second is not None:
                    audio_start = start_second if start_second is not None else 0
                    audio_end = end_second if end_second is not None else audio.metadata.duration_seconds
                    audio = audio.slice(start_seconds=audio_start, end_seconds=audio_end)
            except Exception:
                print(f"No audio found for `{path}`, adding silent track!")
                # Create silent audio based on actual frames read
                segment_duration = frames_read / fps
                audio = Audio.create_silent(duration_seconds=round(segment_duration, 2), stereo=True, sample_rate=44100)

            return cls(frames=frames, fps=fps, audio=audio)

        except VideoMetadataError as e:
            raise ValueError(f"Error getting video metadata: {e}")
        except subprocess.CalledProcessError as e:
            raise ValueError(f"Error processing video file: {e}")
        except Exception as e:
            raise ValueError(f"Error loading video: {e}")

    @classmethod
    def from_frames(cls, frames: np.ndarray, fps: float) -> Video:
        if frames.ndim != 4:
            raise ValueError(f"Unsupported number of dimensions: {frames.shape}!")
        elif frames.shape[-1] == 4:
            frames = frames[:, :, :, :3]
        elif frames.shape[-1] != 3:
            raise ValueError(f"Unsupported number of dimensions: {frames.shape}!")
        return cls(frames=frames, fps=fps)

    @classmethod
    def from_image(cls, image: np.ndarray, fps: float = 24.0, length_seconds: float = 1.0) -> Video:
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        frames = np.repeat(image, round(length_seconds * fps), axis=0)
        return cls(frames=frames, fps=fps)

    def copy(self) -> Video:
        copied = Video.from_frames(self.frames.copy(), self.fps)
        copied.audio = self.audio  # Audio objects are immutable, no need to copy
        return copied

    def is_loaded(self) -> bool:
        return self.fps is not None and self.frames is not None and self.audio is not None

    def split(self, frame_idx: int | None = None) -> tuple[Video, Video]:
        if frame_idx:
            assert 0 <= frame_idx <= len(self.frames)
        else:
            frame_idx = len(self.frames) // 2

        split_videos = (
            self.from_frames(self.frames[:frame_idx], self.fps),
            self.from_frames(self.frames[frame_idx:], self.fps),
        )

        # Split audio at the corresponding time point
        split_time = frame_idx / self.fps
        split_videos[0].audio = self.audio.slice(start_seconds=0, end_seconds=split_time)
        split_videos[1].audio = self.audio.slice(start_seconds=split_time)

        return split_videos

    def save(self, filename: str | Path | None = None, format: ALLOWED_VIDEO_FORMATS = "mp4") -> Path:
        """Save video to file with optimized performance.

        Args:
            filename: Output filename. If None, generates random name
            format: Output format (mp4, avi, mov, mkv, webm)

        Returns:
            Path to saved video file

        Raises:
            RuntimeError: If video is not loaded
            ValueError: If format is not supported
        """
        if not self.is_loaded():
            raise RuntimeError("Video is not loaded, cannot save!")

        if format.lower() not in get_args(ALLOWED_VIDEO_FORMATS):
            raise ValueError(
                f"Unsupported format: {format}. Allowed formats are: {', '.join(get_args(ALLOWED_VIDEO_FORMATS))}"
            )

        if filename is None:
            filename = Path(generate_random_name(suffix=f".{format}"))
        else:
            filename = Path(filename).with_suffix(f".{format}")
            filename.parent.mkdir(parents=True, exist_ok=True)

        # Create a temporary raw video file
        with tempfile.NamedTemporaryFile(suffix=".raw") as raw_video:
            # Convert frames to raw video data
            raw_data = self.frames.astype(np.uint8).tobytes()
            raw_video.write(raw_data)
            raw_video.flush()

            # Save audio to temporary WAV file
            with tempfile.NamedTemporaryFile(suffix=".wav") as temp_audio:
                self.audio.save(temp_audio.name, format="wav")

                # Calculate exact duration
                duration = len(self.frames) / self.fps

                # Construct FFmpeg command for maximum performance
                ffmpeg_command = [
                    "ffmpeg",
                    "-y",
                    # Raw video input settings
                    "-f",
                    "rawvideo",
                    "-pixel_format",
                    "rgb24",
                    "-video_size",
                    f"{self.frame_shape[1]}x{self.frame_shape[0]}",
                    "-framerate",
                    str(self.fps),
                    "-i",
                    raw_video.name,
                    # Audio input
                    "-i",
                    temp_audio.name,
                    # Video encoding settings
                    "-c:v",
                    "libx264",
                    "-preset",
                    "ultrafast",  # Fastest encoding
                    "-tune",
                    "zerolatency",  # Reduce encoding latency
                    "-crf",
                    "23",  # Reasonable quality/size tradeoff
                    # Audio settings
                    "-c:a",
                    "aac",
                    "-b:a",
                    "192k",
                    # Output settings
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",  # Enable fast start for web playback
                    "-t",
                    str(duration),
                    "-vsync",
                    "cfr",
                    str(filename),
                ]

                try:
                    subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True)
                    return filename
                except subprocess.CalledProcessError as e:
                    print(f"Error saving video: {e}")
                    print(f"FFmpeg stderr: {e.stderr}")
                    raise

    def add_audio(self, audio: Audio, overlay: bool = True) -> None:
        video_duration = self.total_seconds
        audio_duration = audio.metadata.duration_seconds

        if audio_duration > video_duration:
            audio = audio.slice(start_seconds=0, end_seconds=video_duration)
        elif audio_duration < video_duration:
            silence_duration = video_duration - audio_duration
            silence = Audio.create_silent(
                duration_seconds=silence_duration,
                stereo=audio.metadata.channels == 2,
                sample_rate=audio.metadata.sample_rate,
            )
            audio = audio.concat(silence)

        if self.audio.is_silent:
            self.audio = audio
        elif overlay:
            self.audio = self.audio.overlay(audio, position=0.0)
        else:
            self.audio = audio

    def add_audio_from_file(self, path: str, overlay: bool = True) -> None:
        try:
            new_audio = Audio.from_file(path)
            self.add_audio(new_audio, overlay)
        except Exception:
            print(f"Audio file `{path}` not found or invalid, skipping!")

    def __add__(self, other: Video) -> Video:
        if self.fps != other.fps:
            raise ValueError("FPS of videos do not match!")
        elif self.frame_shape != other.frame_shape:
            raise ValueError(f"Resolutions do not match: {self.frame_shape} vs {other.frame_shape}")
        new_video = self.from_frames(np.r_["0,2", self.frames, other.frames], fps=self.fps)
        new_video.audio = self.audio.concat(other.audio)
        return new_video

    def __str__(self) -> str:
        return str(self.metadata)

    def __getitem__(self, val: slice) -> Video:
        if not isinstance(val, slice):
            raise ValueError("Only slices are supported for video indexing!")

        # Sub-slice video frames
        sliced = self.from_frames(self.frames[val], fps=self.fps)

        # Handle slicing bounds for audio
        start = val.start if val.start else 0
        stop = val.stop if val.stop else len(self.frames)
        if start < 0:
            start = len(self.frames) + start
        if stop < 0:
            stop = len(self.frames) + stop

        # Slice audio to match video duration
        audio_start = start / self.fps
        audio_end = stop / self.fps
        sliced.audio = self.audio.slice(start_seconds=audio_start, end_seconds=audio_end)
        return sliced

    @property
    def video_shape(self) -> tuple[int, int, int, int]:
        return self.frames.shape

    @property
    def frame_shape(self) -> tuple[int, int, int]:
        return self.frames.shape[1:]

    @property
    def total_seconds(self) -> float:
        return round(self.frames.shape[0] / self.fps, 4)

    @property
    def metadata(self) -> VideoMetadata:
        return VideoMetadata.from_video(self)
