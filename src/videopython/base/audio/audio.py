from __future__ import annotations

import io
import json
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from videopython.base.exceptions import AudioLoadError

if TYPE_CHECKING:
    from videopython.base.audio.analysis import AudioLevels, AudioSegment, AudioSegmentType, SilentSegment


@dataclass
class AudioMetadata:
    """Stores metadata for audio files"""

    sample_rate: int
    channels: int
    sample_width: int  # in bytes
    duration_seconds: float
    frame_count: int

    @property
    def bits_per_sample(self) -> int:
        """Returns the number of bits per sample"""
        return self.sample_width * 8


class Audio:
    """
    A class to handle audio data with numpy arrays

    Attributes:
        data (np.ndarray): Audio data as a numpy array, normalized between -1 and 1
        metadata (AudioMetadata): Metadata about the audio file
    """

    def __init__(self, data: np.ndarray, metadata: AudioMetadata):
        """
        Initialize Audio object

        Args:
            data: Audio data as numpy array, normalized between -1 and 1
            metadata: AudioMetadata object containing audio properties
        """
        self.data = data
        self.metadata = metadata

    @property
    def is_silent(self) -> bool:
        """
        Check if the audio track is silent (all samples are effectively zero)

        Returns:
            bool: True if the audio is silent, False otherwise
        """
        # Use a small threshold to account for floating-point precision
        return bool(np.all(np.abs(self.data) < 1e-7))

    @staticmethod
    def _get_ffmpeg_info(file_path: Path) -> dict:
        """Get audio metadata using ffprobe"""
        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(file_path),
        ]

        try:
            output = subprocess.check_output(cmd)
            info = json.loads(output.decode())

            # Find the audio stream
            audio_stream = None
            for stream in info["streams"]:
                if stream["codec_type"] == "audio":
                    audio_stream = stream
                    break

            if audio_stream is None:
                raise AudioLoadError("No audio stream found")

            return {
                "sample_rate": int(audio_stream["sample_rate"]),
                "channels": int(audio_stream["channels"]),
                "duration": float(info["format"]["duration"]),
                "bit_depth": int(audio_stream.get("bits_per_sample", 16)),
            }
        except subprocess.CalledProcessError as e:
            raise AudioLoadError(f"Error getting audio info: {e}")

    @classmethod
    def create_silent(
        cls, duration_seconds: float, stereo: bool = True, sample_rate: int = 44100, sample_width: int = 2
    ) -> Audio:
        """
        Create a silent audio track.

        Args:
            duration_seconds: Length of the silent track in seconds
            stereo: If True, create stereo track; if False, create mono track (default: True)
            sample_rate: Sample rate in Hz (default: 44100)
            sample_width: Sample width in bytes (default: 2, which is 16-bit)

        Returns:
            Audio: New Audio instance with silent track

        Raises:
            ValueError: If duration is negative or other parameters are invalid
        """
        if duration_seconds <= 0:
            raise ValueError("Duration must be positive")
        if sample_rate <= 0:
            raise ValueError("Sample rate must be positive")
        if sample_width not in {1, 2, 4}:
            raise ValueError("Sample width must be 1, 2, or 4 bytes")

        # Calculate number of frames
        frame_count = int(duration_seconds * sample_rate)

        # Create silent data array
        channels = 2 if stereo else 1
        shape = (frame_count, channels) if stereo else (frame_count,)
        data = np.zeros(shape, dtype=np.float32)

        # Create metadata
        metadata = AudioMetadata(
            sample_rate=sample_rate,
            channels=channels,
            sample_width=sample_width,
            duration_seconds=duration_seconds,
            frame_count=frame_count,
        )

        return cls(data, metadata)

    @classmethod
    def from_path(cls, file_path: str | Path) -> Audio:
        """
        Load audio from a file using ffmpeg

        Args:
            file_path: Path to the audio file

        Returns:
            Audio: New Audio instance

        Raises:
            FileNotFoundError: If the file doesn't exist
            AudioLoadError: If there's an error loading the audio
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Get audio info
        info = cls._get_ffmpeg_info(file_path)

        # Convert to WAV using ffmpeg
        cmd = [
            "ffmpeg",
            "-i",
            str(file_path),
            "-f",
            "wav",
            "-ar",
            str(info["sample_rate"]),  # sample rate
            "-ac",
            str(info["channels"]),  # channels
            "-bits_per_raw_sample",
            str(info["bit_depth"]),
            "-",  # Output to stdout
        ]

        try:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            wav_data, stderr = process.communicate()

            if process.returncode != 0:
                raise AudioLoadError(f"FFmpeg error: {stderr.decode()}")

            # Read WAV data
            with io.BytesIO(wav_data) as wav_io:
                with wave.open(wav_io, "rb") as wav_file:
                    # Get WAV metadata
                    sample_width = wav_file.getsampwidth()
                    channels = wav_file.getnchannels()
                    sample_rate = wav_file.getframerate()
                    n_frames = wav_file.getnframes()

                    # Read raw audio data
                    raw_data = wav_file.readframes(n_frames)

                    # Convert bytes to numpy array based on sample width
                    dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
                    dtype = dtype_map.get(sample_width)
                    if dtype is None:
                        raise AudioLoadError(f"Unsupported sample width: {sample_width}")

                    data = np.frombuffer(raw_data, dtype=dtype)

                    # Reshape if stereo
                    if channels == 2:
                        data = data.reshape(-1, 2)

                    # Convert to float32
                    data = data.astype(np.float32)

                    # Reshape before normalization if stereo
                    if channels == 2:
                        data = data.reshape(-1, 2)

                    # Normalize to float between -1 and 1
                    max_value = float(np.iinfo(dtype).max)  # type: ignore
                    data = data / max_value

                    # Ensure normalization is within bounds due to floating point precision
                    data = np.clip(data, -1.0, 1.0)

                    # Calculate frame count from actual data length
                    # For stereo, len(data) is already correct after reshape
                    frame_count = len(data)

                    metadata = AudioMetadata(
                        sample_rate=sample_rate,
                        channels=channels,
                        sample_width=sample_width,
                        duration_seconds=info["duration"],
                        frame_count=frame_count,
                    )

                    return cls(data, metadata)

        except subprocess.CalledProcessError as e:
            raise AudioLoadError(f"Error running ffmpeg: {e}")

    @classmethod
    def from_file(cls, file_path: str | Path) -> Audio:
        """Deprecated: Use from_path() instead."""
        import warnings

        warnings.warn(
            "Audio.from_file() is deprecated, use Audio.from_path() instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return cls.from_path(file_path)

    @classmethod
    def silence(
        cls,
        duration: float,
        sample_rate: int = 44100,
        channels: int = 2,
    ) -> Audio:
        """Create a silent audio track.

        Args:
            duration: Duration in seconds.
            sample_rate: Sample rate in Hz. Default: 44100.
            channels: Number of channels (1 for mono, 2 for stereo). Default: 2.

        Returns:
            Audio: Silent audio track with the specified parameters.

        Example:
            >>> silence = Audio.silence(duration=5.0)  # 5 seconds of silence
            >>> silence = Audio.silence(duration=2.0, sample_rate=22050, channels=1)
        """
        frame_count = int(duration * sample_rate)
        data = np.zeros((frame_count, channels), dtype=np.float32)

        metadata = AudioMetadata(
            sample_rate=sample_rate,
            channels=channels,
            sample_width=2,
            duration_seconds=duration,
            frame_count=frame_count,
        )

        return cls(data, metadata)

    def to_mono(self) -> Audio:
        """
        Convert stereo audio to mono by averaging channels

        Returns:
            Audio: New Audio instance with mono audio
        """
        if self.metadata.channels == 1:
            return self

        mono_data = self.data.mean(axis=1)

        new_metadata = AudioMetadata(
            sample_rate=self.metadata.sample_rate,
            channels=1,
            sample_width=self.metadata.sample_width,
            duration_seconds=self.metadata.duration_seconds,
            frame_count=len(mono_data),
        )

        return Audio(mono_data, new_metadata)

    def _to_stereo(self) -> Audio:
        """
        Convert mono audio to stereo by duplicating the channel.
        If already stereo, return self.

        Returns:
            Audio: Stereo version of the audio
        """
        if self.metadata.channels == 2:
            return self

        # Reshape mono data to 2D array and duplicate channel
        stereo_data = np.column_stack((self.data, self.data))

        new_metadata = AudioMetadata(
            sample_rate=self.metadata.sample_rate,
            channels=2,
            sample_width=self.metadata.sample_width,
            duration_seconds=self.metadata.duration_seconds,
            frame_count=len(stereo_data),
        )

        return Audio(stereo_data, new_metadata)

    def get_channel(self, channel: int) -> Audio:
        """
        Extract a single channel from the audio

        Args:
            channel: Channel number (0 for left, 1 for right)

        Returns:
            Audio: New Audio instance with single channel

        Raises:
            ValueError: If channel number is invalid
        """
        if self.metadata.channels == 1:
            return self

        if channel not in [0, 1]:
            raise ValueError("Channel must be 0 (left) or 1 (right)")

        channel_data = self.data[:, channel]

        new_metadata = AudioMetadata(
            sample_rate=self.metadata.sample_rate,
            channels=1,
            sample_width=self.metadata.sample_width,
            duration_seconds=self.metadata.duration_seconds,
            frame_count=len(channel_data),
        )

        return Audio(channel_data, new_metadata)

    def resample(self, target_sample_rate: int) -> Audio:
        """
        Resample the audio to a new sample rate

        Args:
            target_sample_rate: New sample rate in Hz

        Returns:
            Audio: New Audio instance with resampled audio
        """
        if target_sample_rate == self.metadata.sample_rate:
            return self

        # Calculate resampling ratio
        ratio = target_sample_rate / self.metadata.sample_rate

        target_length = round(self.data.shape[0] * ratio)

        audio_array = self.data
        if self.metadata.channels == 1:
            audio_array = audio_array.reshape(-1, 1)

        resampled_data = np.zeros((target_length, self.metadata.channels), dtype=np.float32)

        for channel in range(self.metadata.channels):
            resampled_data[:, channel] = self._resample_channel(audio_array[:, channel], target_length)

        new_metadata = AudioMetadata(
            sample_rate=target_sample_rate,
            channels=self.metadata.channels,
            sample_width=self.metadata.sample_width,
            duration_seconds=target_length / target_sample_rate,
            frame_count=target_length,
        )
        if self.metadata.channels == 1:
            resampled_data = resampled_data.flatten()

        return Audio(resampled_data, new_metadata)

    @staticmethod
    def _resample_channel(data: np.ndarray, new_length: int) -> np.ndarray:
        """Resample a single channel of audio data to a new length"""

        data_fourier = np.fft.rfft(data)
        original_length = data.shape[0]

        newshape = [new_length // 2 + 1]

        data_fourier_placeholder = np.zeros(newshape, data_fourier.dtype)

        min_length = min(new_length, original_length)
        nyquist = min_length // 2 + 1
        sl = [slice(0, nyquist)]
        data_fourier_placeholder[tuple(sl)] = data_fourier[tuple(sl)]

        if min_length % 2 == 0:
            if new_length < original_length:
                sl = [slice(min_length // 2, min_length // 2 + 1)]
                data_fourier_placeholder[tuple(sl)] *= 2.0

                sl = [slice(min_length // 2, min_length // 2 + 1)]
                data_fourier_placeholder[tuple(sl)] *= 0.5

        resampled_data = np.fft.irfft(data_fourier_placeholder, new_length)

        resampled_data *= float(new_length) / float(original_length)

        return resampled_data

    def concat(self, other: Audio, crossfade: float = 0.0) -> Audio:
        """
        Concatenate another audio segment to this one.
        If mixing mono and stereo, converts mono to stereo.

        Args:
            other: Another Audio object to concatenate
            crossfade: Duration of crossfade in seconds (default: 0.0 for no crossfade)

        Returns:
            Audio: New Audio object with concatenated data

        Raises:
            ValueError: If audio metadata doesn't match or crossfade duration is invalid
        """
        if abs(self.metadata.sample_rate - other.metadata.sample_rate) > 0:
            raise ValueError("Sample rates must match")
        if self.metadata.sample_width != other.metadata.sample_width:
            raise ValueError("Sample widths must match")

        # Determine output format (mono or stereo)
        output_stereo = self.metadata.channels == 2 or other.metadata.channels == 2

        # Convert to appropriate format if needed
        first = self._to_stereo() if output_stereo and self.metadata.channels == 1 else self
        second = other._to_stereo() if output_stereo and other.metadata.channels == 1 else other

        if first.metadata.channels != second.metadata.channels:
            raise ValueError("Channel counts must match")

        # Handle case with no crossfade
        if crossfade <= 0:
            if first.metadata.channels == 1:
                concatenated_data = np.concatenate([first.data, second.data])
            else:
                concatenated_data = np.vstack([first.data, second.data])

            new_metadata = AudioMetadata(
                sample_rate=first.metadata.sample_rate,
                channels=first.metadata.channels,
                sample_width=first.metadata.sample_width,
                duration_seconds=first.metadata.duration_seconds + second.metadata.duration_seconds,
                frame_count=len(concatenated_data),
            )

            return Audio(concatenated_data, new_metadata)

        # Validate crossfade duration
        if crossfade > min(first.metadata.duration_seconds, second.metadata.duration_seconds):
            raise ValueError("Crossfade duration cannot exceed duration of either audio segment")

        # Calculate crossfade parameters
        crossfade_samples = int(crossfade * first.metadata.sample_rate)

        # Calculate output length and create output array
        total_samples = len(first.data) + len(second.data) - crossfade_samples
        if first.metadata.channels == 1:
            output = np.zeros(total_samples, dtype=np.float32)
        else:
            output = np.zeros((total_samples, 2), dtype=np.float32)  # type: ignore

        # Copy non-crossfaded portions
        crossfade_start = len(first.data) - crossfade_samples
        output[:crossfade_start] = first.data[:crossfade_start]
        output[crossfade_start + crossfade_samples :] = second.data[crossfade_samples:]

        # Create crossfade ramps
        fade_out = np.linspace(1, 0, crossfade_samples)
        fade_in = np.linspace(0, 1, crossfade_samples)

        # Apply crossfade
        if first.metadata.channels == 1:
            output[crossfade_start : crossfade_start + crossfade_samples] = (
                first.data[crossfade_start:] * fade_out + second.data[:crossfade_samples] * fade_in
            )
        else:
            for channel in range(first.metadata.channels):
                output[crossfade_start : crossfade_start + crossfade_samples, channel] = (
                    first.data[crossfade_start:, channel] * fade_out
                    + second.data[:crossfade_samples, channel] * fade_in
                )

        # Create new metadata
        new_duration = total_samples / first.metadata.sample_rate
        new_metadata = AudioMetadata(
            sample_rate=first.metadata.sample_rate,
            channels=first.metadata.channels,
            sample_width=first.metadata.sample_width,
            duration_seconds=new_duration,
            frame_count=total_samples,
        )

        return Audio(output, new_metadata)

    def slice(self, start_seconds: float = 0.0, end_seconds: float | None = None) -> Audio:
        """
        Extract a portion of the audio between start_seconds and end_seconds.

        Args:
            start_seconds: Start time in seconds (default: 0.0)
            end_seconds: End time in seconds (default: None, meaning end of audio)

        Returns:
            Audio: New Audio instance with the extracted portion

        Raises:
            ValueError: If start_seconds or end_seconds are invalid
        """
        # Validate inputs
        if start_seconds < 0:
            raise ValueError("start_seconds must be non-negative")

        if end_seconds is not None:
            if end_seconds < start_seconds:
                raise ValueError("end_seconds must be greater than start_seconds")
            if end_seconds > self.metadata.duration_seconds:
                raise ValueError("end_seconds cannot exceed audio duration")
        else:
            end_seconds = self.metadata.duration_seconds

        # Convert seconds to sample indices
        start_idx = int(start_seconds * self.metadata.sample_rate)
        end_idx = int(end_seconds * self.metadata.sample_rate)

        # Extract the portion of audio data
        sliced_data = self.data[start_idx:end_idx]

        # Calculate new duration
        new_duration = (end_idx - start_idx) / self.metadata.sample_rate

        # Create new metadata
        new_metadata = AudioMetadata(
            sample_rate=self.metadata.sample_rate,
            channels=self.metadata.channels,
            sample_width=self.metadata.sample_width,
            duration_seconds=new_duration,
            frame_count=len(sliced_data) if self.metadata.channels == 1 else len(sliced_data),
        )

        return Audio(sliced_data, new_metadata)

    def scale_volume(self, factor: float) -> Audio:
        """
        Scale audio volume by a factor.

        Args:
            factor: Volume multiplier. 1.0 = no change, 0.5 = half volume,
                    2.0 = double volume (may clip).

        Returns:
            Audio: New Audio object with scaled volume.

        Raises:
            ValueError: If factor is negative.
        """
        if factor < 0:
            raise ValueError("Volume factor must be non-negative")

        scaled_data = self.data * factor
        # Clip to prevent overflow
        scaled_data = np.clip(scaled_data, -1.0, 1.0)

        return Audio(scaled_data.astype(np.float32), self.metadata)

    def fit_to_duration(self, target_duration: float) -> "Audio":
        """
        Adjust audio duration to match a target, slicing or padding with silence.

        If audio is longer than target, it will be sliced.
        If audio is shorter than target, silence will be appended.

        Args:
            target_duration: Target duration in seconds.

        Returns:
            Audio: New Audio object with the target duration.

        Raises:
            ValueError: If target_duration is not positive.
        """
        if target_duration <= 0:
            raise ValueError("Target duration must be positive")

        current_duration = self.metadata.duration_seconds

        if current_duration > target_duration:
            return self.slice(0, target_duration)
        elif current_duration < target_duration:
            silence = Audio.create_silent(
                target_duration - current_duration,
                stereo=self.metadata.channels == 2,
                sample_rate=self.metadata.sample_rate,
                sample_width=self.metadata.sample_width,
            )
            return self.concat(silence)
        return self

    def time_stretch(self, speed: float) -> Audio:
        """
        Time-stretch audio by a speed factor (pitch-preserving).

        Uses ffmpeg's atempo filter for high-quality time stretching.
        For speeds outside the 0.5-2.0 range, multiple atempo filters are chained.

        Args:
            speed: Speed multiplier. 2.0 = twice as fast (half duration),
                   0.5 = half speed (double duration).

        Returns:
            Audio: New Audio object with time-stretched audio.

        Raises:
            ValueError: If speed is not positive.
            AudioLoadError: If ffmpeg fails.
        """
        if speed <= 0:
            raise ValueError("Speed must be positive")

        if abs(speed - 1.0) < 0.001:
            # No change needed
            return Audio(self.data.copy(), self.metadata)

        # Build atempo filter string, chaining for extreme speeds
        # atempo only supports range [0.5, 2.0], so we chain multiple filters
        filters = []
        remaining_speed = speed

        while remaining_speed > 2.0:
            filters.append("atempo=2.0")
            remaining_speed /= 2.0
        while remaining_speed < 0.5:
            filters.append("atempo=0.5")
            remaining_speed /= 0.5

        if abs(remaining_speed - 1.0) > 0.001:
            filters.append(f"atempo={remaining_speed}")

        filter_str = ",".join(filters) if filters else "anull"

        # Save current audio to temp WAV, process with ffmpeg, read back
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as input_file:
            input_path = input_file.name

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as output_file:
            output_path = output_file.name

        try:
            # Save current audio to temp file
            self.save(input_path, format="wav")

            # Run ffmpeg with atempo filter
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                input_path,
                "-af",
                filter_str,
                "-ar",
                str(self.metadata.sample_rate),
                "-ac",
                str(self.metadata.channels),
                output_path,
            ]

            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            _, stderr = process.communicate()

            if process.returncode != 0:
                raise AudioLoadError(f"FFmpeg time stretch failed: {stderr.decode()}")

            # Load the result
            result = Audio.from_path(output_path)
            return result

        finally:
            # Clean up temp files
            import os

            for path in [input_path, output_path]:
                try:
                    os.unlink(path)
                except OSError:
                    pass

    def overlay(self, other: Audio, position: float = 0.0) -> Audio:
        """
        Overlay another audio segment on top of this one, mixing both signals.
        If mixing mono and stereo, converts mono to stereo.

        Args:
            other: Another Audio object to overlay
            position: Start position in seconds for the overlay (default: 0.0)

        Returns:
            Audio: New Audio object with mixed audio

        Raises:
            ValueError: If audio metadata doesn't match or position is invalid
        """
        if abs(self.metadata.sample_rate - other.metadata.sample_rate) > 0:
            raise ValueError("Sample rates must match")
        if self.metadata.sample_width != other.metadata.sample_width:
            raise ValueError("Sample widths must match")
        if position < 0:
            raise ValueError("Position cannot be negative")

        # Determine output format (mono or stereo)
        output_stereo = self.metadata.channels == 2 or other.metadata.channels == 2

        # Convert to appropriate format if needed
        base = self._to_stereo() if output_stereo and self.metadata.channels == 1 else self
        overlay_audio = other._to_stereo() if output_stereo and other.metadata.channels == 1 else other

        if base.metadata.channels != overlay_audio.metadata.channels:
            raise ValueError("Channel counts must match")

        # Convert position to samples, using ceil to ensure we don't cut off any audio
        position_samples = int(np.ceil(position * base.metadata.sample_rate))

        # Calculate the total length needed for the output
        total_length = max(len(base.data), position_samples + len(overlay_audio.data))

        # Create output array with appropriate shape
        if base.metadata.channels == 1:
            output = np.zeros(total_length, dtype=np.float32)
        else:
            output = np.zeros((total_length, 2), dtype=np.float32)  # type: ignore

        # Copy base audio
        output[: len(base.data)] = base.data

        # Add overlay audio at the specified position
        overlay_end = position_samples + len(overlay_audio.data)
        output[position_samples:overlay_end] += overlay_audio.data

        # Prevent clipping by scaling if necessary
        max_amplitude = np.max(np.abs(output))
        if max_amplitude > 1.0:
            output = output / max_amplitude

        # Create new metadata, using actual duration calculation
        new_duration = max(base.metadata.duration_seconds, position + overlay_audio.metadata.duration_seconds)
        new_metadata = AudioMetadata(
            sample_rate=base.metadata.sample_rate,
            channels=base.metadata.channels,
            sample_width=base.metadata.sample_width,
            duration_seconds=new_duration,
            frame_count=total_length,
        )

        return Audio(output, new_metadata)

    def save(self, file_path: str | Path, format: str | None = None) -> None:
        """
        Save audio to a file using ffmpeg

        Args:
            file_path: Path to save the audio file
            format: Output format (e.g., 'mp3', 'wav'). If None, inferred from extension.
        """
        file_path = Path(file_path)

        # Convert data back to int16
        int_data = (self.data * np.iinfo(np.int16).max).astype(np.int16)

        # Create WAV in memory
        wav_io = io.BytesIO()
        with wave.open(wav_io, "wb") as wav_file:
            wav_file.setnchannels(self.metadata.channels)
            wav_file.setsampwidth(self.metadata.sample_width)
            wav_file.setframerate(self.metadata.sample_rate)
            wav_file.writeframes(int_data.tobytes())

        wav_io.seek(0)

        # Check and infer format
        if format is None:
            format = file_path.suffix[1:]  # Remove the dot

        # Validate format
        SUPPORTED_FORMATS = {"mp3", "wav", "ogg", "flac"}
        if format not in SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {format}. Supported formats are: {', '.join(SUPPORTED_FORMATS)}")

        # Build ffmpeg command
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output file
            "-f",
            "wav",  # Input format
            "-i",
            "-",  # Read from stdin
        ]

        if format:
            cmd.extend(["-f", format])

        cmd.append(str(file_path))

        try:
            process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
            _, stderr = process.communicate(wav_io.getvalue())

            if process.returncode != 0:
                raise AudioLoadError(f"Error saving audio: {stderr.decode()}")

        except subprocess.CalledProcessError as e:
            raise AudioLoadError(f"Error running ffmpeg: {e}")

    def __len__(self) -> int:
        """Returns the number of samples"""
        return self.metadata.frame_count

    def __repr__(self) -> str:
        """String representation of the Audio object"""
        return (
            f"Audio(channels={self.metadata.channels}, "
            f"sample_rate={self.metadata.sample_rate}Hz, "
            f"duration={self.metadata.duration_seconds:.2f}s)"
        )

    # -------------------------------------------------------------------------
    # Audio Analysis Methods
    # -------------------------------------------------------------------------

    def get_levels(
        self,
        start_seconds: float = 0.0,
        end_seconds: float | None = None,
    ) -> "AudioLevels":
        """Calculate audio levels for a segment.

        Args:
            start_seconds: Start time in seconds (default: 0.0)
            end_seconds: End time in seconds (default: None, meaning end of audio)

        Returns:
            AudioLevels with RMS, peak, and dB measurements

        Example:
            >>> audio = Audio.from_path("audio.mp3")
            >>> levels = audio.get_levels()
            >>> print(f"Peak: {levels.db_peak:.1f} dB")
        """
        from videopython.base.audio.analysis import AudioLevels

        segment = self.slice(start_seconds, end_seconds)
        data = segment.data.flatten() if segment.metadata.channels == 2 else segment.data

        rms = float(np.sqrt(np.mean(data**2)))
        peak = float(np.max(np.abs(data)))

        # Convert to dB (avoid log of zero)
        db_rms = 20 * np.log10(max(rms, 1e-10))
        db_peak = 20 * np.log10(max(peak, 1e-10))

        return AudioLevels(rms=rms, peak=peak, db_rms=float(db_rms), db_peak=float(db_peak))

    def get_levels_over_time(
        self,
        window_seconds: float = 0.1,
        hop_seconds: float | None = None,
    ) -> list[tuple[float, "AudioLevels"]]:
        """Calculate audio levels over time using a sliding window.

        Args:
            window_seconds: Window size in seconds (default: 0.1)
            hop_seconds: Hop size in seconds (default: window_seconds / 2)

        Returns:
            List of (timestamp, AudioLevels) tuples where timestamp is the
            center of each window

        Example:
            >>> levels_over_time = audio.get_levels_over_time(window_seconds=0.1)
            >>> for timestamp, levels in levels_over_time:
            ...     print(f"{timestamp:.2f}s: {levels.db_rms:.1f} dB")
        """
        if hop_seconds is None:
            hop_seconds = window_seconds / 2

        results = []
        current_time = 0.0

        while current_time + window_seconds <= self.metadata.duration_seconds:
            levels = self.get_levels(current_time, current_time + window_seconds)
            results.append((current_time + window_seconds / 2, levels))
            current_time += hop_seconds

        return results

    def detect_silence(
        self,
        threshold_db: float = -40.0,
        min_duration: float = 0.5,
        window_seconds: float = 0.1,
    ) -> list["SilentSegment"]:
        """Detect silent segments in the audio.

        Args:
            threshold_db: RMS level below which audio is considered silent (default: -40 dB)
            min_duration: Minimum duration for a segment to be classified as silent (default: 0.5s)
            window_seconds: Window size for level analysis (default: 0.1s)

        Returns:
            List of SilentSegment objects representing detected silent regions

        Example:
            >>> silent_segments = audio.detect_silence(threshold_db=-40.0, min_duration=0.5)
            >>> for seg in silent_segments:
            ...     print(f"Silence: {seg.start:.2f}s - {seg.end:.2f}s")
        """
        from videopython.base.audio.analysis import SilentSegment

        levels_over_time = self.get_levels_over_time(window_seconds=window_seconds, hop_seconds=window_seconds / 2)

        silent_segments = []
        in_silence = False
        silence_start = 0.0
        silence_levels: list[float] = []

        for timestamp, levels in levels_over_time:
            is_silent = levels.db_rms < threshold_db

            if is_silent and not in_silence:
                # Start of silence
                in_silence = True
                silence_start = timestamp - window_seconds / 2
                silence_levels = [levels.rms]
            elif is_silent and in_silence:
                # Continue silence
                silence_levels.append(levels.rms)
            elif not is_silent and in_silence:
                # End of silence
                silence_end = timestamp - window_seconds / 2
                duration = silence_end - silence_start

                if duration >= min_duration:
                    avg_level = sum(silence_levels) / len(silence_levels)
                    silent_segments.append(
                        SilentSegment(
                            start=silence_start,
                            end=silence_end,
                            duration=duration,
                            avg_level=avg_level,
                        )
                    )

                in_silence = False
                silence_levels = []

        # Handle case where audio ends in silence
        if in_silence:
            silence_end = self.metadata.duration_seconds
            duration = silence_end - silence_start
            if duration >= min_duration:
                avg_level = sum(silence_levels) / len(silence_levels) if silence_levels else 0.0
                silent_segments.append(
                    SilentSegment(
                        start=silence_start,
                        end=silence_end,
                        duration=duration,
                        avg_level=avg_level,
                    )
                )

        return silent_segments

    def classify_segments(
        self,
        segment_length: float = 2.0,
        overlap: float = 0.5,
    ) -> list["AudioSegment"]:
        """Classify audio segments as speech, music, noise, or silence.

        This uses basic signal processing heuristics (no ML):
        - Zero-crossing rate (higher for speech/noise)
        - Spectral flatness (higher for noise, lower for music)
        - Energy distribution across frequency bands

        Args:
            segment_length: Length of each segment to classify in seconds (default: 2.0)
            overlap: Overlap between segments as fraction (default: 0.5)

        Returns:
            List of AudioSegment objects with classifications

        Example:
            >>> segments = audio.classify_segments(segment_length=2.0)
            >>> for seg in segments:
            ...     print(f"{seg.start:.1f}-{seg.end:.1f}s: {seg.segment_type.value}")
        """
        from videopython.base.audio.analysis import AudioSegment

        hop_length = segment_length * (1 - overlap)
        segments = []
        current_time = 0.0

        while current_time + segment_length <= self.metadata.duration_seconds:
            segment_audio = self.slice(current_time, current_time + segment_length)
            segment_type, confidence = self._classify_segment(segment_audio)
            levels = self.get_levels(current_time, current_time + segment_length)

            segments.append(
                AudioSegment(
                    start=current_time,
                    end=current_time + segment_length,
                    segment_type=segment_type,
                    confidence=confidence,
                    levels=levels,
                )
            )

            current_time += hop_length

        return segments

    def _classify_segment(self, segment: "Audio") -> tuple["AudioSegmentType", float]:
        """Classify a single audio segment using heuristics.

        Uses zero-crossing rate, spectral flatness, and speech band energy
        to make a best-effort classification without ML.

        Args:
            segment: Audio segment to classify

        Returns:
            Tuple of (AudioSegmentType, confidence)
        """
        from videopython.base.audio.analysis import AudioSegmentType

        data = segment.to_mono().data

        # Calculate RMS
        rms = np.sqrt(np.mean(data**2))

        # Check for silence first
        if rms < 0.01:  # Very quiet
            return AudioSegmentType.SILENCE, 0.95

        # Zero-crossing rate (normalized)
        zero_crossings = np.sum(np.abs(np.diff(np.sign(data)))) / 2
        zcr = zero_crossings / len(data)

        # Spectral analysis using FFT
        fft = np.abs(np.fft.rfft(data))
        freqs = np.fft.rfftfreq(len(data), 1.0 / segment.metadata.sample_rate)

        # Spectral flatness (geometric mean / arithmetic mean)
        # Higher values indicate noise-like signal
        log_fft = np.log(fft + 1e-10)
        geometric_mean = np.exp(np.mean(log_fft))
        arithmetic_mean = np.mean(fft)
        spectral_flatness = geometric_mean / (arithmetic_mean + 1e-10)

        # Energy in speech frequency range (300-3400 Hz)
        speech_mask = (freqs >= 300) & (freqs <= 3400)
        speech_energy = np.sum(fft[speech_mask] ** 2)
        total_energy = np.sum(fft**2) + 1e-10
        speech_ratio = speech_energy / total_energy

        # Spectral centroid (where is the "center of mass" of the spectrum)
        spectral_centroid = np.sum(freqs * fft) / (np.sum(fft) + 1e-10)

        # Heuristic classification
        # Noise: high spectral flatness, high ZCR
        if spectral_flatness > 0.5 and zcr > 0.1:
            return AudioSegmentType.NOISE, min(0.7, float(spectral_flatness))

        # Speech: medium ZCR, energy concentrated in 300-3400 Hz
        if speech_ratio > 0.4 and 0.02 < zcr < 0.15:
            return AudioSegmentType.SPEECH, min(0.8, float(speech_ratio))

        # Music: lower ZCR, lower spectral flatness, broader spectrum
        if spectral_flatness < 0.3 and spectral_centroid < 2000:
            return AudioSegmentType.MUSIC, min(0.7, 1 - float(spectral_flatness))

        # Default to speech with lower confidence
        return AudioSegmentType.SPEECH, 0.4

    def normalize(
        self,
        target_db: float = -3.0,
        method: str = "peak",
    ) -> "Audio":
        """Normalize audio to a target level.

        Args:
            target_db: Target level in dB (default: -3.0 dB, allows headroom)
            method: Normalization method, either "peak" or "rms" (default: "peak")

        Returns:
            New Audio object with normalized levels

        Example:
            >>> normalized = audio.normalize(target_db=-3.0, method="peak")
            >>> print(f"New peak: {normalized.get_levels().db_peak:.1f} dB")
        """
        data = self.data.copy()

        if method == "peak":
            current_peak = np.max(np.abs(data))
            if current_peak < 1e-10:
                return Audio(data, self.metadata)

            target_amplitude = 10 ** (target_db / 20)
            scale_factor = target_amplitude / current_peak

        elif method == "rms":
            current_rms = np.sqrt(np.mean(data**2))
            if current_rms < 1e-10:
                return Audio(data, self.metadata)

            target_rms = 10 ** (target_db / 20)
            scale_factor = target_rms / current_rms

        else:
            raise ValueError(f"Unknown method: {method}. Use 'peak' or 'rms'")

        # Apply scaling
        normalized_data = data * scale_factor

        # Clip to prevent overflow (should be rare with proper target_db)
        normalized_data = np.clip(normalized_data, -1.0, 1.0)

        return Audio(normalized_data.astype(np.float32), self.metadata)
