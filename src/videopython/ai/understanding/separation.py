"""Audio source separation using local Demucs models."""

from __future__ import annotations

import logging
from typing import Any

from videopython.ai._device import log_device_initialization, release_device_memory, select_device
from videopython.ai.dubbing.models import SeparatedAudio
from videopython.base.audio import Audio, AudioMetadata

logger = logging.getLogger(__name__)


def _merge_regions(
    regions: list[tuple[float, float]],
    audio_duration: float,
    pad: float = 0.5,
    merge_gap: float = 1.0,
) -> list[tuple[float, float]]:
    """Merge overlapping/adjacent (start, end) ranges and pad each side.

    Args:
        regions: Speech regions in seconds. Order does not matter.
        audio_duration: Total audio duration; output is clamped to ``[0, audio_duration]``.
        pad: Seconds added to each side. Demucs needs context to separate
            cleanly at boundaries; 0.5s avoids clipped onsets/decays.
        merge_gap: Adjacent regions whose padded edges are within this
            many seconds are merged. Avoids running Demucs on very short
            slices (where its temporal context isn't there).

    Returns:
        Sorted list of non-overlapping (start, end) regions covering the
        speech-bearing portion of the audio.
    """
    if not regions:
        return []

    sorted_regions = sorted(regions)

    merged: list[tuple[float, float]] = []
    for start, end in sorted_regions:
        if end <= start:
            continue
        padded_start = max(0.0, start - pad)
        padded_end = min(audio_duration, end + pad)
        if padded_start >= audio_duration or padded_end <= 0.0:
            continue

        if merged and padded_start - merged[-1][1] <= merge_gap:
            merged[-1] = (merged[-1][0], max(merged[-1][1], padded_end))
        else:
            merged.append((padded_start, padded_end))

    return merged


class AudioSeparator:
    """Separates audio into vocals and background components using Demucs."""

    SUPPORTED_MODELS: list[str] = ["htdemucs", "htdemucs_ft", "htdemucs_6s", "mdx_extra"]
    STEM_NAMES = ["drums", "bass", "other", "vocals"]
    STEM_NAMES_6S = ["drums", "bass", "other", "vocals", "guitar", "piano"]

    def __init__(self, model_name: str = "htdemucs", device: str | None = None):
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"Model '{model_name}' not supported. Supported: {self.SUPPORTED_MODELS}")

        self.model_name = model_name
        self.device = device
        self._model: Any = None

    def _init_local(self) -> None:
        """Initialize local Demucs model."""
        from demucs.pretrained import get_model

        requested_device = self.device
        device = select_device(self.device, mps_allowed=False)

        self._model = get_model(self.model_name)
        self._model.to(device)
        self._model.eval()
        self.device = device
        log_device_initialization(
            "AudioSeparator",
            requested_device=requested_device,
            resolved_device=device,
        )

    def _separate_local(self, audio: Audio) -> SeparatedAudio:
        """Separate audio using local Demucs model.

        Keeps the input tensor on CPU and passes ``device=self.device`` to
        ``apply_model`` so per-chunk compute runs on GPU while the full
        ``(stems, channels, samples)`` output is stored in CPU RAM. For long
        sources this is the difference between OOM-on-GPU and running cleanly:
        a 2h stereo @ 44.1kHz output is ~10 GB — too big for an 8 GB card but
        comfortable on a 32 GB host.
        """
        import numpy as np
        import torch
        from demucs.apply import apply_model

        if self._model is None:
            self._init_local()

        target_sr = self._model.samplerate

        if audio.metadata.channels == 1:
            audio = audio._to_stereo()

        if audio.metadata.sample_rate != target_sr:
            audio = audio.resample(target_sr)

        audio_data = audio.data
        if audio_data.ndim == 1:
            audio_data = np.stack([audio_data, audio_data])
        elif audio_data.ndim == 2:
            audio_data = audio_data.T

        wav = torch.tensor(audio_data, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            sources = apply_model(self._model, wav, device=self.device)

        sources_np = sources[0].cpu().numpy()
        del sources

        stem_names = self.STEM_NAMES_6S if self.model_name == "htdemucs_6s" else self.STEM_NAMES
        vocals_idx = stem_names.index("vocals")
        non_vocal_indices = [i for i in range(len(stem_names)) if i != vocals_idx]

        vocals_data = sources_np[vocals_idx].T
        background_data = sources_np[non_vocal_indices].sum(axis=0).T
        del sources_np

        max_val = np.max(np.abs(background_data))
        if max_val > 1.0:
            background_data /= max_val

        metadata = AudioMetadata(
            sample_rate=target_sr,
            channels=2,
            sample_width=2,
            duration_seconds=vocals_data.shape[0] / target_sr,
            frame_count=vocals_data.shape[0],
        )
        vocals = Audio(np.ascontiguousarray(vocals_data, dtype=np.float32), metadata)
        background = Audio(np.ascontiguousarray(background_data, dtype=np.float32), metadata)

        return SeparatedAudio(
            vocals=vocals,
            background=background,
            original=audio,
            music=None,
            effects=None,
        )

    def separate(self, audio: Audio) -> SeparatedAudio:
        """Separate audio into vocals and background components."""
        return self._separate_local(audio)

    def separate_regions(
        self,
        audio: Audio,
        regions: list[tuple[float, float]],
        full_separation_threshold: float = 0.9,
    ) -> SeparatedAudio:
        """Separate only the given (start, end) regions; pass the rest through.

        Demucs is the slowest stage of the dubbing pipeline. On talk-heavy
        sources (podcasts, interviews) most of the track is speech, but
        long pauses, silence, or music-only stretches don't need vocal
        isolation — there's nothing to isolate. We run Demucs only on the
        speech-bearing regions and treat the rest as pure background.

        Output is full-length: vocals are silent outside the given
        regions; background is the original audio outside the given
        regions and the Demucs-separated background inside.

        Args:
            audio: Source audio (typically the full track).
            regions: List of ``(start, end)`` second pairs marking
                speech-bearing portions. Caller is responsible for
                merging/padding (use ``_merge_regions``).
            full_separation_threshold: If the regions cover more than
                this fraction of the audio, fall back to full-track
                ``separate()`` since per-region slicing+stitching
                overhead would exceed the savings. Default 0.9.

        Returns:
            ``SeparatedAudio`` with full-length vocals and background.
        """
        import numpy as np

        if not regions:
            logger.info("separate_regions: no regions, returning silent vocals over original audio")
            return self._passthrough_separation(audio)

        total_duration = audio.metadata.duration_seconds
        speech_duration = sum(end - start for start, end in regions)
        if total_duration > 0 and speech_duration / total_duration >= full_separation_threshold:
            logger.info(
                "separate_regions: speech covers %.0f%% of audio (>=%.0f%%), using full-track separation",
                speech_duration / total_duration * 100,
                full_separation_threshold * 100,
            )
            return self._separate_local(audio)

        logger.info(
            "separate_regions: separating %.1fs of speech across %d region(s) (full duration: %.1fs)",
            speech_duration,
            len(regions),
            total_duration,
        )

        # Build full-length output buffers. Background defaults to the
        # original audio (so non-speech gaps pass through unchanged); vocals
        # default to silence (no speech to isolate outside the regions).
        # Both are stereo to match the full-track separation contract.
        sr = audio.metadata.sample_rate
        stereo_audio = audio if audio.metadata.channels == 2 else audio._to_stereo()

        total_samples = len(stereo_audio.data)
        vocals_full = np.zeros((total_samples, 2), dtype=np.float32)
        background_full = stereo_audio.data.astype(np.float32, copy=True)

        for start, end in regions:
            chunk = audio.slice(start, end)
            separated_chunk = self._separate_local(chunk)
            chunk_vocals = separated_chunk.vocals.data
            chunk_background = separated_chunk.background.data

            # Demucs operates at its model sample rate (typically 44.1 kHz)
            # and returns stereo. The slice of `audio` we passed in may have
            # been resampled inside _separate_local, so resample the chunk
            # outputs back to the source sample rate before splicing.
            chunk_sr = separated_chunk.vocals.metadata.sample_rate
            if chunk_sr != sr:
                chunk_vocals = separated_chunk.vocals.resample(sr).data
                chunk_background = separated_chunk.background.resample(sr).data

            start_sample = int(start * sr)
            end_sample = min(start_sample + len(chunk_vocals), total_samples)
            length = end_sample - start_sample
            if length <= 0:
                continue

            vocals_full[start_sample:end_sample] = chunk_vocals[:length]
            background_full[start_sample:end_sample] = chunk_background[:length]

        metadata = AudioMetadata(
            sample_rate=sr,
            channels=2,
            sample_width=audio.metadata.sample_width,
            duration_seconds=total_samples / sr,
            frame_count=total_samples,
        )
        vocals = Audio(np.ascontiguousarray(vocals_full, dtype=np.float32), metadata)
        background = Audio(np.ascontiguousarray(background_full, dtype=np.float32), metadata)

        return SeparatedAudio(
            vocals=vocals,
            background=background,
            original=stereo_audio,
            music=None,
            effects=None,
        )

    def _passthrough_separation(self, audio: Audio) -> SeparatedAudio:
        """Return the original audio as background with silent vocals.

        Used when no speech regions are present — there's nothing to
        separate, so the entire signal is background by definition.
        """
        import numpy as np

        stereo_audio = audio if audio.metadata.channels == 2 else audio._to_stereo()
        silent_vocals_data = np.zeros_like(stereo_audio.data, dtype=np.float32)
        vocals = Audio(silent_vocals_data, stereo_audio.metadata)

        return SeparatedAudio(
            vocals=vocals,
            background=stereo_audio,
            original=stereo_audio,
            music=None,
            effects=None,
        )

    def extract_vocals(self, audio: Audio) -> Audio:
        """Convenience method to extract only vocals from audio."""
        return self.separate(audio).vocals

    def extract_background(self, audio: Audio) -> Audio:
        """Convenience method to extract only background from audio."""
        return self.separate(audio).background

    def unload(self) -> None:
        """Release the Demucs model so the next separate() re-initializes.

        Used by low-memory dubbing to free VRAM between pipeline stages.
        """
        self._model = None
        release_device_memory(self.device)
