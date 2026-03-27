from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from videopython.base.audio import Audio
from videopython.base.transitions import InstantTransition, Transition
from videopython.base.video import Video, VideoMetadata

__all__ = [
    "CutPoint",
    "MultiCamEdit",
]


@dataclass(frozen=True)
class CutPoint:
    """A camera switch point in a multicam timeline.

    Attributes:
        time: Seconds into the timeline where this cut happens.
        camera: Key into the MultiCamEdit.sources dict.
        transition: Transition to use when switching to this camera.
            None means use the MultiCamEdit.default_transition.
    """

    time: float
    camera: str
    transition: Transition | None = None


class MultiCamEdit:
    """Multicam timeline editor for podcast-style recordings.

    Switches between synchronized camera angles at specified cut points,
    joining segments with transitions and replacing audio with an external
    track (or silence).
    """

    def __init__(
        self,
        sources: dict[str, str | Path],
        cuts: Sequence[CutPoint],
        audio_source: str | Path | None = None,
        default_transition: Transition | None = None,
        source_offsets: dict[str, float] | None = None,
    ):
        if not sources:
            raise ValueError("MultiCamEdit requires at least one source")
        if not cuts:
            raise ValueError("MultiCamEdit requires at least one cut point")

        self.sources: dict[str, Path] = {k: Path(v) for k, v in sources.items()}
        self.cuts: tuple[CutPoint, ...] = tuple(cuts)
        self.audio_source: Path | None = Path(audio_source) if audio_source else None
        self.default_transition: Transition = default_transition or InstantTransition()
        self.source_offsets: dict[str, float] = source_offsets or {}

        self._validate()

    def _validate(self) -> None:
        # Sources must exist
        for name, path in self.sources.items():
            if not path.exists():
                raise FileNotFoundError(f"Source '{name}' not found: {path}")

        # Audio source must exist if provided
        if self.audio_source and not self.audio_source.exists():
            raise FileNotFoundError(f"Audio source not found: {self.audio_source}")

        # First cut must start at time 0
        if self.cuts[0].time != 0.0:
            raise ValueError(f"First cut must start at time 0.0, got {self.cuts[0].time}")

        # Cuts must be in ascending order
        for i in range(1, len(self.cuts)):
            if self.cuts[i].time <= self.cuts[i - 1].time:
                raise ValueError(
                    f"Cuts must be in strictly ascending order: "
                    f"cut {i} time ({self.cuts[i].time}) <= cut {i - 1} time ({self.cuts[i - 1].time})"
                )

        # All camera references must be valid
        for i, cut in enumerate(self.cuts):
            if cut.camera not in self.sources:
                raise ValueError(
                    f"Cut {i} references unknown camera '{cut.camera}'. Available: {sorted(self.sources.keys())}"
                )

        # All offset keys must reference valid sources
        for name in self.source_offsets:
            if name not in self.sources:
                raise ValueError(
                    f"source_offsets references unknown source '{name}'. Available: {sorted(self.sources.keys())}"
                )

        # All sources must have compatible fps and resolution
        metas: dict[str, VideoMetadata] = {}
        for name, path in self.sources.items():
            metas[name] = VideoMetadata.from_path(str(path))

        meta_list = list(metas.values())
        first = meta_list[0]
        for name, meta in metas.items():
            if meta.fps != first.fps:
                raise ValueError(
                    f"Source '{name}' has fps {meta.fps}, expected {first.fps}. All sources must have the same fps."
                )
            if (meta.width, meta.height) != (first.width, first.height):
                raise ValueError(
                    f"Source '{name}' has resolution {meta.width}x{meta.height}, "
                    f"expected {first.width}x{first.height}. "
                    f"All sources must have the same resolution."
                )

        # Cache source metadata for validate() and run()
        self._source_meta = first
        self._source_duration = first.total_seconds
        self._source_metas = metas

        # Build per-camera time ranges (cut start, cut end) from the timeline
        camera_ranges: dict[str, list[tuple[float, float]]] = {}
        for i, cut in enumerate(self.cuts):
            start = cut.time
            end = self.cuts[i + 1].time if i + 1 < len(self.cuts) else self._source_duration
            camera_ranges.setdefault(cut.camera, []).append((start, end))

        # Validate adjusted seek positions per source
        for camera, ranges in camera_ranges.items():
            offset = self.source_offsets.get(camera, 0.0)
            source_dur = metas[camera].total_seconds
            for start, end in ranges:
                adj_start = start - offset
                adj_end = end - offset
                if adj_start < 0:
                    raise ValueError(
                        f"Cut at timeline {start}s for '{camera}' (offset {offset}s) "
                        f"results in negative seek position ({adj_start}s)"
                    )
                if adj_end > source_dur:
                    raise ValueError(
                        f"Cut ending at timeline {end}s for '{camera}' (offset {offset}s) "
                        f"exceeds source duration ({source_dur}s)"
                    )

    def run(self) -> Video:
        """Execute the multicam edit and return the final video."""
        source_duration = self._source_duration

        # Build time ranges: each segment runs from its cut time to the next cut time
        segments: list[tuple[CutPoint, float, float]] = []
        for i, cut in enumerate(self.cuts):
            start = cut.time
            end = self.cuts[i + 1].time if i + 1 < len(self.cuts) else source_duration
            segments.append((cut, start, end))

        # Load and join segments
        result: Video | None = None
        for i, (cut, start, end) in enumerate(segments):
            source_path = self.sources[cut.camera]
            offset = self.source_offsets.get(cut.camera, 0.0)
            segment = Video.from_path(str(source_path), start_second=start - offset, end_second=end - offset)

            if result is None:
                result = segment
            else:
                transition = cut.transition or self.default_transition
                result = transition.apply((result, segment))

        assert result is not None

        # Replace audio
        if self.audio_source:
            audio = Audio.from_path(self.audio_source)
            audio = audio.fit_to_duration(result.total_seconds)
        else:
            audio = Audio.create_silent(
                duration_seconds=result.total_seconds,
                sample_rate=result.audio.metadata.sample_rate,
            )
        result.audio = audio

        return result

    def validate(self) -> VideoMetadata:
        """Validate the plan and predict output metadata without loading frames."""
        total_seconds = self._source_duration
        fps = self._source_meta.fps

        # Subtract overlap consumed by transitions
        for i in range(1, len(self.cuts)):
            transition = self.cuts[i].transition or self.default_transition
            effect_time = getattr(transition, "effect_time_seconds", 0.0)
            if effect_time > 0:
                total_seconds -= effect_time

        total_seconds = round(total_seconds, 4)
        frame_count = math.floor(total_seconds * fps)

        return VideoMetadata(
            width=self._source_meta.width,
            height=self._source_meta.height,
            fps=fps,
            frame_count=frame_count,
            total_seconds=total_seconds,
        )

    @classmethod
    def json_schema(cls) -> dict[str, Any]:
        """Return a JSON Schema for MultiCamEdit plans."""
        transition_schemas = [
            {
                "type": "object",
                "properties": {"type": {"const": "instant"}},
                "required": ["type"],
                "additionalProperties": False,
            },
            {
                "type": "object",
                "properties": {
                    "type": {"const": "fade"},
                    "effect_time_seconds": {
                        "type": "number",
                        "exclusiveMinimum": 0,
                        "description": "Duration of the crossfade in seconds.",
                    },
                },
                "required": ["type", "effect_time_seconds"],
                "additionalProperties": False,
            },
            {
                "type": "object",
                "properties": {
                    "type": {"const": "blur"},
                    "effect_time_seconds": {
                        "type": "number",
                        "exclusiveMinimum": 0,
                        "description": "Duration of the blur transition in seconds.",
                    },
                    "blur_iterations": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Blur strength at peak.",
                    },
                    "blur_kernel_size": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 2,
                        "maxItems": 2,
                        "description": "Gaussian kernel [width, height] in pixels.",
                    },
                },
                "required": ["type"],
                "additionalProperties": False,
            },
        ]

        cut_schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "time": {
                    "type": "number",
                    "minimum": 0,
                    "description": "Seconds into the timeline where this cut happens.",
                },
                "camera": {
                    "type": "string",
                    "description": "Camera name (key into sources).",
                },
                "transition": {
                    "oneOf": transition_schemas,
                    "description": "Transition to use at this cut. Omit to use default_transition.",
                },
            },
            "required": ["time", "camera"],
            "additionalProperties": False,
        }

        return {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {
                "sources": {
                    "type": "object",
                    "description": "Named camera sources. Keys are camera names, values are file paths.",
                    "additionalProperties": {"type": "string"},
                    "minProperties": 1,
                },
                "source_offsets": {
                    "type": "object",
                    "additionalProperties": {"type": "number"},
                    "description": "Per-source time offsets in seconds. "
                    "Positive means the source starts later than the timeline origin.",
                },
                "audio_source": {
                    "type": "string",
                    "description": "Path to external audio track. Omit for silent output.",
                },
                "cuts": {
                    "type": "array",
                    "items": cut_schema,
                    "minItems": 1,
                    "description": "Ordered list of camera switches. First cut must have time=0.",
                },
                "default_transition": {
                    "oneOf": transition_schemas,
                    "description": "Transition used between cuts when not specified per-cut.",
                },
            },
            "required": ["sources", "cuts"],
            "additionalProperties": False,
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        result: dict[str, Any] = {
            "sources": {k: str(v) for k, v in self.sources.items()},
            "cuts": [],
            "default_transition": self.default_transition.to_dict(),
        }
        if self.source_offsets:
            result["source_offsets"] = dict(self.source_offsets)
        if self.audio_source:
            result["audio_source"] = str(self.audio_source)

        for cut in self.cuts:
            cut_dict: dict[str, Any] = {"time": cut.time, "camera": cut.camera}
            if cut.transition is not None:
                cut_dict["transition"] = cut.transition.to_dict()
            result["cuts"].append(cut_dict)

        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MultiCamEdit:
        """Deserialize from a dict."""
        if not isinstance(data, dict):
            raise ValueError("MultiCamEdit plan must be a JSON object")

        sources = data.get("sources")
        if not isinstance(sources, dict) or not sources:
            raise ValueError("MultiCamEdit plan must have a non-empty 'sources' dict")

        cuts_data = data.get("cuts")
        if not isinstance(cuts_data, list) or not cuts_data:
            raise ValueError("MultiCamEdit plan must have a non-empty 'cuts' list")

        cuts: list[CutPoint] = []
        for i, cut_data in enumerate(cuts_data):
            if not isinstance(cut_data, dict):
                raise ValueError(f"cuts[{i}] must be an object")
            transition = None
            if "transition" in cut_data:
                transition = Transition.from_dict(cut_data["transition"])
            cuts.append(
                CutPoint(
                    time=cut_data["time"],
                    camera=cut_data["camera"],
                    transition=transition,
                )
            )

        default_transition = None
        if "default_transition" in data:
            default_transition = Transition.from_dict(data["default_transition"])

        return cls(
            sources=sources,
            cuts=cuts,
            audio_source=data.get("audio_source"),
            default_transition=default_transition,
            source_offsets=data.get("source_offsets"),
        )

    @classmethod
    def from_json(cls, text: str) -> MultiCamEdit:
        """Deserialize from a JSON string."""
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid MultiCamEdit JSON: {e.msg} at line {e.lineno} column {e.colno}") from e
        return cls.from_dict(data)
