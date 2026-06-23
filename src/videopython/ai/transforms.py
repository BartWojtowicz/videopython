"""AI-powered video transforms that require face detection."""

from __future__ import annotations

import logging
import tempfile
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import ClassVar, Literal

import numpy as np
from pydantic import Field
from tqdm import tqdm

from videopython.ai.understanding.faces import FaceSmoothingTracker
from videopython.base._dimensions import floor_to_even
from videopython.base._ffmpeg import escape_filter_value
from videopython.base.video import FrameIterator, VideoMetadata
from videopython.editing.operation import FilterCtx, OpCategory, Operation

logger = logging.getLogger(__name__)


__all__ = [
    "FaceTrackingCrop",
]


class FaceTrackingCrop(Operation):
    """Crops video to follow detected faces.

    Useful for creating vertical (9:16) content from horizontal (16:9) video
    by tracking the speaker's face and keeping it framed.

    The crop window has a fixed size -- the largest ``target_aspect`` box
    that fits the frame (also the output size, so no resampling happens) --
    and its position follows the smoothed face track. On the streaming path
    the detection pass runs at plan-compile time over a bounded decode of
    exactly the frames the filter will see, and the track compiles to a
    per-frame ``crop`` position command file (ffmpeg ``sendcmd``): zero
    per-frame Python at render time.
    """

    op: Literal["face_crop"] = "face_crop"
    category: ClassVar[OpCategory] = OpCategory.TRANSFORM
    compiles_from_source: ClassVar[bool] = True

    target_aspect: tuple[int, int] = Field((9, 16), description="Output aspect ratio as (width, height).")
    face_selection: Literal["largest", "centered", "index"] = Field(
        "largest", description="Strategy for selecting which face to track."
    )
    face_index: int = Field(0, ge=0, description='Index of face to track when using ``face_selection="index"``.')
    padding: float = Field(0.3, ge=0, description="Extra space around face (0.3 = 30% padding on each side).")
    vertical_offset: float = Field(
        -0.1, description='Legacy vertical position offset used by ``framing_rule="offset"``.'
    )
    framing_rule: Literal["offset", "center", "headroom", "thirds", "dynamic"] = Field(
        "offset",
        description=(
            'Subject framing strategy. "offset": legacy ``vertical_offset`` behavior; '
            '"center": keep face centered; "headroom": extra room above the face; '
            '"thirds": face near the upper-third line; "dynamic": currently same as "headroom".'
        ),
    )
    headroom: float = Field(0.15, description="Headroom amount for framing rules that use it.")
    smoothing: float = Field(0.8, ge=0, le=1, description="Position smoothing factor (0-1, higher = smoother).")
    max_speed: float | None = Field(None, gt=0, description="Optional max camera movement per frame (normalized).")
    fallback: Literal["center", "last_position", "full_frame"] = Field(
        "last_position",
        description=(
            'Behavior when no face detected. "center" and "full_frame" both center the crop '
            '("full_frame" kept for plan compatibility); "last_position" holds the last tracked crop.'
        ),
    )
    detection_interval: int = Field(3, ge=1, description="Frames between face detections.")

    def _apply_framing_offset(self, face_cx: float, face_cy: float, face_h: float) -> tuple[float, float]:
        if self.framing_rule == "offset":
            return (face_cx, face_cy + self.vertical_offset)
        if self.framing_rule == "center":
            return (face_cx, face_cy)
        if self.framing_rule == "headroom":
            return (face_cx, face_cy - self.headroom)
        if self.framing_rule == "thirds":
            return (face_cx, face_cy - (1 / 3 - 0.5))
        # "dynamic" — placeholder until motion/look-direction framing is implemented.
        return (face_cx, face_cy - self.headroom)

    def _resolved_output_dims(self, w: int, h: int) -> tuple[int, int]:
        """Output ``(width, height)`` -- the fixed crop-window size.

        The largest ``target_aspect`` box that fits the frame, even-floored.
        A pure function of the input dimensions, shared by
        :meth:`predict_metadata` and :meth:`to_ffmpeg_filter`, so the
        dry-run cannot disagree with the render.
        """
        target_ratio = self.target_aspect[0] / self.target_aspect[1]
        if target_ratio < w / h:
            out_h = floor_to_even(h)
            out_w = floor_to_even(int(out_h * target_ratio))
        else:
            out_w = floor_to_even(w)
            out_h = floor_to_even(int(out_w / target_ratio))
        return out_w, out_h

    def predict_metadata(self, meta: VideoMetadata) -> VideoMetadata:
        out_w, out_h = self._resolved_output_dims(meta.width, meta.height)
        return meta.with_dimensions(out_w, out_h)

    def _clamp_speed(self, current: tuple[float, float], target: tuple[float, float]) -> tuple[float, float]:
        if self.max_speed is None:
            return target
        dx = target[0] - current[0]
        dy = target[1] - current[1]
        distance = (dx**2 + dy**2) ** 0.5
        if distance <= self.max_speed or distance == 0:
            return target
        scale = self.max_speed / distance
        return (current[0] + dx * scale, current[1] + dy * scale)

    def _track_crop_positions(
        self,
        frames: Iterable[np.ndarray],
        frame_w: int,
        frame_h: int,
    ) -> list[tuple[int, int]]:
        """Per-frame crop top-left positions for a fixed-size crop window.

        The single source of the tracking math (detection cadence, EMA
        smoothing, framing offset, speed clamp, frame clamping), run by the
        compile-time detection pass to build the per-frame crop command file.
        """
        out_w, out_h = self._resolved_output_dims(frame_w, frame_h)
        tracker = FaceSmoothingTracker(
            selection_strategy=self.face_selection,
            face_index=self.face_index,
            smoothing=self.smoothing,
            detection_interval=self.detection_interval,
        )
        default = ((frame_w - out_w) // 2, (frame_h - out_h) // 2)
        last = default
        current_position = (0.5, 0.5)
        positions: list[tuple[int, int]] = []
        for i, frame in enumerate(frames):
            face_info = tracker.detect_and_track(frame, i)
            if face_info:
                cx, cy, _fw, fh = face_info
                target = self._apply_framing_offset(cx, cy, fh)
                current_position = self._clamp_speed(current_position, target)
                x = int(current_position[0] * frame_w - out_w / 2)
                y = int(current_position[1] * frame_h - out_h / 2)
                x = max(0, min(x, frame_w - out_w))
                y = max(0, min(y, frame_h - out_h))
                last = (x, y)
                positions.append((x, y))
            elif self.fallback == "last_position":
                positions.append(last)
            else:  # "center" / "full_frame" (the latter kept for plan compat)
                positions.append(default)
        return positions

    def to_ffmpeg_filter(self, ctx: FilterCtx) -> str | None:
        """Compile the face track to a per-frame ``crop`` position command file.

        Runs the detection pass at plan-compile time over a bounded decode of
        the segment (through the same decode-stage filter prefix the render
        will use, so the detector sees identical frames), then emits one
        ``sendcmd`` interval per frame driving a fixed-size ``crop``. Returns
        ``None`` when the input frames are not reproducible at compile time
        (``decode_filters is None`` -- the op sits behind per-frame Python
        effects) or the source is unknown.
        """
        if ctx.source_path is None or ctx.decode_filters is None or ctx.frame_count <= 0:
            return None
        out_w, out_h = self._resolved_output_dims(ctx.width, ctx.height)

        with FrameIterator(
            ctx.source_path,
            start_second=ctx.start_second,
            end_second=ctx.end_second,
            vf_filters=list(ctx.decode_filters),
            output_width=ctx.width,
            output_height=ctx.height,
        ) as decoder:
            frames = (frame for _, frame in decoder)
            positions = self._track_crop_positions(
                tqdm(frames, desc="Face tracking (compile)", total=ctx.frame_count), ctx.width, ctx.height
            )
        if not positions:
            return None

        label = f"fc{uuid.uuid4().hex[:8]}"
        lines = []
        for i, (x, y) in enumerate(positions):
            t0 = i / ctx.fps
            t1 = (i + 1) / ctx.fps
            lines.append(f"{t0:.6f}-{t1:.6f} crop@{label} x {x}, crop@{label} y {y};")
        tmp = tempfile.NamedTemporaryFile("w", suffix=".cmd", delete=False, encoding="utf-8")
        try:
            tmp.write("\n".join(lines) + "\n")
        finally:
            tmp.close()
        cmd_path = Path(tmp.name)
        ctx.owned_files.append(cmd_path)

        x0, y0 = positions[0]
        return f"sendcmd=f={escape_filter_value(str(cmd_path))},crop@{label}=w={out_w}:h={out_h}:x={x0}:y={y0}"
