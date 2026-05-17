"""AI-powered video transforms that require face detection."""

from __future__ import annotations

import logging
from typing import ClassVar, Literal

import cv2
import numpy as np
from pydantic import Field
from tqdm import tqdm

from videopython.ai.understanding.faces import FaceTracker
from videopython.base._dimensions import floor_to_even
from videopython.base.video import Video, VideoMetadata
from videopython.editing.operation import OpCategory, Operation

logger = logging.getLogger(__name__)


__all__ = [
    "FaceTrackingCrop",
]


class FaceTrackingCrop(Operation):
    """Crops video to follow detected faces.

    Useful for creating vertical (9:16) content from horizontal (16:9) video
    by tracking the speaker's face and keeping it framed.

    Supports GPU acceleration for faster processing with optional frame sampling
    and simple cinematographic framing rules (headroom / thirds) plus optional
    movement speed clamping.
    """

    op: Literal["face_crop"] = "face_crop"
    category: ClassVar[OpCategory] = OpCategory.TRANSFORM

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
        "last_position", description="Behavior when no face detected."
    )
    detection_interval: int = Field(3, ge=1, description="Frames between face detections.")
    backend: Literal["cpu", "gpu", "auto"] = Field("auto", description='Detection backend - "cpu", "gpu", or "auto".')
    sample_rate: int = Field(1, ge=1, description="For GPU backend, detect every Nth frame and interpolate.")

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
        """Output ``(width, height)`` after the crop + resize.

        Every frame is resized to this size regardless of the per-frame face
        position, so it is a pure function of the input dimensions and
        ``target_aspect``. Single source of truth shared by :meth:`apply` and
        :meth:`predict_metadata` (mirrors ``Resize._resolve_dims`` /
        ``Crop._resolve_box``), so the dry-run cannot disagree with the render.
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

    def _calculate_crop_region(
        self,
        face_cx: float,
        face_cy: float,
        face_w: float,
        face_h: float,
        frame_w: int,
        frame_h: int,
        center_position: tuple[float, float] | None = None,
    ) -> tuple[int, int, int, int]:
        target_ratio = self.target_aspect[0] / self.target_aspect[1]
        frame_ratio = frame_w / frame_h

        if target_ratio < frame_ratio:
            crop_h = floor_to_even(frame_h)
            crop_w = floor_to_even(int(crop_h * target_ratio))
        else:
            crop_w = floor_to_even(frame_w)
            crop_h = floor_to_even(int(crop_w / target_ratio))

        min_face_dim = max(face_w * frame_w, face_h * frame_h)
        min_crop_dim = min_face_dim * (1 + 2 * self.padding)
        if crop_w < min_crop_dim * target_ratio:
            crop_w = floor_to_even(min(int(min_crop_dim * target_ratio), frame_w))
            crop_h = floor_to_even(min(int(crop_w / target_ratio), frame_h))

        if center_position is None:
            center_position = self._apply_framing_offset(face_cx, face_cy, face_h)

        center_x = center_position[0] * frame_w
        center_y = center_position[1] * frame_h
        x = int(center_x - crop_w / 2)
        y = int(center_y - crop_h / 2)
        x = max(0, min(x, frame_w - crop_w))
        y = max(0, min(y, frame_h - crop_h))
        return (x, y, crop_w, crop_h)

    def apply(self, video: Video) -> Video:
        tracker = FaceTracker(
            selection_strategy=self.face_selection,
            face_index=self.face_index,
            smoothing=self.smoothing,
            detection_interval=self.detection_interval,
            backend=self.backend,
            sample_rate=self.sample_rate,
        )

        h, w = video.frame_shape[:2]
        out_w, out_h = self._resolved_output_dims(w, h)

        default_x = (w - out_w) // 2
        default_y = (h - out_h) // 2
        last_crop = (default_x, default_y, out_w, out_h)
        current_position = (0.5, 0.5)

        framing_label = self.framing_rule if self.framing_rule != "offset" else "legacy-offset"
        logger.info(
            "Face tracking crop: %dx%d -> %dx%d (%d:%d, framing=%s)",
            w,
            h,
            out_w,
            out_h,
            self.target_aspect[0],
            self.target_aspect[1],
            framing_label,
        )

        new_frames = []
        for i in tqdm(range(len(video.frames)), desc="Face tracking crop"):
            frame = video.frames[i]
            face_info = tracker.detect_and_track(frame, i)

            if face_info:
                cx, cy, fw, fh = face_info
                target_position = self._apply_framing_offset(cx, cy, fh)
                current_position = self._clamp_speed(current_position, target_position)
                crop = self._calculate_crop_region(cx, cy, fw, fh, w, h, center_position=current_position)
                last_crop = crop
            else:
                if self.fallback == "center":
                    crop = (default_x, default_y, out_w, out_h)
                elif self.fallback == "last_position":
                    crop = last_crop
                else:  # full_frame
                    crop = (0, 0, w, h)

            x, y, cw, ch = crop
            cropped = frame[y : y + ch, x : x + cw]
            if cropped.shape[1] != out_w or cropped.shape[0] != out_h:
                cropped = cv2.resize(cropped, (out_w, out_h), interpolation=cv2.INTER_AREA)
            new_frames.append(cropped)

        video.frames = np.array(new_frames, dtype=np.uint8)
        return video
