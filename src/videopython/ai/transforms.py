"""AI-powered video transforms that require face detection."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import ClassVar, Literal

import cv2
import numpy as np
from pydantic import Field, model_validator
from tqdm import tqdm

from videopython.ai.understanding.faces import FaceTracker
from videopython.base.operation import OpCategory, Operation
from videopython.base.video import Video

logger = logging.getLogger(__name__)


def _make_even(value: int) -> int:
    """Round down to nearest even number for H.264 compatibility."""
    return value - (value % 2)


__all__ = [
    "FaceTrackingCrop",
    "SplitScreenComposite",
]


class FaceTrackingCrop(Operation):
    """Crops video to follow detected faces.

    Useful for creating vertical (9:16) content from horizontal (16:9) video
    by tracking the speaker's face and keeping it framed.

    Supports GPU acceleration for faster processing with optional frame sampling
    and simple cinematographic framing rules (headroom / thirds) plus optional
    movement speed clamping.

    Args:
        target_aspect: Output aspect ratio as (width, height).
        face_selection: Strategy for selecting which face to track.
        face_index: Index of face to track when using ``face_selection="index"``.
        padding: Extra space around face (0.3 = 30% padding on each side).
        vertical_offset: Legacy vertical position offset used by ``framing_rule="offset"``.
        framing_rule: Subject framing strategy.
            - "offset": Use legacy ``vertical_offset`` behavior.
            - "center": Keep face centered.
            - "headroom": Keep extra room above the face.
            - "thirds": Place face near the upper-third line.
            - "dynamic": Currently same as "headroom".
        headroom: Headroom amount for framing rules that use it.
        smoothing: Position smoothing factor (0-1, higher = smoother).
        max_speed: Optional max camera movement per frame (normalized).
        fallback: Behavior when no face detected.
        detection_interval: Frames between face detections.
        backend: Detection backend - "cpu", "gpu", or "auto".
        sample_rate: For GPU backend, detect every Nth frame and interpolate.
    """

    op: Literal["face_crop"] = "face_crop"
    category: ClassVar[OpCategory] = OpCategory.TRANSFORM

    target_aspect: tuple[int, int] = (9, 16)
    face_selection: Literal["largest", "centered", "index"] = "largest"
    face_index: int = Field(0, ge=0)
    padding: float = Field(0.3, ge=0)
    vertical_offset: float = -0.1
    framing_rule: Literal["offset", "center", "headroom", "thirds", "dynamic"] = "offset"
    headroom: float = 0.15
    smoothing: float = Field(0.8, ge=0, le=1)
    max_speed: float | None = Field(None, gt=0)
    fallback: Literal["center", "last_position", "full_frame"] = "last_position"
    detection_interval: int = Field(3, ge=1)
    backend: Literal["cpu", "gpu", "auto"] = "auto"
    sample_rate: int = Field(1, ge=1)

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
            crop_h = _make_even(frame_h)
            crop_w = _make_even(int(crop_h * target_ratio))
        else:
            crop_w = _make_even(frame_w)
            crop_h = _make_even(int(crop_w / target_ratio))

        min_face_dim = max(face_w * frame_w, face_h * frame_h)
        min_crop_dim = min_face_dim * (1 + 2 * self.padding)
        if crop_w < min_crop_dim * target_ratio:
            crop_w = _make_even(min(int(min_crop_dim * target_ratio), frame_w))
            crop_h = _make_even(min(int(crop_w / target_ratio), frame_h))

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
        target_ratio = self.target_aspect[0] / self.target_aspect[1]
        if target_ratio < w / h:
            out_h = _make_even(h)
            out_w = _make_even(int(out_h * target_ratio))
        else:
            out_w = _make_even(w)
            out_h = _make_even(int(out_w / target_ratio))

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


class SplitScreenComposite(Operation):
    """Arranges multiple face-tracked crops in a grid layout.

    Useful for interview-style videos, reaction videos, or showing multiple
    perspectives simultaneously. Extra source videos beyond the primary input
    are loaded just-in-time from ``additional_sources`` so the op stays
    JSON-serialisable.

    Args:
        layout: Grid layout for cells.
            - "2x1": Two cells side by side (horizontal)
            - "1x2": Two cells stacked (vertical)
            - "2x2": Four cells in 2x2 grid
            - "1+2": One large cell on left, two small on right
            - "2+1": Two small cells on left, one large on right
        additional_sources: Paths to extra source videos used to fill the
            remaining cells (the primary cell is filled by the input ``video``).
        output_size: Output dimensions (width, height). If None, use source size.
        gap: Gap between cells in pixels.
        gap_color: Color of gap between cells (RGB).
        border_width: Border width around each cell.
        border_color: Border color (RGB).
        face_padding: Extra space around face in each cell.
        smoothing: Position smoothing factor.
        detection_interval: Frames between face detections.
        audio_source: Audio handling ("main" uses first source).
        backend: Detection backend - "cpu", "gpu", or "auto".
        sample_rate: For GPU backend, detect every Nth frame and interpolate.
    """

    op: Literal["split_screen"] = "split_screen"
    category: ClassVar[OpCategory] = OpCategory.TRANSFORM

    layout: Literal["2x1", "1x2", "2x2", "1+2", "2+1"] = "2x1"
    additional_sources: list[Path] = Field(default_factory=list)
    output_size: tuple[int, int] | None = None
    gap: int = Field(4, ge=0)
    gap_color: tuple[int, int, int] = (0, 0, 0)
    border_width: int = Field(0, ge=0)
    border_color: tuple[int, int, int] = (255, 255, 255)
    face_padding: float = Field(0.2, ge=0)
    smoothing: float = Field(0.8, ge=0, le=1)
    detection_interval: int = Field(3, ge=1)
    audio_source: Literal["main", "loudest", "mix"] = "main"
    backend: Literal["cpu", "gpu", "auto"] = "auto"
    sample_rate: int = Field(1, ge=1)

    @model_validator(mode="after")
    def _validate_source_count(self) -> SplitScreenComposite:
        required = self._required_extras()
        if len(self.additional_sources) < required:
            raise ValueError(
                f"Layout '{self.layout}' requires {required} additional_sources, got {len(self.additional_sources)}"
            )
        return self

    def _required_extras(self) -> int:
        """Number of EXTRA sources needed beyond the primary input."""
        if self.layout in ("2x1", "1x2"):
            return 1
        if self.layout == "2x2":
            return 3
        # 1+2, 2+1
        return 2

    def _get_cell_rects(self, width: int, height: int) -> list[tuple[int, int, int, int]]:
        gap = self.gap
        if self.layout == "2x1":
            cell_w = (width - gap) // 2
            return [(0, 0, cell_w, height), (cell_w + gap, 0, width - cell_w - gap, height)]
        if self.layout == "1x2":
            cell_h = (height - gap) // 2
            return [(0, 0, width, cell_h), (0, cell_h + gap, width, height - cell_h - gap)]
        if self.layout == "2x2":
            cell_w = (width - gap) // 2
            cell_h = (height - gap) // 2
            return [
                (0, 0, cell_w, cell_h),
                (cell_w + gap, 0, width - cell_w - gap, cell_h),
                (0, cell_h + gap, cell_w, height - cell_h - gap),
                (cell_w + gap, cell_h + gap, width - cell_w - gap, height - cell_h - gap),
            ]
        if self.layout == "1+2":
            large_w = (width - gap) * 2 // 3
            small_w = width - large_w - gap
            small_h = (height - gap) // 2
            return [
                (0, 0, large_w, height),
                (large_w + gap, 0, small_w, small_h),
                (large_w + gap, small_h + gap, small_w, height - small_h - gap),
            ]
        # "2+1"
        large_w = (width - gap) * 2 // 3
        small_w = width - large_w - gap
        small_h = (height - gap) // 2
        return [
            (0, 0, small_w, small_h),
            (0, small_h + gap, small_w, height - small_h - gap),
            (small_w + gap, 0, large_w, height),
        ]

    def apply(self, video: Video) -> Video:
        extras = [Video.from_path(str(p)) for p in self.additional_sources]
        all_videos = [video] + extras
        n_frames = len(video.frames)

        if self.output_size:
            out_w, out_h = self.output_size
        else:
            out_w, out_h = video.frame_shape[1], video.frame_shape[0]
        out_w = _make_even(out_w)
        out_h = _make_even(out_h)

        cell_rects = self._get_cell_rects(out_w, out_h)
        trackers = [
            FaceTracker(
                selection_strategy="largest",
                smoothing=self.smoothing,
                detection_interval=self.detection_interval,
                backend=self.backend,
                sample_rate=self.sample_rate,
            )
            for _ in range(len(cell_rects))
        ]

        logger.info("Creating %s split screen: %dx%d", self.layout, out_w, out_h)

        new_frames = []
        for i in tqdm(range(n_frames), desc="Split screen composite"):
            output = np.full((out_h, out_w, 3), self.gap_color, dtype=np.uint8)

            for cell_idx, (cx, cy, cw, ch) in enumerate(cell_rects):
                if cell_idx >= len(all_videos):
                    break
                src_video = all_videos[cell_idx]
                src_frame = src_video.frames[i % len(src_video.frames)]
                src_h, src_w = src_frame.shape[:2]
                face_info = trackers[cell_idx].detect_and_track(src_frame, i)

                cell_aspect = cw / ch
                src_aspect = src_w / src_h

                if face_info:
                    fcx, fcy, _fw, _fh = face_info
                    if cell_aspect < src_aspect:
                        crop_h = _make_even(src_h)
                        crop_w = _make_even(int(crop_h * cell_aspect))
                    else:
                        crop_w = _make_even(src_w)
                        crop_h = _make_even(int(crop_w / cell_aspect))
                    center_x = int(fcx * src_w)
                    center_y = int(fcy * src_h)
                    crop_x = max(0, min(center_x - crop_w // 2, src_w - crop_w))
                    crop_y = max(0, min(center_y - crop_h // 2, src_h - crop_h))
                else:
                    if cell_aspect < src_aspect:
                        crop_h = _make_even(src_h)
                        crop_w = _make_even(int(crop_h * cell_aspect))
                        crop_x = (src_w - crop_w) // 2
                        crop_y = 0
                    else:
                        crop_w = _make_even(src_w)
                        crop_h = _make_even(int(crop_w / cell_aspect))
                        crop_x = 0
                        crop_y = (src_h - crop_h) // 2

                cropped = src_frame[crop_y : crop_y + crop_h, crop_x : crop_x + crop_w]
                resized = cv2.resize(cropped, (cw, ch), interpolation=cv2.INTER_AREA)

                if self.border_width > 0:
                    bw = self.border_width
                    resized[:bw, :] = self.border_color
                    resized[-bw:, :] = self.border_color
                    resized[:, :bw] = self.border_color
                    resized[:, -bw:] = self.border_color

                output[cy : cy + ch, cx : cx + cw] = resized

            new_frames.append(output)

        video.frames = np.array(new_frames, dtype=np.uint8)
        # Audio handling — keep main video audio (mixing left as follow-up).
        return video
