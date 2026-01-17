"""AI-powered video transforms that require face detection."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import cv2
import numpy as np
from tqdm import tqdm

from videopython.ai.understanding.detection import FaceDetector
from videopython.base.transforms import Transformation
from videopython.base.video import Video

if TYPE_CHECKING:
    pass


def _make_even(value: int) -> int:
    """Round down to nearest even number for H.264 compatibility."""
    return value - (value % 2)


__all__ = [
    "FaceTracker",
    "FaceTrackingCrop",
    "SplitScreenComposite",
    "AutoFramingCrop",
]


class FaceTracker:
    """Utility for tracking faces across video frames with smoothing.

    Provides frame-by-frame face detection with position smoothing using
    exponential moving average to prevent jitter in the tracked position.
    """

    def __init__(
        self,
        selection_strategy: Literal["largest", "centered", "index"] = "largest",
        face_index: int = 0,
        smoothing: float = 0.8,
        detection_interval: int = 3,
        min_face_size: int = 30,
    ):
        """Initialize face tracker.

        Args:
            selection_strategy: How to select which face to track.
                - "largest": Track the face with the largest bounding box.
                - "centered": Track the face closest to frame center.
                - "index": Track the face at a specific index (sorted by area).
            face_index: Index of face to track when using "index" strategy.
            smoothing: Exponential moving average factor (0-1). Higher = smoother.
            detection_interval: Run detection every N frames, interpolate between.
            min_face_size: Minimum face size in pixels for detection.
        """
        self.selection_strategy = selection_strategy
        self.face_index = face_index
        self.smoothing = smoothing
        self.detection_interval = detection_interval
        self.min_face_size = min_face_size

        self._detector: FaceDetector | None = None
        self._last_position: tuple[float, float] | None = None
        self._last_size: tuple[float, float] | None = None
        self._smoothed_position: tuple[float, float] | None = None
        self._smoothed_size: tuple[float, float] | None = None

    def _init_detector(self) -> None:
        """Initialize face detector lazily."""
        self._detector = FaceDetector(min_face_size=self.min_face_size)

    def _select_face(
        self,
        faces: list,
        frame_width: int,
        frame_height: int,
    ) -> tuple[float, float, float, float] | None:
        """Select a face based on the configured strategy.

        Args:
            faces: List of DetectedFace objects.
            frame_width: Width of the frame.
            frame_height: Height of the frame.

        Returns:
            Tuple of (center_x, center_y, width, height) in normalized coords, or None.
        """
        if not faces:
            return None

        if self.selection_strategy == "largest":
            # Faces are already sorted by area (largest first)
            face = faces[0]
        elif self.selection_strategy == "centered":
            # Find face closest to center
            frame_center = (0.5, 0.5)
            face = min(
                faces,
                key=lambda f: (
                    (f.bounding_box.center[0] - frame_center[0]) ** 2
                    + (f.bounding_box.center[1] - frame_center[1]) ** 2
                ),
            )
        elif self.selection_strategy == "index":
            if self.face_index < len(faces):
                face = faces[self.face_index]
            else:
                face = faces[0]  # Fall back to largest
        else:
            face = faces[0]

        bbox = face.bounding_box
        return (bbox.center[0], bbox.center[1], bbox.width, bbox.height)

    def detect_and_track(
        self,
        frame: np.ndarray,
        frame_index: int,
    ) -> tuple[float, float, float, float] | None:
        """Detect face in frame and return smoothed position.

        Args:
            frame: Video frame as numpy array (H, W, 3).
            frame_index: Index of current frame.

        Returns:
            Tuple of (center_x, center_y, width, height) in normalized coords,
            or None if no face detected and no fallback available.
        """
        if self._detector is None:
            self._init_detector()
            assert self._detector is not None

        h, w = frame.shape[:2]

        # Only run detection on interval frames
        should_detect = frame_index % self.detection_interval == 0

        if should_detect:
            faces = self._detector.detect(frame)
            face_info = self._select_face(faces, w, h)

            if face_info:
                cx, cy, fw, fh = face_info
                self._last_position = (cx, cy)
                self._last_size = (fw, fh)
        else:
            # Use last detected position
            face_info = None
            if self._last_position and self._last_size:
                face_info = (*self._last_position, *self._last_size)

        if face_info:
            cx, cy, fw, fh = face_info

            # Apply exponential moving average smoothing
            if self._smoothed_position is None:
                self._smoothed_position = (cx, cy)
                self._smoothed_size = (fw, fh)
            else:
                alpha = 1 - self.smoothing
                self._smoothed_position = (
                    self._smoothed_position[0] * self.smoothing + cx * alpha,
                    self._smoothed_position[1] * self.smoothing + cy * alpha,
                )
                assert self._smoothed_size is not None  # Set alongside _smoothed_position
                self._smoothed_size = (
                    self._smoothed_size[0] * self.smoothing + fw * alpha,
                    self._smoothed_size[1] * self.smoothing + fh * alpha,
                )

            return (*self._smoothed_position, *self._smoothed_size)

        # Return last smoothed position as fallback
        if self._smoothed_position and self._smoothed_size:
            return (*self._smoothed_position, *self._smoothed_size)

        return None

    def reset(self) -> None:
        """Reset tracker state for a new video."""
        self._last_position = None
        self._last_size = None
        self._smoothed_position = None
        self._smoothed_size = None


class FaceTrackingCrop(Transformation):
    """Crops video to follow detected faces.

    Useful for creating vertical (9:16) content from horizontal (16:9) video
    by tracking the speaker's face and keeping it centered.
    """

    def __init__(
        self,
        target_aspect: tuple[int, int] = (9, 16),
        face_selection: Literal["largest", "centered", "index"] = "largest",
        face_index: int | None = None,
        padding: float = 0.3,
        vertical_offset: float = -0.1,
        smoothing: float = 0.8,
        fallback: Literal["center", "last_position", "full_frame"] = "last_position",
        detection_interval: int = 3,
    ):
        """Initialize face tracking crop.

        Args:
            target_aspect: Output aspect ratio as (width, height).
            face_selection: Strategy for selecting which face to track.
            face_index: Index of face to track when using "index" selection.
            padding: Extra space around face (0.3 = 30% padding on each side).
            vertical_offset: Vertical position offset (-0.1 = face in upper third).
            smoothing: Position smoothing factor (0-1, higher = smoother).
            fallback: Behavior when no face detected.
            detection_interval: Frames between face detections.
        """
        self.target_aspect = target_aspect
        self.face_selection = face_selection
        self.face_index = face_index if face_index is not None else 0
        self.padding = padding
        self.vertical_offset = vertical_offset
        self.smoothing = smoothing
        self.fallback = fallback
        self.detection_interval = detection_interval

    def _calculate_crop_region(
        self,
        face_cx: float,
        face_cy: float,
        face_w: float,
        face_h: float,
        frame_w: int,
        frame_h: int,
    ) -> tuple[int, int, int, int]:
        """Calculate crop region centered on face with padding.

        Args:
            face_cx, face_cy: Face center in normalized coords.
            face_w, face_h: Face dimensions in normalized coords.
            frame_w, frame_h: Frame dimensions in pixels.

        Returns:
            Tuple of (x, y, width, height) for crop region in pixels.
        """
        target_ratio = self.target_aspect[0] / self.target_aspect[1]
        frame_ratio = frame_w / frame_h

        # Calculate crop size to achieve target aspect ratio
        # Use _make_even to ensure H.264 compatibility
        if target_ratio < frame_ratio:
            # Target is taller (e.g., 9:16) - height limited
            crop_h = _make_even(frame_h)
            crop_w = _make_even(int(crop_h * target_ratio))
        else:
            # Target is wider - width limited
            crop_w = _make_even(frame_w)
            crop_h = _make_even(int(crop_w / target_ratio))

        # Calculate minimum crop size based on face + padding
        min_face_dim = max(face_w * frame_w, face_h * frame_h)
        min_crop_dim = min_face_dim * (1 + 2 * self.padding)

        # Ensure crop is at least large enough for face with padding
        if crop_w < min_crop_dim * target_ratio:
            crop_w = _make_even(min(int(min_crop_dim * target_ratio), frame_w))
            crop_h = _make_even(min(int(crop_w / target_ratio), frame_h))

        # Center crop on face with vertical offset
        center_x = face_cx * frame_w
        center_y = (face_cy + self.vertical_offset) * frame_h

        x = int(center_x - crop_w / 2)
        y = int(center_y - crop_h / 2)

        # Clamp to frame bounds
        x = max(0, min(x, frame_w - crop_w))
        y = max(0, min(y, frame_h - crop_h))

        return (x, y, crop_w, crop_h)

    def apply(self, video: Video) -> Video:
        """Apply face tracking crop to video.

        Args:
            video: Input video.

        Returns:
            Video cropped to follow faces.
        """
        tracker = FaceTracker(
            selection_strategy=self.face_selection,
            face_index=self.face_index,
            smoothing=self.smoothing,
            detection_interval=self.detection_interval,
        )

        h, w = video.frame_shape[:2]
        target_ratio = self.target_aspect[0] / self.target_aspect[1]

        # Calculate output dimensions maintaining target aspect ratio
        # Use _make_even to ensure H.264 compatibility (requires even dimensions)
        if target_ratio < w / h:
            out_h = _make_even(h)
            out_w = _make_even(int(out_h * target_ratio))
        else:
            out_w = _make_even(w)
            out_h = _make_even(int(out_w / target_ratio))

        # Default crop region (center)
        default_x = (w - out_w) // 2
        default_y = (h - out_h) // 2
        last_crop = (default_x, default_y, out_w, out_h)

        print(f"Face tracking crop: {w}x{h} -> {out_w}x{out_h} ({self.target_aspect[0]}:{self.target_aspect[1]})")

        new_frames = []
        for i in tqdm(range(len(video.frames)), desc="Face tracking crop"):
            frame = video.frames[i]
            face_info = tracker.detect_and_track(frame, i)

            if face_info:
                cx, cy, fw, fh = face_info
                crop = self._calculate_crop_region(cx, cy, fw, fh, w, h)
                last_crop = crop
            else:
                # Fallback behavior
                if self.fallback == "center":
                    crop = (default_x, default_y, out_w, out_h)
                elif self.fallback == "last_position":
                    crop = last_crop
                else:  # full_frame
                    crop = (0, 0, w, h)

            x, y, cw, ch = crop
            cropped = frame[y : y + ch, x : x + cw]

            # Resize to output dimensions if needed
            if cropped.shape[1] != out_w or cropped.shape[0] != out_h:
                cropped = cv2.resize(cropped, (out_w, out_h), interpolation=cv2.INTER_AREA)

            new_frames.append(cropped)

        video.frames = np.array(new_frames, dtype=np.uint8)
        return video


class SplitScreenComposite(Transformation):
    """Arranges multiple face-tracked crops in a grid layout.

    Useful for interview-style videos, reaction videos, or showing
    multiple perspectives simultaneously.
    """

    def __init__(
        self,
        layout: Literal["2x1", "1x2", "2x2", "1+2", "2+1"] = "2x1",
        output_size: tuple[int, int] | None = None,
        gap: int = 4,
        gap_color: tuple[int, int, int] = (0, 0, 0),
        border_width: int = 0,
        border_color: tuple[int, int, int] = (255, 255, 255),
        face_padding: float = 0.2,
        smoothing: float = 0.8,
        detection_interval: int = 3,
        audio_source: Literal["main", "loudest", "mix"] = "main",
    ):
        """Initialize split screen composite.

        Args:
            layout: Grid layout for cells.
                - "2x1": Two cells side by side (horizontal)
                - "1x2": Two cells stacked (vertical)
                - "2x2": Four cells in 2x2 grid
                - "1+2": One large cell on left, two small on right
                - "2+1": Two small cells on left, one large on right
            output_size: Output dimensions (width, height). If None, use source size.
            gap: Gap between cells in pixels.
            gap_color: Color of gap between cells (RGB).
            border_width: Border width around each cell.
            border_color: Border color (RGB).
            face_padding: Extra space around face in each cell.
            smoothing: Position smoothing factor.
            detection_interval: Frames between face detections.
            audio_source: Audio handling ("main" uses first source).
        """
        self.layout = layout
        self.output_size = output_size
        self.gap = gap
        self.gap_color = gap_color
        self.border_width = border_width
        self.border_color = border_color
        self.face_padding = face_padding
        self.smoothing = smoothing
        self.detection_interval = detection_interval
        self.audio_source = audio_source

    def _get_cell_rects(self, width: int, height: int) -> list[tuple[int, int, int, int]]:
        """Calculate cell rectangles for the layout.

        Returns:
            List of (x, y, width, height) tuples for each cell.
        """
        gap = self.gap

        if self.layout == "2x1":
            cell_w = (width - gap) // 2
            return [
                (0, 0, cell_w, height),
                (cell_w + gap, 0, width - cell_w - gap, height),
            ]
        elif self.layout == "1x2":
            cell_h = (height - gap) // 2
            return [
                (0, 0, width, cell_h),
                (0, cell_h + gap, width, height - cell_h - gap),
            ]
        elif self.layout == "2x2":
            cell_w = (width - gap) // 2
            cell_h = (height - gap) // 2
            return [
                (0, 0, cell_w, cell_h),
                (cell_w + gap, 0, width - cell_w - gap, cell_h),
                (0, cell_h + gap, cell_w, height - cell_h - gap),
                (cell_w + gap, cell_h + gap, width - cell_w - gap, height - cell_h - gap),
            ]
        elif self.layout == "1+2":
            # Large cell on left (2/3 width), two small on right (1/3 width)
            large_w = (width - gap) * 2 // 3
            small_w = width - large_w - gap
            small_h = (height - gap) // 2
            return [
                (0, 0, large_w, height),
                (large_w + gap, 0, small_w, small_h),
                (large_w + gap, small_h + gap, small_w, height - small_h - gap),
            ]
        elif self.layout == "2+1":
            # Two small cells on left, large on right
            large_w = (width - gap) * 2 // 3
            small_w = width - large_w - gap
            small_h = (height - gap) // 2
            return [
                (0, 0, small_w, small_h),
                (0, small_h + gap, small_w, height - small_h - gap),
                (small_w + gap, 0, large_w, height),
            ]
        else:
            raise ValueError(f"Unknown layout: {self.layout}")

    def _get_required_sources(self) -> int:
        """Get number of video sources required for the layout."""
        if self.layout in ("2x1", "1x2"):
            return 2
        elif self.layout == "2x2":
            return 4
        else:  # 1+2, 2+1
            return 3

    def apply(self, video: Video, *additional_videos: Video) -> Video:
        """Apply split screen composite to videos.

        Args:
            video: Primary video (used for timing and audio).
            *additional_videos: Additional videos to include in grid.

        Returns:
            Composite video with all sources in grid layout.
        """
        all_videos = [video] + list(additional_videos)
        required = self._get_required_sources()

        if len(all_videos) < required:
            raise ValueError(f"Layout '{self.layout}' requires {required} videos, got {len(all_videos)}")

        # Use first video for timing
        n_frames = len(video.frames)

        # Determine output size
        if self.output_size:
            out_w, out_h = self.output_size
        else:
            out_w, out_h = video.frame_shape[1], video.frame_shape[0]

        cell_rects = self._get_cell_rects(out_w, out_h)

        # Create face trackers for each cell
        trackers = [
            FaceTracker(
                selection_strategy="largest",
                smoothing=self.smoothing,
                detection_interval=self.detection_interval,
            )
            for _ in range(len(cell_rects))
        ]

        print(f"Creating {self.layout} split screen: {out_w}x{out_h}")

        new_frames = []
        for i in tqdm(range(n_frames), desc="Split screen composite"):
            # Create output frame with gap color
            output = np.full((out_h, out_w, 3), self.gap_color, dtype=np.uint8)

            for cell_idx, (cx, cy, cw, ch) in enumerate(cell_rects):
                if cell_idx >= len(all_videos):
                    break

                src_video = all_videos[cell_idx]
                src_idx = i % len(src_video.frames)
                src_frame = src_video.frames[src_idx]
                src_h, src_w = src_frame.shape[:2]

                # Track face in source
                face_info = trackers[cell_idx].detect_and_track(src_frame, i)

                if face_info:
                    fcx, fcy, fw, fh = face_info
                    # Calculate crop to fit face in cell
                    cell_aspect = cw / ch
                    src_aspect = src_w / src_h

                    if cell_aspect < src_aspect:
                        # Cell is taller - crop width
                        crop_h = _make_even(src_h)
                        crop_w = _make_even(int(crop_h * cell_aspect))
                    else:
                        # Cell is wider - crop height
                        crop_w = _make_even(src_w)
                        crop_h = _make_even(int(crop_w / cell_aspect))

                    # Center on face with padding consideration
                    center_x = int(fcx * src_w)
                    center_y = int(fcy * src_h)

                    crop_x = max(0, min(center_x - crop_w // 2, src_w - crop_w))
                    crop_y = max(0, min(center_y - crop_h // 2, src_h - crop_h))

                    cropped = src_frame[crop_y : crop_y + crop_h, crop_x : crop_x + crop_w]
                else:
                    # Center crop as fallback
                    cell_aspect = cw / ch
                    src_aspect = src_w / src_h

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

                # Resize to cell size
                resized = cv2.resize(cropped, (cw, ch), interpolation=cv2.INTER_AREA)

                # Apply border if specified
                if self.border_width > 0:
                    bw = self.border_width
                    resized[:bw, :] = self.border_color
                    resized[-bw:, :] = self.border_color
                    resized[:, :bw] = self.border_color
                    resized[:, -bw:] = self.border_color

                # Place in output
                output[cy : cy + ch, cx : cx + cw] = resized

            new_frames.append(output)

        video.frames = np.array(new_frames, dtype=np.uint8)

        # Audio handling - keep main video audio
        # (mixing multiple audio tracks would require more complex handling)
        return video


class AutoFramingCrop(Transformation):
    """Intelligent cropping with cinematographic framing rules.

    Applies professional framing techniques like rule of thirds,
    headroom, and lead room when tracking subjects.
    """

    def __init__(
        self,
        target_aspect: tuple[int, int] = (9, 16),
        framing_rule: Literal["thirds", "center", "headroom", "dynamic"] = "headroom",
        headroom: float = 0.15,
        lead_room: float = 0.1,
        track_mode: Literal["face", "person", "auto"] = "auto",
        multi_subject: Literal["group", "primary", "alternate"] = "group",
        smoothing: float = 0.85,
        max_speed: float = 0.1,
        detection_interval: int = 5,
    ):
        """Initialize auto framing crop.

        Args:
            target_aspect: Output aspect ratio as (width, height).
            framing_rule: How to frame the subject.
                - "thirds": Place subject on rule of thirds line.
                - "center": Keep subject centered.
                - "headroom": Maintain proper headroom above face.
                - "dynamic": Adjust framing based on motion.
            headroom: Amount of space above head (0.15 = 15% of frame).
            lead_room: Extra space in direction subject is looking.
            track_mode: What to track ("face", "person", or "auto").
            multi_subject: How to handle multiple subjects.
            smoothing: Position smoothing factor (higher = smoother camera).
            max_speed: Maximum camera movement per frame (normalized).
            detection_interval: Frames between detections.
        """
        self.target_aspect = target_aspect
        self.framing_rule = framing_rule
        self.headroom = headroom
        self.lead_room = lead_room
        self.track_mode = track_mode
        self.multi_subject = multi_subject
        self.smoothing = smoothing
        self.max_speed = max_speed
        self.detection_interval = detection_interval

    def _apply_framing_offset(
        self,
        face_cx: float,
        face_cy: float,
        face_h: float,
    ) -> tuple[float, float]:
        """Apply framing rule to get desired subject position.

        Args:
            face_cx, face_cy: Face center in normalized coords.
            face_h: Face height in normalized coords.

        Returns:
            Target (x, y) position for crop center.
        """
        if self.framing_rule == "center":
            return (face_cx, face_cy)
        elif self.framing_rule == "headroom":
            # Position face with headroom above
            # Face should be at (headroom + face_h/2) from top
            target_y = face_cy - self.headroom
            return (face_cx, target_y)
        elif self.framing_rule == "thirds":
            # Place face on upper-third horizontal line
            # Upper third is at 1/3 from top
            target_y = face_cy - (1 / 3 - 0.5)  # Offset to place at 1/3
            return (face_cx, target_y)
        elif self.framing_rule == "dynamic":
            # Similar to headroom but could adjust based on motion
            target_y = face_cy - self.headroom
            return (face_cx, target_y)
        else:
            return (face_cx, face_cy)

    def _clamp_speed(
        self,
        current: tuple[float, float],
        target: tuple[float, float],
    ) -> tuple[float, float]:
        """Clamp movement speed to max_speed.

        Args:
            current: Current position (x, y).
            target: Target position (x, y).

        Returns:
            New position clamped to max_speed from current.
        """
        dx = target[0] - current[0]
        dy = target[1] - current[1]
        distance = (dx**2 + dy**2) ** 0.5

        if distance <= self.max_speed:
            return target

        # Scale movement to max_speed
        scale = self.max_speed / distance
        return (current[0] + dx * scale, current[1] + dy * scale)

    def apply(self, video: Video) -> Video:
        """Apply auto framing crop to video.

        Args:
            video: Input video.

        Returns:
            Video with intelligent framing applied.
        """
        tracker = FaceTracker(
            selection_strategy="largest" if self.multi_subject != "alternate" else "centered",
            smoothing=self.smoothing,
            detection_interval=self.detection_interval,
        )

        h, w = video.frame_shape[:2]
        target_ratio = self.target_aspect[0] / self.target_aspect[1]

        # Calculate output dimensions
        # Use _make_even to ensure H.264 compatibility (requires even dimensions)
        if target_ratio < w / h:
            out_h = _make_even(h)
            out_w = _make_even(int(out_h * target_ratio))
        else:
            out_w = _make_even(w)
            out_h = _make_even(int(out_w / target_ratio))

        # Crop size for extracting from source
        crop_w = out_w
        crop_h = out_h

        # Default center position
        current_position = (0.5, 0.5)

        print(f"Auto framing: {w}x{h} -> {out_w}x{out_h} ({self.framing_rule} framing)")

        new_frames = []
        for i in tqdm(range(len(video.frames)), desc="Auto framing"):
            frame = video.frames[i]
            face_info = tracker.detect_and_track(frame, i)

            if face_info:
                cx, cy, fw, fh = face_info
                target_position = self._apply_framing_offset(cx, cy, fh)
            else:
                target_position = current_position

            # Apply speed limit for smooth camera movement
            current_position = self._clamp_speed(current_position, target_position)

            # Use current position (smoothing already applied in _clamp_speed)
            smooth_x = current_position[0]
            smooth_y = current_position[1]

            # Calculate crop region
            crop_center_x = int(smooth_x * w)
            crop_center_y = int(smooth_y * h)

            crop_x = crop_center_x - crop_w // 2
            crop_y = crop_center_y - crop_h // 2

            # Clamp to frame bounds
            crop_x = max(0, min(crop_x, w - crop_w))
            crop_y = max(0, min(crop_y, h - crop_h))

            cropped = frame[crop_y : crop_y + crop_h, crop_x : crop_x + crop_w]

            # Ensure output size matches
            if cropped.shape[1] != out_w or cropped.shape[0] != out_h:
                cropped = cv2.resize(cropped, (out_w, out_h), interpolation=cv2.INTER_AREA)

            new_frames.append(cropped)

        video.frames = np.array(new_frames, dtype=np.uint8)
        return video
