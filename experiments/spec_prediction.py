"""
Experiment 1: VideoSpec Prediction

Test whether we can accurately predict output metadata for transforms
without executing them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from videopython.base import Video, VideoMetadata
from videopython.base.transforms import CutFrames, CutSeconds, Resize, ResampleFPS, Crop


@dataclass
class VideoSpec:
    """Lightweight video specification for validation without loading frames."""

    height: int
    width: int
    fps: float
    duration_seconds: float

    @classmethod
    def from_path(cls, path: str | Path) -> VideoSpec:
        """Read spec from file without loading frames."""
        meta = VideoMetadata.from_path(str(path))
        return cls(meta.height, meta.width, meta.fps, meta.total_seconds)

    @classmethod
    def from_video(cls, video: Video) -> VideoSpec:
        """Create spec from loaded video."""
        return cls(
            video.metadata.height,
            video.metadata.width,
            video.fps,
            video.total_seconds,
        )

    @classmethod
    def from_metadata(cls, meta: VideoMetadata) -> VideoSpec:
        """Create spec from VideoMetadata."""
        return cls(meta.height, meta.width, meta.fps, meta.total_seconds)

    def can_merge_with(self, other: VideoSpec) -> bool:
        """Check if two specs can be merged (for transitions)."""
        return (
            self.height == other.height
            and self.width == other.width
            and round(self.fps) == round(other.fps)
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, VideoSpec):
            return False
        # Allow small floating point differences in duration
        return (
            self.height == other.height
            and self.width == other.width
            and round(self.fps, 2) == round(other.fps, 2)
            and abs(self.duration_seconds - other.duration_seconds) < 0.1
        )

    def __repr__(self) -> str:
        return f"VideoSpec({self.width}x{self.height}@{self.fps:.2f}fps, {self.duration_seconds:.2f}s)"


# Prediction functions for each transform type


def predict_cut_frames(spec: VideoSpec, start: int, end: int) -> VideoSpec:
    """Predict output of CutFrames transform."""
    frame_count = end - start
    duration = frame_count / spec.fps
    return VideoSpec(spec.height, spec.width, spec.fps, duration)


def predict_cut_seconds(spec: VideoSpec, start: float, end: float) -> VideoSpec:
    """Predict output of CutSeconds transform."""
    duration = end - start
    return VideoSpec(spec.height, spec.width, spec.fps, duration)


def predict_resize(
    spec: VideoSpec, width: int | None = None, height: int | None = None
) -> VideoSpec:
    """Predict output of Resize transform."""
    if width and height:
        return VideoSpec(height, width, spec.fps, spec.duration_seconds)
    elif width:
        ratio = width / spec.width
        new_height = round(spec.height * ratio)
        return VideoSpec(new_height, width, spec.fps, spec.duration_seconds)
    elif height:
        ratio = height / spec.height
        new_width = round(spec.width * ratio)
        return VideoSpec(height, new_width, spec.fps, spec.duration_seconds)
    else:
        raise ValueError("Must provide width or height")


def predict_resample_fps(spec: VideoSpec, target_fps: float) -> VideoSpec:
    """Predict output of ResampleFPS transform."""
    # Duration stays the same, fps changes
    return VideoSpec(spec.height, spec.width, target_fps, spec.duration_seconds)


def predict_crop(spec: VideoSpec, width: int, height: int) -> VideoSpec:
    """Predict output of Crop transform."""
    return VideoSpec(height, width, spec.fps, spec.duration_seconds)


def predict_transition(
    spec1: VideoSpec, spec2: VideoSpec, effect_time_seconds: float = 0.0
) -> VideoSpec | str:
    """Predict output of transition between two videos.

    Returns VideoSpec if compatible, error string if not.
    """
    if not spec1.can_merge_with(spec2):
        return (
            f"Incompatible specs: {spec1.width}x{spec1.height}@{spec1.fps:.0f}fps "
            f"vs {spec2.width}x{spec2.height}@{spec2.fps:.0f}fps"
        )

    combined_duration = spec1.duration_seconds + spec2.duration_seconds - effect_time_seconds
    return VideoSpec(spec1.height, spec1.width, spec1.fps, combined_duration)


def test_prediction_accuracy(video_path: str) -> dict:
    """Test prediction accuracy for all transforms against actual results."""
    results = {}

    # Load video - use only first 2 seconds for faster testing
    print(f"Loading video: {video_path}")
    video = Video.from_path(video_path, start_second=0, end_second=2)
    original_spec = VideoSpec.from_video(video)
    print(f"Test video spec: {original_spec}")

    # Test CutSeconds
    print("\n--- Testing CutSeconds ---")
    cut = CutSeconds(0, 1.0)
    predicted = predict_cut_seconds(original_spec, cut.start, cut.end)
    actual_video = cut.apply(video.copy())
    actual = VideoSpec.from_video(actual_video)
    match = predicted == actual
    results["CutSeconds"] = {"predicted": predicted, "actual": actual, "match": match}
    print(f"  Predicted: {predicted}")
    print(f"  Actual:    {actual}")
    print(f"  Match: {match}")

    # Test CutFrames
    print("\n--- Testing CutFrames ---")
    cut_frames = CutFrames(0, 30)
    predicted = predict_cut_frames(original_spec, cut_frames.start, cut_frames.end)
    actual_video = cut_frames.apply(video.copy())
    actual = VideoSpec.from_video(actual_video)
    match = predicted == actual
    results["CutFrames"] = {"predicted": predicted, "actual": actual, "match": match}
    print(f"  Predicted: {predicted}")
    print(f"  Actual:    {actual}")
    print(f"  Match: {match}")

    # Test Resize (both dimensions)
    print("\n--- Testing Resize (both dims) ---")
    resize = Resize(width=640, height=480)
    predicted = predict_resize(original_spec, width=640, height=480)
    actual_video = resize.apply(video.copy())
    actual = VideoSpec.from_video(actual_video)
    match = predicted == actual
    results["Resize_both"] = {"predicted": predicted, "actual": actual, "match": match}
    print(f"  Predicted: {predicted}")
    print(f"  Actual:    {actual}")
    print(f"  Match: {match}")

    # Test Resize (width only)
    print("\n--- Testing Resize (width only) ---")
    resize = Resize(width=640)
    predicted = predict_resize(original_spec, width=640)
    actual_video = resize.apply(video.copy())
    actual = VideoSpec.from_video(actual_video)
    match = predicted == actual
    results["Resize_width"] = {"predicted": predicted, "actual": actual, "match": match}
    print(f"  Predicted: {predicted}")
    print(f"  Actual:    {actual}")
    print(f"  Match: {match}")

    # Test Resize (height only)
    print("\n--- Testing Resize (height only) ---")
    resize = Resize(height=480)
    predicted = predict_resize(original_spec, height=480)
    actual_video = resize.apply(video.copy())
    actual = VideoSpec.from_video(actual_video)
    match = predicted == actual
    results["Resize_height"] = {"predicted": predicted, "actual": actual, "match": match}
    print(f"  Predicted: {predicted}")
    print(f"  Actual:    {actual}")
    print(f"  Match: {match}")

    # Test ResampleFPS (downsample)
    print("\n--- Testing ResampleFPS (downsample) ---")
    target_fps = 15.0
    resample = ResampleFPS(fps=target_fps)
    predicted = predict_resample_fps(original_spec, target_fps)
    actual_video = resample.apply(video.copy())
    actual = VideoSpec.from_video(actual_video)
    match = predicted == actual
    results["ResampleFPS_down"] = {"predicted": predicted, "actual": actual, "match": match}
    print(f"  Predicted: {predicted}")
    print(f"  Actual:    {actual}")
    print(f"  Match: {match}")

    # Test Crop
    print("\n--- Testing Crop ---")
    crop_w = min(320, video.metadata.width)
    crop_h = min(240, video.metadata.height)
    crop = Crop(width=crop_w, height=crop_h)
    predicted = predict_crop(original_spec, crop_w, crop_h)
    actual_video = crop.apply(video.copy())
    actual = VideoSpec.from_video(actual_video)
    match = predicted == actual
    results["Crop"] = {"predicted": predicted, "actual": actual, "match": match}
    print(f"  Predicted: {predicted}")
    print(f"  Actual:    {actual}")
    print(f"  Match: {match}")

    # Test Transition compatibility (same specs)
    print("\n--- Testing Transition (compatible) ---")
    spec1 = VideoSpec.from_video(video)
    spec2 = VideoSpec.from_video(video)  # Same video = compatible
    effect_time = 0.5
    predicted = predict_transition(spec1, spec2, effect_time)
    # Actually run transition
    from videopython.base import FadeTransition
    fade = FadeTransition(effect_time_seconds=effect_time)
    actual_video = fade.apply((video.copy(), video.copy()))
    actual = VideoSpec.from_video(actual_video)
    match = isinstance(predicted, VideoSpec) and predicted == actual
    results["Transition_compatible"] = {"predicted": predicted, "actual": actual, "match": match}
    print(f"  Predicted: {predicted}")
    print(f"  Actual:    {actual}")
    print(f"  Match: {match}")

    # Test Transition compatibility (incompatible - different dimensions)
    print("\n--- Testing Transition (incompatible) ---")
    spec1 = VideoSpec(1080, 1920, 24, 2.0)  # 1080p
    spec2 = VideoSpec(720, 1280, 24, 2.0)  # 720p
    predicted = predict_transition(spec1, spec2)
    is_error = isinstance(predicted, str) and "Incompatible" in predicted
    results["Transition_incompatible"] = {"predicted": predicted, "is_error": is_error, "match": is_error}
    print(f"  Predicted error: {predicted}")
    print(f"  Correctly detected incompatibility: {is_error}")

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    all_match = all(r["match"] for r in results.values())
    passed = sum(1 for r in results.values() if r["match"])
    total = len(results)
    print(f"Passed: {passed}/{total}")

    if not all_match:
        print("\nFailed tests:")
        for name, result in results.items():
            if not result["match"]:
                print(f"  {name}:")
                print(f"    Predicted: {result['predicted']}")
                print(f"    Actual:    {result['actual']}")

    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python spec_prediction.py <video_path>")
        print("\nLooking for test videos...")
        test_videos = list(Path(".").glob("*.mp4")) + list(Path(".").glob("test_video*.mp4"))
        if test_videos:
            video_path = str(test_videos[0])
            print(f"Found: {video_path}")
        else:
            print("No test video found. Please provide a video path.")
            sys.exit(1)
    else:
        video_path = sys.argv[1]

    results = test_prediction_accuracy(video_path)

    # Exit with error code if any test failed
    if not all(r["match"] for r in results.values()):
        sys.exit(1)
