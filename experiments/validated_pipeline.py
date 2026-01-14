"""
Experiment 3: ValidatedPipeline

Build a pipeline that validates compatibility before execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from videopython.base import Video, VideoMetadata
from videopython.base.transforms import (
    Transformation,
    CutFrames,
    CutSeconds,
    Resize,
    ResampleFPS,
    Crop,
)
from videopython.base.transitions import Transition


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

    def can_merge_with(self, other: VideoSpec) -> bool:
        """Check if two specs can be merged (for transitions)."""
        return (
            self.height == other.height
            and self.width == other.width
            and round(self.fps) == round(other.fps)
        )

    def __repr__(self) -> str:
        return f"VideoSpec({self.width}x{self.height}@{self.fps:.1f}fps, {self.duration_seconds:.2f}s)"


# Transform prediction functions


def predict_transform_output(transform: Transformation, spec: VideoSpec) -> VideoSpec:
    """Predict output spec for a transform."""
    if isinstance(transform, CutSeconds):
        duration = transform.end - transform.start
        return VideoSpec(spec.height, spec.width, spec.fps, duration)

    elif isinstance(transform, CutFrames):
        frame_count = transform.end - transform.start
        duration = frame_count / spec.fps
        return VideoSpec(spec.height, spec.width, spec.fps, duration)

    elif isinstance(transform, Resize):
        if transform.width and transform.height:
            return VideoSpec(transform.height, transform.width, spec.fps, spec.duration_seconds)
        elif transform.width:
            ratio = transform.width / spec.width
            new_height = round(spec.height * ratio)
            return VideoSpec(new_height, transform.width, spec.fps, spec.duration_seconds)
        else:
            ratio = transform.height / spec.height
            new_width = round(spec.width * ratio)
            return VideoSpec(transform.height, new_width, spec.fps, spec.duration_seconds)

    elif isinstance(transform, ResampleFPS):
        return VideoSpec(spec.height, spec.width, transform.fps, spec.duration_seconds)

    elif isinstance(transform, Crop):
        return VideoSpec(transform.height, transform.width, spec.fps, spec.duration_seconds)

    else:
        # Unknown transform - assume no change (conservative)
        return spec


def predict_transition_output(
    spec1: VideoSpec, spec2: VideoSpec, transition: Transition
) -> VideoSpec | str:
    """Predict output of transition. Returns error string if incompatible."""
    if not spec1.can_merge_with(spec2):
        return (
            f"Incompatible specs for merge: "
            f"{spec1.width}x{spec1.height}@{round(spec1.fps)}fps vs "
            f"{spec2.width}x{spec2.height}@{round(spec2.fps)}fps"
        )

    effect_time = getattr(transition, "effect_time_seconds", 0.0)
    combined_duration = spec1.duration_seconds + spec2.duration_seconds - effect_time
    return VideoSpec(spec1.height, spec1.width, spec1.fps, combined_duration)


@dataclass
class ValidationError:
    """A single validation error."""

    step: str
    message: str

    def __str__(self) -> str:
        return f"[{self.step}] {self.message}"


@dataclass
class ValidationResult:
    """Result of pipeline validation."""

    valid: bool
    output_spec: VideoSpec | None
    errors: list[ValidationError] = field(default_factory=list)

    def __str__(self) -> str:
        if self.valid:
            return f"Valid -> {self.output_spec}"
        return f"Invalid: {len(self.errors)} error(s)\n" + "\n".join(f"  - {e}" for e in self.errors)


@dataclass
class PipelineStep:
    """A single step in the pipeline."""

    step_type: str  # "transform" or "transition"
    inputs: list[str]
    output: str
    operation: Any  # Transformation or Transition


class ValidatedPipeline:
    """Pipeline that validates before execution."""

    def __init__(self):
        self.sources: dict[str, str] = {}  # name -> path
        self.steps: list[PipelineStep] = []

    def add_source(self, name: str, path: str) -> ValidatedPipeline:
        """Add a video source."""
        self.sources[name] = path
        return self

    def add_transform(
        self, input_name: str, output_name: str, transform: Transformation
    ) -> ValidatedPipeline:
        """Add a transform step."""
        self.steps.append(
            PipelineStep(
                step_type="transform",
                inputs=[input_name],
                output=output_name,
                operation=transform,
            )
        )
        return self

    def add_transition(
        self, input1: str, input2: str, output: str, transition: Transition
    ) -> ValidatedPipeline:
        """Add a transition step (merges two videos)."""
        self.steps.append(
            PipelineStep(
                step_type="transition",
                inputs=[input1, input2],
                output=output,
                operation=transition,
            )
        )
        return self

    def validate(self) -> ValidationResult:
        """Validate the pipeline without executing."""
        specs: dict[str, VideoSpec] = {}
        errors: list[ValidationError] = []

        # Load source specs (fast - just reads metadata)
        for name, path in self.sources.items():
            try:
                specs[name] = VideoSpec.from_path(path)
            except Exception as e:
                errors.append(ValidationError(f"source:{name}", f"Failed to read: {e}"))

        if errors:
            return ValidationResult(False, None, errors)

        # Process each step
        for i, step in enumerate(self.steps):
            step_id = f"step{i}:{step.step_type}"

            # Check inputs exist
            missing_inputs = [inp for inp in step.inputs if inp not in specs]
            if missing_inputs:
                errors.append(
                    ValidationError(step_id, f"Missing inputs: {missing_inputs}")
                )
                continue

            if step.step_type == "transform":
                input_spec = specs[step.inputs[0]]
                try:
                    output_spec = predict_transform_output(step.operation, input_spec)
                    specs[step.output] = output_spec
                except Exception as e:
                    errors.append(ValidationError(step_id, f"Transform failed: {e}"))

            elif step.step_type == "transition":
                spec1 = specs[step.inputs[0]]
                spec2 = specs[step.inputs[1]]
                result = predict_transition_output(spec1, spec2, step.operation)

                if isinstance(result, str):
                    errors.append(ValidationError(step_id, result))
                else:
                    specs[step.output] = result

        if errors:
            return ValidationResult(False, None, errors)

        # Return spec of final output
        if self.steps:
            final_output = self.steps[-1].output
            return ValidationResult(True, specs[final_output], [])
        elif self.sources:
            # No steps, just return first source
            first_source = next(iter(self.sources.keys()))
            return ValidationResult(True, specs[first_source], [])
        else:
            return ValidationResult(False, None, [ValidationError("pipeline", "Empty pipeline")])

    def execute(self) -> Video:
        """Execute the pipeline after validation."""
        result = self.validate()
        if not result.valid:
            raise ValueError(f"Pipeline validation failed:\n{result}")

        # Load videos
        videos: dict[str, Video] = {}
        for name, path in self.sources.items():
            videos[name] = Video.from_path(path)

        # Execute steps
        for step in self.steps:
            if step.step_type == "transform":
                input_video = videos[step.inputs[0]]
                videos[step.output] = step.operation.apply(input_video)

            elif step.step_type == "transition":
                video1 = videos[step.inputs[0]]
                video2 = videos[step.inputs[1]]
                videos[step.output] = step.operation.apply((video1, video2))

        # Return final output
        return videos[self.steps[-1].output]

    def describe(self) -> str:
        """Describe the pipeline structure."""
        lines = ["Pipeline:"]
        lines.append(f"  Sources: {list(self.sources.keys())}")
        for i, step in enumerate(self.steps):
            if step.step_type == "transform":
                lines.append(
                    f"  {i}. {step.inputs[0]} -> {step.operation.__class__.__name__} -> {step.output}"
                )
            else:
                lines.append(
                    f"  {i}. ({step.inputs[0]}, {step.inputs[1]}) -> {step.operation.__class__.__name__} -> {step.output}"
                )
        return "\n".join(lines)


def test_validated_pipeline(video_path: str):
    """Test ValidatedPipeline with various scenarios."""
    print("=" * 60)
    print("EXPERIMENT 3: ValidatedPipeline")
    print("=" * 60)

    # Test 1: Simple transform pipeline
    print("\n--- Test 1: Simple transform pipeline ---")
    pipeline = (
        ValidatedPipeline()
        .add_source("v1", video_path)
        .add_transform("v1", "v1_cut", CutSeconds(0, 1))
        .add_transform("v1_cut", "v1_resized", Resize(640, 480))
    )
    print(pipeline.describe())
    result = pipeline.validate()
    print(f"Validation: {result}")
    assert result.valid, "Should be valid"
    assert result.output_spec.width == 640, "Width should be 640"
    print("PASS")

    # Test 2: Pipeline with compatible transition
    print("\n--- Test 2: Compatible transition ---")
    pipeline = (
        ValidatedPipeline()
        .add_source("v1", video_path)
        .add_source("v2", video_path)
        .add_transform("v1", "v1_cut", CutSeconds(0, 1))
        .add_transform("v2", "v2_cut", CutSeconds(0, 1))
        .add_transition("v1_cut", "v2_cut", "merged", __import__("videopython.base", fromlist=["FadeTransition"]).FadeTransition(0.5))
    )
    print(pipeline.describe())
    result = pipeline.validate()
    print(f"Validation: {result}")
    assert result.valid, "Should be valid (same video = compatible)"
    print("PASS")

    # Test 3: Pipeline with incompatible transition (different sizes)
    print("\n--- Test 3: Incompatible transition (different sizes) ---")
    from videopython.base import FadeTransition
    pipeline = (
        ValidatedPipeline()
        .add_source("v1", video_path)
        .add_source("v2", video_path)
        .add_transform("v1", "v1_resized", Resize(640, 480))  # 640x480
        .add_transform("v2", "v2_resized", Resize(1280, 720))  # 1280x720 - different!
        .add_transition("v1_resized", "v2_resized", "merged", FadeTransition(0.5))
    )
    print(pipeline.describe())
    result = pipeline.validate()
    print(f"Validation: {result}")
    assert not result.valid, "Should be invalid (different sizes)"
    assert any("Incompatible" in str(e) for e in result.errors), "Should have incompatibility error"
    print("PASS - correctly caught incompatibility")

    # Test 4: Missing input reference
    print("\n--- Test 4: Missing input reference ---")
    pipeline = (
        ValidatedPipeline()
        .add_source("v1", video_path)
        .add_transform("v1", "v1_cut", CutSeconds(0, 1))
        .add_transform("nonexistent", "v2_cut", CutSeconds(0, 1))  # Bad reference
    )
    print(pipeline.describe())
    result = pipeline.validate()
    print(f"Validation: {result}")
    assert not result.valid, "Should be invalid (missing input)"
    assert any("Missing inputs" in str(e) for e in result.errors), "Should have missing input error"
    print("PASS - correctly caught missing input")

    # Test 5: Execute valid pipeline
    print("\n--- Test 5: Execute valid pipeline ---")
    pipeline = (
        ValidatedPipeline()
        .add_source("v1", video_path)
        .add_transform("v1", "v1_cut", CutSeconds(0, 1))
        .add_transform("v1_cut", "v1_small", Resize(320, 240))
    )
    result = pipeline.validate()
    print(f"Validation: {result}")
    assert result.valid

    print("Executing pipeline...")
    video = pipeline.execute()
    actual_spec = VideoSpec.from_video(video)
    print(f"Actual output: {actual_spec}")
    print(f"Predicted output: {result.output_spec}")

    # Check prediction matches actual
    assert actual_spec.width == result.output_spec.width
    assert actual_spec.height == result.output_spec.height
    assert round(actual_spec.fps) == round(result.output_spec.fps)
    print("PASS - execution matches prediction")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        # Try to find a test video
        test_videos = list(Path(".").glob("*.mp4"))
        if test_videos:
            video_path = str(test_videos[0])
            print(f"Using test video: {video_path}")
        else:
            print("Usage: python validated_pipeline.py <video_path>")
            sys.exit(1)
    else:
        video_path = sys.argv[1]

    test_validated_pipeline(video_path)
