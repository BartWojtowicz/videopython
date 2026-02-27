"""Tests for object swapping functionality."""

import numpy as np
import pytest

from videopython.ai.swapping.models import (
    InpaintingConfig,
    ObjectMask,
    ObjectTrack,
    SegmentationConfig,
    SwapConfig,
    SwapResult,
)


@pytest.fixture
def sample_mask():
    """Create a sample binary mask."""
    mask = np.zeros((100, 100), dtype=bool)
    mask[30:70, 30:70] = True  # 40x40 square in center
    return mask


@pytest.fixture
def sample_object_mask(sample_mask):
    """Create a sample ObjectMask."""
    return ObjectMask(
        frame_index=0,
        mask=sample_mask,
        confidence=0.95,
        bounding_box=(0.3, 0.3, 0.7, 0.7),
    )


@pytest.fixture
def sample_frames():
    """Create sample video frames."""
    # Create 10 frames of 100x100 RGB
    frames = np.random.randint(0, 255, (10, 100, 100, 3), dtype=np.uint8)
    return frames


class TestObjectMask:
    """Tests for ObjectMask data model."""

    def test_creation(self, sample_mask):
        """Test creating an ObjectMask."""
        mask = ObjectMask(
            frame_index=5,
            mask=sample_mask,
            confidence=0.9,
        )

        assert mask.frame_index == 5
        assert mask.confidence == 0.9
        assert mask.bounding_box is None
        assert mask.height == 100
        assert mask.width == 100

    def test_creation_with_bbox(self, sample_mask):
        """Test creating an ObjectMask with bounding box."""
        mask = ObjectMask(
            frame_index=0,
            mask=sample_mask,
            confidence=0.95,
            bounding_box=(0.1, 0.2, 0.8, 0.9),
        )

        assert mask.bounding_box == (0.1, 0.2, 0.8, 0.9)

    def test_area(self, sample_mask):
        """Test mask area calculation."""
        mask = ObjectMask(
            frame_index=0,
            mask=sample_mask,
            confidence=0.9,
        )

        # 40x40 square = 1600 pixels
        assert mask.area == 1600

    def test_invalid_mask_shape(self):
        """Test that 3D mask raises error."""
        invalid_mask = np.zeros((100, 100, 3), dtype=bool)

        with pytest.raises(ValueError, match="must be 2D"):
            ObjectMask(frame_index=0, mask=invalid_mask, confidence=0.9)

    def test_invalid_confidence(self, sample_mask):
        """Test that invalid confidence raises error."""
        with pytest.raises(ValueError, match="between 0 and 1"):
            ObjectMask(frame_index=0, mask=sample_mask, confidence=1.5)

        with pytest.raises(ValueError, match="between 0 and 1"):
            ObjectMask(frame_index=0, mask=sample_mask, confidence=-0.1)

    def test_dilate(self, sample_object_mask):
        """Test mask dilation."""
        dilated = sample_object_mask.dilate(kernel_size=5)

        assert dilated.frame_index == sample_object_mask.frame_index
        assert dilated.confidence == sample_object_mask.confidence
        # Dilated mask should be larger
        assert dilated.area > sample_object_mask.area


class TestObjectTrack:
    """Tests for ObjectTrack data model."""

    def test_creation(self, sample_object_mask):
        """Test creating an ObjectTrack."""
        track = ObjectTrack(
            object_id="test-id",
            masks=[sample_object_mask],
            label="person",
            start_frame=0,
            end_frame=0,
        )

        assert track.object_id == "test-id"
        assert track.label == "person"
        assert track.num_frames == 1
        assert track.start_frame == 0
        assert track.end_frame == 0

    def test_frame_indices(self, sample_mask):
        """Test getting frame indices."""
        masks = [ObjectMask(frame_index=i, mask=sample_mask, confidence=0.9) for i in [0, 2, 5, 7]]
        track = ObjectTrack(
            object_id="test",
            masks=masks,
            label="car",
            start_frame=0,
            end_frame=7,
        )

        assert track.frame_indices == [0, 2, 5, 7]

    def test_average_confidence(self, sample_mask):
        """Test average confidence calculation."""
        masks = [
            ObjectMask(frame_index=0, mask=sample_mask, confidence=0.8),
            ObjectMask(frame_index=1, mask=sample_mask, confidence=0.9),
            ObjectMask(frame_index=2, mask=sample_mask, confidence=1.0),
        ]
        track = ObjectTrack(
            object_id="test",
            masks=masks,
            label="car",
            start_frame=0,
            end_frame=2,
        )

        assert track.average_confidence == pytest.approx(0.9, abs=0.01)

    def test_average_confidence_empty(self):
        """Test average confidence with no masks."""
        track = ObjectTrack(
            object_id="test",
            masks=[],
            label="car",
            start_frame=0,
            end_frame=0,
        )

        assert track.average_confidence == 0.0

    def test_get_mask_for_frame(self, sample_mask):
        """Test getting mask for specific frame."""
        masks = [
            ObjectMask(frame_index=0, mask=sample_mask, confidence=0.9),
            ObjectMask(frame_index=5, mask=sample_mask, confidence=0.8),
        ]
        track = ObjectTrack(
            object_id="test",
            masks=masks,
            label="car",
            start_frame=0,
            end_frame=5,
        )

        found = track.get_mask_for_frame(5)
        assert found is not None
        assert found.confidence == 0.8

        not_found = track.get_mask_for_frame(3)
        assert not_found is None

    def test_get_masks_array(self, sample_mask):
        """Test getting masks as stacked array."""
        masks = [ObjectMask(frame_index=i, mask=sample_mask, confidence=0.9) for i in range(3)]
        track = ObjectTrack(
            object_id="test",
            masks=masks,
            label="car",
            start_frame=0,
            end_frame=2,
        )

        array = track.get_masks_array()

        assert array.shape == (3, 100, 100)

    def test_get_masks_array_empty(self):
        """Test that empty track raises error."""
        track = ObjectTrack(
            object_id="test",
            masks=[],
            label="car",
            start_frame=0,
            end_frame=0,
        )

        with pytest.raises(ValueError, match="No masks"):
            track.get_masks_array()


class TestSwapResult:
    """Tests for SwapResult data model."""

    def test_creation(self, sample_frames, sample_object_mask):
        """Test creating a SwapResult."""
        track = ObjectTrack(
            object_id="test",
            masks=[sample_object_mask],
            label="person",
            start_frame=0,
            end_frame=0,
        )

        result = SwapResult(
            swapped_frames=sample_frames,
            object_track=track,
            source_prompt="person",
            target_prompt="robot",
        )

        assert result.num_frames == 10
        assert result.frame_size == (100, 100)
        assert result.source_prompt == "person"
        assert result.target_prompt == "robot"
        assert not result.has_inpainted_frames

    def test_with_inpainted_frames(self, sample_frames, sample_object_mask):
        """Test SwapResult with inpainted frames."""
        track = ObjectTrack(
            object_id="test",
            masks=[sample_object_mask],
            label="person",
            start_frame=0,
            end_frame=0,
        )

        result = SwapResult(
            swapped_frames=sample_frames,
            object_track=track,
            inpainted_frames=sample_frames.copy(),
            source_prompt="person",
        )

        assert result.has_inpainted_frames

    def test_with_replacement_image(self, sample_frames, sample_object_mask):
        """Test SwapResult with replacement image path."""
        track = ObjectTrack(
            object_id="test",
            masks=[sample_object_mask],
            label="logo",
            start_frame=0,
            end_frame=0,
        )

        result = SwapResult(
            swapped_frames=sample_frames,
            object_track=track,
            source_prompt="old logo",
            replacement_image="/path/to/new_logo.png",
        )

        assert result.replacement_image == "/path/to/new_logo.png"


class TestSegmentationConfig:
    """Tests for SegmentationConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = SegmentationConfig()

        assert config.model_size == "large"
        assert config.points_per_side == 32
        assert config.pred_iou_thresh == 0.88
        assert config.stability_score_thresh == 0.95

    def test_custom_values(self):
        """Test custom configuration values."""
        config = SegmentationConfig(
            model_size="tiny",
            points_per_side=16,
            pred_iou_thresh=0.9,
        )

        assert config.model_size == "tiny"
        assert config.points_per_side == 16
        assert config.pred_iou_thresh == 0.9

    def test_invalid_model_size(self):
        """Test that invalid model size raises error."""
        with pytest.raises(ValueError, match="model_size must be one of"):
            SegmentationConfig(model_size="invalid")

    def test_valid_model_sizes(self):
        """Test all valid model sizes."""
        for size in ["tiny", "small", "base", "large"]:
            config = SegmentationConfig(model_size=size)
            assert config.model_size == size


class TestInpaintingConfig:
    """Tests for InpaintingConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = InpaintingConfig()

        assert config.model_id == "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
        assert config.num_inference_steps == 25
        assert config.guidance_scale == 7.5
        assert config.mask_dilation == 5
        assert config.batch_size == 1

    def test_custom_values(self):
        """Test custom configuration values."""
        config = InpaintingConfig(
            model_id="custom/model",
            num_inference_steps=50,
            guidance_scale=10.0,
            mask_dilation=10,
        )

        assert config.model_id == "custom/model"
        assert config.num_inference_steps == 50
        assert config.guidance_scale == 10.0
        assert config.mask_dilation == 10


class TestSwapConfig:
    """Tests for SwapConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = SwapConfig()

        assert config.composite_blend == 0.5
        assert config.reference_frame == 0
        assert isinstance(config.segmentation, SegmentationConfig)
        assert isinstance(config.inpainting, InpaintingConfig)

    def test_custom_nested_config(self):
        """Test custom nested configurations."""
        seg_config = SegmentationConfig(model_size="tiny")
        inp_config = InpaintingConfig(num_inference_steps=10)

        config = SwapConfig(
            segmentation=seg_config,
            inpainting=inp_config,
            composite_blend=0.8,
            reference_frame=5,
        )

        assert config.segmentation.model_size == "tiny"
        assert config.inpainting.num_inference_steps == 10
        assert config.composite_blend == 0.8
        assert config.reference_frame == 5


class TestObjectSwapper:
    """Tests for ObjectSwapper class."""

    def test_initialization_default(self):
        """Test default initialization."""
        from videopython.ai.swapping import ObjectSwapper

        swapper = ObjectSwapper()

        assert isinstance(swapper.config, SwapConfig)

    def test_initialization_with_device(self):
        """Test initialization with explicit device."""
        from videopython.ai.swapping import ObjectSwapper

        swapper = ObjectSwapper(device="cpu")

        assert swapper.device == "cpu"

    def test_initialization_custom_config(self):
        """Test initialization with custom config."""
        from videopython.ai.swapping import ObjectSwapper

        config = SwapConfig(
            composite_blend=0.9,
            reference_frame=10,
        )
        swapper = ObjectSwapper(config=config)

        assert swapper.config.composite_blend == 0.9
        assert swapper.config.reference_frame == 10

    def test_visualize_track(self, sample_frames, sample_mask):
        """Test visualize_track static method."""
        from videopython.ai.swapping import ObjectSwapper

        masks = [ObjectMask(frame_index=i, mask=sample_mask, confidence=0.9) for i in range(10)]
        track = ObjectTrack(
            object_id="test",
            masks=masks,
            label="object",
            start_frame=0,
            end_frame=9,
        )

        visualized = ObjectSwapper.visualize_track(sample_frames, track, color=(255, 0, 0), alpha=0.5)

        assert visualized.shape == sample_frames.shape
        # Check that some pixels in masked region are colored
        # (they should be different from original in masked area)


class TestObjectSegmenter:
    """Tests for ObjectSegmenter class."""

    def test_initialization_default(self):
        """Test default initialization."""
        from videopython.ai.swapping import ObjectSegmenter

        segmenter = ObjectSegmenter()

        assert isinstance(segmenter.config, SegmentationConfig)

    def test_initialization_with_device(self):
        """Test initialization with explicit device."""
        from videopython.ai.swapping import ObjectSegmenter

        segmenter = ObjectSegmenter(device="cpu")

        assert segmenter.device == "cpu"

    def test_initialization_custom_config(self):
        """Test initialization with custom config."""
        from videopython.ai.swapping import ObjectSegmenter

        config = SegmentationConfig(model_size="tiny")
        segmenter = ObjectSegmenter(config=config)

        assert segmenter.config.model_size == "tiny"


class TestVideoInpainter:
    """Tests for VideoInpainter class."""

    def test_initialization_default(self):
        """Test default initialization."""
        from videopython.ai.swapping import VideoInpainter

        inpainter = VideoInpainter()

        assert isinstance(inpainter.config, InpaintingConfig)

    def test_initialization_with_device(self):
        """Test initialization with explicit device."""
        from videopython.ai.swapping import VideoInpainter

        inpainter = VideoInpainter(device="cpu")

        assert inpainter.device == "cpu"

    def test_initialization_custom_config(self):
        """Test initialization with custom config."""
        from videopython.ai.swapping import VideoInpainter

        config = InpaintingConfig(num_inference_steps=10)
        inpainter = VideoInpainter(config=config)

        assert inpainter.config.num_inference_steps == 10
