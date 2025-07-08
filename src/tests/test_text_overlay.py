import numpy as np
import pytest

from videopython.base.effects import FullImageOverlay
from videopython.base.text.overlay import AnchorPoint, ImageText, TextAlign

from .test_config import TEST_FONT_PATH


def test_text_is_rendered_correctly():
    """Test that text is actually rendered and visible in the image."""
    # Create a black background image
    img_size = (400, 200)
    my_overlay = ImageText(image_size=img_size, background=(0, 0, 0, 255))

    test_text = "Hello World"
    text_color = (255, 255, 255)  # White text

    # Write text in the center
    my_overlay.write_text(
        text=test_text,
        font_filename=TEST_FONT_PATH,
        font_size=30,
        color=text_color,
        xy=(200, 100),
        anchor=AnchorPoint.CENTER,
    )

    # Convert to array
    img_array = my_overlay.img_array

    # Check that we have some white pixels (text)
    white_pixels = np.sum(img_array[:, :, 0:3] == 255, axis=2) == 3
    assert np.any(white_pixels), "No white pixels found - text not rendered"

    # Count white pixels - should have a reasonable number for the text
    white_pixel_count = np.sum(white_pixels)
    assert white_pixel_count > 100, f"Only {white_pixel_count} white pixels found, expected more"

    # Verify image is not all white (which would indicate a rendering error)
    assert white_pixel_count < (img_size[0] * img_size[1]), "Too many white pixels found"


@pytest.mark.parametrize("place", ["left", "center", "right"])
def test_overlaying_video_with_text(place, small_video):
    my_overlay = ImageText(image_size=(800, 500))
    my_overlay.write_text_box(
        "Test test test test test test test",
        box_width=800,
        font_filename=TEST_FONT_PATH,
        font_size=100,
        text_color=(255, 255, 255),
        place=place,
        xy=(0, 0),
    )

    # Check overlay has text pixels (non-zero alpha)
    overlay_array = my_overlay.img_array
    assert np.any(overlay_array[:, :, 3] > 0), "No visible pixels found in the overlay"

    # Apply overlay to video
    overlay = FullImageOverlay(overlay_image=overlay_array)
    original_video = small_video.copy()
    overlayed_video = overlay.apply(original_video)

    # Check video shape hasn't changed
    assert overlayed_video.video_shape == small_video.video_shape

    # Check that the overlay actually modified pixel values in the video
    first_frame_before = original_video.frames[0].copy()
    first_frame_after = overlayed_video.frames[0]

    # There should be differences between original and overlayed frames
    assert np.any(first_frame_before != first_frame_after), "Overlay did not modify any pixel values"


@pytest.mark.parametrize(
    "anchor_point",
    [
        AnchorPoint.TOP_LEFT,
        AnchorPoint.TOP_CENTER,
        AnchorPoint.TOP_RIGHT,
        AnchorPoint.CENTER_LEFT,
        AnchorPoint.CENTER,
        AnchorPoint.CENTER_RIGHT,
        AnchorPoint.BOTTOM_LEFT,
        AnchorPoint.BOTTOM_CENTER,
        AnchorPoint.BOTTOM_RIGHT,
    ],
)
def test_text_anchor_positioning(anchor_point):
    """Test that text is positioned correctly based on anchor point."""
    # Create a black background image
    img_size = (500, 500)
    my_overlay = ImageText(image_size=img_size, background=(0, 0, 0, 255))
    text = "Anchor"
    font_size = 40
    text_color = (255, 255, 255)  # White

    # Central reference point
    center_x, center_y = 250, 250

    # Render text with specified anchor point
    my_overlay.write_text(
        text=text,
        font_filename=TEST_FONT_PATH,
        font_size=font_size,
        color=text_color,
        xy=(center_x, center_y),
        anchor=anchor_point,
    )

    # Get the image array
    img_array = my_overlay.img_array

    # Find text bounding box
    text_pixels = np.sum(img_array[:, :, 0:3] == 255, axis=2) == 3
    if not np.any(text_pixels):
        pytest.fail("No text pixels found")

    # Get row and column indices of text pixels
    rows, cols = np.where(text_pixels)
    text_bbox = {
        "min_y": np.min(rows),
        "max_y": np.max(rows),
        "min_x": np.min(cols),
        "max_x": np.max(cols),
        "center_x": (np.min(cols) + np.max(cols)) // 2,
        "center_y": (np.min(rows) + np.max(rows)) // 2,
    }

    # Check positioning based on anchor point
    if anchor_point == AnchorPoint.TOP_LEFT:
        # Text should start near the center point
        assert abs(text_bbox["min_x"] - center_x) < 10, "Text not aligned to top left"
        assert abs(text_bbox["min_y"] - center_y) < 10, "Text not aligned to top left"

    elif anchor_point == AnchorPoint.TOP_CENTER:
        # Text horizontal center should be near the center point
        assert abs(text_bbox["center_x"] - center_x) < 10, "Text not aligned to top center"
        assert abs(text_bbox["min_y"] - center_y) < 10, "Text not aligned to top center"

    elif anchor_point == AnchorPoint.TOP_RIGHT:
        # Text should end near the center point
        assert abs(text_bbox["max_x"] - center_x) < 10, "Text not aligned to top right"
        assert abs(text_bbox["min_y"] - center_y) < 10, "Text not aligned to top right"

    elif anchor_point == AnchorPoint.CENTER_LEFT:
        # Text should start horizontally near center point and be vertically centered
        assert abs(text_bbox["min_x"] - center_x) < 10, "Text not aligned to center left"
        assert abs(text_bbox["center_y"] - center_y) < 10, "Text not vertically centered"

    elif anchor_point == AnchorPoint.CENTER:
        # Text should be centered on the center point
        assert abs(text_bbox["center_x"] - center_x) < 10, "Text not horizontally centered"
        assert abs(text_bbox["center_y"] - center_y) < 10, "Text not vertically centered"

    elif anchor_point == AnchorPoint.CENTER_RIGHT:
        # Text should end horizontally near center point and be vertically centered
        assert abs(text_bbox["max_x"] - center_x) < 10, "Text not aligned to center right"
        assert abs(text_bbox["center_y"] - center_y) < 10, "Text not vertically centered"

    elif anchor_point == AnchorPoint.BOTTOM_LEFT:
        # Text should start horizontally near center point and end vertically near it
        assert abs(text_bbox["min_x"] - center_x) < 10, "Text not aligned to bottom left"
        assert abs(text_bbox["max_y"] - center_y) < 10, "Text not aligned to bottom left"

    elif anchor_point == AnchorPoint.BOTTOM_CENTER:
        # Text horizontal center should be near the center point and end vertically near it
        assert abs(text_bbox["center_x"] - center_x) < 10, "Text not aligned to bottom center"
        assert abs(text_bbox["max_y"] - center_y) < 10, "Text not aligned to bottom center"

    elif anchor_point == AnchorPoint.BOTTOM_RIGHT:
        # Text should end near the center point (both horizontally and vertically)
        assert abs(text_bbox["max_x"] - center_x) < 10, "Text not aligned to bottom right"
        assert abs(text_bbox["max_y"] - center_y) < 10, "Text not aligned to bottom right"


def test_relative_positioning():
    """Test that relative positioning works correctly."""
    img_size = (500, 300)
    my_overlay = ImageText(image_size=img_size, background=(0, 0, 0, 255))

    # Define text and styling
    text = "Relative Position"
    text_color = (255, 255, 255)

    # Use relative positioning (0.5, 0.5 should be center of image)
    rel_x, rel_y = 0.5, 0.5
    expected_x, expected_y = int(img_size[0] * rel_x), int(img_size[1] * rel_y)

    # Add relative width (50% of image width)
    rel_width = 0.5
    expected_width = int(img_size[0] * rel_width)

    # Write text using relative positioning and width
    my_overlay.write_text_box(
        text=text,
        font_filename=TEST_FONT_PATH,
        font_size=24,
        text_color=text_color,
        xy=(rel_x, rel_y),  # Relative position
        box_width=rel_width,  # Relative width
        anchor=AnchorPoint.CENTER,
        place=TextAlign.CENTER,
        margin=10,
    )

    # Get the image array
    img_array = my_overlay.img_array

    # Find text pixels
    text_pixels = np.sum(img_array[:, :, 0:3] == 255, axis=2) == 3
    assert np.any(text_pixels), "No text pixels found"

    # Get text bounding box
    rows, cols = np.where(text_pixels)
    text_center_x = (np.min(cols) + np.max(cols)) // 2
    text_center_y = (np.min(rows) + np.max(rows)) // 2
    text_width = np.max(cols) - np.min(cols)

    # Check that text is centered correctly
    assert abs(text_center_x - expected_x) < 20, (
        f"Text center x position {text_center_x} too far from expected {expected_x}"
    )
    assert abs(text_center_y - expected_y) < 20, (
        f"Text center y position {text_center_y} too far from expected {expected_y}"
    )

    # Check that text width respects the relative width setting (with some tolerance)
    assert text_width <= expected_width + 10, f"Text width {text_width} exceeds expected width {expected_width}"


def test_margin_handling():
    """Test that margins are respected in text positioning."""
    img_size = (400, 300)
    my_overlay = ImageText(image_size=img_size, background=(0, 0, 0, 255))

    text = "Margin Test"
    text_color = (255, 255, 255)
    margin = 50  # 50px margin on all sides

    # Expected available area
    # These variables are kept for documentation purposes
    _ = img_size[0] - (2 * margin)  # available_width
    _ = img_size[1] - (2 * margin)  # available_height

    # Write text in top-left corner with margin
    my_overlay.write_text(
        text=text,
        font_filename=TEST_FONT_PATH,
        font_size=24,
        color=text_color,
        xy=(0, 0),  # Top-left relative to available area
        anchor=AnchorPoint.TOP_LEFT,
        margin=margin,
    )

    # Get the image array
    img_array = my_overlay.img_array

    # Find text pixels
    text_pixels = np.sum(img_array[:, :, 0:3] == 255, axis=2) == 3
    assert np.any(text_pixels), "No text pixels found"

    # Get text bounding box
    rows, cols = np.where(text_pixels)
    min_x, max_x = np.min(cols), np.max(cols)
    min_y, max_y = np.min(rows), np.max(rows)

    # Text should start near the margin
    assert abs(min_x - margin) < 10, f"Text x position {min_x} too far from margin {margin}"
    assert abs(min_y - margin) < 10, f"Text y position {min_y} too far from margin {margin}"

    # Text should not exceed the available area
    assert max_x <= img_size[0] - margin, f"Text exceeds right margin: {max_x} > {img_size[0] - margin}"
    assert max_y <= img_size[1] - margin, f"Text exceeds bottom margin: {max_y} > {img_size[1] - margin}"


def test_text_highlighting_basic():
    """Test basic word highlighting functionality."""
    img_size = (600, 300)
    my_overlay = ImageText(image_size=img_size, background=(0, 0, 0, 255))

    text = "Hello world test"
    base_color = (255, 255, 255)  # White
    highlight_color = (255, 0, 0)  # Red

    # Highlight the second word (index 1)
    my_overlay.write_text_box(
        text=text,
        font_filename=TEST_FONT_PATH,
        font_size=50,
        text_color=base_color,
        xy=(50, 50),
        box_width=500,  # Specify box width to avoid out of bounds
        highlight_word_index=1,  # "world"
        highlight_color=highlight_color,
        highlight_size_multiplier=1.5,
    )

    # Get the image array
    img_array = my_overlay.img_array

    # Check that we have both white and red pixels
    white_pixels = np.sum(img_array[:, :, 0:3] == [255, 255, 255], axis=2) == 3
    red_pixels = np.sum(img_array[:, :, 0:3] == [255, 0, 0], axis=2) == 3

    assert np.any(white_pixels), "No white pixels found - base text not rendered"
    assert np.any(red_pixels), "No red pixels found - highlighted text not rendered"

    # Count pixels to ensure we have reasonable amounts
    white_count = np.sum(white_pixels)
    red_count = np.sum(red_pixels)

    assert white_count > 100, f"Too few white pixels: {white_count}"
    assert red_count > 50, f"Too few red pixels: {red_count}"


def test_text_highlighting_different_multipliers():
    """Test highlighting with different size multipliers."""
    img_size = (500, 400)
    my_overlay = ImageText(image_size=img_size, background=(0, 0, 0, 255))

    text = "Small big text"
    base_color = (100, 100, 100)  # Gray
    highlight_color = (0, 255, 0)  # Green

    # Test with size multiplier of 2.0
    my_overlay.write_text_box(
        text=text,
        font_filename=TEST_FONT_PATH,
        font_size=30,
        text_color=base_color,
        xy=(30, 30),
        box_width=400,
        highlight_word_index=1,  # "big"
        highlight_color=highlight_color,
        highlight_size_multiplier=2.0,
    )

    # Get the image array
    img_array = my_overlay.img_array

    # Check that we have both gray and green pixels
    gray_pixels = np.sum(img_array[:, :, 0:3] == [100, 100, 100], axis=2) == 3
    green_pixels = np.sum(img_array[:, :, 0:3] == [0, 255, 0], axis=2) == 3

    assert np.any(gray_pixels), "No gray pixels found"
    assert np.any(green_pixels), "No green pixels found"


def test_text_highlighting_edge_cases():
    """Test highlighting edge cases and error conditions."""
    img_size = (400, 200)
    my_overlay = ImageText(image_size=img_size, background=(0, 0, 0, 255))

    text = "One two three"
    base_color = (255, 255, 255)
    highlight_color = (255, 0, 0)

    # Test invalid word index - should raise ValueError
    with pytest.raises(ValueError, match="highlight_word_index.*out of range"):
        my_overlay.write_text_box(
            text=text,
            font_filename=TEST_FONT_PATH,
            font_size=20,
            text_color=base_color,
            xy=(50, 50),
            highlight_word_index=5,  # Out of range
            highlight_color=highlight_color,
        )

    # Test negative word index - should raise ValueError
    with pytest.raises(ValueError, match="highlight_word_index.*out of range"):
        my_overlay.write_text_box(
            text=text,
            font_filename=TEST_FONT_PATH,
            font_size=20,
            text_color=base_color,
            xy=(50, 50),
            highlight_word_index=-1,
            highlight_color=highlight_color,
        )

    # Test invalid size multiplier - should raise ValueError
    with pytest.raises(ValueError, match="highlight_size_multiplier must be positive"):
        my_overlay.write_text_box(
            text=text,
            font_filename=TEST_FONT_PATH,
            font_size=20,
            text_color=base_color,
            xy=(50, 50),
            highlight_word_index=1,
            highlight_color=highlight_color,
            highlight_size_multiplier=0,
        )


def test_text_highlighting_default_color():
    """Test that highlight_color defaults to text_color when not specified."""
    img_size = (400, 200)
    my_overlay = ImageText(image_size=img_size, background=(0, 0, 0, 255))

    text = "Default color test"
    base_color = (200, 100, 50)  # Custom color

    # Don't specify highlight_color - should default to base_color
    my_overlay.write_text_box(
        text=text,
        font_filename=TEST_FONT_PATH,
        font_size=30,
        text_color=base_color,
        xy=(30, 30),
        box_width=300,
        highlight_word_index=1,  # "color"
        highlight_size_multiplier=1.8,
    )

    # Get the image array
    img_array = my_overlay.img_array

    # Check that we have pixels with the base color (both normal and highlighted text)
    base_color_pixels = np.sum(img_array[:, :, 0:3] == base_color, axis=2) == 3
    assert np.any(base_color_pixels), "No pixels found with base color"


def test_text_highlighting_multiline():
    """Test highlighting words across multiple lines."""
    img_size = (300, 400)
    my_overlay = ImageText(image_size=img_size, background=(0, 0, 0, 255))

    # Long text that will wrap to multiple lines
    text = "This is a very long text that should wrap across multiple lines when rendered"
    base_color = (255, 255, 255)
    highlight_color = (0, 0, 255)  # Blue

    # Highlight a word that should be on the second line
    my_overlay.write_text_box(
        text=text,
        font_filename=TEST_FONT_PATH,
        font_size=25,
        text_color=base_color,
        xy=(20, 20),
        box_width=250,  # Force wrapping
        highlight_word_index=8,  # "wrap" (approximate position after wrapping)
        highlight_color=highlight_color,
        highlight_size_multiplier=1.4,
    )

    # Get the image array
    img_array = my_overlay.img_array

    # Check that we have both white and blue pixels
    white_pixels = np.sum(img_array[:, :, 0:3] == [255, 255, 255], axis=2) == 3
    blue_pixels = np.sum(img_array[:, :, 0:3] == [0, 0, 255], axis=2) == 3

    assert np.any(white_pixels), "No white pixels found"
    assert np.any(blue_pixels), "No blue pixels found - multiline highlighting failed"


def test_text_highlighting_first_and_last_word():
    """Test highlighting the first and last words."""
    img_size = (500, 200)

    text = "First middle last"
    base_color = (255, 255, 255)
    highlight_color = (255, 255, 0)  # Yellow

    # Test highlighting first word
    my_overlay1 = ImageText(image_size=img_size, background=(0, 0, 0, 255))
    my_overlay1.write_text_box(
        text=text,
        font_filename=TEST_FONT_PATH,
        font_size=40,
        text_color=base_color,
        xy=(30, 30),
        box_width=400,
        highlight_word_index=0,  # "First"
        highlight_color=highlight_color,
    )

    img_array1 = my_overlay1.img_array
    yellow_pixels1 = np.sum(img_array1[:, :, 0:3] == [255, 255, 0], axis=2) == 3
    assert np.any(yellow_pixels1), "First word highlighting failed"

    # Test highlighting last word
    my_overlay2 = ImageText(image_size=img_size, background=(0, 0, 0, 255))
    my_overlay2.write_text_box(
        text=text,
        font_filename=TEST_FONT_PATH,
        font_size=40,
        text_color=base_color,
        xy=(30, 30),
        box_width=400,
        highlight_word_index=2,  # "last"
        highlight_color=highlight_color,
    )

    img_array2 = my_overlay2.img_array
    yellow_pixels2 = np.sum(img_array2[:, :, 0:3] == [255, 255, 0], axis=2) == 3
    assert np.any(yellow_pixels2), "Last word highlighting failed"
