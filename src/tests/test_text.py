import numpy as np
import pytest

from videopython.base.effects import FullImageOverlay
from videopython.utils.text import AnchorPoint, ImageText

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
        place="center",
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
