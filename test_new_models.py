#!/usr/bin/env python3
"""
Test script for newly updated AI models.
Run this on a GPU-enabled machine (H100 recommended) with:
    uv run python test_new_models.py

Tests:
1. TextToImage (FLUX.1-dev)
2. ImageToText (Qwen2.5-VL-7B)
3. TextToSpeech (Kokoro-82M)
"""

import sys
from pathlib import Path

import torch


def test_text_to_image():
    """Test FLUX.1-dev text-to-image generation."""
    print("\n" + "=" * 80)
    print("Testing TextToImage (FLUX.1-dev)")
    print("=" * 80)

    from videopython.ai.generation.image import TextToImage

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. FLUX.1-dev requires GPU.")
        return False

    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    try:
        print("\nInitializing TextToImage model...")
        model = TextToImage()
        print("Model loaded successfully!")

        print("\nGenerating image with prompt: 'A cat holding a sign that says hello world'")
        image = model.generate_image(
            prompt="A cat holding a sign that says hello world",
            height=1024,
            width=1024,
            num_inference_steps=30,  # Reduced for faster testing
        )

        # Save the image
        output_path = Path("test_output_flux_image.png")
        image.save(output_path)
        print(f"Image saved to: {output_path.absolute()}")
        print("TextToImage test PASSED")
        return True

    except Exception as e:
        print(f"ERROR in TextToImage test: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_image_to_text():
    """Test Qwen2.5-VL-7B image-to-text understanding."""
    print("\n" + "=" * 80)
    print("Testing ImageToText (Qwen2.5-VL-7B)")
    print("=" * 80)

    from videopython.ai.understanding.image import ImageToText

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    try:
        print("\nInitializing ImageToText model...")
        model = ImageToText(device=device)
        print("Model loaded successfully!")

        # Create a simple test image
        from PIL import Image

        print("\nCreating test image (blue square with red circle)...")
        test_image = Image.new("RGB", (512, 512), color="blue")
        from PIL import ImageDraw

        draw = ImageDraw.Draw(test_image)
        draw.ellipse([128, 128, 384, 384], fill="red")

        # Save test image
        test_image.save("test_input_image.png")
        print("Test image saved to: test_input_image.png")

        print("\nDescribing the image...")
        description = model.describe_image(test_image)
        print(f"Description: {description}")

        print("\nDescribing with custom prompt...")
        description_custom = model.describe_image(test_image, prompt="What colors are in this image?")
        print(f"Custom description: {description_custom}")

        print("ImageToText test PASSED")
        return True

    except Exception as e:
        print(f"ERROR in ImageToText test: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_text_to_speech():
    """Test Kokoro-82M text-to-speech generation."""
    print("\n" + "=" * 80)
    print("Testing TextToSpeech (Kokoro-82M)")
    print("=" * 80)

    from videopython.ai.generation.audio import TextToSpeech

    try:
        print("\nInitializing TextToSpeech model...")
        model = TextToSpeech()
        print("Model loaded successfully!")

        test_text = "Hello world! This is a test of the Kokoro text-to-speech model."
        print(f"\nGenerating speech from text: '{test_text}'")
        audio = model.generate_audio(test_text)

        print(f"Generated audio: {audio.metadata.duration_seconds:.2f} seconds")
        print(f"Sample rate: {audio.metadata.sample_rate} Hz")
        print(f"Channels: {audio.metadata.channels}")

        # Save the audio
        output_path = Path("test_output_kokoro_speech.wav")
        audio.save(str(output_path))
        print(f"Audio saved to: {output_path.absolute()}")

        print("TextToSpeech test PASSED")
        return True

    except Exception as e:
        print(f"ERROR in TextToSpeech test: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests and report results."""
    print("=" * 80)
    print("Testing Updated AI Models")
    print("=" * 80)

    results = {
        "TextToImage (FLUX.1-dev)": test_text_to_image(),
        "ImageToText (Qwen2.5-VL-7B)": test_image_to_text(),
        "TextToSpeech (Kokoro-82M)": test_text_to_speech(),
    }

    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    for model_name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"{model_name}: {status}")

    all_passed = all(results.values())
    print("\n" + "=" * 80)
    if all_passed:
        print("ALL TESTS PASSED")
        print("=" * 80)
        return 0
    else:
        print("SOME TESTS FAILED")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
