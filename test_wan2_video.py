#!/usr/bin/env python3
"""
Test script for the updated Wan2.2 video generation implementation
"""

import os
import sys
import time
from pathlib import Path

# Add the src directory to the path so we can import videopython
sys.path.insert(0, str(Path(__file__).parent / "src"))

from videopython.ai.generation.video import TextToVideo, ImageToVideo
from PIL import Image


def test_wan2_video_generation():
    """Test the Wan2.2 video generation implementation"""

    print("🎬 Testing Wan2.2 Video Generation")
    print("=" * 50)

    # Initialize the model
    print("📦 Initializing TextToVideo model...")
    try:
        start_time = time.time()
        model = TextToVideo()
        init_time = time.time() - start_time
        print(f"✅ Model initialized successfully in {init_time:.2f} seconds")
    except Exception as e:
        print(f"❌ Failed to initialize model: {e}")
        return False

    # Test prompt
    prompt = "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
    negative_prompt = "blurry, low quality, static, watermark, text, subtitle"

    print(f"\n🎯 Generating video with prompt: '{prompt}'")
    print(f"🚫 Negative prompt: '{negative_prompt}'")

    # Generate video
    try:
        start_time = time.time()
        video = model.generate_video(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_steps=10,
            height=480,
            width=832,
            num_frames=49,
            guidance_scale=4.0,
        )
        generation_time = time.time() - start_time

        print(f"✅ Video generated successfully in {generation_time:.2f} seconds")
        print(f"📊 Video info: {video.width}x{video.height}, {len(video.frames)} frames, {video.fps} fps")

        # Save the video
        output_path = "test_wan2_output.mp4"
        video.save(output_path)
        print(f"💾 Video saved to: {output_path}")

        return True

    except Exception as e:
        print(f"❌ Failed to generate video: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_different_parameters():
    """Test with different generation parameters"""

    print("\n🎛️  Testing Different Parameters")
    print("=" * 50)

    try:
        model = TextToVideo()

        # Test smaller video
        print("🔸 Testing smaller resolution (320x576)...")
        video = model.generate_video(
            prompt="A cute cat playing with a ball of yarn", height=320, width=576, num_frames=25, num_steps=8
        )
        video.save("test_wan2_small.mp4")
        print(f"✅ Small video: {video.width}x{video.height}, {len(video.frames)} frames")

        # Test with higher guidance
        print("🔸 Testing higher guidance scale (7.0)...")
        video = model.generate_video(
            prompt="A majestic eagle soaring through mountain peaks", guidance_scale=7.0, num_steps=12, num_frames=37
        )
        video.save("test_wan2_high_guidance.mp4")
        print(f"✅ High guidance video: {video.width}x{video.height}, {len(video.frames)} frames")

        return True

    except Exception as e:
        print(f"❌ Parameter test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_image_to_video():
    """Test Image-to-Video generation with Wan2.2"""
    
    print("\n🖼️  Testing Image-to-Video Generation")
    print("=" * 50)
    
    # Create a simple test image
    test_image = Image.new('RGB', (512, 512), color='blue')
    test_image.save("test_input_image.png")
    print("📸 Created test input image: test_input_image.png")
    
    try:
        print("📦 Initializing ImageToVideo model...")
        model = ImageToVideo()
        print("✅ ImageToVideo model initialized")
        
        # Generate video from image
        prompt = "The blue square transforms into a vibrant animated scene"
        negative_prompt = "blurry, low quality, static, watermark"
        
        print(f"🎯 Generating video from image with prompt: '{prompt}'")
        
        start_time = time.time()
        video = model.generate_video(
            image=test_image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=720,
            width=1280,
            num_frames=41,  # Smaller for faster testing
            guidance_scale=4.0,
            guidance_scale_2=3.0,
            num_inference_steps=20,  # Reduced for testing
            fps=16.0
        )
        generation_time = time.time() - start_time
        
        print(f"✅ Image-to-video generated in {generation_time:.2f} seconds")
        print(f"📊 Video info: {video.width}x{video.height}, {len(video.frames)} frames, {video.fps} fps")
        
        # Save the video
        output_path = "test_image_to_video.mp4"
        video.save(output_path)
        print(f"💾 Image-to-video saved to: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Image-to-video test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function"""
    print("🚀 Starting Wan2.2 Video Generation Tests")
    print(f"📍 Working directory: {os.getcwd()}")
    print(f"🐍 Python path: {sys.executable}")

    # Run basic test
    success = test_wan2_video_generation()

    if success:
        print("\n" + "=" * 50)
        print("🎉 Text-to-video test passed! Running parameter tests...")
        test_different_parameters()
        
        print("\n" + "=" * 50)
        print("🖼️  Running image-to-video test...")
        test_image_to_video()
        
        print("\n🏁 All tests completed!")
    else:
        print("\n❌ Basic test failed. Skipping other tests.")
        sys.exit(1)


if __name__ == "__main__":
    main()
