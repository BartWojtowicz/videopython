#!/usr/bin/env python3
"""Test script to demonstrate scene detection and frame understanding."""

from videopython.base.video import Video
from videopython.ai.understanding.scenes import SceneDetector
from videopython.ai.understanding.frames import ImageToText

# Load video
print("Loading video...")
video = Video.from_path("src/tests/test_data/big_video.mp4")
print(f"Video loaded: {video.metadata}\n")

# Detect scenes
print("Detecting scenes...")
detector = SceneDetector(threshold=0.3, min_scene_length=0.5)
scenes = detector.detect(video)
print(f"Detected {len(scenes)} scenes\n")

# Initialize image-to-text model
print("Initializing ImageToText model (this may take a moment)...")
image_to_text = ImageToText(device="cpu")
print("Model loaded\n")

# Describe each scene
print("=" * 80)
for i, scene in enumerate(scenes):
    print(f"\nScene {i+1}:")
    print(f"  Duration: {scene.start:.2f}s - {scene.end:.2f}s ({scene.duration:.2f}s)")
    print(f"  Frames: {scene.start_frame} - {scene.end_frame} ({scene.frame_count} frames)")

    print(f"\n  Analyzing frames (sampling at 1 fps)...")
    frame_descriptions = image_to_text.describe_scene(video, scene, frames_per_second=1.0)

    print(f"  Sampled {len(frame_descriptions)} frames:")
    for fd in frame_descriptions:
        print(f"    - Frame {fd.frame_index} ({fd.timestamp:.2f}s): {fd.description}")

    print("-" * 80)

print("\nDone!")
