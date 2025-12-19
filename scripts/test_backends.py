#!/usr/bin/env python3
"""Integration test script for all AI backends.

Requires the following environment variables:
- OPENAI_API_KEY
- GOOGLE_API_KEY
- LUMA_API_KEY
- RUNWAY_API_KEY (optional, API not publicly available yet)
- REPLICATE_API_TOKEN
- ELEVENLABS_API_KEY

Usage:
    uv run python scripts/test_backends.py
    uv run python scripts/test_backends.py --skip-local  # Skip local GPU models
    uv run python scripts/test_backends.py --only openai  # Test only OpenAI backends
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import tempfile
import time
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from PIL import Image
import numpy as np


def create_test_image() -> Image.Image:
    """Create a simple test image."""
    img = Image.new("RGB", (512, 512), color=(100, 150, 200))
    return img


def create_test_audio():
    """Create a simple test audio."""
    from soundpython import Audio, AudioMetadata

    # Generate 2 seconds of simple sine wave
    sample_rate = 24000
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    audio_data = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave

    metadata = AudioMetadata(
        sample_rate=sample_rate,
        channels=1,
        sample_width=2,
        duration_seconds=duration,
        frame_count=len(audio_data),
    )
    return Audio(audio_data, metadata)


class BackendTester:
    def __init__(self, skip_local: bool = False, only_backend: str | None = None):
        self.skip_local = skip_local
        self.only_backend = only_backend
        self.results: list[dict] = []

    def should_test(self, backend: str) -> bool:
        if self.only_backend and backend != self.only_backend:
            return False
        if self.skip_local and backend == "local":
            return False
        return True

    async def run_test(self, name: str, coro):
        """Run a single test and record results."""
        print(f"\n{'='*60}")
        print(f"Testing: {name}")
        print("=" * 60)

        start = time.time()
        try:
            result = await coro
            elapsed = time.time() - start
            print(f"SUCCESS in {elapsed:.2f}s")
            print(f"Result type: {type(result).__name__}")
            if hasattr(result, "__len__"):
                print(f"Result length/size: {len(result)}")
            self.results.append({"name": name, "status": "PASS", "time": elapsed})
            return result
        except Exception as e:
            elapsed = time.time() - start
            print(f"FAILED in {elapsed:.2f}s")
            print(f"Error: {type(e).__name__}: {e}")
            self.results.append({"name": name, "status": "FAIL", "time": elapsed, "error": str(e)})
            return None

    async def test_text_to_image(self):
        """Test TextToImage with all backends."""
        from videopython.ai.generation import TextToImage

        prompt = "A beautiful sunset over mountains, digital art"

        # OpenAI (DALL-E 3)
        if self.should_test("openai"):
            generator = TextToImage(backend="openai")
            image = await self.run_test("TextToImage [openai/DALL-E 3]", generator.generate_image(prompt))
            if image:
                image.save("/tmp/test_text_to_image_openai.png")
                print(f"Saved to /tmp/test_text_to_image_openai.png ({image.size})")

        # Replicate (Flux)
        if self.should_test("replicate"):
            generator = TextToImage(backend="replicate")
            image = await self.run_test("TextToImage [replicate/flux]", generator.generate_image(prompt))
            if image:
                image.save("/tmp/test_text_to_image_replicate.png")
                print(f"Saved to /tmp/test_text_to_image_replicate.png ({image.size})")

        # Local (requires CUDA)
        if self.should_test("local"):
            generator = TextToImage(backend="local")
            image = await self.run_test("TextToImage [local/SDXL]", generator.generate_image(prompt))
            if image:
                image.save("/tmp/test_text_to_image_local.png")
                print(f"Saved to /tmp/test_text_to_image_local.png ({image.size})")

    async def test_text_to_video(self):
        """Test TextToVideo with all backends."""
        from videopython.ai.generation import TextToVideo

        prompt = "A dog running on a beach, slow motion"

        # Luma (Dream Machine)
        if self.should_test("luma"):
            generator = TextToVideo(backend="luma")
            video = await self.run_test("TextToVideo [luma/Dream Machine]", generator.generate_video(prompt))
            if video:
                path = video.save("/tmp/test_text_to_video_luma.mp4")
                print(f"Saved to {path} ({len(video.frames)} frames)")

        # Replicate
        if self.should_test("replicate"):
            generator = TextToVideo(backend="replicate")
            video = await self.run_test("TextToVideo [replicate]", generator.generate_video(prompt, num_frames=24))
            if video:
                path = video.save("/tmp/test_text_to_video_replicate.mp4")
                print(f"Saved to {path} ({len(video.frames)} frames)")

        # Local (requires CUDA)
        if self.should_test("local"):
            generator = TextToVideo(backend="local")
            video = await self.run_test(
                "TextToVideo [local/Zeroscope]", generator.generate_video(prompt, num_frames=24)
            )
            if video:
                path = video.save("/tmp/test_text_to_video_local.mp4")
                print(f"Saved to {path} ({len(video.frames)} frames)")

    async def test_image_to_video(self):
        """Test ImageToVideo with all backends."""
        from videopython.ai.generation import ImageToVideo

        image = create_test_image()

        # Luma (Dream Machine)
        if self.should_test("luma"):
            generator = ImageToVideo(backend="luma")
            video = await self.run_test("ImageToVideo [luma/Dream Machine]", generator.generate_video(image))
            if video:
                path = video.save("/tmp/test_image_to_video_luma.mp4")
                print(f"Saved to {path} ({len(video.frames)} frames)")

        # Replicate
        if self.should_test("replicate"):
            generator = ImageToVideo(backend="replicate")
            video = await self.run_test("ImageToVideo [replicate]", generator.generate_video(image))
            if video:
                path = video.save("/tmp/test_image_to_video_replicate.mp4")
                print(f"Saved to {path} ({len(video.frames)} frames)")

        # Local (requires CUDA)
        if self.should_test("local"):
            generator = ImageToVideo(backend="local")
            video = await self.run_test("ImageToVideo [local/SVD]", generator.generate_video(image))
            if video:
                path = video.save("/tmp/test_image_to_video_local.mp4")
                print(f"Saved to {path} ({len(video.frames)} frames)")

    async def test_text_to_speech(self):
        """Test TextToSpeech with all backends."""
        from videopython.ai.generation import TextToSpeech

        text = "Hello! This is a test of the text to speech system. How does it sound?"

        # OpenAI TTS
        if self.should_test("openai"):
            generator = TextToSpeech(backend="openai", voice="nova")
            audio = await self.run_test("TextToSpeech [openai/TTS]", generator.generate_audio(text))
            if audio:
                audio.to_wav("/tmp/test_tts_openai.wav")
                print(f"Saved to /tmp/test_tts_openai.wav ({audio.metadata.duration_seconds:.2f}s)")

        # ElevenLabs
        if self.should_test("elevenlabs"):
            generator = TextToSpeech(backend="elevenlabs", voice="Rachel")
            audio = await self.run_test("TextToSpeech [elevenlabs]", generator.generate_audio(text))
            if audio:
                audio.to_wav("/tmp/test_tts_elevenlabs.wav")
                print(f"Saved to /tmp/test_tts_elevenlabs.wav ({audio.metadata.duration_seconds:.2f}s)")

        # Local (Bark)
        if self.should_test("local"):
            generator = TextToSpeech(backend="local", model_size="small")
            audio = await self.run_test("TextToSpeech [local/Bark]", generator.generate_audio(text))
            if audio:
                audio.to_wav("/tmp/test_tts_local.wav")
                print(f"Saved to /tmp/test_tts_local.wav ({audio.metadata.duration_seconds:.2f}s)")

    async def test_text_to_music(self):
        """Test TextToMusic with all backends."""
        from videopython.ai.generation import TextToMusic

        prompt = "Happy upbeat electronic music with synths"

        # Replicate (MusicGen)
        if self.should_test("replicate"):
            generator = TextToMusic(backend="replicate")
            audio = await self.run_test("TextToMusic [replicate/MusicGen]", generator.generate_audio(prompt, max_new_tokens=256))
            if audio:
                audio.to_wav("/tmp/test_music_replicate.wav")
                print(f"Saved to /tmp/test_music_replicate.wav ({audio.metadata.duration_seconds:.2f}s)")

        # Local (MusicGen)
        if self.should_test("local"):
            generator = TextToMusic(backend="local")
            audio = await self.run_test("TextToMusic [local/MusicGen]", generator.generate_audio(prompt, max_new_tokens=256))
            if audio:
                audio.to_wav("/tmp/test_music_local.wav")
                print(f"Saved to /tmp/test_music_local.wav ({audio.metadata.duration_seconds:.2f}s)")

    async def test_image_to_text(self):
        """Test ImageToText with all backends."""
        from videopython.ai.understanding import ImageToText

        image = create_test_image()

        # OpenAI (GPT-4o Vision)
        if self.should_test("openai"):
            analyzer = ImageToText(backend="openai")
            desc = await self.run_test("ImageToText [openai/GPT-4o]", analyzer.describe_image(image))
            if desc:
                print(f"Description: {desc[:200]}...")

        # Gemini
        if self.should_test("gemini"):
            analyzer = ImageToText(backend="gemini")
            desc = await self.run_test("ImageToText [gemini]", analyzer.describe_image(image))
            if desc:
                print(f"Description: {desc[:200]}...")

        # Local (BLIP)
        if self.should_test("local"):
            analyzer = ImageToText(backend="local")
            desc = await self.run_test("ImageToText [local/BLIP]", analyzer.describe_image(image))
            if desc:
                print(f"Description: {desc[:200]}...")

    async def test_audio_to_text(self):
        """Test AudioToText with all backends."""
        from videopython.ai.understanding import AudioToText

        # For this test we need actual speech audio
        # We'll use a simple test or skip if no test file available
        test_audio_path = Path(__file__).parent.parent / "src/tests/test_data/test_audio.mp3"

        if not test_audio_path.exists():
            print("Skipping AudioToText tests - no test audio file available")
            return

        from soundpython import Audio

        audio = Audio.from_file(str(test_audio_path))

        # OpenAI (Whisper API)
        if self.should_test("openai"):
            transcriber = AudioToText(backend="openai")
            transcription = await self.run_test("AudioToText [openai/Whisper API]", transcriber.transcribe(audio))
            if transcription:
                text = " ".join([s.text for s in transcription.segments])
                print(f"Transcription: {text[:200]}...")

        # Gemini
        if self.should_test("gemini"):
            transcriber = AudioToText(backend="gemini")
            transcription = await self.run_test("AudioToText [gemini]", transcriber.transcribe(audio))
            if transcription:
                text = " ".join([s.text for s in transcription.segments])
                print(f"Transcription: {text[:200]}...")

        # Local (Whisper)
        if self.should_test("local"):
            transcriber = AudioToText(backend="local", model_name="tiny")
            transcription = await self.run_test("AudioToText [local/Whisper]", transcriber.transcribe(audio))
            if transcription:
                text = " ".join([s.text for s in transcription.segments])
                print(f"Transcription: {text[:200]}...")

    async def test_llm_summarizer(self):
        """Test LLMSummarizer with all backends."""
        from videopython.ai.understanding import LLMSummarizer

        frame_descriptions = [
            (0.0, "A person walking into a room"),
            (1.0, "The person sits down at a desk"),
            (2.0, "They open a laptop and start typing"),
            (3.0, "Close-up of hands on keyboard"),
            (4.0, "The person smiles at the screen"),
        ]

        # OpenAI
        if self.should_test("openai"):
            summarizer = LLMSummarizer(backend="openai")
            summary = await self.run_test("LLMSummarizer [openai/GPT-4o]", summarizer.summarize_scene(frame_descriptions))
            if summary:
                print(f"Summary: {summary}")

        # Gemini
        if self.should_test("gemini"):
            summarizer = LLMSummarizer(backend="gemini")
            summary = await self.run_test("LLMSummarizer [gemini]", summarizer.summarize_scene(frame_descriptions))
            if summary:
                print(f"Summary: {summary}")

        # Local (Ollama) - requires Ollama running
        if self.should_test("local"):
            summarizer = LLMSummarizer(backend="local")
            summary = await self.run_test("LLMSummarizer [local/Ollama]", summarizer.summarize_scene(frame_descriptions))
            if summary:
                print(f"Summary: {summary}")

    async def test_concurrent_generation(self):
        """Test concurrent generation with multiple backends."""
        from videopython.ai.generation import TextToImage, TextToSpeech
        from videopython.ai.understanding import LLMSummarizer

        if not (self.should_test("openai")):
            print("Skipping concurrent test - requires openai backend")
            return

        print("\n" + "=" * 60)
        print("Testing: Concurrent Generation")
        print("=" * 60)

        start = time.time()

        # Run multiple generations concurrently
        image_gen = TextToImage(backend="openai")
        tts_gen = TextToSpeech(backend="openai")
        summarizer = LLMSummarizer(backend="openai")

        try:
            image, audio, summary = await asyncio.gather(
                image_gen.generate_image("A cozy coffee shop interior"),
                tts_gen.generate_audio("Welcome to our coffee shop!"),
                summarizer.summarize_scene([
                    (0.0, "Interior of a warm coffee shop"),
                    (1.0, "Barista making coffee"),
                    (2.0, "Customer receiving their order"),
                ]),
            )

            elapsed = time.time() - start
            print(f"SUCCESS - All 3 generations completed in {elapsed:.2f}s")

            if image:
                image.save("/tmp/test_concurrent_image.png")
            if audio:
                audio.to_wav("/tmp/test_concurrent_audio.wav")
            if summary:
                print(f"Summary: {summary}")

            self.results.append({"name": "Concurrent Generation [openai]", "status": "PASS", "time": elapsed})

        except Exception as e:
            elapsed = time.time() - start
            print(f"FAILED in {elapsed:.2f}s: {e}")
            self.results.append({
                "name": "Concurrent Generation [openai]",
                "status": "FAIL",
                "time": elapsed,
                "error": str(e),
            })

    def print_summary(self):
        """Print test summary."""
        print("\n")
        print("=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)

        passed = [r for r in self.results if r["status"] == "PASS"]
        failed = [r for r in self.results if r["status"] == "FAIL"]

        print(f"\nTotal: {len(self.results)} | Passed: {len(passed)} | Failed: {len(failed)}")

        if passed:
            print("\nPASSED:")
            for r in passed:
                print(f"  - {r['name']} ({r['time']:.2f}s)")

        if failed:
            print("\nFAILED:")
            for r in failed:
                print(f"  - {r['name']}: {r.get('error', 'Unknown error')}")

        total_time = sum(r["time"] for r in self.results)
        print(f"\nTotal time: {total_time:.2f}s")

    async def run_all_tests(self):
        """Run all backend tests."""
        print("=" * 60)
        print("VIDEOPYTHON BACKEND INTEGRATION TESTS")
        print("=" * 60)

        # Check environment variables
        env_vars = {
            "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
            "GOOGLE_API_KEY": os.environ.get("GOOGLE_API_KEY"),
            "LUMA_API_KEY": os.environ.get("LUMA_API_KEY"),
            "REPLICATE_API_TOKEN": os.environ.get("REPLICATE_API_TOKEN"),
            "ELEVENLABS_API_KEY": os.environ.get("ELEVENLABS_API_KEY"),
        }

        print("\nEnvironment variables:")
        for key, value in env_vars.items():
            status = "SET" if value else "NOT SET"
            print(f"  {key}: {status}")

        print(f"\nSkip local: {self.skip_local}")
        print(f"Only backend: {self.only_backend or 'all'}")

        # Run tests
        await self.test_text_to_image()
        await self.test_text_to_speech()
        await self.test_text_to_music()
        await self.test_image_to_text()
        await self.test_audio_to_text()
        await self.test_llm_summarizer()

        # Video generation tests (slower, run last)
        await self.test_text_to_video()
        await self.test_image_to_video()

        # Concurrent test
        await self.test_concurrent_generation()

        self.print_summary()


def main():
    parser = argparse.ArgumentParser(description="Test all AI backends")
    parser.add_argument("--skip-local", action="store_true", help="Skip local GPU models")
    parser.add_argument("--only", type=str, help="Test only this backend (openai, gemini, luma, replicate, elevenlabs, local)")
    args = parser.parse_args()

    tester = BackendTester(skip_local=args.skip_local, only_backend=args.only)
    asyncio.run(tester.run_all_tests())


if __name__ == "__main__":
    main()
