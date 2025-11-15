from __future__ import annotations

from unittest.mock import patch

from videopython.ai.understanding.summarize import LLMSummarizer
from videopython.base.frames import FrameDescription
from videopython.base.scene_description import SceneDescription
from videopython.base.scenes import Scene
from videopython.base.video_description import VideoDescription


class TestLLMSummarizer:
    def test_initialization(self):
        """Test LLMSummarizer initialization."""
        summarizer = LLMSummarizer(model="llama3.2", timeout=60.0)
        assert summarizer.model == "llama3.2"
        assert summarizer.timeout == 60.0

    def test_initialization_defaults(self):
        """Test LLMSummarizer with default parameters."""
        summarizer = LLMSummarizer()
        assert summarizer.model == "llama3.2"
        assert summarizer.timeout == 30.0

    @patch("videopython.ai.understanding.summarize.ollama.generate")
    def test_summarize_scene_success(self, mock_generate):
        """Test scene summarization with successful LLM response."""
        mock_generate.return_value = {"response": "A dog runs and plays in the park."}

        summarizer = LLMSummarizer()
        frame_descriptions = [
            (0.0, "a dog running in a park"),
            (1.0, "a dog jumping"),
            (2.0, "a dog playing with a ball"),
        ]

        summary = summarizer.summarize_scene(frame_descriptions)
        assert summary == "A dog runs and plays in the park."
        mock_generate.assert_called_once()

    @patch("videopython.ai.understanding.summarize.ollama.generate")
    def test_summarize_scene_fallback(self, mock_generate):
        """Test scene summarization falls back to concatenation on error."""
        mock_generate.side_effect = Exception("Ollama not available")

        summarizer = LLMSummarizer()
        frame_descriptions = [
            (0.0, "a dog running"),
            (1.0, "a dog jumping"),
        ]

        summary = summarizer.summarize_scene(frame_descriptions)
        assert summary == "a dog running a dog jumping"

    def test_summarize_scene_empty(self):
        """Test scene summarization with empty frame list."""
        summarizer = LLMSummarizer()
        summary = summarizer.summarize_scene([])
        assert summary == "Empty scene with no frames."

    @patch("videopython.ai.understanding.summarize.ollama.generate")
    def test_summarize_video_success(self, mock_generate):
        """Test video summarization with successful LLM response."""
        mock_generate.return_value = {"response": "The video shows a person's daily routine from morning to evening."}

        summarizer = LLMSummarizer()
        scene_summaries = [
            (0.0, 5.0, "Person wakes up and gets ready"),
            (5.0, 10.0, "Person eats breakfast"),
            (10.0, 15.0, "Person goes to work"),
        ]

        summary = summarizer.summarize_video(scene_summaries)
        assert summary == "The video shows a person's daily routine from morning to evening."
        mock_generate.assert_called_once()

    @patch("videopython.ai.understanding.summarize.ollama.generate")
    def test_summarize_video_fallback(self, mock_generate):
        """Test video summarization falls back to concatenation on error."""
        mock_generate.side_effect = Exception("Ollama not available")

        summarizer = LLMSummarizer()
        scene_summaries = [
            (0.0, 5.0, "Morning routine"),
            (5.0, 10.0, "Breakfast time"),
        ]

        summary = summarizer.summarize_video(scene_summaries)
        assert summary == "Morning routine Breakfast time"

    def test_summarize_video_empty(self):
        """Test video summarization with empty scene list."""
        summarizer = LLMSummarizer()
        summary = summarizer.summarize_video([])
        assert summary == "Empty video with no scenes."

    @patch("videopython.ai.understanding.summarize.ollama.generate")
    def test_summarize_scene_description(self, mock_generate):
        """Test summarizing a SceneDescription object."""
        mock_generate.return_value = {"response": "A red and blue scene."}

        scene = Scene(start=0.0, end=5.0, start_frame=0, end_frame=120)
        frame_desc1 = FrameDescription(frame_index=0, timestamp=0.0, description="red background")
        frame_desc2 = FrameDescription(frame_index=24, timestamp=1.0, description="blue foreground")
        scene_desc = SceneDescription(scene=scene, frame_descriptions=[frame_desc1, frame_desc2])

        summarizer = LLMSummarizer()
        summary = summarizer.summarize_scene_description(scene_desc)

        assert summary == "A red and blue scene."
        mock_generate.assert_called_once()

    @patch("videopython.ai.understanding.summarize.ollama.generate")
    def test_summarize_video_description(self, mock_generate):
        """Test summarizing a VideoDescription object."""
        mock_generate.return_value = {"response": "A complete video analysis."}

        scene1 = Scene(start=0.0, end=5.0, start_frame=0, end_frame=120)
        frame_desc1 = FrameDescription(frame_index=0, timestamp=0.0, description="scene one")
        scene_desc1 = SceneDescription(scene=scene1, frame_descriptions=[frame_desc1])

        scene2 = Scene(start=5.0, end=10.0, start_frame=120, end_frame=240)
        frame_desc2 = FrameDescription(frame_index=120, timestamp=5.0, description="scene two")
        scene_desc2 = SceneDescription(scene=scene2, frame_descriptions=[frame_desc2])

        video_desc = VideoDescription(scene_descriptions=[scene_desc1, scene_desc2])

        summarizer = LLMSummarizer()
        summary = summarizer.summarize_video_description(video_desc)

        assert summary == "A complete video analysis."
        mock_generate.assert_called_once()

    @patch("videopython.ai.understanding.summarize.ollama.generate")
    def test_temperature_setting(self, mock_generate):
        """Test that temperature is set correctly in LLM options."""
        mock_generate.return_value = {"response": "Summary"}

        summarizer = LLMSummarizer()
        summarizer.summarize_scene([(0.0, "test")])

        call_args = mock_generate.call_args
        assert call_args[1]["options"]["temperature"] == 0.3

    @patch("videopython.ai.understanding.summarize.ollama.generate")
    def test_custom_model(self, mock_generate):
        """Test using a custom model."""
        mock_generate.return_value = {"response": "Summary"}

        summarizer = LLMSummarizer(model="llama3.1")
        summarizer.summarize_scene([(0.0, "test")])

        call_args = mock_generate.call_args
        assert call_args[1]["model"] == "llama3.1"
