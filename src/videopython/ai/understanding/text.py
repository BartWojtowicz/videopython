"""Text/LLM understanding with multi-backend support."""

from __future__ import annotations

from videopython.ai.backends import LLMBackend, UnsupportedBackendError, get_api_key
from videopython.ai.config import get_default_backend
from videopython.base.description import SceneDescription, VideoDescription


class LLMSummarizer:
    """Generates coherent summaries of video content using LLMs."""

    SUPPORTED_BACKENDS: list[str] = ["local", "openai", "gemini"]

    def __init__(
        self,
        backend: LLMBackend | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float = 30.0,
    ):
        """Initialize the LLM summarizer.

        Args:
            backend: Backend to use. If None, uses config default or 'local'.
            model: Model name (backend-specific). If None, uses default per backend.
            api_key: API key for cloud backends. If None, reads from environment.
            timeout: Request timeout in seconds.
        """
        resolved_backend: str = backend if backend is not None else get_default_backend("llm_summarizer")
        if resolved_backend not in self.SUPPORTED_BACKENDS:
            raise UnsupportedBackendError(resolved_backend, self.SUPPORTED_BACKENDS)

        self.backend: LLMBackend = resolved_backend  # type: ignore[assignment]
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def _get_model_name(self) -> str:
        """Get the model name for the current backend."""
        if self.model:
            return self.model

        if self.backend == "local":
            return "llama3.2"
        elif self.backend == "openai":
            return "gpt-4o"
        elif self.backend == "gemini":
            return "gemini-2.0-flash"
        else:
            return "llama3.2"

    def _generate_local(self, prompt: str) -> str:
        """Generate text using local Ollama."""
        import ollama

        model = self._get_model_name()

        response = ollama.generate(
            model=model,
            prompt=prompt,
            options={"temperature": 0.3, "num_predict": 150},
        )
        return response["response"].strip()

    def _generate_openai(self, prompt: str) -> str:
        """Generate text using OpenAI."""
        from openai import OpenAI

        api_key = get_api_key("openai", self.api_key)
        client = OpenAI(api_key=api_key)

        model = self._get_model_name()

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.3,
        )

        return response.choices[0].message.content or ""

    def _generate_gemini(self, prompt: str) -> str:
        """Generate text using Google Gemini."""
        import google.generativeai as genai

        api_key = get_api_key("gemini", self.api_key)
        genai.configure(api_key=api_key)

        model_name = self._get_model_name()
        model = genai.GenerativeModel(model_name)

        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.3,
                max_output_tokens=150,
            ),
        )
        return response.text

    def _generate(self, prompt: str) -> str:
        """Generate text using the configured backend."""
        if self.backend == "local":
            return self._generate_local(prompt)
        elif self.backend == "openai":
            return self._generate_openai(prompt)
        elif self.backend == "gemini":
            return self._generate_gemini(prompt)
        else:
            raise UnsupportedBackendError(self.backend, self.SUPPORTED_BACKENDS)

    def summarize_scene(self, frame_descriptions: list[tuple[float, str]]) -> str:
        """Generate a coherent summary of a scene from frame descriptions.

        Args:
            frame_descriptions: List of (timestamp, description) tuples for frames.

        Returns:
            2-3 sentence coherent summary of the scene.
        """
        if not frame_descriptions:
            return "Empty scene with no frames."

        frames_text = "\n".join([f"- At {ts:.2f}s: {desc}" for ts, desc in frame_descriptions])

        prompt = f"""You are analyzing a video scene. Below are descriptions of individual frames sampled from \
this scene:

{frames_text}

Task: Write a coherent 2-3 sentence summary of what happens in this scene. Focus on:
- Main actions or events
- Key objects or people present
- Any changes or progression within the scene

Remove redundancy and synthesize the information into a flowing narrative. Be concise and specific.

Summary:"""

        try:
            return self._generate(prompt)
        except Exception:
            # Fallback: return concatenated descriptions
            return " ".join([desc for _, desc in frame_descriptions])

    def summarize_video(self, scene_summaries: list[tuple[float, float, str]]) -> str:
        """Generate a high-level summary of the entire video from scene summaries.

        Args:
            scene_summaries: List of (start_time, end_time, summary) tuples for each scene.

        Returns:
            Paragraph describing the entire video narrative.
        """
        if not scene_summaries:
            return "Empty video with no scenes."

        scenes_text = "\n".join(
            [f"- Scene at {start:.2f}s-{end:.2f}s: {summary}" for start, end, summary in scene_summaries]
        )

        prompt = f"""You are analyzing a video. Below are summaries of different scenes in the video:

{scenes_text}

Task: Write a coherent paragraph (3-5 sentences) summarizing the entire video. Focus on:
- Overall narrative or theme
- Main events in chronological order
- Key subjects or topics covered
- The progression from beginning to end

Synthesize the scenes into a high-level overview that captures the video's essence.

Summary:"""

        try:
            return self._generate(prompt)
        except Exception:
            # Fallback: return concatenated scene summaries
            return " ".join([summary for _, _, summary in scene_summaries])

    def summarize_scene_description(self, scene_description: SceneDescription) -> str:
        """Generate summary from a SceneDescription object.

        Args:
            scene_description: SceneDescription object with frame descriptions.

        Returns:
            Coherent summary of the scene.
        """
        frame_descriptions = [(fd.timestamp, fd.description) for fd in scene_description.frame_descriptions]
        return self.summarize_scene(frame_descriptions)

    def summarize_video_description(self, video_description: VideoDescription) -> str:
        """Generate summary from a VideoDescription object.

        Args:
            video_description: VideoDescription object with scene descriptions.

        Returns:
            High-level summary of the entire video.
        """
        scene_summaries = [
            (sd.start, sd.end, sd.get_description_summary()) for sd in video_description.scene_descriptions
        ]
        return self.summarize_video(scene_summaries)
