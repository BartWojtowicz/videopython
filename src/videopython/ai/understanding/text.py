from __future__ import annotations

import ollama

from videopython.base.description import SceneDescription, VideoDescription


class LLMSummarizer:
    """Generates coherent summaries of video content using LLMs via Ollama."""

    def __init__(self, model: str = "llama3.2", timeout: float = 30.0):
        """Initialize the LLM summarizer.

        Args:
            model: Ollama model name to use (default: "llama3.2")
            timeout: Request timeout in seconds (default: 30.0)
        """
        self.model = model
        self.timeout = timeout

    def summarize_scene(self, frame_descriptions: list[tuple[float, str]]) -> str:
        """Generate a coherent summary of a scene from frame descriptions.

        Args:
            frame_descriptions: List of (timestamp, description) tuples for frames in the scene

        Returns:
            2-3 sentence coherent summary of the scene
        """
        if not frame_descriptions:
            return "Empty scene with no frames."

        # Build the frame descriptions text
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
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                options={"temperature": 0.3, "num_predict": 100},
            )
            summary = response["response"].strip()
            return summary
        except Exception:
            # Fallback: return concatenated descriptions
            return " ".join([desc for _, desc in frame_descriptions])

    def summarize_video(self, scene_summaries: list[tuple[float, float, str]]) -> str:
        """Generate a high-level summary of the entire video from scene summaries.

        Args:
            scene_summaries: List of (start_time, end_time, summary) tuples for each scene

        Returns:
            Paragraph describing the entire video narrative
        """
        if not scene_summaries:
            return "Empty video with no scenes."

        # Build the scene summaries text
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
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                options={"temperature": 0.3, "num_predict": 150},
            )
            summary = response["response"].strip()
            return summary
        except Exception:
            # Fallback: return concatenated scene summaries
            return " ".join([summary for _, _, summary in scene_summaries])

    def summarize_scene_description(self, scene_description: SceneDescription) -> str:
        """Generate summary from a SceneDescription object.

        Args:
            scene_description: SceneDescription object with frame descriptions

        Returns:
            Coherent summary of the scene
        """
        frame_descriptions = [(fd.timestamp, fd.description) for fd in scene_description.frame_descriptions]
        return self.summarize_scene(frame_descriptions)

    def summarize_video_description(self, video_description: VideoDescription) -> str:
        """Generate summary from a VideoDescription object.

        Args:
            video_description: VideoDescription object with scene descriptions

        Returns:
            High-level summary of the entire video
        """
        scene_summaries = [
            (sd.scene.start, sd.scene.end, sd.get_description_summary()) for sd in video_description.scene_descriptions
        ]
        return self.summarize_video(scene_summaries)
