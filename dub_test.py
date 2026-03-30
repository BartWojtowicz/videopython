"""Test dubbing first 5s of cam1_1min.mp4 into English with diarization."""

from videopython.ai.dubbing import VideoDubber
from videopython.base import Video

video = Video.from_path("cam1_1min.mp4").cut(0, 5)
dubber = VideoDubber()

print(f"Video duration: {video.total_seconds:.1f}s")
print("Starting dubbing to English (with diarization)...")


def progress(stage: str, pct: float) -> None:
    print(f"  [{pct:5.1f}%] {stage}")


result = dubber.dub(
    video=video,
    target_lang="en",
    preserve_background=True,
    voice_clone=True,
    enable_diarization=True,
    progress_callback=progress,
)

print()
print(f"Translated segments: {len(result.translated_segments)}")
for seg in result.translated_segments:
    print(f"  [{seg.start:.1f}s - {seg.end:.1f}s] speaker={seg.speaker} | {seg.original_text} -> {seg.translated_text}")

dubbed_video = video.add_audio(result.dubbed_audio, overlay=False)
dubbed_video.save("cam1_5s_dubbed_en.mp4")
print()
print("Saved to cam1_5s_dubbed_en.mp4")
