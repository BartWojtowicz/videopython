"""Test dubbing first 10s of cam1_1min.mp4 into English."""

from videopython.ai.dubbing import VideoDubber
from videopython.base import Video

video = Video.from_path("cam1_1min.mp4").cut(0, 10)
dubber = VideoDubber()

print(f"Video duration: {video.total_seconds:.1f}s")
print("Starting dubbing to English...")


def progress(stage: str, pct: float) -> None:
    print(f"  [{pct:5.1f}%] {stage}")


result = dubber.dub(
    video=video,
    target_lang="en",
    preserve_background=True,
    voice_clone=True,
    progress_callback=progress,
)

print()
print(f"Translated segments: {len(result.translated_segments)}")
for seg in result.translated_segments:
    print(f"  [{seg.start:.1f}s - {seg.end:.1f}s] {seg.original_text} -> {seg.translated_text}")

dubbed_video = video.add_audio(result.dubbed_audio, overlay=False)
dubbed_video.save("cam1_10s_dubbed_en.mp4")
print()
print("Saved to cam1_10s_dubbed_en.mp4")
