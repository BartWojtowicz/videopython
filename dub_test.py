"""Test dubbing cam1_1min.mp4 (first 10s) -- diagnose timing issues."""

from videopython.ai.generation._qwen3_tts_compat import apply_patches

apply_patches()

from videopython.ai.dubbing.pipeline import LocalDubbingPipeline
from videopython.ai.dubbing.timing import TimingSynchronizer
from videopython.base import Video

video = Video.from_path("cam1_1min.mp4").cut(0, 10)
print(f"Video duration: {video.total_seconds:.1f}s")

pipeline = LocalDubbingPipeline()

# Transcribe
pipeline._init_transcriber()
source_audio = video.audio
transcription = pipeline._transcriber.transcribe(source_audio)
print(f"Detected language: {transcription.language}")
print(f"Segments: {len(transcription.segments)}")

# Separate
pipeline._init_separator()
separated = pipeline._separator.separate(source_audio)

# Voice samples
voice_samples = pipeline._extract_voice_samples(separated.vocals, transcription)

# Translate
pipeline._init_translator()
translated_segments = pipeline._translator.translate_segments(
    segments=transcription.segments, target_lang="en", source_lang=transcription.language
)

# Generate TTS and measure durations
from videopython.ai.generation.audio import TextToSpeech

tts = TextToSpeech(model_size="qwen3", device=None, language="en")

print("\n--- TTS Generation vs Target Durations ---")
dubbed_segments = []
target_durations = []
start_times = []

for seg in translated_segments:
    speaker = seg.speaker or "speaker_0"
    voice_sample = voice_samples.get(speaker)
    dubbed = tts.generate_audio(seg.translated_text, voice_sample=voice_sample)
    target_dur = seg.duration
    gen_dur = dubbed.metadata.duration_seconds
    ratio = gen_dur / target_dur if target_dur > 0 else 0
    print(
        f"  [{seg.start:.1f}-{seg.end:.1f}s] target={target_dur:.1f}s  generated={gen_dur:.1f}s  "
        f"ratio={ratio:.2f}x  sr={dubbed.metadata.sample_rate}"
    )
    dubbed_segments.append(dubbed)
    target_durations.append(target_dur)
    start_times.append(seg.start)

# Synchronize
sync = TimingSynchronizer()
synced_segments, adjustments = sync.synchronize_segments(dubbed_segments, target_durations)

print("\n--- Timing Adjustments ---")
for adj in adjustments:
    print(
        f"  seg {adj.segment_index}: orig={adj.original_duration:.1f}s -> target={adj.target_duration:.1f}s  "
        f"actual={adj.actual_duration:.1f}s  speed={adj.speed_factor:.2f}x  truncated={adj.was_truncated}"
    )

# Assemble
total_duration = source_audio.metadata.duration_seconds
dubbed_speech = sync.assemble_with_timing(synced_segments, start_times, total_duration)

background_sr = separated.background.metadata.sample_rate
if dubbed_speech.metadata.sample_rate != background_sr:
    print(f"\nResampling dubbed speech from {dubbed_speech.metadata.sample_rate} to {background_sr}")
    dubbed_speech = dubbed_speech.resample(background_sr)

final_audio = separated.background.overlay(dubbed_speech, position=0.0)

dubbed_video = video.add_audio(final_audio, overlay=False)
dubbed_video.save("cam1_10s_dubbed_en.mp4")
print(f"\nSaved to cam1_10s_dubbed_en.mp4")
