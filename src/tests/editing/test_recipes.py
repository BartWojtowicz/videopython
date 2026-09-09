import runpy
from pathlib import Path

import numpy as np

from videopython.audio import Audio, AudioMetadata
from videopython.base import Transcription, TranscriptionWord, Video
from videopython.base.video import extract_frames_at_times
from videopython.editing import VideoEdit


def test_two_pass_summary_reorders_cuts_and_ducks_mapped_speech(tmp_path):
    recipes = runpy.run_path(str(Path(__file__).parents[3] / "examples" / "editing_recipes.py"))
    frames = np.zeros((40, 48, 64, 3), dtype=np.uint8)
    frames[:20, :, :, 0] = 255
    frames[20:, :, :, 2] = 255
    source = Video(frames, fps=10).save(tmp_path / "source.mp4")
    rate = 16000
    tone = (0.25 * np.sin(2 * np.pi * 440 * np.arange(rate * 4) / rate)).astype(np.float32)
    music = tmp_path / "music.wav"
    Audio(
        tone,
        AudioMetadata(sample_rate=rate, channels=1, sample_width=2, duration_seconds=4, frame_count=len(tone)),
    ).save(music)
    transcription = Transcription(
        words=[
            TranscriptionWord(word="first", start=0.6, end=1.0),
            TranscriptionWord(word="second", start=2.6, end=3.0),
        ],
        language="en",
    )
    ranges = [(2.0, 4.0), (0.0, 2.0)]
    cuts = recipes["summary_cuts"](source, ranges, width=64, height=48)
    assembled = VideoEdit.from_dict(cuts.to_dict()).run_to_file(tmp_path / "cuts.mp4", preset="ultrafast")
    mapped = recipes["summary_transcription"](transcription, ranges)
    assert [word.word for word in mapped.words] == ["second", "first"]
    np.testing.assert_allclose([word.start for word in mapped.words], [0.6, 2.6])
    assert [word.start for word in transcription.words] == [0.6, 2.6]
    edit = recipes["ducked_music"](assembled, music, gain=0.5, duck=0.8)
    output = VideoEdit.from_dict(edit.to_dict()).run_to_file(
        tmp_path / "summary.mp4", context={"transcription": mapped}, preset="ultrafast"
    )
    selected = extract_frames_at_times(output, [0.2, 2.2])
    assert selected[0].mean(axis=(0, 1)).argmax() == 2
    assert selected[1].mean(axis=(0, 1)).argmax() == 0
    samples = Audio.from_path(output, sample_rate=rate, channels=1).data
    bed_rms = np.sqrt(np.mean(samples[int(0.05 * rate) : int(0.15 * rate)] ** 2))
    speech_rms = np.sqrt(np.mean(samples[int(0.7 * rate) : int(0.9 * rate)] ** 2))
    assert bed_rms > 0.03
    assert 0 < speech_rms < bed_rms * 0.4
