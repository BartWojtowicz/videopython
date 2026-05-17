"""Subtitle fit safety: the validate/run gap and its preconditions.

Covers the layered design in TODO.md:

* Step 0 -- ``face_crop`` predicts real (cropped) dimensions, so the dry-run
  measures subtitles against the frame they will actually render on.
* Step 3 -- geometry is resolution-relative by default (``font_scale``).
* Step 4 -- a near-miss auto-shrinks instead of crashing.
* Step 2 -- an un-fittable plan fails fast in ``validate()``, not mid-render,
  and ``predict_metadata``/``apply`` never disagree (parity).
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.test_config import SMALL_VIDEO_METADATA, SMALL_VIDEO_PATH
from videopython.base.transcription import Transcription, TranscriptionSegment, TranscriptionWord
from videopython.base.video import Video, VideoMetadata
from videopython.editing.transcription_overlay import (
    SubtitleRegion,
    SubtitleStyle,
    TranscriptionOverlay,
)
from videopython.editing.video_edit import VideoEdit


def _tx(words: list[tuple[float, float, str]]) -> Transcription:
    w = [TranscriptionWord(start=s, end=e, word=t) for s, e, t in words]
    return Transcription(segments=[TranscriptionSegment.from_words(w)])


_WORDS = [(0.0, 0.4, "Hello"), (0.4, 0.8, "there"), (0.8, 1.2, "this"), (1.2, 1.6, "really"), (1.6, 2.0, "works")]


class TestStylePresetsBackCompat:
    def test_boxed_preset_reproduces_legacy_defaults(self):
        """Upgrading without touching fields is a visual no-op except sizing."""
        sp = TranscriptionOverlay(style=SubtitleStyle.BOXED)._style_params()
        assert sp.text_color == (255, 235, 59)
        assert sp.highlight_color == (76, 175, 80)
        assert sp.border == 2
        assert sp.background_color == (0, 0, 0, 100)
        assert sp.background_padding == 15
        assert sp.highlight_size_multiplier == 1.2

    def test_explicit_overrides_beat_preset(self):
        op = TranscriptionOverlay(style=SubtitleStyle.OUTLINE, text_color=(1, 2, 3), background_color=(9, 9, 9, 9))
        sp = op._style_params()
        assert sp.text_color == (1, 2, 3)
        assert sp.background_color == (9, 9, 9, 9)
        assert sp.border == 4  # still from the OUTLINE preset

    def test_background_none_explicitly_disables(self):
        sp = TranscriptionOverlay(background_color=None)._style_params()
        assert sp.background_color is None

    def test_region_maps_to_position(self):
        assert TranscriptionOverlay(region=SubtitleRegion.TOP)._resolve_config().position == (0.5, 0.18)
        assert TranscriptionOverlay(region=SubtitleRegion.BOTTOM)._resolve_config().position == (0.5, 0.82)
        assert TranscriptionOverlay(position=(0.1, 0.2))._resolve_config().position == (0.1, 0.2)


class TestResolutionIndependence:
    def test_font_scales_with_frame_height(self):
        op = TranscriptionOverlay()  # font_scale=0.055
        t = _tx(_WORDS)
        assert op._resolve_layout(1920, 1080, t).font_px == round(0.055 * 1080)
        assert op._resolve_layout(640, 360, t).font_px == round(0.055 * 360)

    def test_explicit_font_size_overrides_scale(self):
        op = TranscriptionOverlay(font_size=24)
        assert op._resolve_layout(1920, 1080, _tx(_WORDS)).font_px == 24


class TestGracefulAutoFit:
    def test_oversized_font_shrinks_instead_of_raising(self):
        """A near-miss auto-fits within the legible band (Step 4)."""
        op = TranscriptionOverlay(font_size=400)
        layout = op._resolve_layout(640, 360, _tx(_WORDS))
        assert layout.fits
        assert layout.font_px < 400
        assert layout.font_px >= round(op.min_font_scale * 360)

    def test_unfittable_reports_offending_cue(self):
        """Only an un-fittable-at-min case errors, and it names the cue."""
        op = TranscriptionOverlay(font_size=300, min_font_scale=0.5)
        layout = op._resolve_layout(160, 120, _tx(_WORDS))
        assert layout.fits is False
        assert "cannot fit" in (layout.error or "")
        assert "Hello there this really works" in (layout.error or "")


class TestPredictApplyParity:
    @pytest.mark.parametrize("w, h", [(1920, 1080), (640, 360), (320, 240), (606, 1080)])
    def test_predict_says_fits_then_apply_does_not_raise(self, w, h):
        op = TranscriptionOverlay()
        t = _tx(_WORDS)
        meta = VideoMetadata(height=h, width=w, fps=5, frame_count=10, total_seconds=2.0)

        op.predict_metadata(meta, transcription=t)  # must not raise

        layout = op._resolve_layout(w, h, t)
        video = Video.from_frames(np.zeros((10, h, w, 3), dtype=np.uint8), fps=5)
        result = op.apply(video, transcription=t)  # must not raise (parity)

        assert result.video_shape == video.video_shape
        assert layout.fits
        assert (result.frames != 0).any(), "subtitles were not drawn"

    def test_predict_metadata_is_identity_when_it_fits(self):
        op = TranscriptionOverlay()
        meta = VideoMetadata(height=1080, width=1920, fps=24, frame_count=48, total_seconds=2.0)
        assert op.predict_metadata(meta, transcription=_tx(_WORDS)) is meta

    def test_predict_metadata_no_transcription_is_identity(self):
        """Mirrors SilenceRemoval: no context -> no-op identity, not an error."""
        op = TranscriptionOverlay(font_size=9999)
        meta = VideoMetadata(height=10, width=10, fps=24, frame_count=48, total_seconds=2.0)
        assert op.predict_metadata(meta) is meta


class TestValidateRunGapClosed:
    """End-to-end through ``VideoEdit.validate`` (Step 2).

    Uses the core ``crop`` transform so the editing suite stays free of the
    optional ``[ai]`` extra; the ``face_crop``-specific Step 0 + Step 2
    integration lives in ``tests/ai/test_transforms.py``.
    """

    def test_overflow_after_upstream_crop_fails_in_validate_with_predicted_dims(self):
        """An upstream transform shrinks the frame; subtitles that no longer
        fit are rejected before any frame/GPU work, and the error names the
        *predicted* (post-crop) frame -- proving the check sees the geometry
        the render will actually use, not the source dimensions."""
        plan = {
            "segments": [
                {
                    "source": "fake.mp4",
                    "start": 0.0,
                    "end": 2.0,
                    "operations": [
                        {"op": "crop", "width": 0.2, "height": 0.2},
                        {"op": "add_subtitles", "font_size": 200, "min_font_scale": 0.5},
                    ],
                }
            ]
        }
        source = VideoMetadata(height=1080, width=1920, fps=30, frame_count=60, total_seconds=2.0)
        with pytest.raises(ValueError, match="add_subtitles.*cannot fit") as exc:
            VideoEdit.from_dict(plan).validate_with_metadata(source, context={"transcription": _tx(_WORDS)})
        # crop 0.2 of 1920x1080 -> 384x216, NOT the 1920x1080 source.
        assert "384x216 frame" in str(exc.value)

    def test_reasonable_subtitles_pass_validate(self):
        plan = {
            "segments": [
                {
                    "source": "fake.mp4",
                    "start": 0.0,
                    "end": 2.0,
                    "operations": [{"op": "add_subtitles"}],
                }
            ]
        }
        source = VideoMetadata(height=1080, width=1920, fps=30, frame_count=60, total_seconds=2.0)
        out = VideoEdit.from_dict(plan).validate_with_metadata(source, context={"transcription": _tx(_WORDS)})
        assert (out.width, out.height) == (1920, 1080)

    def test_run_end_to_end_auto_fits(self):
        """Real decode path: a sane plan renders without raising."""
        plan = {
            "segments": [
                {
                    "source": SMALL_VIDEO_PATH,
                    "start": 0.0,
                    "end": 2.0,
                    "operations": [{"op": "add_subtitles"}],
                }
            ]
        }
        edit = VideoEdit.from_dict(plan)
        context = {"transcription": _tx(_WORDS)}
        edit.validate_with_metadata(SMALL_VIDEO_METADATA, context=context)
        result = edit.run(context=context)
        assert result.video_shape[1:3] == (SMALL_VIDEO_METADATA.height, SMALL_VIDEO_METADATA.width)
