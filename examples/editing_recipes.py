from pathlib import Path

from videopython.base import Transcription, VideoMetadata
from videopython.editing import (
    AnchorPoint,
    Crop,
    ImageOverlay,
    Operation,
    Resize,
    SegmentConfig,
    SubtitleStyle,
    TextOverlay,
    TranscriptionOverlay,
    VideoEdit,
)
from videopython.editing.audio_ops import MusicBed


def _framing(source: Path, width: int, height: int) -> list[Operation]:
    metadata = VideoMetadata.from_path(source)
    resize = Resize(height=height) if metadata.width / metadata.height >= width / height else Resize(width=width)
    return [resize, Crop(width=width, height=height, mode="center")]


def captioned_interview(
    source: Path,
    start: float,
    end: float,
    *,
    width: int,
    height: int,
    style: SubtitleStyle = SubtitleStyle.BOXED,
    font_scale: float = 0.055,
    margin: float = 0.08,
) -> VideoEdit:
    """Build a centered excerpt; supply source-timed transcription when rendering."""
    return VideoEdit(
        segments=[
            SegmentConfig(
                source=source,
                start=start,
                end=end,
                operations=[
                    *_framing(source, width, height),
                    TranscriptionOverlay(
                        style=style,
                        font_scale=font_scale,
                        position=(0.5, 1 - margin),
                        anchor=AnchorPoint.BOTTOM_CENTER,
                        box_width=1 - 2 * margin,
                    ),
                ],
            )
        ]
    )


def branded_excerpt(
    source: Path,
    start: float,
    end: float,
    *,
    logo: Path,
    title: str,
    width: int,
    height: int,
    margin: float = 0.08,
    logo_width: float = 0.15,
    font_size: int = 32,
    text_color: tuple[int, int, int] = (255, 255, 255),
) -> VideoEdit:
    """Build a centered excerpt with a top-left logo and a wrapped bottom title."""
    return VideoEdit(
        segments=[
            SegmentConfig(
                source=source,
                start=start,
                end=end,
                operations=[
                    *_framing(source, width, height),
                    ImageOverlay(source=logo, scale=logo_width, position=(margin, margin), anchor="top_left"),
                    TextOverlay(
                        text=title,
                        font_size=font_size,
                        text_color=text_color,
                        position=(0.5, 1 - margin),
                        anchor="bottom_center",
                        max_width=1 - 2 * margin,
                    ),
                ],
            )
        ]
    )


def summary_cuts(source: Path, ranges: list[tuple[float, float]], *, width: int, height: int) -> VideoEdit:
    """Assemble frame-aligned ranges from one source, without transitions or retiming."""
    return VideoEdit(
        segments=[
            SegmentConfig(source=source, start=start, end=end, operations=_framing(source, width, height))
            for start, end in ranges
        ]
    )


def summary_transcription(transcription: Transcription, ranges: list[tuple[float, float]]) -> Transcription:
    """Map timed words to the same ordered cuts used by summary_cuts."""
    words = []
    offset = 0.0
    for start, end in ranges:
        for word in transcription.words:
            if word.end > start and word.start < end:
                words.append(
                    word.model_copy(
                        update={
                            "start": offset + max(word.start, start) - start,
                            "end": offset + min(word.end, end) - start,
                        }
                    )
                )
        offset += end - start
    return Transcription(words=words, language=transcription.language)


def ducked_music(assembled: Path, music: Path, *, gain: float = 0.2, duck: float = 0.8) -> VideoEdit:
    """Add a bed to rendered cuts; supply their mapped transcription when rendering."""
    metadata = VideoMetadata.from_path(assembled)
    return VideoEdit(
        segments=[SegmentConfig(source=assembled, start=0, end=metadata.total_seconds)],
        music_bed=MusicBed(source=music, gain=gain, duck=duck),
    )
