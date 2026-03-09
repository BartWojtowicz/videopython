# Auto-Subtitles

Automatically transcribe speech and add word-level subtitles to any video.

## Goal

Take a video with speech, transcribe the audio using AI, and overlay synchronized subtitles with word-by-word highlighting.

## Full Example

```python
from videopython import Video
from videopython.ai import AudioToText
from videopython.base import TranscriptionOverlay

def add_subtitles(input_path: str, output_path: str):
    # Load video
    video = Video.from_path(input_path)

    # Transcribe audio
    transcriber = AudioToText()
    transcription = transcriber.transcribe(video)

    # Apply subtitle overlay
    overlay = TranscriptionOverlay(
        font_filename="/path/to/font.ttf",
        font_size=48,
        text_color=(255, 255, 255),
        highlight_color=(255, 200, 0),
        position=(0.5, 0.85),  # centered, near bottom
        margin=100,
    )
    video = overlay.apply(video, transcription)

    # Save with burned-in subtitles
    video.save(output_path)

add_subtitles("interview.mp4", "interview_subtitled.mp4")
```

## Step-by-Step Breakdown

### 1. Transcribe Audio

```python
transcriber = AudioToText()  # Local Whisper model
transcription = transcriber.transcribe(video)
```

The transcription includes word-level timestamps:

```python
# transcription structure:
# [
#     {"word": "Hello", "start": 0.0, "end": 0.5},
#     {"word": "world", "start": 0.6, "end": 1.0},
#     ...
# ]
```

Model options:

- `tiny`, `base`, `small`, `medium`, `large`, `turbo`
- Enable diarization with `AudioToText(enable_diarization=True)` when needed.

### 2. Configure Subtitle Style

```python
overlay = TranscriptionOverlay(
    font_filename="/path/to/font.ttf",  # Required: path to a .ttf font file
    font_size=48,                        # Text size in pixels
    text_color=(255, 255, 255),          # White text (RGB)
    highlight_color=(255, 200, 0),       # Yellow highlight for current word
    position=(0.5, 0.85),               # Relative position (x, y) -- 0.0 to 1.0
    box_width=0.6,                       # Text box width as fraction of video width
    margin=100,                          # Distance from edge in pixels
)
```

Key parameters:

- `font_filename` -- **Required.** Path to a TrueType font file (`.ttf`).
- `position` -- Tuple of `(x, y)` as relative (0.0-1.0) or absolute pixel values. `(0.5, 0.85)` places text centered, near the bottom.
- `text_color` / `highlight_color` -- RGB tuples for normal and highlighted words.
- `box_width` -- Width of the subtitle box (relative or absolute).
- `background_color` -- RGBA tuple for background behind text, or `None` for no background. Default: `(0, 0, 0, 100)`.

### 3. Apply Overlay

```python
video = overlay.apply(video, transcription)
```

The overlay renders each word at its exact timestamp, highlighting the current word being spoken.

## Customization

### Styling Options

```python
# Minimal white subtitles near the bottom
overlay = TranscriptionOverlay(
    font_filename="/path/to/font.ttf",
    font_size=36,
    text_color=(255, 255, 255),
    highlight_color=(255, 255, 255),  # No highlight distinction
    position=(0.5, 0.85),
    margin=80,
)

# Bold yellow subtitles centered on screen
overlay = TranscriptionOverlay(
    font_filename="/path/to/font.ttf",
    font_size=64,
    text_color=(255, 255, 0),
    highlight_color=(255, 50, 50),
    position=(0.5, 0.5),
    margin=0,
)
```

### Processing Long Videos

For long videos, transcription can take time. Consider processing in segments:

```python
from videopython import Video

# Process first 5 minutes
video = Video.from_path("long_video.mp4", start_second=0, end_second=300)
```

## Tips

- **Font Size**: Use 36-48px for 1080p, 48-64px for 4K. Larger is better for mobile viewing.
- **Contrast**: White text on a semi-transparent background works on most backgrounds (the default `background_color` handles this).
- **Position**: `(0.5, 0.85)` is standard for bottom subtitles. `(0.5, 0.5)` works for short-form vertical videos.
- **Languages**: Whisper supports 90+ languages. The API auto-detects language by default.
