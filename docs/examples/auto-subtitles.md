# Auto-Subtitles

Automatically transcribe speech and add word-level subtitles to any video.

## Goal

Take a video with speech, transcribe the audio using AI, and overlay synchronized subtitles with word-by-word highlighting.

## Full Example

```python
from videopython.base import Video
from videopython.ai import AudioToText
from videopython.base.text import TranscriptionOverlay

def add_subtitles(input_path: str, output_path: str):
    # Load video
    video = Video.from_path(input_path)

    # Transcribe audio
    transcriber = AudioToText(backend="openai")
    transcription = transcriber.transcribe(video)

    # Apply subtitle overlay
    overlay = TranscriptionOverlay(
        font_size=48,
        font_color=(255, 255, 255),
        highlight_color=(255, 200, 0),
        position="bottom",
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
transcriber = AudioToText(backend="openai")  # Uses Whisper API
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

Available backends:

| Backend | Model | Notes |
|---------|-------|-------|
| `openai` | Whisper API | Best accuracy, requires API key |
| `gemini` | Gemini | Good accuracy, requires API key |
| `local` | Whisper | Free, runs locally, slower |

### 2. Configure Subtitle Style

```python
overlay = TranscriptionOverlay(
    font_size=48,                    # Text size in pixels
    font_color=(255, 255, 255),      # White text (RGB)
    highlight_color=(255, 200, 0),   # Yellow highlight for current word
    position="bottom",               # "top", "center", or "bottom"
    margin=100,                      # Distance from edge in pixels
)
```

### 3. Apply Overlay

```python
video = overlay.apply(video, transcription)
```

The overlay renders each word at its exact timestamp, highlighting the current word being spoken.

## Customization

### Styling Options

```python
# Minimal white subtitles
overlay = TranscriptionOverlay(
    font_size=36,
    font_color=(255, 255, 255),
    highlight_color=(255, 255, 255),  # No highlight distinction
    position="bottom",
    margin=80,
)

# Bold yellow subtitles with red highlight
overlay = TranscriptionOverlay(
    font_size=64,
    font_color=(255, 255, 0),
    highlight_color=(255, 50, 50),
    position="center",
    margin=0,
)
```

### Processing Long Videos

For long videos, transcription can take time. Consider processing in segments:

```python
from videopython.base import CutSeconds

# Process first 5 minutes
video = Video.from_path("long_video.mp4", start_second=0, end_second=300)
```

## Tips

- **Font Size**: Use 36-48px for 1080p, 48-64px for 4K. Larger is better for mobile viewing.
- **Contrast**: White text with a subtle shadow or outline works on most backgrounds.
- **Position**: "bottom" is standard, but "center" works well for short-form vertical videos.
- **Languages**: Whisper supports 90+ languages. The API auto-detects language by default.
