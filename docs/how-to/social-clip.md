# Build styled excerpts and summaries

Use the functions in
[`examples/editing_recipes.py`](https://github.com/bartwojtowicz/videopython/blob/main/examples/editing_recipes.py)
to build ordinary `VideoEdit` plans. Copy the example module into your project, or
run these snippets from the repository root. Each recipe takes caller-owned media
and selected source ranges. It does not run models or download assets.

## Caption an interview excerpt

Prepare a source-timed `Transcription` as shown in the
[subtitle tutorial](../tutorials/subtitles.md), or load one you saved earlier:

```python
from pathlib import Path

from examples.editing_recipes import captioned_interview
from videopython.base import Transcription

transcription = Transcription.model_validate_json(Path("transcription.json").read_text())
edit = captioned_interview(
    Path("interview.mp4"), 30.0, 45.0,
    width=1080, height=1920,
    font_scale=0.055, margin=0.08,
)
context = {"transcription": transcription}
edit.validate(context=context)
edit.run_to_file("captioned.mp4", context=context)
```

The runner maps the source timestamps to the excerpt. Supply the full source
transcription; do not subtract the cut start yourself.

## Add a logo and title

```python
from pathlib import Path

from examples.editing_recipes import branded_excerpt

edit = branded_excerpt(
    Path("interview.mp4"), 30.0, 45.0,
    logo=Path("logo.png"), title="Nowe pomysły na kolejny rok",
    width=1920, height=1080,
    margin=0.08, logo_width=0.15, font_size=48,
)
edit.validate()
edit.run_to_file("branded.mp4")
```

The logo sits at the top left. The title wraps above the bottom margin. Use a
shorter title or smaller font if the rendered text covers the subject.

## Assemble a summary with ducked music

This example uses two explicit passes because music ducking accepts a single
segment. The first pass assembles cuts. The second adds music to that rendered
file, using a transcription mapped to the cuts.

Choose frame-aligned ranges from one source. Keep this recipe to cuts and framing;
transitions, speed changes, and cuts from sources with different frame rates need
a different transcript mapping.

```python
from pathlib import Path

from examples.editing_recipes import ducked_music, summary_cuts, summary_transcription
from videopython.base import Transcription

source = Path("interview.mp4")
transcription = Transcription.model_validate_json(Path("transcription.json").read_text())
ranges = [(30.0, 40.0), (60.0, 75.0), (15.0, 20.0)]

cuts = summary_cuts(source, ranges, width=1080, height=1920)
cuts.validate()
assembled = cuts.run_to_file("summary-cuts.mp4")

mapped = summary_transcription(transcription, ranges)
edit = ducked_music(assembled, Path("music.wav"), gain=0.2, duck=0.8)
context = {"transcription": mapped}
edit.validate(context=context)
edit.run_to_file("summary.mp4", context=context)
```

Supply actual timed words for ducking. The mapping preserves cut order, including
repeated ranges, and clips words at selected boundaries. It does not infer missing
speech timing. Select complete words when choosing the cuts.

Keep `summary-cuts.mp4` until the second pass finishes. This approach needs an
intermediate file and a second encode. Choose source and bed levels that leave
headroom in the mix. For bed gain, looping, and attack/release
behavior, see [MusicBed](../reference/video-edit.md#musicbed).

## Choose framing and style

All three recipes resize with the source aspect ratio, then center-crop to the
requested dimensions. Use positive, even output dimensions. Center cropping can
remove an off-center subject; inspect each selected range. Face tracking is an
alternative described in [AI operations](../reference/ai/operations.md), but face
selection does not identify the active speaker.

`margin` is a fraction of output width and height, between zero and one half.
`logo_width` is a fraction of output width; logo height follows its aspect ratio.
`font_size` is in output pixels. Caption styles and fonts are described once in the
[subtitle tutorial](../tutorials/subtitles.md#step-4-restyle).

The functions return serializable plans:

```python
import json
from videopython.editing import VideoEdit

saved = json.dumps(edit.to_dict())
restored = VideoEdit.from_dict(json.loads(saved))
```

Transcriptions remain separate render context. Inspect landscape and portrait
outputs for text fit and subject framing, and listen to the music balance before
publishing. The local [recipe verification](../reference/verification.md#editing-recipes)
records the tested compositions and limits.
