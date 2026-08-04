# Architecture

videopython is four subpackages in a strict dependency order. The order is the point: it
is what lets a video-editing install stay free of PyTorch.

```
videopython.base       Video, VideoMetadata, FrameIterator, Transcription,
                       shared result types.            No AI imports.
        ↑
videopython.audio      Audio container and analysis.   Depends on base.
        ↑
videopython.editing    Operation/Effect foundation,    Depends on base, audio.
                       the VideoEdit plan runner.
        ↑
videopython.ai         Generation, understanding,      Depends on base, audio,
                       dubbing, AI operations.         optionally editing.
                                                       Needs the [ai] extra.
```

`videopython.mcp` sits alongside as a thin server over `ai` + `editing`, behind the
`[mcp]` extra.

## Why the layering is enforced, not just intended

Only `videopython.ai` may import ML dependencies. If that invariant erodes — a single
`import torch` inside `editing/` — then `pip install videopython` starts dragging a
multi-gigabyte CUDA stack behind it for users who only wanted to crop a video.

The invariant is not left to code review: `src/tests/test_import_isolation.py` fails the
build if `base`, `audio` or `editing` gain an AI import.

## Why `import videopython` is fast even with `[ai]` installed

`[ai]` is a single extra covering every AI capability, so installing it pulls in torch,
transformers, diffusers and chatterbox. Importing all of that eagerly would add seconds to
every process start, including ones that never touch a model.

So `videopython/ai/__init__.py` has no top-level imports of its submodules. Every public
symbol is re-exported lazily through PEP 562 `__getattr__`, mapped to the one leaf module
that defines it. `from videopython.ai import AudioToText` imports the understanding leaf
and nothing else — not diffusers, not chatterbox.

Two consequences worth knowing:

- **Import errors arrive at first use, not at import.** Without `[ai]` installed,
  `import videopython.ai` succeeds and touching a symbol raises an `ImportError` that
  names the extra to install. That is deliberate: a missing optional dependency should not
  break the whole package import.
- **AI operations register lazily.** `face_crop` and `object_detection_overlay` only
  appear in the operation registry (and therefore the LLM schema) after `videopython.ai`
  has been imported. See [LLM-first design](llm-first-design.md).

## Why AI effects live in `ai`, not `editing`

`ObjectDetectionOverlay` is an effect in every structural sense — shape-preserving,
per-frame, windowed. It nonetheless lives in `videopython.ai.effects`, because it runs a
detection model per frame and the layering above admits no exceptions. The drawing half
of it is a pure, AI-free function (`videopython.base.draw_detections`) that anything can
reuse with any list of `DetectedObject`.

The same split explains `FaceTrackingCrop`: the crop transform is in
`videopython.ai.transforms`, while the geometry types it produces
(`BoundingBox`, `FaceTrack`) are ordinary `base` result types with no AI dependency.

## Where the repository puts things

```
src/
├── stubs        # mypy stubs for untyped third-party packages
├── tests        # mirrors the package tree
└── videopython  # library code
```

Contributor workflow, test commands and the release process are in
[`DEVELOPMENT.md`](https://github.com/bartwojtowicz/videopython/blob/main/DEVELOPMENT.md).
