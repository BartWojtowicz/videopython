"""Centralized HuggingFace model-revision pins.

The AI predictors load weights via ``from_pretrained(...)``,
``hf_hub_download(...)``, and ``YOLO(...)`` with no revision pinning by
default, so a silent upstream weight change can alter production analysis
without any code change on our side. This module maps each *fixed* model
identifier to the commit SHA we want to pin to, so callers can pass
``revision=pinned(model_id)`` to lock the download to a known-good commit.

Resolution: each SHA below is the ``sha`` field returned by the HuggingFace
model API (``https://huggingface.co/api/models/<repo_id>``), i.e. the latest
commit on the repo's ``main`` branch at the time it was captured.

Refreshing SHAs:
    For each ``model_id`` in ``MODEL_REVISIONS`` (and the TODO list, if any),
    fetch ``https://huggingface.co/api/models/<model_id>`` and copy the JSON
    ``sha`` field into the dict. Bump deliberately -- the whole point of a pin
    is that it does not move on its own. After editing, re-run the AI suite
    so the new revision is exercised before release.

Usage:
    >>> from videopython.ai._revisions import pinned
    >>> pinned("facebook/musicgen-small")  # -> "<sha>" or None
    Pass the result straight through:
    ``AutoProcessor.from_pretrained(model_id, revision=pinned(model_id))``.
    ``revision=None`` is the library default and means "current behavior"
    (track ``main``), so an unpinned or dynamic model stays safe.

# ---------------------------------------------------------------------------
# Unpinnable-by-design (intentionally absent from MODEL_REVISIONS)
# ---------------------------------------------------------------------------
# These load paths cannot be pinned to a HuggingFace commit SHA, so they are
# documented here rather than registered above. ``pinned()`` returns None for
# them, which leaves the caller on its current (unpinned) behavior.
#
#   * ultralytics YOLO asset -- ai/understanding/objects.py loads
#     ``YOLO(self.model_name)`` (e.g. "yolov8n.pt"). Ultralytics resolves and
#     downloads this from its own GitHub release assets, not from a HF repo,
#     so there is no HF revision to pin. (The face detector in
#     ai/understanding/faces.py is different: it pulls a YOLOv8 *checkpoint*
#     from the HF repo ``arnabdhar/YOLOv8-Face-Detection`` via
#     hf_hub_download, and that one IS pinned below.)
#
#   * Chatterbox internal load -- ai/generation/audio.py calls
#     ``ChatterboxMultilingualTTS.from_pretrained(device=...)`` with no repo
#     argument. The repo id + revision are resolved internally by the
#     chatterbox package, so there is nothing for us to pass a ``revision`` to.
#
#   * openai-whisper CDN models -- ai/understanding/audio.py loads transcription
#     weights via ``whisper.load_model(name="turbo", ...)`` (tiny/base/small/
#     medium/large/turbo). openai-whisper downloads these from OpenAI's own CDN
#     by name, NOT through a HF ``from_pretrained`` repo, so there is no HF
#     commit SHA to pin. (The faster-whisper backend, if/when used, WOULD map to
#     a HF repo and could be pinned -- this code path does not use it.)
# ---------------------------------------------------------------------------
"""

from __future__ import annotations

# Exact model-identifier string -> pinned commit SHA. The key must match the
# literal value passed to from_pretrained/hf_hub_download at the call site so
# ``pinned(model_id)`` resolves with a plain dict lookup. SHAs captured from
# the HuggingFace model API (see module docstring for refresh instructions).
MODEL_REVISIONS: dict[str, str] = {
    # Speaker diarization (ai/understanding/audio.py: PYANNOTE_DIARIZATION_MODEL).
    # Gated repo (auto-approved); from_pretrained needs an accepted HF token.
    "pyannote/speaker-diarization-community-1": "3533c8cf8e369892e6b79ff1bf80f7b0286a54ee",
    # Audio event classifier (ai/understanding/audio.py: AudioClassifier)
    "MIT/ast-finetuned-audioset-10-10-0.4593": "f826b80d28226b62986cc218e5cec390b1096902",
    # Face detection checkpoint (ai/understanding/faces.py: hf_hub_download)
    "arnabdhar/YOLOv8-Face-Detection": "52fa54977207fa4f021de949b515fb19dcab4488",
    # MusicGen (ai/generation/audio.py: TextToMusic)
    "facebook/musicgen-small": "4c8334b02c6ec4e8664a91979669a501ec497792",
    # SDXL (ai/generation/image.py: TextToImage)
    "stabilityai/stable-diffusion-xl-base-1.0": "462165984030d82259a11f4367a4eed129e94a7b",
    # CogVideoX (ai/generation/video.py: TextToVideo / ImageToVideo)
    "THUDM/CogVideoX1.5-5B": "fdc5267c90b5c06492985b966e43aae984e189e0",
    "THUDM/CogVideoX1.5-5B-I2V": "46c90528707aebbe69066390b4fe7e7d24c9c2a4",
}


def pinned(model_id: str) -> str | None:
    """Return the pinned commit SHA for ``model_id``, or ``None`` if unpinned.

    ``None`` is a valid, safe value to forward as ``revision=`` to
    ``from_pretrained`` / ``hf_hub_download`` -- it is their default and means
    "no pin" (track the repo's ``main`` branch, i.e. the current behavior).
    Dynamic and unpinnable-by-design models (see the module docstring) fall
    into this case.

    Args:
        model_id: The exact repo id / model identifier string passed to the
            loader at the call site (e.g. ``"facebook/musicgen-small"``).

    Returns:
        The pinned SHA, or ``None`` when no pin is registered for ``model_id``.
    """
    return MODEL_REVISIONS.get(model_id)
