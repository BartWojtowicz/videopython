#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

_DESCRIPTION = "Run semantic checks against the real local AI models."

# --------------------------------------------------------------------------- #
# Harness
# --------------------------------------------------------------------------- #


@dataclass
class Context:
    """Everything a check may need from the invocation."""

    video: Path | None
    workdir: Path
    device: str | None
    ollama_model: str | None
    source_lang: str
    target_lang: str
    enable_diarization: bool


@dataclass
class Outcome:
    """What a check concluded.

    ``detail`` is one line for the report -- state the *evidence*, not "ok".
    ``measurements`` are values worth tracking across releases even when the
    check passes (a truncation rate, a correlation), rendered under the table.
    """

    passed: bool
    detail: str
    measurements: dict[str, Any] = field(default_factory=dict)


@dataclass
class Check:
    name: str
    summary: str
    fn: Callable[[Context], Outcome]
    needs_video: bool = False
    needs_cuda: bool = False

    def unmet_requirement(self, ctx: Context) -> str | None:
        """Why this check cannot run here, or ``None`` if it can.

        Unmet requirements SKIP rather than FAIL: a missing GPU or an unsupplied
        clip says nothing about whether the code works, and a run that reports
        failures for them trains you to ignore the report.
        """
        if self.needs_video and ctx.video is None:
            return "no --video given"
        if self.needs_cuda and not _cuda_available():
            return "no CUDA device"
        return None


_CHECKS: dict[str, Check] = {}


def _cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def check(name: str, summary: str, *, needs_video: bool = False, needs_cuda: bool = False) -> Callable[..., Any]:
    def register(fn: Callable[[Context], Outcome]) -> Callable[[Context], Outcome]:
        _CHECKS[name] = Check(name=name, summary=summary, fn=fn, needs_video=needs_video, needs_cuda=needs_cuda)
        return fn

    return register


def free_weights() -> None:
    """Drop model weights and empty the CUDA cache between checks.

    A verification run loads several multi-billion-parameter models in one
    process; without this the second one OOMs on anything short of a datacentre
    card. Cheap no-op when torch is absent.
    """
    import gc

    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #


@check("env", "Environment: torch, CUDA, key package versions")
def check_env(ctx: Context) -> Outcome:
    """Record what we are actually running against.

    Informational by design: it is the provenance line for the report, so that a
    later "it passed last time" is comparable. It does not fail on a CPU box --
    checks that genuinely need a GPU declare ``needs_cuda`` and skip instead.
    """
    import importlib.metadata as md

    import torch

    versions = {}
    for dist in ("torch", "diffusers", "transformers", "safetensors", "ollama"):
        try:
            versions[dist] = md.version(dist)
        except md.PackageNotFoundError:
            versions[dist] = "absent"

    cuda = torch.cuda.is_available()
    if cuda:
        versions["gpu"] = torch.cuda.get_device_name(0)
        major, minor = torch.cuda.get_device_capability(0)
        versions["compute_capability"] = f"{major}.{minor}"

    return Outcome(
        passed=True,
        detail=f"torch {versions['torch']}, " + (versions.get("gpu", "no CUDA device (GPU checks will skip)")),
        measurements=versions,
    )


@check("imports", "Every public AI entrypoint resolves against real deps")
def check_imports(ctx: Context) -> Outcome:
    """Touch every lazily-exported symbol so a moved upstream API surfaces.

    This is the one check that covers what CI gave up when the AI test job was
    folded into the base job: nothing in CI imports ``[ai]`` against real
    dependency versions any more, so a ``transformers`` rename reaches users
    silently. PEP 562 laziness means a plain ``import videopython.ai`` proves
    nothing -- each attribute has to be touched.
    """
    import importlib

    failures: list[str] = []
    total = 0
    for mod_name in ("videopython.ai", "videopython.ai.dubbing"):
        mod = importlib.import_module(mod_name)
        for symbol in getattr(mod, "__all__", []):
            total += 1
            try:
                getattr(mod, symbol)
            except Exception as exc:
                failures.append(f"{mod_name}.{symbol}: {type(exc).__name__}: {exc}")

    return Outcome(
        passed=not failures,
        detail=(f"{total - len(failures)}/{total} entrypoints resolve" + ("" if not failures else f"; {failures[0]}")),
        measurements={"entrypoints": total, "failures": failures},
    )


@check("ollama", "Live Ollama returns usable JSON under a schema")
def check_ollama(ctx: Context) -> Outcome:
    """Guards the 0.53.0 class of bug: a live daemon returning empty content.

    The unit tests pin that ``think=False`` is *sent* to a thinking model. They
    cannot show that the response is usable, because the fake client returns
    whatever the test supplied. Only a real daemon can.

    Semantic, not liveness: the call must produce a dict that satisfies the
    requested schema AND carries the value we asked for. An empty ``{}`` parses
    as JSON and would sail past a liveness check -- that is exactly what shipped
    broken.
    """
    from videopython.ai._ollama import OllamaStructuredClient
    from videopython.ai.dubbing.translation import DEFAULT_TRANSLATION_MODEL

    model = ctx.ollama_model or DEFAULT_TRANSLATION_MODEL
    client = OllamaStructuredClient(model=model)
    schema = {
        "type": "object",
        "properties": {"translation": {"type": "string"}},
        "required": ["translation"],
    }
    data = client.generate_json(
        system="Translate the user's text to Spanish. Reply only with the JSON object.",
        text="The cat sits on the mat.",
        schema=schema,
    )

    translation = str(data.get("translation", "")).strip()
    if not translation:
        return Outcome(
            passed=False,
            detail=f"{model} returned empty content -- the 0.53.0 reasoning-model bug is back",
            measurements={"model": model, "raw": data},
        )
    # A reasoning model that leaks its chain-of-thought produces something far
    # longer than the sentence; a hard cap catches that without pinning wording.
    reasonable = len(translation) < 200
    return Outcome(
        passed=reasonable,
        detail=(
            f"{model} -> {translation!r}" if reasonable else f"{model} returned {len(translation)} chars (leaking?)"
        ),
        measurements={"model": model, "translation": translation},
    )


def _grey(frame: Any) -> Any:
    """Frame or PIL image -> flat float32 luma vector, for correlation."""
    import numpy as np

    arr = np.asarray(frame, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[-1] >= 3:
        arr = arr[..., :3].mean(axis=-1)
    return arr.reshape(-1)


def _corr(a: Any, b: Any) -> float:
    """Pearson correlation of two equal-length signals; 0.0 if either is flat.

    The workhorse assertion of this file. Two outputs that *should* differ
    (same seed, different prompt) correlating at ~1.0 means the conditioning
    never reached the model -- a dead text encoder produces beautiful,
    confidently wrong output that every liveness check waves through.
    """
    import numpy as np

    x, y = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    n = min(len(x), len(y))
    x, y = x[:n] - x[:n].mean(), y[:n] - y[:n].mean()
    denom = float(np.sqrt((x * x).sum() * (y * y).sum()))
    return 0.0 if denom == 0 else float((x * y).sum() / denom)


def _mono_audio(samples: Any, sample_rate: int) -> Any:
    """Wrap a float32 mono array as an ``Audio`` with consistent metadata."""
    from videopython.audio.audio import Audio, AudioMetadata

    return Audio(
        data=samples,
        metadata=AudioMetadata(
            sample_rate=sample_rate,
            channels=1,
            sample_width=2,
            duration_seconds=len(samples) / sample_rate,
            frame_count=len(samples),
        ),
    )


def _transcript_text(transcription: Any) -> str:
    """Flatten a ``Transcription`` to plain text.

    It has no ``.text``; the text lives on ``.segments[].text``. Falling back to
    ``str()`` yields the object repr, which silently scores 0% word overlap and
    looks exactly like a model that produced gibberish.
    """
    segments = getattr(transcription, "segments", None)
    if segments:
        return " ".join(s.text for s in segments).strip()
    words = getattr(transcription, "words", None)
    if words:
        return " ".join(w.word for w in words).strip()
    return ""


def _envelope(audio: Any, buckets: int = 200) -> Any:
    """Coarse RMS envelope of an Audio, for cross-stem correlation."""
    import numpy as np

    data = np.asarray(audio.data, dtype=np.float32)
    if data.ndim > 1:
        data = data.mean(axis=1)
    usable = (len(data) // buckets) * buckets
    if usable == 0:
        return np.zeros(buckets, dtype=np.float32)
    return np.sqrt((data[:usable].reshape(buckets, -1) ** 2).mean(axis=1))


def _motion_fraction(first: Any, last: Any) -> float:
    """Fraction of pixels that visibly changed between two frames.

    Deliberately NOT a global correlation. Pearson correlation is dominated by
    a scene's fixed composition, so a locked-camera shot scores ~0.997 however
    much the mist drifts -- a real Wan2.2 clip measured 0.0033 divergence while
    75% of its pixels were moving, and a correlation threshold called that
    static. Counting changed pixels separates real motion (0.75) from a genuinely
    repeated frame (0.000) with three orders of magnitude to spare.
    """
    import numpy as np

    a = np.asarray(first, dtype=np.float32)
    b = np.asarray(last, dtype=np.float32)
    return float((np.abs(a - b).max(axis=-1) > 2).mean())


def _save_frames(video: Any, out: Path, stem: str) -> list[str]:
    """Write first/middle/last frames as PNGs so a human can look at them.

    Every generative check writes artifacts: the numbers below can only say
    "something changed", never "it drew what was asked for". That judgement
    needs eyes on the pixels.
    """
    from PIL import Image as PILImage

    frames = video.frames
    picks = {"first": 0, "mid": len(frames) // 2, "last": len(frames) - 1}
    written = []
    for label, idx in picks.items():
        path = out / f"{stem}_{label}.png"
        PILImage.fromarray(frames[idx]).save(path)
        written.append(path.name)
    return written


@check("t2i", "Text-to-image: prompt actually conditions the output", needs_cuda=True)
def check_text_to_image(ctx: Context) -> Outcome:
    """Qwen-Image, asserted on conditioning rather than on liveness.

    Two generations at the SAME seed with DIFFERENT prompts. If the text
    encoder is wired up, they diverge; if it is dead or the conditioning is
    dropped, the seed alone decides the image and the two come out nearly
    identical. That failure mode has shipped here before, and it is invisible
    to any "is the image non-black" check -- the output is a perfectly good
    image of the wrong thing.

    Both PNGs are written to the workdir; correlation cannot tell you the
    image matches the words, so look at them.
    """
    from videopython.ai import TextToImage

    model = TextToImage()
    a = model.generate_image(
        prompt="a red vintage bicycle leaning against a white wall, sunny day",
        num_inference_steps=30,
        width=1024,
        height=1024,
        seed=42,
    )
    b = model.generate_image(
        prompt="a bowl of green apples on a rustic wooden table, soft light",
        num_inference_steps=30,
        width=1024,
        height=1024,
        seed=42,
    )
    a.save(ctx.workdir / "t2i_bicycle.png")
    b.save(ctx.workdir / "t2i_apples.png")

    corr = _corr(_grey(a), _grey(b))
    free_weights()
    # Same seed gives the same starting noise, so some structural similarity is
    # expected; near-unity means the prompt never reached the model.
    conditioned = corr < 0.9
    return Outcome(
        passed=conditioned,
        detail=(
            f"same-seed different-prompt corr={corr:.3f} (prompt conditions output)"
            if conditioned
            else f"corr={corr:.3f} -- prompt is NOT conditioning; text encoder likely dead"
        ),
        measurements={"same_seed_diff_prompt_corr": round(corr, 4), "artifacts": "t2i_bicycle.png, t2i_apples.png"},
    )


@check("t2v", "Text-to-video: renders and actually moves", needs_cuda=True)
def check_text_to_video(ctx: Context) -> Outcome:
    """Wan2.2 T2V. One generation -- a second at 14B is not worth the minutes.

    Asserts the clip is not a still: consecutive-frame correlation below unity
    means real motion. A pipeline that emits the same frame 81 times passes
    every shape and non-black check ever written.
    """
    from videopython.ai import TextToVideo

    model = TextToVideo()
    video = model.generate_video(
        prompt="a misty mountain lake at dawn, mist drifting slowly over the water",
        num_steps=20,
        num_frames=49,
    )
    written = _save_frames(video, ctx.workdir, "t2v")
    video.save(ctx.workdir / "t2v.mp4")

    frames = video.frames
    moved = _motion_fraction(frames[0], frames[-1])
    divergence = 1.0 - _corr(_grey(frames[0]), _grey(frames[-1]))
    free_weights()
    moving = moved > 0.05
    return Outcome(
        passed=moving and len(frames) > 1,
        detail=(
            f"{len(frames)} frames, {moved:.1%} of pixels move"
            if moving
            else f"static output: only {moved:.2%} of pixels differ between first and last frame"
        ),
        measurements={
            "frames": len(frames),
            "moving_pixel_fraction": round(moved, 4),
            "first_last_divergence": round(divergence, 5),
            "artifacts": ", ".join(written),
        },
    )


@check("i2v", "Image-to-video: the input image conditions frame 0", needs_cuda=True)
def check_image_to_video(ctx: Context) -> Outcome:
    """Wan2.2 I2V, asserted on conditioning AND motion, which pull opposite ways.

    Frame 0 must resemble the supplied image (the image is actually driving the
    generation) while later frames must diverge from it (it is animating rather
    than freezing). Checking only one of those passes a model that ignores the
    image, or one that returns it 81 times.

    Uses the t2i output when a previous check produced one, else a synthetic
    image, so this check can run standalone.
    """
    import numpy as np
    from PIL import Image as PILImage

    from videopython.ai import ImageToVideo

    source = ctx.workdir / "t2i_bicycle.png"
    if source.exists():
        image = PILImage.open(source).convert("RGB").resize((832, 480))
        origin = source.name
    else:
        grad = np.linspace(0, 255, 832, dtype=np.uint8)
        arr = np.repeat(grad[None, :], 480, axis=0)
        image = PILImage.fromarray(np.stack([arr, arr[:, ::-1], arr], axis=-1))
        image.save(ctx.workdir / "i2v_input.png")
        origin = "i2v_input.png (synthetic)"

    model = ImageToVideo()
    video = model.generate_video(
        image=image, prompt="gentle camera push in, subtle motion", num_steps=20, num_frames=49
    )
    written = _save_frames(video, ctx.workdir, "i2v")
    video.save(ctx.workdir / "i2v.mp4")

    frames = video.frames
    ref = np.asarray(image.resize((frames.shape[2], frames.shape[1])), dtype=np.float32)
    conditioning = _corr(_grey(frames[0]), _grey(ref))
    moved = _motion_fraction(frames[0], frames[-1])
    # Deliberately NOT measured here: last-frame-vs-input correlation, as a proxy
    # for "did the subject survive the clip".
    #
    # It was measured on the 2026-08-04 run, read 0.342, and was written up as the
    # subject dissolving. That was wrong -- reviewing the full frame sequence showed
    # the bicycle fully intact and simply translating out of frame under a strong
    # camera move. The low correlation was the camera, not the subject.
    #
    # That is the same blind spot that made this harness call a moving t2v clip
    # static (global correlation is dominated by a locked camera's composition),
    # just with the sign flipped. Global frame correlation cannot separate "the
    # camera moved" from "the subject degraded", so no threshold on it is sound and
    # reporting it unasserted only invites the same misreading. If subject
    # persistence is ever worth asserting, it needs a translation-invariant measure
    # (edge energy, or correlation maximised over a translation search) validated on
    # more than one clip. Look at the mp4, not three stills -- see REPORT.md.
    free_weights()

    ok = conditioning > 0.5 and moved > 0.05
    if conditioning <= 0.5:
        detail = f"frame 0 does not resemble the input image (corr={conditioning:.3f}) -- image not conditioning"
    elif moved <= 0.05:
        detail = f"conditioned (corr={conditioning:.3f}) but static -- only {moved:.2%} of pixels differ"
    else:
        detail = (
            f"frame0-vs-input corr={conditioning:.3f}, {moved:.1%} of pixels move "
            f"(WATCH i2v.mp4: these say the image conditioned frame 0 and something "
            f"moved, never that the model drew what was asked)"
        )
    return Outcome(
        passed=ok,
        detail=detail,
        measurements={
            "input": origin,
            "frame0_input_corr": round(conditioning, 4),
            "moving_pixel_fraction": round(moved, 4),
            "artifacts": ", ".join(written),
        },
    )


@check("tts", "Speech synthesis is intelligible when transcribed back", needs_cuda=True)
def check_tts(ctx: Context) -> Outcome:
    """Chatterbox, verified by round-trip through Whisper.

    "Produced non-silent audio" is not evidence of speech -- noise is
    non-silent. Transcribing the output and comparing word overlap with the
    input text is, and it is the only check here that would notice the voice
    saying something else entirely.
    """
    from videopython.ai import AudioToText, TextToSpeech

    text = "The quick brown fox jumps over the lazy dog near the river bank."
    tts = TextToSpeech()
    audio = tts.generate_audio(text=text)
    wav = ctx.workdir / "tts_default.wav"
    audio.save(str(wav))

    # Voice cloning is the feature dubbing actually depends on, so exercise it
    # too: re-synthesise conditioned on the default voice's own output. Saved
    # for listening -- similarity of timbre is not something a number settles.
    cloned_note = "not attempted"
    try:
        cloned = tts.generate_audio(text=text, voice_sample=audio)
        cloned.save(str(ctx.workdir / "tts_cloned.wav"))
        cloned_note = "tts_cloned.wav written"
    except Exception as exc:  # noqa: BLE001 - reported, not fatal to the check
        cloned_note = f"cloning raised {type(exc).__name__}: {exc}"
    free_weights()

    heard = AudioToText().transcribe(audio)
    spoken = _transcript_text(heard).lower()
    wanted = {w.strip(".,").lower() for w in text.split()}
    got = {w.strip(".,").lower() for w in spoken.split()}
    overlap = len(wanted & got) / max(1, len(wanted))
    free_weights()

    ok = overlap >= 0.6
    return Outcome(
        passed=ok,
        detail=(
            f"round-trip word overlap {overlap:.0%}: {spoken[:70]!r}"
            if ok
            else f"unintelligible (overlap {overlap:.0%}): {spoken[:70]!r}"
        ),
        measurements={
            "word_overlap": round(overlap, 3),
            "transcript": spoken[:200],
            "voice_clone": cloned_note,
            "artifacts": "tts_default.wav, tts_cloned.wav",
        },
    )


@check("separation", "Source separation puts the voice in the vocals stem", needs_cuda=True)
def check_separation(ctx: Context) -> Outcome:
    """Demucs, asserted by correlating stems against signals we control.

    Builds a mix from a known speech track and a known synthetic music bed,
    separates it, and requires the vocals stem to track the speech envelope far
    more closely than the music envelope. A separator that returns the input
    unchanged, or writes empty stems, fails this -- and both have happened.
    """
    import numpy as np

    from videopython.ai import TextToSpeech
    from videopython.ai.dubbing.separation import AudioSeparator

    speech = TextToSpeech().generate_audio(text="This sentence should end up in the vocals stem, not the music one.")
    free_weights()

    sr = speech.metadata.sample_rate
    voice = np.asarray(speech.data, dtype=np.float32)
    if voice.ndim > 1:
        voice = voice.mean(axis=1)
    t = np.arange(len(voice), dtype=np.float32) / sr
    # A chord plus a pulsing envelope: broadband and rhythmic, so a separator
    # cannot trivially split it from speech by bandwidth alone.
    music = 0.25 * (np.sin(2 * np.pi * 220 * t) + np.sin(2 * np.pi * 277 * t) + np.sin(2 * np.pi * 330 * t))
    music *= 0.6 + 0.4 * np.sin(2 * np.pi * 0.7 * t)
    mixed = np.clip(voice + music, -1.0, 1.0)

    mix_audio = _mono_audio(mixed, sr)
    mix_audio.save(str(ctx.workdir / "sep_mix.wav"))
    stems = AudioSeparator().separate(mix_audio)
    free_weights()

    vocals_env = _envelope(stems.vocals)
    voice_env = _envelope(_mono_audio(voice, sr))
    music_env = _envelope(_mono_audio(music, sr))
    to_voice, to_music = _corr(vocals_env, voice_env), _corr(vocals_env, music_env)

    stems.vocals.save(str(ctx.workdir / "sep_vocals.wav"))
    if stems.background is not None:
        stems.background.save(str(ctx.workdir / "sep_background.wav"))

    ok = to_voice > 0.7 and to_voice > to_music + 0.2
    return Outcome(
        passed=ok,
        detail=(
            f"vocals envelope corr {to_voice:.3f} vs voice / {to_music:.3f} vs music"
            if ok
            else f"vocals stem does not track the voice (corr {to_voice:.3f} vs voice, {to_music:.3f} vs music)"
        ),
        measurements={
            "corr_vocals_vs_voice": round(to_voice, 4),
            "corr_vocals_vs_music": round(to_music, 4),
            "artifacts": "sep_mix.wav, sep_vocals.wav, sep_background.wav",
        },
    )


@check("music", "Music generation is audible and prompt-conditioned", needs_cuda=True)
def check_music(ctx: Context) -> Outcome:
    """MusicGen. Same structure as t2i: different prompts must give different audio."""
    from videopython.ai import TextToMusic

    model = TextToMusic()
    a = model.generate_audio(text="slow melancholic solo piano", max_new_tokens=256)
    b = model.generate_audio(text="fast aggressive electronic drum and bass", max_new_tokens=256)
    a.save(str(ctx.workdir / "music_piano.wav"))
    b.save(str(ctx.workdir / "music_dnb.wav"))
    free_weights()

    import numpy as np

    peak = float(np.abs(np.asarray(a.data, dtype=np.float32)).max())
    corr = _corr(_envelope(a), _envelope(b))
    ok = peak > 1e-3 and corr < 0.9
    return Outcome(
        passed=ok,
        detail=(
            f"audible (peak {peak:.3f}), different prompts diverge (env corr {corr:.3f})"
            if ok
            else (f"silent output (peak {peak:.5f})" if peak <= 1e-3 else f"prompts do not condition (corr {corr:.3f})")
        ),
        measurements={
            "peak": round(peak, 5),
            "cross_prompt_env_corr": round(corr, 4),
            "artifacts": "music_piano.wav, music_dnb.wav",
        },
    )


@check("detect", "Object detection finds the right objects in a known image", needs_cuda=True)
def check_detection(ctx: Context) -> Outcome:
    """D-FINE against the COCO sample every detector demo uses.

    Semantic: the labels must include the things actually in the picture. A
    detector returning an empty list, or boxes with garbage labels, passes any
    "it returned a result" check.
    """
    import io
    import urllib.request

    import numpy as np
    from PIL import Image as PILImage

    from videopython.ai import ObjectDetector

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"  # two cats on a sofa, remotes
    with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310 - fixed, well-known URL
        image = PILImage.open(io.BytesIO(resp.read())).convert("RGB")
    image.save(ctx.workdir / "detect_input.png")

    detections = ObjectDetector().detect(np.asarray(image))
    labels = sorted({str(getattr(d, "label", d)) for d in detections})
    free_weights()

    found_cat = any("cat" in lab.lower() for lab in labels)
    return Outcome(
        passed=found_cat,
        detail=(f"{len(detections)} objects: {', '.join(labels[:6])}" if found_cat else f"no cat found; got {labels}"),
        measurements={"count": len(detections), "labels": labels, "artifacts": "detect_input.png"},
    )


_MAX_DUB_TRUNCATION_SECONDS = 3.0


@check("dub", "Full dub end-to-end, translated and audible", needs_video=True, needs_cuda=True)
def check_dub(ctx: Context) -> Outcome:
    """The integration that has broken most often, asserted on its output.

    Checks, in order of how badly each has bitten:

    1. ``translation_failures`` is empty. This is the exact signal the 0.53.0
       reasoning-model bug produced -- every segment failed and was dubbed with
       empty text, while the pipeline reported success.
    2. The dubbed track is not silent. A pipeline that "succeeds" into silence
       has happened via the separator writing stems to the wrong directory.
    3. The worst timing truncation does not exceed the manual-run baseline.
    """
    import numpy as np

    from videopython.ai.dubbing import VideoDubber
    from videopython.base.video import Video

    assert ctx.video is not None  # guaranteed by needs_video
    video = Video.from_path(str(ctx.video))
    dubber = VideoDubber(device=ctx.device) if ctx.device else VideoDubber()
    result = dubber.dub(
        video,
        target_lang=ctx.target_lang,
        source_lang=ctx.source_lang,
        voice_clone=True,
        enable_diarization=ctx.enable_diarization,
    )

    segments = len(result.translated_segments)
    failures = list(result.translation_failures or [])
    peak = float(np.abs(np.asarray(result.dubbed_audio.data, dtype=np.float32)).max())
    timing = result.timing_summary

    # Written before any assertion: a FAILING dub is exactly the one worth
    # listening to, and the run may be the only time these models are loaded.
    result.dubbed_audio.save(str(ctx.workdir / "dub_audio.wav"))
    dubbed_video = video.add_audio(result.dubbed_audio, overlay=False)  # replace, not mix
    dubbed_video.save(ctx.workdir / "dub_video.mp4")
    transcript_path = ctx.workdir / "dub_transcript.txt"
    transcript_path.write_text(
        "SOURCE ({}):\n{}\n\nTRANSLATED SEGMENTS:\n{}\n".format(
            result.source_lang,
            _transcript_text(result.source_transcription),
            "\n".join(f"[{s.start:7.2f} {s.end:7.2f}] {s.translated_text}" for s in result.translated_segments),
        ),
        encoding="utf-8",
    )

    measurements: dict[str, Any] = {
        "segments": segments,
        "translation_failures": len(failures),
        "dubbed_peak_amplitude": round(peak, 6),
        "diarization": ctx.enable_diarization,
        "speakers": sorted(result.source_transcription.speakers),
        "voice_sample_speakers": sorted(result.voice_samples),
        "artifacts": "dub_audio.wav, dub_video.mp4, dub_transcript.txt",
    }
    if timing is not None:
        measurements |= {
            "truncated": f"{timing.truncated_count}/{timing.total_segments}",
            "mean_speed_factor": round(timing.mean_speed_factor, 3),
            "max_truncation_seconds": round(timing.max_truncation_seconds, 3),
        }

    if failures:
        return Outcome(
            passed=False,
            detail=f"{len(failures)}/{segments} segments failed to translate (dubbed with empty text)",
            measurements=measurements,
        )
    if ctx.enable_diarization and not result.source_transcription.speakers:
        return Outcome(passed=False, detail="diarization returned no speakers", measurements=measurements)
    if ctx.enable_diarization and not result.voice_samples:
        return Outcome(passed=False, detail="voice cloning produced no speaker samples", measurements=measurements)
    if peak <= 1e-4:
        return Outcome(passed=False, detail="dubbed track is silent", measurements=measurements)
    if timing is None:
        return Outcome(passed=False, detail="timing summary is missing", measurements=measurements)
    if timing.max_truncation_seconds > _MAX_DUB_TRUNCATION_SECONDS:
        return Outcome(
            passed=False,
            detail=(
                f"worst timing truncation {timing.max_truncation_seconds:.3f}s exceeds "
                f"{_MAX_DUB_TRUNCATION_SECONDS:.1f}s"
            ),
            measurements=measurements,
        )

    free_weights()
    return Outcome(
        passed=True,
        detail=(
            f"{segments} segments {ctx.source_lang}->{ctx.target_lang}, all translated, audible, "
            f"worst truncation {timing.max_truncation_seconds:.3f}s"
        ),
        measurements=measurements,
    )


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

_PASS, _FAIL, _SKIP = "PASS", "FAIL", "SKIP"


def render(results: list[tuple[Check, str, str, dict[str, Any]]], total_elapsed_seconds: float) -> str:
    """Render the verification report as Markdown."""
    lines = ["| # | Check | Result | Detail |", "|---|---|---|---|"]
    for i, (chk, status, detail, _) in enumerate(results, 1):
        lines.append(f"| {i} | `{chk.name}` — {chk.summary} | **{status}** | {detail} |")

    measured = [(c, m) for c, _, _, m in results if m]
    if measured:
        lines.append("")
        lines.append("<details><summary>Measurements</summary>")
        lines.append("")
        for chk, m in measured:
            lines.append(f"- **{chk.name}**: " + ", ".join(f"{k}={v}" for k, v in m.items()))
        lines.append("")
        lines.append("</details>")
    lines.append("")
    lines.append(f"**Total elapsed:** {total_elapsed_seconds:.3f} seconds")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=_DESCRIPTION, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--all", action="store_true", help="run every registered check")
    parser.add_argument("--only", default="", help="comma-separated check names (see --list)")
    parser.add_argument("--list", action="store_true", help="list checks and exit")
    parser.add_argument("--video", type=Path, help="source clip for checks that need one")
    parser.add_argument("--workdir", type=Path, default=Path("./verify-out"), help="where checks write artifacts")
    parser.add_argument("--device", help="torch device override (default: auto)")
    parser.add_argument("--ollama-model", help="override the Ollama model under test")
    parser.add_argument("--source-lang", default="pl")
    parser.add_argument("--target-lang", default="es")
    parser.add_argument(
        "--enable-diarization",
        action="store_true",
        help="enable per-speaker diarization and voice cloning for the dub check",
    )
    args = parser.parse_args(argv)

    if args.list:
        width = max(len(n) for n in _CHECKS)
        for name, chk in _CHECKS.items():
            needs = [n for n, on in (("--video", chk.needs_video), ("CUDA", chk.needs_cuda)) if on]
            flag = f"  (needs {' + '.join(needs)})" if needs else ""
            print(f"{name:{width}}  {chk.summary}{flag}")
        return 0

    if args.all:
        selected = list(_CHECKS.values())
    elif args.only:
        unknown = [n for n in args.only.split(",") if n.strip() and n.strip() not in _CHECKS]
        if unknown:
            parser.error(f"unknown check(s): {', '.join(unknown)}. See --list.")
        selected = [_CHECKS[n.strip()] for n in args.only.split(",") if n.strip()]
    else:
        parser.error("pass --all or --only (see --list)")

    args.workdir.mkdir(parents=True, exist_ok=True)
    ctx = Context(
        video=args.video,
        workdir=args.workdir,
        device=args.device,
        ollama_model=args.ollama_model,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        enable_diarization=args.enable_diarization,
    )

    results: list[tuple[Check, str, str, dict[str, Any]]] = []
    run_started = perf_counter()
    for chk in selected:
        unmet = chk.unmet_requirement(ctx)
        if unmet is not None:
            results.append((chk, _SKIP, unmet, {}))
            print(f"[skip] {chk.name} ({unmet})", file=sys.stderr, flush=True)
            continue
        print(f"[ run] {chk.name} ...", file=sys.stderr, flush=True)
        started = perf_counter()
        try:
            outcome = chk.fn(ctx)
            status = _PASS if outcome.passed else _FAIL
            detail = outcome.detail
            measurements = dict(outcome.measurements)
        except Exception as exc:
            # A raise is a failure, not a crash: one broken model must not cost
            # the other results on a box that is billed by the hour.
            traceback.print_exc()
            status = _FAIL
            detail = f"raised {type(exc).__name__}: {exc}"
            measurements = {}
        elapsed = perf_counter() - started
        measurements["elapsed_seconds"] = round(elapsed, 3)
        results.append((chk, status, detail, measurements))
        print(f"[{status.lower():>4}] {chk.name} ({elapsed:.3f}s)", file=sys.stderr, flush=True)

    total_elapsed = perf_counter() - run_started
    print()
    print(render(results, total_elapsed))

    failed = [c.name for c, s, _, _ in results if s == _FAIL]
    skipped = [c.name for c, s, _, _ in results if s == _SKIP]
    if skipped:
        print(f"\nSkipped: {', '.join(skipped)}", file=sys.stderr)
    if failed:
        print(f"\nFAILED: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
