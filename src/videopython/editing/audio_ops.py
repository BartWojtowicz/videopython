"""Plan-level audio: the music bed and its transcription-derived ducking.

Unlike the per-segment audio twins (``to_ffmpeg_audio_filter`` on transforms /
audio-coupled effects, P1.9b), the music bed is a property of the WHOLE
assembled timeline, not of any one segment. It is mixed in a FINAL ffmpeg pass
*after* assembly (concat / xfade transitions), in one ``amix`` over the
assembled program audio plus the bed input.

:class:`MusicBed` is a frozen, closed pydantic model carried on
:attr:`VideoEdit.music_bed`. ``run_to_file`` routes through the
single :func:`build_music_bed_filter_complex` builder for the mix.

Ducking is *transcription-derived* and deterministic: when ``duck`` is set the
bed is lowered under the speech windows derived from the context transcription
(via the shared :func:`videopython.editing.transforms.speech_windows` helper),
with attack/release ramps -- no live sidechain key signal. It is only
well-defined when the assembled timeline is a single segment's timeline (so the
rebased transcription maps directly); a multi-segment plan with ``duck`` set is
rejected up front with a structured :class:`PlanError`.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from videopython.base import _ffmpeg
from videopython.base.exceptions import PlanError, PlanErrorCode, PlanValidationError

__all__ = [
    "MusicBed",
    "build_music_bed_filter_complex",
    "duck_volume_expression",
    "volume_envelope",
]


def volume_envelope(terms: list[tuple[str, str]]) -> str:
    """Build a nested-if gain envelope expression for volume automation.

    Takes a list of ``(condition, gain_expr)`` tuples and folds them into a
    nested ``if()`` expression evaluated outermost-first (the terms in
    chronological order), with each condition mapping to its gain expression and
    ``"1"`` as the final fallback (no change). The result is wrapped in the
    ffmpeg ``volume`` filter format.

    Args:
        terms: List of ``(condition_str, gain_expr_str)`` tuples, where each
            condition is an ffmpeg expression (e.g. ``"between(t,0,1)"``) and
            ``gain_expr`` is the volume multiplier for that condition.

    Returns:
        Complete ffmpeg volume filter string: ``volume=volume='...':eval=frame``.
    """
    expr = "1"
    for cond, gain in reversed(terms):
        expr = f"if({cond},{gain},{expr})"
    return f"volume=volume='{expr}':eval=frame"


def duck_volume_expression(
    speech: list[tuple[float, float]],
    *,
    duck: float,
    attack: float,
    release: float,
) -> str:
    """A ``volume=...:eval=frame`` automation that lowers the bed under speech.

    Returns the gain expression that holds ``1`` (full bed) outside speech and
    ``1 - duck`` (the ducked level) inside each speech window, with linear
    attack/release ramps of ``attack`` / ``release`` seconds at the leading /
    trailing edges so the duck eases in and out rather than stepping. The ramps
    are expressed as clamped linear interpolations on ``t`` (the per-sample-frame
    time ffmpeg's ``volume`` filter exposes under ``eval=frame``), so the whole
    duck is one deterministic, transcription-grounded expression -- no live key
    signal, no ``sidechaincompress``.

    The expression is built as nested ``if(between(t,...),ramp,...)`` terms in
    chronological order, evaluated outermost-first, falling back to ``1``; an
    empty ``speech`` list yields the constant ``1`` (the bed is untouched).
    """
    floor = 1.0 - duck  # the ducked gain
    # Each speech window [s, e] contributes three regions: an attack ramp
    # 1 -> floor over [s, s+attack], a hold at floor over [s+attack, e], and a
    # release ramp floor -> 1 over [e, e+release]. Build them in time order, then
    # fold into nested if()s evaluated outermost (earliest) first.
    terms: list[tuple[str, str]] = []
    for s, e in speech:
        atk_end = s + attack
        rel_end = e + release
        # Attack: linearly drop from 1 to floor across [s, atk_end].
        atk = f"(1-{duck:.6f}*(t-{s:.6f})/{attack:.6f})"
        terms.append((f"between(t,{s:.6f},{atk_end:.6f})", atk))
        # Hold the ducked floor across [atk_end, e].
        terms.append((f"between(t,{atk_end:.6f},{e:.6f})", f"{floor:.6f}"))
        # Release: linearly rise from floor back to 1 across [e, rel_end].
        rel = f"({floor:.6f}+{duck:.6f}*(t-{e:.6f})/{release:.6f})"
        terms.append((f"between(t,{e:.6f},{rel_end:.6f})", rel))
    return volume_envelope(terms)


class MusicBed(BaseModel):
    """A music bed mixed under the WHOLE assembled program in a final pass.

    The bed spans the entire timeline (after concat / transitions) and is mixed
    in one ``amix`` over the assembled program audio plus this bed input -- not a
    per-segment op. Frozen and closed so it surfaces as a constrained object in
    :meth:`VideoEdit.json_schema` and never accepts stray fields.

    ``source`` is validated cheaply (an ffprobe header probe, like
    :class:`ImageOverlay`'s source check) at validate/check time, surfacing
    ``SOURCE_UNREADABLE`` before any decode. ``loop`` loops/trims the bed to the
    program duration so it neither truncates early nor extends the output.
    ``duck`` (when set) lowers the bed under transcription-derived speech windows
    with ``duck_attack``/``duck_release`` ramps; ducking requires a
    single-segment assembled timeline (see module docstring).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: Path = Field(description="Path to the music bed audio file (mixed under the whole program).")
    gain: float = Field(0.25, ge=0, description="Bed level multiplier (0 = silent, 1 = unchanged source level).")
    loop: bool = Field(True, description="Loop/trim the bed to the program duration so it spans the whole timeline.")
    fade_in: float = Field(0.0, ge=0, description="Seconds to fade the bed in at the start of the program.")
    fade_out: float = Field(0.0, ge=0, description="Seconds to fade the bed out at the end of the program.")
    duck: float | None = Field(
        None,
        ge=0,
        le=1,
        description=(
            "How far to lower the bed under speech (0 = no ducking, 1 = fully duck). "
            "None disables ducking. Requires a single-segment plan."
        ),
    )
    duck_attack: float = Field(0.2, gt=0, description="Seconds for the duck to ramp in at each speech onset.")
    duck_release: float = Field(0.5, gt=0, description="Seconds for the duck to ramp back out after speech ends.")

    def validate_source(self) -> None:
        """Reject an unreadable bed ``source`` with ``SOURCE_UNREADABLE``.

        Mirrors :meth:`ImageOverlay.predict_metadata`: a cheap ffprobe header
        probe (no decode) catches a missing / non-audio file at validate time,
        before ``run_to_file()`` would crash mid-stream after assembling the program.
        """
        try:
            info = _ffmpeg.probe(self.source)
            has_audio = any(s.get("codec_type") == "audio" for s in info.get("streams", []))
        except Exception as exc:
            message = f"music_bed source {str(self.source)!r} is not a readable audio file: {exc}"
            raise PlanValidationError(
                message,
                [PlanError(code=PlanErrorCode.SOURCE_UNREADABLE, field="source")],
            ) from exc
        if not has_audio:
            message = f"music_bed source {str(self.source)!r} has no audio stream"
            raise PlanValidationError(
                message,
                [PlanError(code=PlanErrorCode.SOURCE_UNREADABLE, field="source")],
            )

    def bed_stages(self, program_seconds: float, speech: list[tuple[float, float]] | None) -> list[str]:
        """The ordered ``filter_complex`` fragments applied to the bed stream.

        ``[gain] -> [fade in/out] -> [duck automation] -> [loop/trim pin]``: the
        bed is scaled to ``gain``, faded, ducked under ``speech`` (when given and
        ``duck`` is set), then pinned to exactly ``program_seconds`` (``atrim``
        end + ``apad`` whole_dur) so a looped bed neither truncates early nor
        extends the output past the program. Used by the file
        mix path via :func:`build_music_bed_filter_complex`.
        """
        stages: list[str] = [f"volume={self.gain:.6f}"]
        if self.fade_in > 0:
            stages.append(f"afade=t=in:st=0:d={self.fade_in:.6f}")
        if self.fade_out > 0:
            fade_start = max(0.0, program_seconds - self.fade_out)
            stages.append(f"afade=t=out:st={fade_start:.6f}:d={self.fade_out:.6f}")
        if self.duck is not None and speech:
            stages.append(
                duck_volume_expression(speech, duck=self.duck, attack=self.duck_attack, release=self.duck_release)
            )
        # Pin the bed to the program length: trim a too-long (looped) bed and pad
        # a too-short one, so amix's program input is exactly matched.
        stages.append(f"atrim=end={program_seconds:.6f}")
        stages.append(f"apad=whole_dur={program_seconds:.6f}")
        return stages


def build_music_bed_filter_complex(
    bed: MusicBed,
    program_seconds: float,
    *,
    speech: list[tuple[float, float]] | None = None,
    bed_input_index: int = 1,
    prog_label: str = "0:a",
    sample_rate: int = 44100,
) -> tuple[list[str], list[str], str]:
    """Compile the music-bed input args + ``filter_complex`` mix graph.

    The single source of the bed mix, used by ``run_to_file`` (file pass). Returns
    ``(input_args, graph_statements, out_label)``:

    - ``input_args`` are the ffmpeg ``-i`` argv for the bed: ``-stream_loop -1``
      (loop indefinitely; the ``atrim`` in :meth:`MusicBed.bed_stages` clamps it
      to the program length) when ``loop`` is set, then ``-i <source>``. The bed
      lands at ffmpeg input index ``bed_input_index``.
    - ``graph_statements`` route the bed through its gain/fade/duck/length stages
      to ``[bed]``, then ``[<prog>][bed]amix=inputs=2:duration=first:
      dropout_transition=0`` -- ``duration=first`` clamps the output to the
      program length so the bed never extends it -- then an ``aresample`` to a
      final ``[mixout]``.
    - ``out_label`` is always ``"[mixout]"``.

    ``prog_label`` is the ffmpeg label of the already-assembled program audio
    (``"0:a"`` for the assembled file/Video at input 0). ``speech`` is the
    transcription-derived speech windows for ducking (ignored when ``duck`` is
    ``None``).
    """
    input_args: list[str] = []
    if bed.loop:
        # Loop the bed indefinitely on the input side; the per-stream atrim pins
        # it to the program length, so a short bed tiles under a long program.
        input_args.extend(["-stream_loop", "-1"])
    input_args.extend(["-i", str(bed.source)])

    statements: list[str] = []
    prev = f"{bed_input_index}:a"
    stages = bed.bed_stages(program_seconds, speech)
    for i, stage in enumerate(stages):
        out = "bed" if i == len(stages) - 1 else f"b{i}"
        statements.append(f"[{prev}]{stage}[{out}]")
        prev = out
    # normalize=0 is REQUIRED: amix's default normalize=1 divides every input
    # (including the program/dialogue) by the input count, so attaching any bed
    # would halve the video's own audio. With normalize=0 the program passes
    # through at full level and the bed mixes in at its own `gain`.
    statements.append(f"[{prog_label}][bed]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[mixed]")
    statements.append(f"[mixed]aresample={sample_rate}[mixout]")
    return input_args, statements, "[mixout]"
