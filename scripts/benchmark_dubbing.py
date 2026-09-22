#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
import traceback
import urllib.request
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass
class Sample:
    elapsed_s: float
    stage: str
    system_ram_used_mib: float
    system_ram_available_mib: float
    swap_used_mib: float
    process_tree_rss_mib: float
    gpu_memory_used_mib: float | None
    gpu_utilization_pct: float | None


def normalize_stage(stage: str) -> str:
    if stage.startswith("Translating text"):
        return "Translating text"
    if stage.startswith("Generating speech") or stage == "Generating dubbed speech":
        return "Generating dubbed speech"
    if stage == "Complete":
        return "Muxing output"
    return stage


def parse_meminfo(text: str) -> tuple[float, float, float]:
    values = {
        key.rstrip(":"): int(value)
        for line in text.splitlines()
        if len(parts := line.split()) >= 2
        for key, value in [parts[:2]]
    }
    used_kib = values["MemTotal"] - values["MemAvailable"]
    swap_used_kib = values["SwapTotal"] - values["SwapFree"]
    return used_kib / 1024, values["MemAvailable"] / 1024, swap_used_kib / 1024


def read_system_memory() -> tuple[float, float, float]:
    return parse_meminfo(Path("/proc/meminfo").read_text())


def process_tree_rss(root_pid: int) -> float:
    processes: dict[int, tuple[int, int]] = {}
    for status_path in Path("/proc").glob("[0-9]*/status"):
        try:
            fields = {}
            for line in status_path.read_text().splitlines():
                if line.startswith(("Pid:", "PPid:", "VmRSS:")):
                    key, value = line.split(":", 1)
                    fields[key] = int(value.split()[0])
            processes[fields["Pid"]] = (fields["PPid"], fields.get("VmRSS", 0))
        except (FileNotFoundError, KeyError, ProcessLookupError, ValueError):
            continue

    pids = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, (parent_pid, _) in processes.items():
            if parent_pid in pids and pid not in pids:
                pids.add(pid)
                changed = True
    return sum(processes.get(pid, (0, 0))[1] for pid in pids) / 1024


def read_gpu() -> tuple[float | None, float | None]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            check=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None, None

    rows = [tuple(float(value.strip()) for value in line.split(",")) for line in result.stdout.splitlines() if line]
    if not rows:
        return None, None
    return sum(row[0] for row in rows), max(row[1] for row in rows)


def read_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events = []
    for line in path.read_text().splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def stage_intervals(events: list[dict[str, Any]], total_s: float) -> list[tuple[str, float, float]]:
    starts: list[tuple[str, float]] = []
    for event in events:
        if event.get("type") != "stage":
            continue
        stage = str(event["stage"])
        elapsed = float(event["elapsed_s"])
        if not starts or starts[-1][0] != stage:
            starts.append((stage, elapsed))
    return [
        (stage, start, starts[index + 1][1] if index + 1 < len(starts) else total_s)
        for index, (stage, start) in enumerate(starts)
    ]


def summarize_stages(
    events: list[dict[str, Any]], samples: list[Sample], total_s: float, baseline: Sample
) -> list[dict[str, Any]]:
    summaries = []
    for stage, start, end in stage_intervals(events, total_s):
        stage_samples = [sample for sample in samples if start <= sample.elapsed_s <= end]
        summaries.append(
            {
                "stage": stage,
                "duration_s": end - start,
                "peak_process_rss_mib": max((sample.process_tree_rss_mib for sample in stage_samples), default=None),
                "peak_system_ram_mib": max((sample.system_ram_used_mib for sample in stage_samples), default=None),
                "system_ram_delta_mib": max(
                    (sample.system_ram_used_mib - baseline.system_ram_used_mib for sample in stage_samples),
                    default=None,
                ),
                "peak_vram_mib": max(
                    (sample.gpu_memory_used_mib for sample in stage_samples if sample.gpu_memory_used_mib is not None),
                    default=None,
                ),
                "vram_delta_mib": max(
                    (
                        sample.gpu_memory_used_mib - baseline.gpu_memory_used_mib
                        for sample in stage_samples
                        if sample.gpu_memory_used_mib is not None and baseline.gpu_memory_used_mib is not None
                    ),
                    default=None,
                ),
            }
        )
    return summaries


def _format_number(value: float | None, digits: int = 0) -> str:
    return "n/a" if value is None else f"{value:,.{digits}f}"


def render_report(run: dict[str, Any], stages: list[dict[str, Any]]) -> str:
    config = run["config"]
    environment = run["environment"]
    peaks = run["peaks"]
    lines = [
        "# Dubbing benchmark",
        "",
        f"- Result: **{run['status']}** (worker exit code `{run['exit_code']}`)",
        f"- Total wall time: **{run['total_s']:.2f} s**",
        f"- Input: `{config['input']}` ({run['input']['duration_s']:.2f} s, SHA-256 `{run['input']['sha256']}`)",
        f"- Commit: `{environment['git_commit']}`",
        f"- Host: {environment['platform']}; {environment['gpu']}",
        (
            f"- Python / VideoPython / PyTorch / Ollama: {environment['python']} / "
            f"{environment['videopython']} / {environment['torch']} / {environment['ollama_server']}"
        ),
        (
            f"- Peaks: worker tree RSS **{_format_number(peaks['process_tree_rss_mib'])} MiB**, "
            f"system RAM **{_format_number(peaks['system_ram_used_mib'])} MiB** "
            f"(**{_format_number(peaks['system_ram_delta_mib'])} MiB** over baseline), "
            f"VRAM **{_format_number(peaks['gpu_memory_used_mib'])} MiB** "
            f"(**{_format_number(peaks['gpu_memory_delta_mib'])} MiB** over baseline), "
            f"swap **{_format_number(peaks['swap_used_mib'])} MiB**"
        ),
        (
            f"- Configuration: `{config['source_lang'] or 'auto'} -> {config['target_lang']}`, "
            f"Whisper `{config['whisper_model']}`, translator `{config['translator_model']}`, "
            f"low-memory enabled, background `{config['preserve_background']}`, "
            f"voice clone `{config['voice_clone']}`, diarization `{config['enable_diarization']}`"
        ),
        "",
        "| Stage | Time (s) | Peak worker tree RSS (MiB) | Peak system RAM (MiB) | "
        "RAM over baseline (MiB) | Peak VRAM (MiB) | VRAM over baseline (MiB) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    if result := run.get("result_event"):
        lines.insert(
            4,
            f"- Segments: {result['translated_segments']} translated; "
            f"{len(result['translation_failures'])} translation failures; "
            f"{len(result['synthesis_failures'])} synthesis failures",
        )
    for stage in stages:
        lines.append(
            f"| {stage['stage']} | {stage['duration_s']:.2f} | "
            f"{_format_number(stage['peak_process_rss_mib'])} | "
            f"{_format_number(stage['peak_system_ram_mib'])} | "
            f"{_format_number(stage['system_ram_delta_mib'])} | "
            f"{_format_number(stage['peak_vram_mib'])} | "
            f"{_format_number(stage['vram_delta_mib'])} |"
        )
    lines.extend(
        [
            "",
            f"Sampling interval: {config['sample_interval_s']:.2f} s. System RAM and VRAM include Ollama and "
            "other host processes; baseline deltas reduce idle usage but do not isolate concurrent work.",
            "Raw samples are in `samples.csv`; progress events are in `events.jsonl`; worker output is in "
            "`worker.log`; full metadata is in `run.json`.",
            "",
        ]
    )
    if run.get("error"):
        lines.extend(["Failure: `" + str(run["error"]).replace("`", "'") + "`", ""])
    return "\n".join(lines)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _video_duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True,
        check=True,
        text=True,
    )
    return float(result.stdout.strip())


def _command_output(command: list[str], fallback: str) -> str:
    try:
        return subprocess.run(command, capture_output=True, check=True, text=True).stdout.strip() or fallback
    except (FileNotFoundError, subprocess.CalledProcessError):
        return fallback


def _ollama_version(host: str | None) -> str:
    url = (host or "http://127.0.0.1:11434").rstrip("/") + "/api/version"
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            return str(json.load(response).get("version", "unknown"))
    except (OSError, ValueError):
        return "unavailable"


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "source checkout" if name == "videopython" else "absent"


def _environment(translator_host: str | None) -> dict[str, str]:
    gpu = _command_output(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], "no NVIDIA GPU"
    )
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "videopython": _package_version("videopython"),
        "torch": _package_version("torch"),
        "ollama_server": _ollama_version(translator_host),
        "gpu": gpu.replace("\n", "; "),
        "git_commit": _command_output(["git", "rev-parse", "HEAD"], "unknown"),
    }


def _write_event(stream: Any, started: float, event_type: str, **values: Any) -> None:
    record = {
        "type": event_type,
        "elapsed_s": time.monotonic() - started,
        "time_utc": datetime.now(UTC).isoformat(),
        **values,
    }
    stream.write(json.dumps(record) + "\n")
    stream.flush()


def run_worker(config_path: Path) -> int:
    config = json.loads(config_path.read_text())
    events_path = Path(config["events_path"])
    started = float(config["benchmark_started_monotonic"])
    current_stage: str | None = None

    with events_path.open("a", buffering=1) as events:

        def set_stage(stage: str, progress: float | None = None) -> None:
            nonlocal current_stage
            normalized = normalize_stage(stage)
            if normalized != current_stage:
                current_stage = normalized
                _write_event(events, started, "stage", stage=normalized)
            if progress is not None:
                _write_event(events, started, "progress", stage=normalized, raw_stage=stage, progress=progress)

        try:
            from videopython.ai.dubbing import VideoDubber

            set_stage("Loading source audio")
            dubber = VideoDubber(
                device=config["device"],
                low_memory=True,
                whisper_model=config["whisper_model"],
                translator_model=config["translator_model"],
                translator_host=config["translator_host"],
            )
            result = dubber.dub_file(
                input_path=config["input"],
                output_path=config["output"],
                target_lang=config["target_lang"],
                source_lang=config["source_lang"],
                preserve_background=config["preserve_background"],
                voice_clone=config["voice_clone"],
                enable_diarization=config["enable_diarization"],
                keep_original_audio=config["keep_original_audio"],
                progress_callback=set_stage,
            )
            _write_event(
                events,
                started,
                "result",
                translated_segments=len(result.translated_segments),
                translation_failures=result.translation_failures,
                synthesis_failures=result.synthesis_failures,
            )
            return 0
        except BaseException as exc:
            _write_event(events, started, "error", error=f"{type(exc).__name__}: {exc}")
            traceback.print_exc()
            return 1


def _sample(elapsed: float, stage: str, root_pid: int, gpu: tuple[float | None, float | None]) -> Sample:
    used, available, swap = read_system_memory()
    return Sample(elapsed, stage, used, available, swap, process_tree_rss(root_pid), gpu[0], gpu[1])


def _write_sample(writer: csv.DictWriter[Any], stream: Any, sample: Sample) -> None:
    writer.writerow(asdict(sample))
    stream.flush()


def run_benchmark(args: argparse.Namespace) -> int:
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    if not input_path.is_file():
        raise SystemExit(f"Input video does not exist: {input_path}")
    if args.result_dir.exists() and any(args.result_dir.iterdir()):
        raise SystemExit(f"Result directory must be empty: {args.result_dir}")
    args.result_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    events_path = args.result_dir / "events.jsonl"
    samples_path = args.result_dir / "samples.csv"
    log_path = args.result_dir / "worker.log"
    config_path = args.result_dir / "worker-config.json"
    config = {
        "input": str(input_path),
        "output": str(output_path),
        "target_lang": args.target_language,
        "source_lang": args.source_language,
        "device": args.device,
        "whisper_model": args.whisper_model,
        "translator_model": args.translator_model,
        "translator_host": args.translator_host,
        "preserve_background": args.preserve_background,
        "voice_clone": args.voice_clone,
        "enable_diarization": args.enable_diarization,
        "keep_original_audio": args.keep_original_audio,
        "sample_interval_s": args.sample_interval,
        "events_path": str(events_path),
    }
    baseline_gpu = read_gpu()
    baseline_used, baseline_available, baseline_swap = read_system_memory()
    baseline = Sample(0.0, "Baseline", baseline_used, baseline_available, baseline_swap, 0.0, *baseline_gpu)
    started = time.monotonic()
    config["benchmark_started_monotonic"] = started
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    samples: list[Sample] = []
    current_stage = "Starting worker"

    with log_path.open("w") as log, samples_path.open("w", newline="") as sample_stream:
        writer = csv.DictWriter(sample_stream, fieldnames=list(Sample.__dataclass_fields__))
        writer.writeheader()
        worker = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--worker-config", str(config_path)],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        while worker.poll() is None:
            events = read_events(events_path)
            stage_events = [event for event in events if event.get("type") == "stage"]
            if stage_events:
                current_stage = str(stage_events[-1]["stage"])
            sample = _sample(time.monotonic() - started, current_stage, worker.pid, read_gpu())
            samples.append(sample)
            _write_sample(writer, sample_stream, sample)
            time.sleep(args.sample_interval)
        exit_code = worker.returncode

    total_s = time.monotonic() - started
    events = read_events(events_path)
    error_events = [event for event in events if event.get("type") == "error"]
    status = "passed" if exit_code == 0 else "failed"
    error: str | None
    if exit_code < 0:
        error = f"worker terminated by signal {-exit_code}"
    elif error_events:
        error = error_events[-1]["error"]
    else:
        error = None if exit_code == 0 else "worker exited without an error event"

    peak_ram = max((sample.system_ram_used_mib for sample in samples), default=baseline.system_ram_used_mib)
    peak_vram = max(
        (sample.gpu_memory_used_mib for sample in samples if sample.gpu_memory_used_mib is not None),
        default=baseline.gpu_memory_used_mib,
    )
    peaks = {
        "process_tree_rss_mib": max((sample.process_tree_rss_mib for sample in samples), default=0.0),
        "system_ram_used_mib": peak_ram,
        "system_ram_delta_mib": peak_ram - baseline.system_ram_used_mib,
        "swap_used_mib": max((sample.swap_used_mib for sample in samples), default=baseline.swap_used_mib),
        "gpu_memory_used_mib": peak_vram,
        "gpu_memory_delta_mib": (
            peak_vram - baseline.gpu_memory_used_mib
            if peak_vram is not None and baseline.gpu_memory_used_mib is not None
            else None
        ),
    }
    run = {
        "status": status,
        "exit_code": exit_code,
        "total_s": total_s,
        "error": error,
        "config": config,
        "environment": _environment(config["translator_host"]),
        "input": {"duration_s": _video_duration(input_path), "sha256": _sha256(input_path)},
        "baseline": asdict(baseline),
        "peaks": peaks,
        "result_event": next((event for event in reversed(events) if event.get("type") == "result"), None),
    }
    stages = summarize_stages(events, samples, total_s, baseline)
    run["stages"] = stages
    (args.result_dir / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    (args.result_dir / "report.md").write_text(render_report(run, stages))
    print(f"Benchmark {status}: {args.result_dir / 'report.md'}")
    return 0 if status == "passed" else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark VideoDubber stage time, RAM, and NVIDIA VRAM usage.")
    parser.add_argument("input", nargs="?", type=Path)
    parser.add_argument("output", nargs="?", type=Path)
    parser.add_argument("--result-dir", type=Path)
    parser.add_argument("--target-language", default="pl")
    parser.add_argument("--source-language", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--whisper-model", default="turbo")
    parser.add_argument("--translator-model", default="qwen3.5:4b")
    parser.add_argument("--translator-host", default=None)
    parser.add_argument("--sample-interval", type=float, default=1.0)
    parser.add_argument("--no-preserve-background", dest="preserve_background", action="store_false")
    parser.add_argument("--no-voice-clone", dest="voice_clone", action="store_false")
    parser.add_argument("--enable-diarization", action="store_true")
    parser.add_argument("--keep-original-audio", action="store_true")
    parser.add_argument("--worker-config", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_config:
        return args
    if args.input is None or args.output is None or args.result_dir is None:
        parser.error("input, output, and --result-dir are required")
    if args.sample_interval <= 0:
        parser.error("--sample-interval must be positive")
    return args


if __name__ == "__main__":
    parsed_args = parse_args()
    raise SystemExit(run_worker(parsed_args.worker_config) if parsed_args.worker_config else run_benchmark(parsed_args))
