from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _load_script() -> ModuleType:
    path = Path(__file__).parents[2] / "scripts" / "benchmark_dubbing.py"
    spec = importlib.util.spec_from_file_location("benchmark_dubbing", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_normalize_stage_groups_progress_updates() -> None:
    script = _load_script()

    assert script.normalize_stage("Translating text (42%)") == "Translating text"
    assert script.normalize_stage("Generating speech (3/10)") == "Generating dubbed speech"
    assert script.normalize_stage("Complete") == "Muxing output"


def test_parse_meminfo_reports_used_and_swap_mib() -> None:
    script = _load_script()
    meminfo = """\
MemTotal:       32768 kB
MemAvailable:   12288 kB
SwapTotal:       8192 kB
SwapFree:        2048 kB
"""

    assert script.parse_meminfo(meminfo) == (20.0, 12.0, 6.0)


def test_stage_summary_uses_event_boundaries_and_baseline_deltas() -> None:
    script = _load_script()
    events = [
        {"type": "stage", "stage": "Transcribing audio", "elapsed_s": 1.0},
        {"type": "progress", "stage": "Transcribing audio", "elapsed_s": 1.5},
        {"type": "stage", "stage": "Translating text", "elapsed_s": 3.0},
    ]
    baseline = script.Sample(0, "Baseline", 100, 900, 0, 0, 200, 0)
    samples = [
        script.Sample(2, "Transcribing audio", 130, 870, 0, 20, 250, 80),
        script.Sample(4, "Translating text", 160, 840, 1, 40, 300, 90),
    ]

    stages = script.summarize_stages(events, samples, 5.0, baseline)

    assert stages == [
        {
            "stage": "Transcribing audio",
            "duration_s": 2.0,
            "peak_process_rss_mib": 20,
            "peak_system_ram_mib": 130,
            "system_ram_delta_mib": 30,
            "peak_vram_mib": 250,
            "vram_delta_mib": 50,
        },
        {
            "stage": "Translating text",
            "duration_s": 2.0,
            "peak_process_rss_mib": 40,
            "peak_system_ram_mib": 160,
            "system_ram_delta_mib": 60,
            "peak_vram_mib": 300,
            "vram_delta_mib": 100,
        },
    ]
