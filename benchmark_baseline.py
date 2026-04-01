"""Baseline benchmark: measure time and memory for VideoEdit on cam1_1min.mp4.

Runs the same plan shape as the KurosawaAI production case (color_adjust + volume_adjust + fade)
but on a 1-minute 720p video that fits in memory.
"""

import os
import resource
import time

from videopython.editing import VideoEdit

PLAN = {
    "segments": [
        {
            "source": "cam1_1min.mp4",
            "start": 0,
            "end": 59.9,
            "transforms": [],
            "effects": [
                {
                    "op": "color_adjust",
                    "args": {"saturation": 0, "contrast": 1.15, "brightness": 0.02, "temperature": 0},
                },
                {"op": "volume_adjust", "args": {"volume": 1.6, "ramp_duration": 0.2}},
                {"op": "fade", "args": {"mode": "in_out", "duration": 0.6, "curve": "sqrt"}},
            ],
        }
    ],
    "post_transforms": [],
    "post_effects": [],
}

OUTPUT_FILE = "benchmark_output.mp4"


def get_peak_memory_mb():
    """Get peak RSS in MB (macOS/Linux)."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # macOS reports in bytes, Linux in KB
    if os.uname().sysname == "Darwin":
        return usage.ru_maxrss / (1024 * 1024)
    else:
        return usage.ru_maxrss / 1024


def main():
    print("=" * 60)
    print("Baseline Benchmark: cam1_1min.mp4 (720p, 25fps, 60s)")
    print("Plan: color_adjust + volume_adjust + fade")
    print("=" * 60)

    mem_before = get_peak_memory_mb()

    # Phase 1: Parse plan
    t0 = time.perf_counter()
    edit = VideoEdit.from_dict(PLAN)
    edit.validate()
    t_parse = time.perf_counter() - t0
    print(f"\nParse + validate: {t_parse:.2f}s")

    # Phase 2: Run (load + effects)
    t0 = time.perf_counter()
    result = edit.run()
    t_run = time.perf_counter() - t0
    print(f"Run (load + effects): {t_run:.2f}s")

    mem_after_run = get_peak_memory_mb()

    # Phase 3: Save
    t0 = time.perf_counter()
    result.save(OUTPUT_FILE, format="mp4", crf=20, preset="medium")
    t_save = time.perf_counter() - t0
    print(f"Save: {t_save:.2f}s")

    mem_after_save = get_peak_memory_mb()

    print(f"\nTotal: {t_parse + t_run + t_save:.2f}s")
    print(f"\nPeak memory (RSS):")
    print(f"  Before: {mem_before:.0f} MB")
    print(f"  After run: {mem_after_run:.0f} MB")
    print(f"  After save: {mem_after_save:.0f} MB")

    # Frame math for reference
    n_frames = len(result.frames)
    frame_bytes = result.frames.nbytes
    print(f"\nFrame data: {n_frames} frames, {frame_bytes / 1024 / 1024 / 1024:.2f} GB")

    # Cleanup
    if os.path.exists(OUTPUT_FILE):
        size = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
        print(f"Output file: {size:.1f} MB")
        os.remove(OUTPUT_FILE)


if __name__ == "__main__":
    main()
