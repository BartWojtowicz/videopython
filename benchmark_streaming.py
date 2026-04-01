"""Streaming benchmark: measure time and memory for VideoEdit.run_to_file() on cam1_1min.mp4.

Compares streaming pipeline vs eager (run + save) on the same plan.
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


def get_peak_memory_mb():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    if os.uname().sysname == "Darwin":
        return usage.ru_maxrss / (1024 * 1024)
    else:
        return usage.ru_maxrss / 1024


def main():
    print("=" * 60)
    print("Streaming Benchmark: cam1_1min.mp4 (720p, 25fps, 60s)")
    print("Plan: color_adjust + volume_adjust + fade")
    print("=" * 60)

    mem_before = get_peak_memory_mb()

    edit = VideoEdit.from_dict(PLAN)
    edit.validate()

    output_file = "benchmark_streaming_output.mp4"

    t0 = time.perf_counter()
    edit.run_to_file(output_file, format="mp4", crf=20, preset="medium")
    t_total = time.perf_counter() - t0

    mem_after = get_peak_memory_mb()

    print(f"\nrun_to_file: {t_total:.2f}s")
    print(f"\nPeak memory (RSS):")
    print(f"  Before: {mem_before:.0f} MB")
    print(f"  After: {mem_after:.0f} MB")

    if os.path.exists(output_file):
        size = os.path.getsize(output_file) / (1024 * 1024)
        print(f"\nOutput file: {size:.1f} MB")
        os.remove(output_file)


if __name__ == "__main__":
    main()
