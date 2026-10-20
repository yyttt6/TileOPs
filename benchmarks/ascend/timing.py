"""Fixed Ascend profiler protocol shared by operators, baselines and tuning.

One call per profiler window; reset, a 1 GiB cache eviction and synchronization
precede every call outside that window. The score is the arithmetic mean of
device execution interval unions. All raw repeats are retained. No event timer is used.
"""

import contextlib
import statistics
from pathlib import Path

from benchmarks.ascend.guard import require_reservation
from benchmarks.ascend.profile_parser import _parse_ffts_profile

PROTOCOL = "ascend950-profiler-locked-v1"
WARMUP = 3
REPEATS = 15
FLUSH_BYTES = 1 << 30
FLUSH_NAME = "_tilelang_profiler_cache_flush"


@contextlib.contextmanager
def _profile(directory):
    import torch_npu

    directory.mkdir(parents=True, exist_ok=False)
    handler = torch_npu.profiler.tensorboard_trace_handler(
        str(directory.resolve()), analyse_flag=False
    )
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.NPU],
        schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1, skip_first=0),
        on_trace_ready=handler,
        experimental_config=torch_npu.profiler._ExperimentalConfig(
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
            l2_cache=False,
            data_simplification=False,
        ),
    ) as prof:
        yield prof


class ProfilerTimer:
    """A timer scoped to one guarded run and one already applied clock policy."""

    def __init__(self, directory: Path, clocks):
        require_reservation()
        import torch
        from tilelang.profiler.msprof import get_msprof_cache_flush_kernel

        self.directory = directory
        self.clocks = clocks
        self.flush_buffer = torch.empty(FLUSH_BYTES // 4, dtype=torch.int32, device="npu")
        self.flush = get_msprof_cache_flush_kernel(FLUSH_BYTES // 4, "int32")
        # Resolve the actual compiled name before trusting contamination detection.
        with _profile(directory / "flush-check") as prof:
            self.flush(self.flush_buffer)
            torch.npu.synchronize()
            prof.step()
        _parse_ffts_profile(
            Path(prof.prof_if.prof_path),
            1,
            exclude_name_marker=FLUSH_NAME,
            require_marker_seen=True,
        )

    def _prepare(self, reset):
        import torch

        if reset is not None:
            reset()
        self.flush(self.flush_buffer)
        torch.npu.synchronize()
        self.clocks.check()

    def measure(self, fn, args=(), *, reset=None, name="call", search=False):
        """Measure a verified callable; short search records never carry the formal protocol."""
        require_reservation()
        import torch

        warmup, repeats = (1, 3) if search else (WARMUP, REPEATS)
        for _ in range(warmup):
            self._prepare(reset)
            output = fn(*args)
            torch.npu.synchronize()
            del output
        records = []
        for index in range(repeats):
            self._prepare(reset)
            with _profile(self.directory / name / f"repeat-{index:02d}") as prof:
                output = fn(*args)
                torch.npu.synchronize()
                prof.step()
            del output
            self.clocks.check()
            parsed = _parse_ffts_profile(
                Path(prof.prof_if.prof_path), 1, exclude_name_marker=FLUSH_NAME
            )
            if (
                parsed["profile_excluded_kernel_count"]
                or parsed["profile_kernel_count"] <= 0
                or parsed["latency_ms"] <= 0
            ):
                raise RuntimeError(
                    "Profiler window contains flush work or no positive operator activity"
                )
            records.append(parsed)
        values = [r["latency_ms"] for r in records]
        return {
            "protocol": PROTOCOL if not search else PROTOCOL + "-search",
            "profiler": "torch_npu.profiler.profile",
            "metric": "mean_device_busy_union_ms",
            "warmup": warmup,
            "repeats": repeats,
            "flush_bytes": FLUSH_BYTES,
            "latency_ms": statistics.mean(values),
            "samples_ms": values,
            "median_ms": statistics.median(values),
            "min_ms": min(values),
            "max_ms": max(values),
            "stdev_ms": statistics.stdev(values),
            "records": records,
        }
