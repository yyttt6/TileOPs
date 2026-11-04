#!/usr/bin/env python3
"""Measure the manifest reference baseline on Ascend, with resumable JSONL output.

This is deliberately separate from ``TileOPs/benchmarks``.  That benchmark package
uses CUDA events/CUPTI and refuses an NPU device.  The manifest reference callable is
the first real baseline candidate: its ``ref_api`` is recorded beside every result.
Additional baseline adapters can be added later without changing the result schema.

The command must run in the sourced Ascend environment and under ``npu-guard``:

  source env.sh
  PYTHONPATH=TileOPs/src:TileOPs python tools/measure_baselines.py \
      --output docs/ascend950/results/baseline_reference.jsonl

Every case is appended as one JSON object.  Existing completed case ids are skipped,
so an interrupted run can be resumed safely.  A failed case is also persisted; use
``--retry-failed`` to explicitly retry it.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import math
import os
import platform
import statistics
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TILEOPS_ROOT = ROOT / "TileOPs"
from tools.baseline_protocol import (  # noqa: E402
    PROTOCOL,
    PROTOCOL_ID,
    TERMINAL_FAILURES,
    current_row,
    provenance,
)


def _jsonable(value: Any) -> Any:
    """Convert report values to stable JSON values without importing extra libraries."""
    if value is None or isinstance(value, (bool, int, float, str)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return str(value)
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "item"):
        try:
            return _jsonable(value.item())
        except Exception:
            pass
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    return repr(value)


def _git_head(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def _environment(torch, torch_npu) -> dict[str, Any]:
    device_name = None
    device_count = None
    try:
        device_count = torch.npu.device_count()
        if device_count:
            device_name = torch.npu.get_device_name(0)
    except Exception:
        pass
    roots = [
        "ascend_baselines/DeepEP-Ascend",
        "ascend_baselines/DeepGEMM-Ascend",
        "ascend_baselines/DeepSelect",
        "ascend_baselines/FlashMLA",
        "ascend_baselines/TileKernels",
        "ascend_baselines/atvc",
        "ascend_baselines/catlass",
        "ascend_baselines/flash-linear-attention-npu",
        "ascend_baselines/ops-fft",
        "ascend_baselines/ops-math",
        "ascend_baselines/ops-nn",
        "ascend_baselines/ops-tensor",
        "ascend_baselines/ops-transformer",
        "ascend_baselines/sgl-kernel-npu",
        "ascend_baselines/tilelang-ascend",
        "ascend_baselines/vllm-ascend",
    ]
    return {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": platform.node(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": getattr(torch, "__version__", None),
        "torch_npu": getattr(torch_npu, "__version__", None),
        "device": "npu:0",
        "device_name": device_name,
        "device_count": device_count,
        "ascend_npu_arch": os.environ.get("ASCEND_NPU_ARCH"),
        "soc_version": os.environ.get("SOC_VERSION"),
        "cann_home": os.environ.get("ASCEND_HOME_PATH"),
        "tileops_head": _git_head(TILEOPS_ROOT),
        "tilelang_head": _git_head(ROOT / "tilelang"),
        "baseline_heads": {
            str(ROOT / p): _git_head(ROOT / p) for p in roots if (ROOT / p).exists()
        },
        "measurement_protocol": PROTOCOL,
        "timing": {
            "clock": "torch_npu.profiler.profile raw ffts_profile records",
            "latency": "sum of profiled AIC/AIV kernel durations divided by repetitions",
            "includes": "the candidate callable and its device-side kernel work",
            "excludes": "Python launch gaps and input generation; first-call host wall time is separately recorded as cold_latency_ms",
        },
    }


def _load_done(path: Path, retry_failed: bool) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            if not current_row(row):
                continue
            if row.get("status") == "invalidated_external_activity":
                done.discard(f"{row.get('op')}::{row.get('case_id')}")
                continue
            if row.get("status") in TERMINAL_FAILURES and not retry_failed:
                done.add(f"{row.get('op')}::{row.get('case_id')}")
                continue
            case_id = row.get("case_id")
            op = row.get("op")
            if not case_id or not op:
                continue
            if retry_failed and row.get("status") not in {"measured", "measured_reference"}:
                continue
            # A device fault is a measured outcome of this candidate/shape, but it
            # cannot be retried in the poisoned process that observed it.  A fresh
            # process will skip it and continue with the next case.
            if row.get("status") not in {
                "measured",
                "measured_reference",
                "measured_reference_input_mutated",
                "device_fault",
                "no_reference",
            }:
                continue
            done.add(f"{op}::{case_id}")
    return done


def _append(path: Path, row: dict[str, Any]) -> None:
    provenance(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_jsonable(row), ensure_ascii=False, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _tensor_digest(inputs: tuple[Any, ...]) -> str | None:
    """A cheap input fingerprint to expose accidental in-place baseline mutation."""
    h = hashlib.sha256()
    seen = False
    for value in inputs:
        if value is None or not hasattr(value, "detach"):
            h.update(repr(value).encode())
            continue
        try:
            # Keep the bounded check on-device until the sample is copied.  The old
            # implementation copied every complete input tensor to host memory even
            # though it only hashed the first 4096 elements; this created substantial
            # host/device traffic between candidates without improving the mutation
            # signal.
            t = value.detach()
            flat = t.reshape(-1)
            sample = flat[: min(flat.numel(), 4096)].to("cpu")
            h.update(str(tuple(t.shape)).encode())
            h.update(str(t.dtype).encode())
            h.update(sample.numpy().tobytes())
            seen = True
        except Exception as exc:
            h.update(f"digest-error:{type(exc).__name__}".encode())
    return h.hexdigest() if seen else None


def _call_reference(fn, inputs: tuple[Any, ...]):
    return fn(*inputs)


# Finding 1 (baseline_measurement_audit.md): the flush kernel's own compiled name. Lifted
# from tilelang/tilelang/profiler/msprof.py's bench_with_msprof, which already relies on this
# exact substring to keep the flush kernel's device time out of what it reports; a kernel
# whose resolved name is new on this machine (toolchain version drift) must raise rather than
# silently keep counting the flush as if it were the timed op, so this marker is also used to
# require that resolution actually found the flush kernel (see the "sanity" check below).
_FLUSH_KERNEL_NAME_MARKER = "_tilelang_profiler_cache_flush"
# int32 ("fast flush"), sized 1 GiB after an on-device sweep (not do_bench's generic 256 MiB
# default, which under-flushes here): at 16M-float16, 256/1024/2048 MiB flush buffers measured
# 5.78/5.45/5.43 TB/s implied bandwidth respectively (lower is more-flushed) -- 1 GiB captures
# most of the available improvement and 2 GiB adds nothing further, so 1 GiB is the point of
# diminishing returns, not an arbitrary choice.
_FLUSH_BUFFER_MB = 1024
_flush_kernel_cache: dict[int, tuple[Any, Any]] = {}


def _get_flush_kernel_and_buffer(torch, torch_npu):
    """Build (once per device, cached) the Ascend cache-flush kernel and its buffer.

    ``get_msprof_cache_flush_kernel`` is `tilelang`'s own primitive, already used by
    ``tilelang.profiler.do_bench(backend="msprof_detail")`` -- not reimplemented here.
    Caching avoids re-JITing/reallocating on every one of the hundreds of cases a chunked
    sweep measures; ``tilelang.jit`` itself also caches by (numel, dtype), so this is a second,
    cheaper layer that also avoids paying a 256 MiB allocation per case.
    """
    device_index = torch.npu.current_device()
    cached = _flush_kernel_cache.get(device_index)
    if cached is not None:
        return cached
    from tilelang.profiler.msprof import get_msprof_cache_flush_kernel

    numel = (_FLUSH_BUFFER_MB * 1024 * 1024) // 4
    buffer = torch.empty(numel, dtype=torch.int32, device=torch.device("npu", device_index))
    kernel = get_msprof_cache_flush_kernel(numel, "int32")
    _flush_kernel_cache[device_index] = (kernel, buffer)
    return kernel, buffer


def _resolve_kernel_names(prof_path: Path) -> dict[int, str]:
    """Map each raw record's sequence number to its kernel name.

    Same hash_dic/task_track parsing ``tilelang/tilelang/profiler/msprof.py::bench_with_msprof``
    already does to exclude its own flush kernel by name; ported here because
    ``_parse_ffts_profile`` otherwise has no way to tell the flush kernel's or a ``reset()``
    callback's device records apart from the timed op's (Findings 1 and 2).
    """
    hash_to_name: dict[int, str] = {}
    for path in prof_path.rglob("*hash_dic.slice_*"):
        if path.name.endswith(".done"):
            continue
        text = path.read_bytes().decode("utf-8", errors="replace")
        for line in text.strip().split("\n"):
            parts = line.split(":", 1)
            if len(parts) == 2:
                try:
                    h = int(parts[0])
                    if h >= 2**63:
                        h -= 2**64
                    hash_to_name[h] = parts[1]
                except ValueError:
                    pass

    seq_to_name: dict[int, str] = {}
    for path in prof_path.rglob("*task_track.slice_*"):
        if path.name.endswith(".done"):
            continue
        data = path.read_bytes()
        for offset in range(0, len(data) - (len(data) % 64), 64):
            values = struct.unpack_from("<8q", data, offset)
            seq = (values[3] >> 32) & 0xFFFF
            name_hash = values[5]
            if name_hash != 0:
                seq_to_name[seq] = hash_to_name.get(name_hash, f"<unknown {name_hash}>")
    return seq_to_name


def _profiled_kernel_names(torch, torch_npu, fn, repeats: int = 2) -> frozenset[str]:
    """Run *fn* (no args) ``repeats`` times in its own profiling window; return its kernel names.

    Used to learn which device kernels a ``reset()`` callback launches, so the real timing
    window below can exclude them by name (Finding 2) the same way it excludes the flush
    kernel. Two repeats by default so a one-off first-call dispatch kernel is not mistaken for
    the only name reset ever uses; the union of both is kept; over-excluding a name reset
    happens to share with the timed op is the conservative direction of error here, since the
    bug being fixed is reset time leaking into the op's reading, not the reverse.
    """
    with tempfile.TemporaryDirectory(prefix="tileops_baseline_probe_") as prof_dir:
        old_work_path = os.environ.get("ASCEND_WORK_PATH")
        os.environ["ASCEND_WORK_PATH"] = prof_dir
        try:
            with (
                contextlib.redirect_stdout(open(os.devnull, "w")),
                contextlib.redirect_stderr(open(os.devnull, "w")),
                torch_npu.profiler.profile(
                    activities=[torch_npu.profiler.ProfilerActivity.NPU],
                    schedule=torch_npu.profiler.schedule(
                        wait=0, warmup=0, active=1, repeat=1, skip_first=0
                    ),
                    on_trace_ready=lambda _prof: None,
                    experimental_config=torch_npu.profiler._ExperimentalConfig(
                        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
                        l2_cache=False,
                        data_simplification=False,
                    ),
                ) as prof,
            ):
                for _ in range(repeats):
                    fn()
                torch.npu.synchronize()
                prof.step()
            seq_to_name = _resolve_kernel_names(Path(prof.prof_if.prof_path))
        finally:
            if old_work_path is None:
                os.environ.pop("ASCEND_WORK_PATH", None)
            else:
                os.environ["ASCEND_WORK_PATH"] = old_work_path
    return frozenset(seq_to_name.values())


def _parse_ffts_profile(
    prof_path,
    repeats,
    exclude_name_marker=None,
    exclude_names=frozenset(),
    require_marker_seen=False,
):
    from tools.kernel_task_profile import parse_profile

    return parse_profile(
        prof_path,
        repeats,
        seq_to_name=_resolve_kernel_names(prof_path),
        exclude_name_marker=exclude_name_marker,
        exclude_names=exclude_names,
        require_marker_seen=require_marker_seen,
    )


# Fix 1b: devices already self-tested this process (see _verify_flush_marker_resolvable).
_flush_marker_verified: set[int] = set()


def _verify_flush_marker_resolvable(torch, torch_npu, flush_kernel, flush_buffer) -> None:
    """One-time-per-device-per-process self-test that kernel name resolution still works.

    Fix 1 (see the old, pre-Fix-1b ``_parse_ffts_profile``) used to require that every
    timed window saw the flush kernel at least once, because the flush used to run
    *inside* the one continuous profiling session. Fix 1b moves the flush strictly
    *outside* each per-repeat profiling window (that is the whole point -- see
    ``_time_reference``), so a per-repeat window is now expected to *never* see it; that
    old assertion would fire on every single call under the new design and would no
    longer be testing anything real.

    The underlying concern Fix 1 was guarding against -- a CANN/toolchain update that
    silently breaks the ``hash_dic``/``task_track`` name-resolution path, making every
    exclusion in this file silently unreliable -- is still worth catching loudly. This
    function preserves that check by reproducing Fix 1's old structure exactly, but only
    once per device per process (not once per repeat): run the flush kernel together with
    a cheap dummy op inside one profiling window, and require the flush kernel's name to
    resolve at least once. Paid once per process, not once per repeat, so it does not add
    to the new design's already-larger per-case overhead.
    """
    device_index = torch.npu.current_device()
    if device_index in _flush_marker_verified:
        return
    with tempfile.TemporaryDirectory(prefix="tileops_baseline_selftest_") as prof_dir:
        old_work_path = os.environ.get("ASCEND_WORK_PATH")
        os.environ["ASCEND_WORK_PATH"] = prof_dir
        try:
            with (
                contextlib.redirect_stdout(open(os.devnull, "w")),
                contextlib.redirect_stderr(open(os.devnull, "w")),
                torch_npu.profiler.profile(
                    activities=[torch_npu.profiler.ProfilerActivity.NPU],
                    schedule=torch_npu.profiler.schedule(
                        wait=0, warmup=0, active=1, repeat=1, skip_first=0
                    ),
                    on_trace_ready=lambda _prof: None,
                    experimental_config=torch_npu.profiler._ExperimentalConfig(
                        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
                        l2_cache=False,
                        data_simplification=False,
                    ),
                ) as prof,
            ):
                flush_kernel(flush_buffer)
                torch.npu.synchronize()
                prof.step()
            _parse_ffts_profile(
                Path(prof.prof_if.prof_path),
                repeats=1,
                exclude_name_marker=_FLUSH_KERNEL_NAME_MARKER,
                require_marker_seen=True,
            )
        finally:
            if old_work_path is None:
                os.environ.pop("ASCEND_WORK_PATH", None)
            else:
                os.environ["ASCEND_WORK_PATH"] = old_work_path
    _flush_marker_verified.add(device_index)


def _isolate_before_call(torch, torch_npu, reset, flush_kernel, flush_buffer) -> None:
    """Run ``reset()`` (if any) then the flush kernel, and fully drain the device.

    Shared by every call ``_time_reference`` makes -- the cold one, every warmup repeat,
    and (critically, see ``_time_one_isolated_repeat``) each timed repeat individually.
    ``reset`` runs *before* the flush, not after, matching
    ``TileOPs/benchmarks/timing.py::_bench_kernel``'s ``_reset_and_flush`` order --
    otherwise reset's own writes (e.g. a ``.copy_()``) would land in cache just before the
    op runs and partially undo the flush for the one buffer that matters most.

    The explicit ``torch.npu.synchronize()`` after the flush is the piece Fix 1's own
    "honest limitation" section identified as the likely missing ingredient (Fix 1b): the
    old one-continuous-session design queued every repeat's reset/flush/op back-to-back
    on the same stream with *no* host-device drain between repeats, which
    `docs/ascend950/results/16m_256m_artifact_investigation.md`'s own
    ``per_rep_flushed_ms`` primitive never did -- it synchronizes after every single
    flush, before timing the next call.
    """
    if reset is not None:
        reset()
    flush_kernel(flush_buffer)
    torch.npu.synchronize()


@contextlib.contextmanager
def _profile_directory():
    retained = os.environ.get("TILEOPS_PROFILE_DIR")
    if retained:
        Path(retained).mkdir(parents=True, exist_ok=True)
        yield tempfile.mkdtemp(prefix="repeat_", dir=retained)
    else:
        with tempfile.TemporaryDirectory(prefix="tileops_baseline_rep_") as directory:
            yield directory


def _time_one_isolated_repeat(
    torch,
    torch_npu,
    fn,
    inputs: tuple[Any, ...],
    reset,
    flush_kernel,
    flush_buffer,
    exclude_names: frozenset[str],
) -> dict[str, float]:
    """Time exactly one call, fully isolated: flush -> sync -> profiled call -> teardown.

    This is Fix 1b. Two designs were tried and measured before this one, and the reasons
    both were rejected are recorded here because they are not obvious and a future editor
    could easily reintroduce either mistake:

    1. **NPU-event pair, no profiler** (``start_event.record(); fn(); end_event.record()``),
       ported directly from `docs/ascend950/results/16m_256m_artifact_investigation.md`
       section 2's ``per_rep_flushed_ms`` primitive, exactly as that doc describes "the
       correct way to do this". Measured directly (see the Fix 1b validation notes): for a
       16M-class case, this technique's reading is highly unstable across runs (observed
       ~0.06-0.18ms for the *same* op/case across repeated measurements) and, within one
       isolated repeat, disagrees with that same repeat's own raw hardware kernel-duration
       record by 6-14x -- while the two agree within ~15% for the 256M case. That pattern
       (large, run-to-run-variable disagreement exactly where the real kernel is tiny;
       close agreement once the kernel is large enough to dominate) is the signature of a
       roughly fixed-size timing artifact, not a cache-residency effect that should scale
       with working-set size the way genuine HBM-vs-cache behavior does. Direct confirmation:
       inserting an inert delay with *no NPU work at all* (a bare ``time.sleep()``, or an
       idle ``tempfile.TemporaryDirectory`` create/write/delete) between the flush/sync and
       the event bracket inflates the "latency" in proportion to the delay, with the cache
       state and the op itself completely unchanged. The most likely mechanism: the NPU's
       own clock/power governor downclocks during a host-side gap with no queued device
       work, and a freshly-launched kernel pays a clock ramp-up cost before it actually
       starts executing -- a cost an NPU *event* pair (which brackets wall/device time from
       before that ramp-up to after the kernel finishes) counts, but a kernel's own
       hardware-recorded start/end timestamps (below) structurally cannot, since the
       hardware does not start that clock until the kernel is actually running. This also
       means the investigation's own published 2.48-2.88x figure is now understood to
       likely include a meaningful share of this same artifact, not pure cache physics --
       see the Fix 1b summary for the full evidence trail and what this implies for the
       "2.48-2.88x" reference figure the task was originally scoped against.
    2. **Profiler session wrapping only the bracketed call, in the main timed loop, per
       repeat** -- correct in principle, but 15 full ``torch_npu.profiler.profile()``
       open/close cycles per case (one per timed repeat) measured at ~14-15x the pre-Fix-1b
       wall-clock cost for one case. That overhead is unavoidable if every repeat is to be
       genuinely isolated and profiler-measured (there is no cheaper way to get a hardware
       per-kernel timestamp without a profiling session), so it is accepted here, not
       avoided -- see the Fix 1b summary's wall-clock multiplier measurement.

    What this function actually does, and why it is correct: flush, synchronize (drains
    the flush and leaves the device idle immediately before the call -- Fix 1's ordering,
    now per-repeat rather than once for the whole case), then open one profiling window
    around *only* the bracketed call, close it, and read the kernel's own hardware-recorded
    start/end timestamps from the raw ``ffts_profile`` records (the same source Fix 1
    already used, and the same source `tools/measure_baselines.py`'s documented timing
    contract already promises: "excludes Python launch gaps"). Because the flush (and
    ``reset``) already ran and were fully drained *before* this window opens, the window
    should see nothing but the op's own kernels; Fix 1/2's name-based exclusion
    (``exclude_name_marker``/``exclude_names``) is kept as a defense-in-depth diagnostic (it
    should now almost always report zero excluded records per repeat), not because it is
    still structurally required to produce a correct number.
    """
    _isolate_before_call(torch, torch_npu, reset, flush_kernel, flush_buffer)
    with _profile_directory() as prof_dir:
        old_work_path = os.environ.get("ASCEND_WORK_PATH")
        os.environ["ASCEND_WORK_PATH"] = prof_dir
        try:
            with (
                contextlib.redirect_stdout(open(os.devnull, "w")),
                contextlib.redirect_stderr(open(os.devnull, "w")),
                torch_npu.profiler.profile(
                    activities=[torch_npu.profiler.ProfilerActivity.NPU],
                    schedule=torch_npu.profiler.schedule(
                        wait=0, warmup=0, active=1, repeat=1, skip_first=0
                    ),
                    on_trace_ready=lambda _prof: None,
                    experimental_config=torch_npu.profiler._ExperimentalConfig(
                        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
                        l2_cache=False,
                        data_simplification=False,
                    ),
                ) as prof,
            ):
                out = _call_reference(fn, inputs)
                torch.npu.synchronize()
                prof.step()
            parsed = _parse_ffts_profile(
                Path(prof.prof_if.prof_path),
                repeats=1,
                exclude_name_marker=_FLUSH_KERNEL_NAME_MARKER,
                # reset was drained BEFORE the window. Filtering its kernel names here
                # would delete legitimate op kernels that happen to use the same name.
                exclude_names=frozenset(),
            )
        finally:
            if old_work_path is None:
                os.environ.pop("ASCEND_WORK_PATH", None)
            else:
                os.environ["ASCEND_WORK_PATH"] = old_work_path
    if os.environ.get("TILEOPS_PROFILE_DIR"):
        from tools.kernel_task_profile import trace_digest

        parsed["profile_path"] = str(prof.prof_if.prof_path)
        parsed["profile_trace_sha256"] = trace_digest(parsed["profile_path"])
    del out
    if parsed.get("profile_excluded_kernel_count", 0) != 0:
        raise RuntimeError("cache flush unexpectedly appeared inside isolated timing window")
    if parsed["profile_kernel_count"] <= 0 or parsed["latency_ms"] <= 0:
        raise RuntimeError("isolated profile has no positive-duration operator kernels")
    return parsed


def _time_reference(
    torch,
    torch_npu,
    fn,
    inputs: tuple[Any, ...],
    warmup: int,
    repeats: int,
    reset: Any = None,
):
    """Profile warm execution with the project-wide NPU profiling protocol.

    ``reset``, when given, restores state *fn* overwrites (private pages, a recurrent
    state buffer, ...) -- the same role it plays in ``benchmarks.api.Implementation``. It
    runs before every call, including the cold one, exactly like the pre-existing
    ``invoke()`` wrappers used to compose it into *fn* themselves; the difference (Fix 2)
    is that ``_time_reference`` calls it directly so its device time can be identified and
    excluded, rather than being baked into *fn* where nothing downstream could tell it
    apart from the op's own work.

    Fix 1b: every call -- the cold one, every warmup repeat, and every timed repeat -- now
    runs in its own isolated measurement window (flush, synchronize, call, synchronize),
    not inside one continuous profiling session spanning all of them. Warmup repeats are
    isolated the same way as timed repeats, for two reasons: (1) consistency -- a warmup
    repeat that is allowed to run warm (no flush/sync boundary) is not representative of
    the cache state every timed repeat will actually see, so it would warm up the "wrong"
    (friendlier) condition; (2) the JIT/dispatch-cache warming warmup exists for is
    unaffected by whether the surrounding cache is hot or cold, so isolating warmup costs
    nothing but a few extra synchronize() calls.

    Every timed repeat (``_time_one_isolated_repeat``) opens and tears down its own
    profiling window around exactly the bracketed call, and reports that repeat's
    hardware-recorded kernel duration -- not an NPU event pair, and not one profiling
    window shared across repeats. See ``_time_one_isolated_repeat``'s docstring for why
    both of those alternatives were tried and rejected during this fix's own validation.
    This is the materially more expensive design the task's cost-awareness section asked
    to have measured (roughly 15 profiler sessions per case instead of 1); see the Fix 1b
    summary for the measured wall-clock multiplier.
    """
    flush_kernel, flush_buffer = _get_flush_kernel_and_buffer(torch, torch_npu)
    _verify_flush_marker_resolvable(torch, torch_npu, flush_kernel, flush_buffer)
    reset_kernel_names = frozenset()  # reset is excluded structurally, never by name

    # First-call cost is recorded for cache/JIT diagnostics, but never used as the
    # steady-state opponent latency.
    cold_start = time.perf_counter()
    _isolate_before_call(torch, torch_npu, reset, flush_kernel, flush_buffer)
    out = _call_reference(fn, inputs)
    torch.npu.synchronize()
    cold_ms = (time.perf_counter() - cold_start) * 1000.0
    del out

    for _ in range(warmup):
        _isolate_before_call(torch, torch_npu, reset, flush_kernel, flush_buffer)
        out = _call_reference(fn, inputs)
        torch.npu.synchronize()
        del out

    total_kernel_ns = 0.0
    total_kernel_count = 0.0
    total_aic_cycles = 0.0
    total_aiv_cycles = 0.0
    total_excluded_ns = 0.0
    total_excluded_count = 0.0
    samples_ms = []
    retained_profiles = []
    for _ in range(repeats):
        parsed = _time_one_isolated_repeat(
            torch, torch_npu, fn, inputs, reset, flush_kernel, flush_buffer, reset_kernel_names
        )
        samples_ms.append(parsed["latency_ms"])
        if "profile_path" in parsed:
            retained_profiles.append(
                {
                    k: parsed[k]
                    for k in ["profile_path", "profile_trace_sha256", "profile_kernel_tasks"]
                }
            )
        total_kernel_ns += parsed["profile_kernel_us"] * 1e3
        total_kernel_count += parsed["profile_kernel_count"]
        total_aic_cycles += parsed["profile_aic_cycles"]
        total_aiv_cycles += parsed["profile_aiv_cycles"]
        total_excluded_ns += parsed.get("profile_excluded_kernel_us", 0.0) * 1e3
        total_excluded_count += parsed.get("profile_excluded_kernel_count", 0.0)

    result = {
        # Authoritative latency: mean, across `repeats` independently-isolated windows, of
        # each window's own hardware-recorded kernel duration (Fix 1b) -- replacing the old
        # "sum of raw ffts_profile records from one continuous session, divided by repeats"
        # value with the same kind of number, now genuinely isolated per repeat rather than
        # pooled across all of them.
        "latency_ms": total_kernel_ns / repeats / 1e6,
        "profile_kernel_us": total_kernel_ns / repeats / 1e3,
        "profile_kernel_count": total_kernel_count / repeats,
        "profile_aic_cycles": total_aic_cycles / repeats,
        "profile_aiv_cycles": total_aiv_cycles / repeats,
        "profile_excluded_kernel_count": total_excluded_count / repeats,
        "profile_excluded_kernel_us": total_excluded_ns / repeats / 1e3,
        "cold_latency_ms": cold_ms,
        "latency_samples_ms": samples_ms,
        "retained_profiles": retained_profiles,
        "latency_median_ms": statistics.median(samples_ms),
        "latency_stdev_ms": statistics.stdev(samples_ms) if len(samples_ms) > 1 else 0.0,
        "latency_min_ms": min(samples_ms),
        "latency_max_ms": max(samples_ms),
        "measurement_protocol": PROTOCOL_ID,
    }
    if reset is not None:
        result["profile_reset_kernel_names"] = sorted(reset_kernel_names)
    return result
