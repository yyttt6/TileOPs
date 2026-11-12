"""Bracket each flushed repeat inside one profiler session; retain replayable windows."""

import collections
import hashlib
import json
import os
import struct
import time
import types
from pathlib import Path

PROTOCOL = "ascend950-marker-session-v4"
COVERAGE = "v4-bracketed-ffts-plus-named-aicpu-v1"


class UnsupportedSessionError(RuntimeError):
    """A captured topology cannot be unambiguously partitioned."""


def partition(path, repeats):
    from complete_profile import decode
    from tools import measure_baselines as base

    path = Path(path)
    whole = base._parse_ffts_profile(path, 1)
    extra = decode(path)
    tasks = whole["profile_kernel_tasks"]
    markers = sorted(
        (t for t in tasks if base._FLUSH_KERNEL_NAME_MARKER in t["name"]),
        key=lambda t: t["start_ns"],
    )
    assert len(markers) == repeats * 2, ("marker count", len(markers), repeats)
    seqs = [t["task_id"] & 65535 for t in markers]
    assert seqs == sorted(set(seqs)), "wrapped/reused task IDs unsupported"
    assert all(
        len(t["stream_ids"]) == 1 and t["name"] and not t["name"].startswith("<unknown")
        for t in tasks
    )
    streams = {s for t in tasks for s in t["stream_ids"]}
    assert len(streams) == 1, "multiple streams require the isolated profiler"
    cycles = collections.defaultdict(lambda: [0, 0])
    for p in path.rglob("ffts_profile*.slice_*"):
        if p.name.endswith(".done"):
            continue
        for v in struct.iter_unpack("<16q", p.read_bytes()):
            if v[1]:
                cycles[(v[0] >> 32) & 65535][int(v[2] >= 1 << 31)] += v[1]
    star_by_seq = collections.defaultdict(list)
    for t in extra["tasks"]:
        star_by_seq[t["seq"]].append(t)
    samples = []
    assigned = set()
    for i in range(repeats):
        left, right = markers[2 * i : 2 * i + 2]
        lo, hi = seqs[2 * i : 2 * i + 2]
        body = [t for t in tasks if lo < (t["task_id"] & 65535) < hi]
        assert all(
            left["end_ns"] <= t["start_ns"] <= t["end_ns"] <= right["start_ns"] for t in body
        ), "timestamp/sequence disagreement"
        added = [t for t in extra["added_aicpu"] if lo < t["seq"] < hi]
        if added:
            ls, rs = star_by_seq[lo], star_by_seq[hi]
            assert ls and rs, "missing marker STARS timestamps"
            assert all(
                max(t["end_ticks"] for t in ls)
                <= a["start_ticks"]
                <= a["end_ticks"]
                <= min(t["start_ticks"] for t in rs)
                for a in added
            )
        us = sum(t["duration_us"] for t in body + added)
        assert us > 0 and body + added, "empty measurement window"
        ids = {t["task_id"] & 65535 for t in body} | {t["seq"] for t in added}
        assert not assigned & ids
        assigned |= ids
        samples.append(
            dict(
                latency_ms=us / 1000,
                profile_kernel_us=us,
                profile_kernel_count=len(body) + len(added),
                profile_aic_cycles=sum(cycles[t][0] for t in ids),
                profile_aiv_cycles=sum(cycles[t][1] for t in ids),
                profile_excluded_kernel_count=0,
                profile_excluded_kernel_us=0,
                profile_kernel_tasks=body,
                added_aicpu=added,
                boundary=dict(start=left, end=right),
            )
        )
    excluded = [t for t in tasks if (t["task_id"] & 65535) not in assigned]
    # Every non-body task is retained as preparation/marker evidence, never name-filtered from a body.
    return samples, dict(samples=samples, preparation_and_markers=excluded, full_decode=extra)


def require_qualified_topology(proof):
    # Keep the established complete AICPU adapter until a real marker-session
    # AICPU control is qualified on this toolchain. Synthetic partition tests
    # establish parser invariants, not hardware qualification.
    if any(t.get("name") and t["type"] in (4, 5) for t in proof["full_decode"]["tasks"]):
        raise ValueError("Named AICPU work uses the qualified isolated profiler")


def make_timer(evidence_dir):
    from complete_profile import complete_digest
    from tools import measure_baselines as base

    evidence_dir = Path(evidence_dir)
    evidence_dir.mkdir(parents=True, exist_ok=False)

    def wrapped(torch, torch_npu, fn, inputs, warmup, repeats, reset=None):
        pending = []
        parsed_samples = []
        session_info = {}

        def one(torch, torch_npu, fn, inputs, reset, flush_kernel, flush_buffer, exclude_names):
            if not pending:
                from tilelang.profiler.msprof import get_msprof_cache_flush_kernel

                end_buffer = torch.empty(524288, dtype=torch.int32, device=flush_buffer.device)
                end_marker = get_msprof_cache_flush_kernel(524288, "int32")
                end_marker(end_buffer)
                torch.npu.synchronize()
                profile_root = evidence_dir / "raw"
                profile_root.mkdir()
                old = os.environ.get("ASCEND_WORK_PATH")
                os.environ["ASCEND_WORK_PATH"] = str(profile_root)
                started = time.monotonic()
                try:
                    with torch_npu.profiler.profile(
                        activities=[torch_npu.profiler.ProfilerActivity.NPU],
                        schedule=torch_npu.profiler.schedule(
                            wait=0, warmup=0, active=1, repeat=1, skip_first=0
                        ),
                        on_trace_ready=lambda _: None,
                        experimental_config=torch_npu.profiler._ExperimentalConfig(
                            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
                            l2_cache=False,
                            data_simplification=False,
                        ),
                    ) as prof:
                        for _ in range(repeats):
                            base._isolate_before_call(
                                torch, torch_npu, reset, flush_kernel, flush_buffer
                            )
                            out = base._call_reference(fn, inputs)
                            torch.npu.synchronize()
                            end_marker(end_buffer)
                            torch.npu.synchronize()
                            del out
                        prof.step()
                    session_info["capture_wall_seconds"] = time.monotonic() - started
                    path = Path(prof.prof_if.prof_path)
                    try:
                        parsed, proof = partition(path, repeats)
                        require_qualified_topology(proof)
                    except (AssertionError, ValueError) as exc:
                        (evidence_dir / "unsupported.json").write_text(
                            json.dumps({"reason": str(exc), "path": str(path)})
                        )
                        raise UnsupportedSessionError(str(exc)) from exc
                    session_info.update(
                        path=str(path), sha256=complete_digest(path), repeats=repeats
                    )
                    (evidence_dir / "partition.json").write_text(json.dumps(proof, indent=2))
                    parsed_samples.extend(parsed)
                    pending.extend(parsed)
                finally:
                    if old is None:
                        os.environ.pop("ASCEND_WORK_PATH", None)
                    else:
                        os.environ["ASCEND_WORK_PATH"] = old
            return pending.pop(0)

        timer = types.FunctionType(
            base._time_reference.__code__,
            dict(base._time_reference.__globals__, _time_one_isolated_repeat=one),
            base._time_reference.__name__,
            base._time_reference.__defaults__,
        )
        result = timer(torch, torch_npu, fn, inputs, warmup, repeats, reset)
        assert not pending
        result["measurement_protocol"] = PROTOCOL
        result["profile_coverage_protocol"] = COVERAGE
        result["profile_coverage_adapter"] = str(Path(__file__).resolve())
        records = [
            dict(
                kind="marker-session",
                window=i,
                path=session_info["path"],
                complete_sha256=session_info["sha256"],
                result=sample,
            )
            for i, sample in enumerate(parsed_samples)
        ]
        (evidence_dir / "coverage.json").write_text(json.dumps(records, indent=2))
        result["profile_coverage_evidence"] = str(evidence_dir / "coverage.json")
        result["single_session"] = session_info
        result["profile_coverage_adapter_sha256"] = hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest()
        return result

    return wrapped
