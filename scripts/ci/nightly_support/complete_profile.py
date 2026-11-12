"""Logical reductions: retain v3 distinct-task timing and add missing named AICPU tasks.

CPU-only decoder follows CANN 9.2 AcsqTaskV6 (4HQHH3L), StarsCommon
unique task ID, ChipV6SqeType, and info.json HWTS MHz. No installed/global
profiler code is changed. Unmatched or uncovered named tasks fail closed.
"""

import hashlib
import json
import shutil
import struct
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def names_for(path):
    from tools.measure_baselines import _resolve_kernel_names

    return _resolve_kernel_names(path)


def decode(path):
    path = Path(path)
    names = names_for(path)
    ffts = set()
    for f in path.rglob("ffts_profile*.slice_*"):
        if f.name.endswith(".done"):
            continue
        data = f.read_bytes()
        assert len(data) % 128 == 0
        for v in struct.iter_unpack("<16q", data):
            if v[1] and v[15] >= v[14]:
                ffts.add((v[0] >> 32) & 65535)
    tasks = []
    for f in path.rglob("stars_soc.data.*.slice_*"):
        if f.name.endswith(".done"):
            continue
        info = json.loads((f.parent.parent / "info.json.0").read_text())
        assert str(info["platform_version"]) == "15", "decoder limited to verified V6 platform"
        mhz = float(info["DeviceInfo"][0]["hwts_frequency"])
        assert mhz > 0
        data = f.read_bytes()
        assert len(data) % 32 == 0
        pending = {}
        for v in struct.iter_unpack("<4HQHH3L", data):
            assert v[1] == 0x6BD3
            func = v[0] & 63
            if func not in (0, 1):
                continue
            seq = v[2] + (v[3] << 16)
            kind = v[0] >> 10
            key = (seq, kind)
            if func == 0:
                assert key not in pending
                pending[key] = v[4]
            else:
                assert key in pending, ("unmatched end", key)
                start = pending.pop(key)
                assert v[4] >= start
                tasks.append(
                    dict(
                        seq=seq,
                        type=kind,
                        name=names.get(seq),
                        duration_us=(v[4] - start) / mhz,
                        start_ticks=start,
                        end_ticks=v[4],
                        ffts_counted=seq in ffts,
                    )
                )
        assert not pending, ("unmatched start", pending)
    covered = ffts | {t["seq"] for t in tasks if t["type"] in (4, 5)}
    missing = {seq: name for seq, name in names.items() if seq not in covered}
    assert not missing, ("uncovered tracked kernels", missing)
    additions = [t for t in tasks if t["name"] and not t["ffts_counted"]]
    assert all(t["type"] in (4, 5) for t in additions), additions
    return dict(
        tasks=tasks,
        ffts_sequences=sorted(ffts),
        added_aicpu=additions,
        added_us=sum(t["duration_us"] for t in additions),
    )


def complete_digest(root):
    root = Path(root)
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if p.is_file() and not p.name.endswith((".done", ".complete")):
            h.update(str(p.relative_to(root)).encode())
            h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


def pure_aicpu_base(extra):
    assert not extra["ffts_sequences"] and extra["added_aicpu"], "No complete operator evidence"
    assert len({(t["seq"], t["type"]) for t in extra["added_aicpu"]}) == len(extra["added_aicpu"])
    assert extra["added_us"] > 0
    return dict(
        latency_ms=0.0,
        profile_kernel_us=0.0,
        profile_kernel_count=0.0,
        profile_raw_record_count=0.0,
        profile_aic_cycles=0.0,
        profile_aiv_cycles=0.0,
        profile_excluded_kernel_count=0.0,
        profile_excluded_kernel_us=0.0,
        profile_kernel_tasks=[],
        pure_aicpu_window=True,
    )


def make_isolated_timer(evidence_dir):
    from tools import measure_baselines as base

    evidence_dir = Path(evidence_dir)
    evidence_dir.mkdir(parents=True, exist_ok=False)
    records = []

    def parse(
        path,
        repeats,
        exclude_name_marker=None,
        exclude_names=frozenset(),
        require_marker_seen=False,
    ):
        extra = decode(path)
        try:
            result = base._parse_ffts_profile(
                path, repeats, exclude_name_marker, exclude_names, require_marker_seen
            )
        except ValueError as exc:
            if str(exc) != "No valid FFTS activity":
                raise
            assert not require_marker_seen, "Do not bypass marker validation"
            result = pure_aicpu_base(extra)
        additions = [
            t
            for t in extra["added_aicpu"]
            if t["name"] not in exclude_names
            and not (exclude_name_marker and exclude_name_marker in t["name"])
        ]
        delta = sum(t["duration_us"] for t in additions)
        if result.get("pure_aicpu_window"):
            assert delta > 0
        result["latency_ms"] += delta / repeats / 1000
        result["profile_kernel_us"] += delta / repeats
        result["profile_kernel_count"] += len(additions) / repeats
        saved = evidence_dir / f"sample-{len(records):03d}"
        shutil.copytree(path, saved)
        records.append(
            dict(
                original_path=str(Path(path).resolve()),
                path=str(saved),
                complete_sha256=complete_digest(saved),
                decoded=extra,
                result=result,
            )
        )
        (evidence_dir / "coverage.json").write_text(json.dumps(records, indent=2) + "\n")
        return result

    one_globals = dict(base._time_one_isolated_repeat.__globals__, _parse_ffts_profile=parse)
    one = types.FunctionType(
        base._time_one_isolated_repeat.__code__,
        one_globals,
        base._time_one_isolated_repeat.__name__,
        base._time_one_isolated_repeat.__defaults__,
    )
    timer_globals = dict(base._time_reference.__globals__, _time_one_isolated_repeat=one)
    timer = types.FunctionType(
        base._time_reference.__code__,
        timer_globals,
        base._time_reference.__name__,
        base._time_reference.__defaults__,
    )

    def wrapped(*args, **kwargs):
        result = timer(*args, **kwargs)
        result["profile_coverage_protocol"] = "v3-distinct-ffts-plus-missing-named-aicpu-v1"
        result["profile_coverage_adapter"] = str(Path(__file__).resolve())
        result["profile_coverage_adapter_sha256"] = hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest()
        result["profile_coverage_evidence"] = str(evidence_dir / "coverage.json")
        return result

    return wrapped


def make_timer(evidence_dir, mode="auto"):
    if mode == "isolated":
        return make_isolated_timer(evidence_dir)
    if mode not in ("auto", "session"):
        raise ValueError(f"Unknown timer mode: {mode}")
    from session_profile import make_timer as session_timer

    return session_timer(evidence_dir)


def replay_coverage(measurement):
    """Replay every retained repeat, including shared-session window boundaries."""
    import math

    from tools.measure_baselines import _parse_ffts_profile

    records = json.loads(Path(measurement["profile_coverage_evidence"]).read_text())
    repeats = measurement["repeats"]
    assert len(records) == repeats
    if records[0].get("kind") == "marker-session":
        from session_profile import COVERAGE, PROTOCOL, partition

        assert measurement["measurement_protocol"] == PROTOCOL
        assert measurement["profile_coverage_protocol"] == COVERAGE
        path = records[0]["path"]
        sha = complete_digest(path)
        assert all(r["path"] == path and r["complete_sha256"] == sha for r in records)
        assert [r["window"] for r in records] == list(range(repeats))
        samples, _ = partition(path, repeats)
        for r, sample in zip(records, samples, strict=True):
            assert r["result"] == sample
    else:
        assert not any(r.get("kind") == "marker-session" for r in records)
        samples = []
        for r in records:
            assert complete_digest(r["path"]) == r["complete_sha256"]
            extra = decode(r["path"])
            try:
                sample = _parse_ffts_profile(Path(r["path"]), 1)
            except ValueError as exc:
                if str(exc) != "No valid FFTS activity":
                    raise
                sample = pure_aicpu_base(extra)
            sample["latency_ms"] += extra["added_us"] / 1000
            samples.append(sample)
    assert len(measurement["latency_samples_ms"]) == repeats
    for r, sample, value in zip(records, samples, measurement["latency_samples_ms"], strict=True):
        assert math.isclose(sample["latency_ms"], r["result"]["latency_ms"], rel_tol=1e-10)
        assert math.isclose(sample["latency_ms"], value, rel_tol=1e-10)
    assert math.isclose(
        sum(measurement["latency_samples_ms"]) / repeats, measurement["latency_ms"], rel_tol=1e-10
    )
    return records
