"""Trace partitioning rejects ambiguous windows and retains named AICPU work."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.smoke
SUPPORT = Path(__file__).resolve().parents[1] / "nightly_support"
SPEC = importlib.util.spec_from_file_location("session_test", SUPPORT / "session_profile.py")
session = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(session)


def fixture(monkeypatch, *, aicpu=False):
    def task(seq, start, end, name="body", stream=0):
        return dict(
            task_id=seq, start_ns=start, end_ns=end, duration_us=5, name=name, stream_ids=[stream]
        )

    tasks = [
        task(1, 0, 1, "flush"),
        task(2, 2, 3),
        task(5, 9, 10, "flush"),
        task(6, 11, 12, "flush"),
        task(7, 13, 14),
        task(10, 20, 21, "flush"),
    ]
    added = [dict(seq=3, start_ticks=4, end_ticks=5, duration_us=7, name="aicpu")] if aicpu else []
    extra = dict(added_aicpu=added, tasks=[dict(seq=1, end_ticks=1), dict(seq=5, start_ticks=9)])
    monkeypatch.setitem(
        sys.modules,
        "tools",
        SimpleNamespace(
            measure_baselines=SimpleNamespace(
                _parse_ffts_profile=lambda *args: dict(profile_kernel_tasks=tasks),
                _FLUSH_KERNEL_NAME_MARKER="flush",
            )
        ),
    )
    monkeypatch.setitem(sys.modules, "complete_profile", SimpleNamespace(decode=lambda _: extra))
    return tasks, added


def test_partitions_repeats_and_keeps_aicpu(monkeypatch, tmp_path):
    fixture(monkeypatch, aicpu=True)
    samples, proof = session.partition(tmp_path, 2)
    assert [s["latency_ms"] for s in samples] == [0.012, 0.005]
    assert samples[0]["profile_kernel_count"] == 2
    assert len(proof["preparation_and_markers"]) == 4


@pytest.mark.parametrize(
    "damage", ["stream", "wrap", "missing-marker", "overlap", "unknown-name", "aicpu-outside"]
)
def test_rejects_ambiguous_trace(monkeypatch, tmp_path, damage):
    tasks, added = fixture(monkeypatch, aicpu=True)
    if damage == "stream":
        tasks[1]["stream_ids"] = [1]
    if damage == "wrap":
        tasks[3]["task_id"] = 65536
    if damage == "missing-marker":
        tasks.pop(0)
    if damage == "overlap":
        tasks[1]["end_ns"] = 11
    if damage == "unknown-name":
        tasks[1]["name"] = "<unknown task>"
    if damage == "aicpu-outside":
        added[0]["end_ticks"] = 20
    with pytest.raises(AssertionError):
        session.partition(tmp_path, 2)


@pytest.mark.parametrize("damage", ["none", "window", "hash", "sample", "body", "protocol"])
def test_replay_rejects_tampered_shared_session(monkeypatch, tmp_path, damage):
    spec = importlib.util.spec_from_file_location("complete_test", SUPPORT / "complete_profile.py")
    complete = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(complete)
    samples = [
        dict(latency_ms=0.01, profile_kernel_count=1),
        dict(latency_ms=0.02, profile_kernel_count=1),
    ]
    entries = [
        dict(kind="marker-session", window=i, path="raw", complete_sha256="a" * 64, result=dict(x))
        for i, x in enumerate(samples)
    ]
    measurement = dict(
        repeats=2,
        measurement_protocol=session.PROTOCOL,
        profile_coverage_protocol=session.COVERAGE,
        latency_samples_ms=[0.01, 0.02],
        latency_ms=0.015,
        profile_coverage_evidence=str(tmp_path / "coverage.json"),
    )
    if damage == "window":
        entries[1]["window"] = 0
    if damage == "hash":
        entries[1]["complete_sha256"] = "b" * 64
    if damage == "sample":
        measurement["latency_samples_ms"][1] = 0.01
    if damage == "body":
        entries[1]["result"]["profile_kernel_count"] = 0
    if damage == "protocol":
        measurement["measurement_protocol"] = "old"
    (tmp_path / "coverage.json").write_text(json.dumps(entries))
    monkeypatch.setattr(complete, "complete_digest", lambda _: "a" * 64)
    monkeypatch.setitem(
        sys.modules,
        "session_profile",
        SimpleNamespace(
            partition=lambda *_: (samples, {}), PROTOCOL=session.PROTOCOL, COVERAGE=session.COVERAGE
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "tools",
        SimpleNamespace(measure_baselines=SimpleNamespace(_parse_ffts_profile=None)),
    )
    monkeypatch.setitem(
        sys.modules, "tools.measure_baselines", SimpleNamespace(_parse_ffts_profile=None)
    )
    if damage == "none":
        assert complete.replay_coverage(measurement) == entries
    else:
        with pytest.raises(AssertionError):
            complete.replay_coverage(measurement)


@pytest.mark.parametrize("kind", [4, 5])
def test_unqualified_named_aicpu_uses_isolated_backend(kind):
    with pytest.raises(ValueError, match="qualified isolated"):
        session.require_qualified_topology(
            dict(full_decode=dict(tasks=[dict(type=kind, name="Cast")]))
        )
    session.require_qualified_topology(dict(full_decode=dict(tasks=[dict(type=1, name="main")])))


def test_names_and_hardware_join_on_full_task_id(tmp_path, monkeypatch):
    import json
    import struct

    monkeypatch.syspath_prepend(str(SUPPORT))
    from complete_profile import decode
    from tools.measure_baselines import _parse_ffts_profile

    data = tmp_path / "device_0/data"
    data.mkdir(parents=True)
    (data.parent / "info.json.0").write_text(
        json.dumps({"platform_version": "15", "DeviceInfo": [{"hwts_frequency": 1000}]})
    )
    (data / "hash_dic.slice_0").write_text("1:main\n2:LeftShift\n")

    def track(task, stream, name):
        return struct.pack("<8Q", 0, 0, 0, (task << 32) | (stream << 16), 0, name, 0, 0)

    (data / "task_track.slice_0").write_bytes(track(73, 3, 1) + track(65609, 4, 2))
    a = [0] * 16
    a[0] = (73 << 32) | 0x2A
    a[1] = 100
    a[14:16] = [1000, 2000]
    (data / "ffts_profile.data.0.slice_0").write_bytes(struct.pack("<16q", *a))

    def event(func, ticks):
        return struct.pack("<4HQHH3L", (5 << 10) | func, 0x6BD3, 73, 1, ticks, 0, 0, 0, 0, 0)

    stars = data / "stars_soc.data.0.slice_0"
    stars.write_bytes(event(0, 3000) + event(1, 6000))
    decoded = decode(tmp_path)
    assert [(t["name"], t["seq"], t["duration_us"]) for t in decoded["added_aicpu"]] == [
        ("LeftShift", 65609, 3.0)
    ]
    (task,) = _parse_ffts_profile(tmp_path, 1)["profile_kernel_tasks"]
    assert (task["name"], task["task_id"], task["stream_ids"]) == ("main", 73, [3])
    stars.write_bytes(b"")
    with pytest.raises(AssertionError, match="uncovered tracked kernels"):
        decode(tmp_path)
