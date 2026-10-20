"""Ascend measurement gates: missing evidence cannot produce an accepted score."""

import contextlib
import struct
from types import SimpleNamespace

import pytest

from benchmarks.ascend import timing
from benchmarks.ascend.profile_parser import _parse_ffts_profile
from benchmarks.ascend.runner import benchmark_case
from workloads.numerics import CheckResult


def _record(duration, cycles=1, start=100):
    values = [0] * 16
    values[1] = cycles
    values[14] = start
    values[15] = start + duration
    return struct.pack("<16q", *values)


def test_raw_kernel_durations_exclude_host_gaps(tmp_path):
    (tmp_path / "ffts_profile.slice_0").write_bytes(_record(2000) + _record(3000, start=10000))
    parsed = _parse_ffts_profile(tmp_path, 1)
    assert parsed["latency_ms"] == pytest.approx(0.005)
    assert parsed["profile_kernel_count"] == 2


def test_overlapping_kernels_count_once(tmp_path):
    (tmp_path / "ffts_profile.slice_0").write_bytes(_record(2000) + _record(3000))
    parsed = _parse_ffts_profile(tmp_path, 1)
    assert parsed["latency_ms"] == pytest.approx(0.003)
    assert parsed["profile_kernel_us"] == 5


def test_truncated_profiler_data_fails(tmp_path):
    (tmp_path / "ffts_profile.slice_0").write_bytes(_record(2000) + b"partial")
    with pytest.raises(RuntimeError, match="Truncated"):
        _parse_ffts_profile(tmp_path, 1)


def test_missing_device_records_fail(tmp_path):
    (tmp_path / "ffts_profile.slice_0").write_bytes(_record(2000, cycles=0))
    with pytest.raises(RuntimeError, match="no ffts_profile records"):
        _parse_ffts_profile(tmp_path, 1)


def test_flush_name_must_be_resolved_by_self_check(tmp_path):
    (tmp_path / "ffts_profile.slice_0").write_bytes(_record(2000))
    with pytest.raises(RuntimeError, match="never matched"):
        _parse_ffts_profile(
            tmp_path, 1, exclude_name_marker=timing.FLUSH_NAME, require_marker_seen=True
        )


@pytest.mark.parametrize("checked,total,reason", [(0, 1, None), (1, 2, "partial"), (1, None, None)])
def test_incomplete_correctness_never_reaches_timer(monkeypatch, checked, total, reason):
    from workloads import numerics

    monkeypatch.setattr(
        numerics, "verify", lambda *a, **k: {"tileops": CheckResult(checked, total, None, reason)}
    )
    case = SimpleNamespace(reference=lambda: None, inputs=(), verification=None)
    impl = SimpleNamespace(run=lambda: None, args=None, reset=None, noncomparable_reason=None)
    with pytest.raises(AssertionError, match="incomplete correctness"):
        benchmark_case(case, {"tileops": impl}, None, name="case")


def test_isolated_windows_have_fixed_counts_and_no_event_timer(monkeypatch, tmp_path):
    import torch

    calls = []
    active = False
    fake_npu = SimpleNamespace(synchronize=lambda: calls.append(("sync", active)))
    fake_npu.Event = lambda *a, **k: pytest.fail("event timer used")
    monkeypatch.setattr(torch, "npu", fake_npu, raising=False)
    monkeypatch.setattr(timing, "require_reservation", lambda: {})

    @contextlib.contextmanager
    def profile(path):
        nonlocal active
        active = True
        yield SimpleNamespace(prof_if=SimpleNamespace(prof_path=str(path)), step=lambda: None)
        active = False

    monkeypatch.setattr(timing, "_profile", profile)
    durations = iter(range(1, 16))
    monkeypatch.setattr(
        timing,
        "_parse_ffts_profile",
        lambda *a, **k: {
            "latency_ms": next(durations),
            "profile_excluded_kernel_count": 0,
            "profile_kernel_count": 1,
        },
    )
    timer = timing.ProfilerTimer.__new__(timing.ProfilerTimer)
    timer.directory = tmp_path
    timer.clocks = SimpleNamespace(check=lambda: calls.append(("clock", active)))
    timer.flush_buffer = None
    timer.flush = lambda _: calls.append(("flush", active))
    result = timer.measure(
        lambda: calls.append(("op", active)), reset=lambda: calls.append(("reset", active))
    )
    assert result["samples_ms"] == list(range(1, 16))
    assert result["latency_ms"] == 8
    assert [kind for kind, inside in calls if inside and kind != "sync"] == ["op"] * 15
    assert sum(kind == "op" and not inside for kind, inside in calls) == 3
    assert sum(kind == "flush" for kind, _ in calls) == 18
    for index, (kind, _) in enumerate(calls):
        if kind == "flush":
            assert calls[index - 1] == ("reset", False)
            assert calls[index + 1] == ("sync", False)


@pytest.mark.parametrize("failure", [None, "lock", "readback", "measurement"])
def test_clock_policy_restores_after_success_or_failure(monkeypatch, failure):
    from benchmarks.ascend import clocks

    commands = []
    policy = clocks.ClockPolicy.__new__(clocks.ClockPolicy)
    policy.spec = {"lock": ["/site/lock"], "restore": ["/site/restore"]}
    policy.targets = [1000, 1000]
    policy.samples = []
    policy.read = lambda: [{"rated_mhz": 1200, "mhz": 1000}] * 2
    monkeypatch.setattr(clocks, "require_reservation", lambda: {})

    def run(command, **kwargs):
        commands.append(command)
        if failure == "lock" and command == policy.spec["lock"]:
            raise RuntimeError("lock failed")

    monkeypatch.setattr(clocks.subprocess, "run", run)
    if failure == "readback":
        policy.targets = [1100, 1100]

    def measure():
        with policy:
            if failure == "measurement":
                raise RuntimeError("measurement failed")

    if failure:
        with pytest.raises(RuntimeError):
            measure()
    else:
        measure()
        assert policy.after == policy.before
    assert commands == [policy.spec["lock"], policy.spec["restore"]]


def test_clock_policy_rejects_above_rated_before_setting(monkeypatch):
    from benchmarks.ascend import clocks

    policy = clocks.ClockPolicy.__new__(clocks.ClockPolicy)
    policy.targets = [1300, 1000]
    policy.read = lambda: [{"rated_mhz": 1200, "mhz": 1000}] * 2
    monkeypatch.setattr(clocks, "require_reservation", lambda: {})
    monkeypatch.setattr(
        clocks.subprocess, "run", lambda *a, **k: pytest.fail("clock command executed")
    )
    with pytest.raises(ValueError, match="rated"), policy:
        pytest.fail("above-rated policy accepted")


def test_runtime_versions_retain_unavailable_queries():
    from benchmarks.ascend.runner import runtime_versions

    def get_version(*, module):
        if module == "DRIVER":
            raise RuntimeError("unavailable")
        return "9.2.0"

    versions = runtime_versions(get_version)
    assert versions["CANN"] == "9.2.0"
    assert versions["DRIVER"] == {"error": "unavailable"}
    assert set(versions) == {"CANN", "RUNTIME", "COMPILER", "DRIVER", "TOOLKIT", "OPP"}


def test_source_fingerprint_includes_native_code(monkeypatch, tmp_path):
    from benchmarks.ascend import runner

    monkeypatch.setattr(runner, "ROOT", tmp_path)
    header = tmp_path / "src/tileops/csrc/kernel.h"
    header.parent.mkdir(parents=True)
    header.write_text("void kernel();\n")
    for name in ("pyproject.toml", "constraints.txt", "constraints-runner-lock.txt"):
        (tmp_path / name).write_text("")
    before = runner.source_fingerprint()
    assert "src/tileops/csrc/kernel.h" in before
    header.write_text("void kernel(int);\n")
    assert before != runner.source_fingerprint()
