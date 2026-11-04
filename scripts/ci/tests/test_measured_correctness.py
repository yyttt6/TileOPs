"""CPU negative controls for verification of the actual final timed output."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from workloads.numerics import Custom, Exact, NegativeControl, Partial, Unestablished  # noqa: E402

SPEC = importlib.util.spec_from_file_location(
    "measured_correctness", ROOT / "scripts/ci/measured_correctness.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def execute(
    run,
    *,
    reference=lambda x: x * 2,
    evidence=None,
    reset=None,
    args=None,
    inputs=None,
    repeats=19,
    expected_calls=19,
):
    inputs = (torch.tensor([1.0, 3.0]),) if inputs is None else inputs
    case = SimpleNamespace(
        inputs=inputs, reference=reference, verification=Exact() if evidence is None else evidence
    )
    impl = SimpleNamespace(run=run, args=args, reset=reset, noncomparable_reason=None)

    def measure(capture):
        for _ in range(repeats):
            if reset:
                reset()
            capture(*(inputs if args is None else args))
        return {"latency_ms": 1.25}

    return MODULE.measure_and_verify(case, impl, measure, expected_calls=expected_calls)


def test_only_final_timed_output_checked_without_extra_call():
    calls = []
    references = []

    def run(x):
        calls.append(1)
        return x * (2 if len(calls) == 19 else 9)

    def reference(x):
        references.append(len(calls))
        return x * 2

    check, measurement = execute(run, reference=reference)
    assert len(calls) == 19
    assert references == [19]
    assert check["checked_outputs"] == check["total_outputs"] == 1
    assert measurement["latency_ms"] == 1.25


def test_bad_final_output_rejected_even_if_earlier_outputs_correct():
    calls = []

    def run(x):
        calls.append(1)
        return x * (9 if len(calls) == 19 else 2)

    with pytest.raises(AssertionError):
        execute(run)
    assert len(calls) == 19


def test_mutation_rejected_and_inputs_restored():
    x = torch.tensor([1.0, 3.0])

    def run(x):
        x[-1] += 1
        return x * 2

    with pytest.raises(AssertionError, match="overwrote the shared inputs"):
        execute(run, inputs=(x,))
    assert torch.equal(x, torch.tensor([1.0, 3.0]))


def test_private_reset_and_alias_output_survive_reference_mutation():
    x = torch.tensor([1.0, 3.0])
    private = x.clone()
    resets = []

    def reset():
        resets.append(1)
        private.copy_(x)

    def run(value):
        return value.mul_(2)

    def reference(value):
        return value.mul_(2)

    execute(run, inputs=(x,), args=(private,), reset=reset, reference=reference)
    assert len(resets) == 19
    assert torch.equal(x, torch.tensor([1.0, 3.0]))
    assert torch.equal(private, x)


@pytest.mark.parametrize("evidence", [Partial(1, "unchecked second output"), Unestablished()])
def test_incomplete_coverage_rejected(evidence):
    with pytest.raises(AssertionError, match="incomplete correctness coverage"):
        execute(lambda x: (x * 2, x), reference=lambda x: (x * 2, x), evidence=evidence)


def test_second_output_is_checked():
    with pytest.raises(AssertionError):
        execute(lambda x: (x * 2, x + 1), reference=lambda x: (x * 2, x))


def test_negative_control_still_rejected():
    # A control indistinguishable from the oracle must invalidate the contract.
    evidence = Exact(controls=(NegativeControl("ineffective", lambda ref, args: ref(*args)),))
    with pytest.raises(ValueError, match="negative control.*was accepted"):
        execute(lambda x: x * 2, evidence=evidence)


def test_custom_probe_uses_real_implementation():
    calls = []

    def run(x):
        calls.append(1)
        return x * 2

    def probe(fn, args):
        assert len(calls) == 19
        torch.testing.assert_close(fn(args[0] + 1), (args[0] + 1) * 2)

    execute(run, evidence=Custom(torch.testing.assert_close, "probe", probe=probe))
    assert len(calls) == 20


def test_timer_count_mismatch_fails_closed():
    with pytest.raises(AssertionError, match="timer call count changed"):
        execute(lambda x: x * 2, repeats=20)


def test_device_failure_keeps_receipt_and_does_not_run_next_case(tmp_path, monkeypatch):
    """Exercise the actual device loop with a CPU timer and fake device discovery."""
    import json

    released = []
    cases = [
        SimpleNamespace(
            id=name,
            params={},
            arguments={},
            inputs=(torch.tensor([1.0]),),
            reference=lambda x: x * 2,
            verification=Exact(),
            _release=lambda name=name: released.append(name),
        )
        for name in ("bad", "must-not-run")
    ]
    impl = SimpleNamespace(run=lambda x: x * 9, args=None, reset=None, noncomparable_reason=None)
    modules = {
        "benchmarks.ascend.guard": SimpleNamespace(require_reservation=lambda: None),
        "benchmarks._cases": SimpleNamespace(
            entry=lambda _: SimpleNamespace(binder=lambda *_: impl)
        ),
        "benchmarks.api": SimpleNamespace(cases=lambda _: cases),
        "benchmarks.ascend.runner": SimpleNamespace(
            benchmark_case=None, source_fingerprint=lambda: {}
        ),
        "tileops.backend": SimpleNamespace(
            registry=SimpleNamespace(
                ensure_loaded=lambda: None,
                LOAD_FAILURES=[],
                REGISTERED_KERNEL_TYPES={"BrokenOp": {"ascend"}},
            )
        ),
        "tileops.manifest": SimpleNamespace(load_manifest=lambda: {"BrokenOp": {"family": "fake"}}),
        "tileops.fake": SimpleNamespace(BrokenOp=lambda: None),
        "workloads.device": SimpleNamespace(set_run_device=lambda _: None),
        "torch_npu": SimpleNamespace(),
        "measured_correctness": MODULE,
    }

    def make_timer(path):
        path.mkdir(parents=True)
        (path / "raw-evidence").write_text("kept even on failed comparison")

        def timer(_torch, _npu, run, args, warmup, repeats, reset=None):
            for _ in range(1 + warmup + repeats):
                run(*args)
            return {"latency_ms": 1.0}

        return timer

    modules["complete_profile"] = SimpleNamespace(make_timer=make_timer)
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(
            set_device=lambda _: None,
            synchronize=lambda: None,
            get_device_name=lambda _: "Ascend950",
            get_device_properties=lambda _: SimpleNamespace(
                cube_core_num=32,
                vector_core_num=64,
            ),
        ),
        raising=False,
    )
    spec = importlib.util.spec_from_file_location(
        "device_loop", ROOT / "scripts/ci/nightly_device.py"
    )
    device = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(device)
    out = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "nightly_device.py",
            "--source",
            str(ROOT),
            "--output",
            str(out),
            "--op",
            "BrokenOp",
            "--verify-measured-output",
        ],
    )
    with pytest.raises(SystemExit) as exc:
        device.main()
    assert exc.value.code == 1
    assert released == ["bad"]
    assert json.loads((out / "result.json").read_text())["failed"] == 1
    rows = [json.loads(line) for line in (out / "results.jsonl").read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["status"] == "failed"
    assert "measurement" not in rows[0]
    assert (out / "profiles/0000/raw-evidence").exists()


def test_optional_outputs_count_as_validated_after_structural_checks():
    check, _ = execute(lambda x: (x * 2, None, None), reference=lambda x: (x * 2, None, None))
    assert check["checked_outputs"] == check["total_outputs"] == 3


@pytest.mark.parametrize(
    "bad", [lambda x: (x * 2, x, None), lambda x: (None, None, None), lambda x: (x * 9, None, None)]
)
def test_optional_outputs_do_not_hide_wrong_tensor_or_structure(bad):
    with pytest.raises(AssertionError):
        execute(bad, reference=lambda x: (x * 2, None, None))
