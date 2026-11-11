"""Golden reuse must not bypass correctness or mix actual inputs/contracts."""

import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/ci"))
from types import SimpleNamespace  # noqa: E402

from golden_cache import GoldenCache, input_digest  # noqa: E402
from measured_correctness import measure_and_verify  # noqa: E402

from workloads.numerics import Custom, Exact, Partial, zeroed_input  # noqa: E402

pytestmark = pytest.mark.smoke


def cache(path, **kwargs):
    return GoldenCache(path, {"reference": "a", "case": "case"}, min_free_bytes=0, **kwargs)


def execute(c, ref, run=lambda x: x * 2, evidence=None):
    inputs = (torch.tensor([1.0, 3.0]),)
    case = SimpleNamespace(inputs=inputs, reference=ref, verification=evidence or Exact())
    impl = SimpleNamespace(run=run, args=None, reset=None, noncomparable_reason=None)

    def measure(fn):
        for _ in range(3):
            fn(*inputs)
        return {"latency_ms": 1}

    return measure_and_verify(case, impl, measure, expected_calls=3, golden_cache=c)


def test_hit_skips_reference_but_bad_kernel_is_still_rejected(tmp_path):
    calls = []

    def ref(x):
        calls.append(1)
        return x * 2

    execute(cache(tmp_path), ref)
    execute(cache(tmp_path), ref)
    assert len(calls) == 1
    with pytest.raises(AssertionError):
        execute(cache(tmp_path), ref, run=lambda x: x * 3)
    assert len(calls) == 1


def test_failed_validation_does_not_publish(tmp_path):
    with pytest.raises(AssertionError):
        execute(cache(tmp_path), lambda x: x * 2, run=lambda x: x * 3)
    assert not list(tmp_path.glob("*/outputs.pt"))


def test_partial_does_not_publish(tmp_path):
    with pytest.raises(AssertionError, match="incomplete"):
        execute(
            cache(tmp_path),
            lambda x: (x, x),
            run=lambda x: (x, x),
            evidence=Partial(1, "second missing"),
        )
    assert not list(tmp_path.glob("*/outputs.pt"))


def test_negative_control_uses_different_input_key(tmp_path):
    calls = []

    def ref(x):
        calls.append(x.clone())
        return x * 2

    evidence = Exact(controls=(zeroed_input(0, "zero"),))
    execute(cache(tmp_path), ref, evidence=evidence)
    assert len(calls) == 2
    assert len(list(tmp_path.glob("*/outputs.pt"))) == 2
    execute(cache(tmp_path), ref, evidence=evidence)
    assert len(calls) == 2


def test_reference_failure_propagates(tmp_path):
    def bad(x):
        raise RuntimeError("original reference failure")

    with pytest.raises(RuntimeError, match="original reference failure"):
        execute(cache(tmp_path), bad)
    assert not list(tmp_path.glob("*/outputs.pt"))


def test_changed_input_dtype_stride_and_signed_zero_change_key():
    x = torch.tensor([0.0, 1.0])
    assert input_digest((x,)) != input_digest((x + 1,))
    assert input_digest((x,)) != input_digest((x.double(),))
    assert input_digest((x,)) != input_digest((torch.tensor([-0.0, 1.0]),))
    assert input_digest((x, 1)) != input_digest((x, 2))
    m = torch.ones(2, 2)
    assert input_digest((m,)) != input_digest((m.T,))


def test_corrupt_entry_falls_back(tmp_path):
    calls = []

    def ref(x):
        calls.append(1)
        return x * 2

    execute(cache(tmp_path), ref)
    next(tmp_path.glob("*/outputs.pt")).write_bytes(b"broken")
    execute(cache(tmp_path), ref)
    assert len(calls) == 2


def test_contract_change_misses(tmp_path):
    execute(cache(tmp_path), lambda x: x * 2)
    c = GoldenCache(tmp_path, {"reference": "changed"}, min_free_bytes=0)
    execute(c, lambda x: x * 2)
    assert len(list(tmp_path.glob("*/outputs.pt"))) == 2


def test_mutation_and_rng_reference_not_cached(tmp_path):
    for ref in [lambda x: x.add_(1), lambda x: x + torch.rand_like(x)]:
        c = cache(tmp_path)
        c.wrap(ref)(torch.ones(2))
        c.commit()
    assert not list(tmp_path.glob("*/outputs.pt"))


def test_budget_and_free_space_preserve_fallback(tmp_path):
    execute(cache(tmp_path, max_bytes=1), lambda x: x * 2)
    c = GoldenCache(tmp_path, {}, min_free_bytes=1 << 100)
    execute(c, lambda x: x * 2)
    assert not list(tmp_path.glob("*/outputs.pt"))


def test_structure_and_nonfinite(tmp_path):
    c = cache(tmp_path)
    f = c.wrap(lambda x: (x, None, {"state": x + 1}))
    a = f(torch.ones(2))
    c.commit()
    b = cache(tmp_path).wrap(lambda x: pytest.fail("cache miss"))(torch.ones(2))
    assert b[1] is None and torch.equal(a[2]["state"], b[2]["state"])
    c = cache(tmp_path / "nan")
    c.wrap(lambda x: x * float("nan"))(torch.ones(2))
    c.commit()
    assert not list((tmp_path / "nan").glob("*/outputs.pt"))


def test_custom_probe_failure_not_published(tmp_path):
    def check(got, expected):
        torch.testing.assert_close(got, expected)

    def probe(run, args):
        raise AssertionError("probe failed")

    with pytest.raises(AssertionError, match="probe failed"):
        execute(
            cache(tmp_path), lambda x: x * 2, evidence=Custom(check, "probe contract", probe=probe)
        )
    assert not list(tmp_path.glob("*/outputs.pt"))


def test_policy_complete():
    from tileops.manifest import load_manifest

    p = json.loads((ROOT / "scripts/ci/golden_cache_policy.json").read_text())
    assert set(p["operators"]) == set(load_manifest())
    assert len(p["operators"]) == 206
    assert sum(v in {"high", "medium"} for v in p["operators"].values()) == 49


def test_manual_correctness_cache_and_wrong_output(tmp_path):
    from benchmarks.api import Implementation
    from benchmarks.ascend.runner import benchmark_case

    calls = []

    def ref(x):
        calls.append(1)
        return x * 2

    case = SimpleNamespace(inputs=(torch.ones(2),), reference=ref, verification=Exact())
    for _ in range(2):
        benchmark_case(
            case,
            {"tileops": Implementation(lambda x: x * 2)},
            None,
            name="case",
            verify_only=True,
            golden_cache=cache(tmp_path),
        )
    assert len(calls) == 1
    with pytest.raises(AssertionError):
        benchmark_case(
            case,
            {"tileops": Implementation(lambda x: x * 3)},
            None,
            name="case",
            verify_only=True,
            golden_cache=cache(tmp_path),
        )
    assert len(calls) == 1


def test_reference_identity_reuses_kernel_edits_but_not_reference_changes(tmp_path):
    from golden_cache import reference_source_digest

    w = tmp_path / "workloads/reference.py"
    k = tmp_path / "src/tileops/_kernels/ascend/kernel.py"
    w.parent.mkdir(parents=True)
    k.parent.mkdir(parents=True)
    w.write_text("from tileops.manifest import load_manifest\ndef ref(x): return x*2\n")
    k.write_text("value = 1\n")
    original, mode = reference_source_digest(tmp_path)
    assert mode == "independent-reference"
    k.write_text("value = 2\n")
    assert reference_source_digest(tmp_path)[0] == original
    w.write_text("def ref(x): return x*3\n")
    assert reference_source_digest(tmp_path)[0] != original
    w.write_text("from tileops.linear_attention import KDAFwdOp\n")
    before, mode = reference_source_digest(tmp_path)
    assert mode == "full-source"
    k.write_text("value = 3\n")
    assert reference_source_digest(tmp_path)[0] != before


def test_current_tree_is_independent():
    from golden_cache import reference_source_digest

    assert reference_source_digest(ROOT)[1] == "independent-reference"
