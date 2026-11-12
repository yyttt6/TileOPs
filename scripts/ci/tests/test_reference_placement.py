"""Keep the declared oracle and every numerical/negative-control check intact."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "scripts/ci"), str(ROOT / "scripts/ci/nightly_support")]
from golden_cache import GoldenCache  # noqa: E402
from measured_correctness import measure_and_verify  # noqa: E402
from reference_placement import ReferenceCase, cpu_tree, needs_cpu, prepare_reference  # noqa: E402
from workloads.numerics import Exact, zeroed_input  # noqa: E402

pytestmark = pytest.mark.smoke


def inputs(length=4, offsets=None):
    q = SimpleNamespace(shape=(1, length, 2, 4), device=SimpleNamespace(type="npu"))
    return (q, None, None, None, None, None, offsets, offsets)


def test_only_long_npu_gdn_uses_cpu_oracle():
    assert needs_cpu("GDNFwdOp", inputs())
    assert not needs_cpu("GLAFwdOp", inputs())
    assert not needs_cpu("GDNFwdOp", inputs(length=1))
    assert not needs_cpu("GDNFwdOp", inputs(offsets=torch.tensor([0, 1, 2, 3, 4])))
    assert needs_cpu("GDNFwdOp", inputs(offsets=torch.tensor([0, 1, 4])))
    case = SimpleNamespace(inputs=(torch.ones(1, 4),))
    plan = prepare_reference("GDNFwdOp", case)
    assert plan.case is case and plan.comparison_transform is None and not plan.cache_identity


def test_cpu_tree_preserves_structure_dtype_and_separates_storage():
    value = (
        torch.tensor([1.0, 2.0], dtype=torch.float64),
        None,
        {"index": torch.tensor([2], dtype=torch.int32)},
    )
    saved = cpu_tree(value)
    assert saved[0].dtype == torch.float64 and saved[2]["index"].dtype == torch.int32
    saved[0].add_(1)
    torch.testing.assert_close(value[0], torch.tensor([1.0, 2.0], dtype=torch.float64))
    with pytest.raises(ValueError, match="autograd"):
        cpu_tree(torch.ones(2, requires_grad=True))


def test_original_oracle_and_controls_still_run_and_bad_output_is_rejected(tmp_path):
    calls = []
    x = torch.tensor([1.0, 2.0])

    def oracle(t):
        assert t.device.type == "cpu"
        calls.append(t.clone())
        return (t.double() * 2).float()

    case = SimpleNamespace(
        inputs=(x,), reference=oracle, verification=Exact(controls=(zeroed_input(0, "zero"),))
    )
    view = ReferenceCase(case)
    impl = SimpleNamespace(args=None, run=lambda t: t * 2, reset=None, noncomparable_reason=None)

    def measure(fn):
        fn(x)
        return {"prewarm_only": True}

    def execute(c):
        return measure_and_verify(
            view, impl, measure, expected_calls=1, golden_cache=c, comparison_transform=cpu_tree
        )

    identity = {"reference_placement": "gdn-existing-fp64-reference-on-cpu-v1"}
    execute(GoldenCache(tmp_path, identity, min_free_bytes=0))
    assert len(calls) == 2
    execute(GoldenCache(tmp_path, identity, min_free_bytes=0))
    assert len(calls) == 2  # Main oracle and negative control are both true cache hits.
    impl.run = lambda t: t * 3
    with pytest.raises(AssertionError):
        execute(GoldenCache(tmp_path, identity, min_free_bytes=0))
    assert len(calls) == 2
