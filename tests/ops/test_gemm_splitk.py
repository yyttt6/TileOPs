"""Correctness for split-K dense GEMM (T267, op 113).

The shapes below are picked to cover both sides of the kernel's split gate,
because "the numbers are right" has to mean both paths:

  * ``1024x1024x1024`` NN plans 32 logical blocks -- above the gate, so the op
    falls through to the ordinary dense GEMM (``split_k == 1``).
  * ``128x2112x7168`` NT plans 9 -- the most starved shape in the GemmFwdOp
    workload set, and the one that takes the deepest split.
  * ``16x4096x7168`` NT plans 16 -- splits by 2.

``n = 2112`` against ``block_n = 256`` also makes one case exercise the ragged
trailing N tile, which the fp32 partials workspace has to survive.
"""

import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.ops import GemmSplitKFwdOp
from workloads.gemm import GemmWorkload


class GemmSplitKTest(GemmWorkload, TestBase):
    pass


class GemmSplitKFixture(FixtureBase):
    PARAMS = [
        (
            "m, n, k, trans_b, dtype, expect_split",
            [
                pytest.param(
                    1024, 1024, 1024, False, torch.float16, False,
                    marks=pytest.mark.smoke, id="smoke-fp16-nn-1k-no-split",
                ),
                # tests/conftest.py requires a smoke case per dtype, so the
                # bf16 split case carries the marker too (and it is the case
                # that actually exercises the split-K path).
                pytest.param(
                    128, 2112, 7168, True, torch.bfloat16, True,
                    marks=pytest.mark.smoke, id="smoke-bf16-nt-decode-split",
                ),
                pytest.param(
                    16, 4096, 7168, True, torch.bfloat16, True,
                    marks=pytest.mark.full, id="bf16-nt-mid-m16-split",
                ),
            ],
        )
    ]


# NOTE: `@GemmSplitKFixture`, not `@GemmSplitKFixture.apply`.  FixtureBase's
# metaclass (workloads/workload_base.py:51 FixtureMeta.__call__) makes the class
# itself the decorator; there is no `apply` attribute.  tests/ops/
# test_gemm_epilogue.py (T264) uses `.apply` and therefore FAILS COLLECTION --
# see docs/reports/R268.md section 10.
@GemmSplitKFixture
def test_gemm_splitk(m, n, k, trans_b, dtype, expect_split):
    workload = GemmSplitKTest(m, n, k, dtype, trans_a=False, trans_b=trans_b)
    # target="ascend" explicitly: tileops.backend.dispatch.detect_target returns
    # None for an "npu" device in this distribution, so an op constructed
    # without it raises OpNotAvailableError at the first forward (this is what
    # PHASE_B_CONTRACT section 4 warns about, and what makes every test in
    # tests/ops/test_convolution.py fail in this environment).
    op = GemmSplitKFwdOp(trans_a=False, trans_b=trans_b, target="ascend")
    inputs = workload.gen_inputs()
    tol = 1.6e-2 if dtype is torch.bfloat16 else 1e-2
    got = op(*inputs)
    torch.testing.assert_close(
        got, workload.ref_program(*inputs), rtol=tol, atol=tol
    )
    # A split that silently did not happen would still pass the numbers above,
    # so assert which path ran.  This is the same failure mode as a baseline
    # that silently falls back (PROJECT_STATE 13.90).
    assert (op.split_k > 1) is expect_split, (
        f"expected split_k>1 to be {expect_split} for {m}x{n}x{k}, got "
        f"split_k={op.split_k} ({op.split_reason})"
    )


@pytest.mark.smoke
def test_trans_a_is_rejected():
    """The split-K partial kernel is non-transposed-A only; the op must say so."""
    with pytest.raises(ValueError, match="trans_a=True"):
        GemmSplitKFwdOp(trans_a=True)
