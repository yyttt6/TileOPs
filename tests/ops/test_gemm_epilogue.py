"""Correctness for the fused GEMM epilogue ops (T264, ops 110-112)."""

import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.ops import GemmBiasFwdOp, GemmBiasGeluFwdOp, GemmBiasReluFwdOp
from workloads.gemm import (
    GemmBiasGeluWorkload,
    GemmBiasReluWorkload,
    GemmBiasWorkload,
)


class GemmBiasTest(GemmBiasWorkload, TestBase):
    pass


class GemmBiasReluTest(GemmBiasReluWorkload, TestBase):
    pass


class GemmBiasGeluTest(GemmBiasGeluWorkload, TestBase):
    pass


class GemmBiasFixture(FixtureBase):
    # tests/conftest.py::pytest_collection_modifyitems enforces three rules on
    # this list and T264's version of the file broke all three (it also used
    # ``@GemmBiasFixture.apply``, which does not exist, so nothing here was
    # collectable at all -- R268 section 10.3).  The rules: every case carries
    # exactly one tier marker; every dtype has at least one smoke case; and all
    # smoke cases come first.  A ``full`` case may not differ from a smoke case
    # only by dtype, so the bf16 smoke case is the decode shape, not a bf16
    # copy of the fp16 one.
    PARAMS = [
        (
            "m, n, k, dtype, trans_b",
            [
                pytest.param(
                    1024, 1024, 1024, torch.float16, False,
                    marks=[pytest.mark.smoke], id="smoke-fp16-nn-1k",
                ),
                pytest.param(
                    128, 2112, 7168, torch.bfloat16, True,
                    marks=[pytest.mark.smoke], id="smoke-bf16-nt-decode",
                ),
                pytest.param(
                    4096, 2112, 7168, torch.bfloat16, True,
                    marks=[pytest.mark.full], id="bf16-nt-prefill",
                ),
            ],
        )
    ]


# Decorator ORDER matters here.  The bottom-most parametrize varies slowest and
# its id comes first, and tests/conftest.py requires every smoke case to come
# before every non-smoke one within a test.  With the shape fixture on top the
# order is (bias-fp16, bias-bf16, bias-full, bias_relu-fp16, ...) and the full
# case of the first epilogue precedes the smoke cases of the other two.  Putting
# the shape fixture at the bottom makes the shape the slow axis, so all six
# smoke cases come first.
@pytest.mark.parametrize(
    "op_cls, test_cls",
    [
        (GemmBiasFwdOp, GemmBiasTest),
        (GemmBiasReluFwdOp, GemmBiasReluTest),
        (GemmBiasGeluFwdOp, GemmBiasGeluTest),
    ],
    ids=["bias", "bias_relu", "bias_gelu"],
)
@GemmBiasFixture
def test_gemm_epilogue(op_cls, test_cls, m, n, k, dtype, trans_b):
    workload = test_cls(m, n, k, dtype, trans_a=False, trans_b=trans_b)
    # target="ascend" explicitly: tileops.backend.dispatch.detect_target returns
    # None for an "npu" device in this distribution, so an op constructed
    # without it raises OpNotAvailableError at the first forward
    # (PHASE_B_CONTRACT section 4; the same omission is why every test in
    # tests/ops/test_convolution.py failed -- R268 section 10.3).
    op = op_cls(trans_a=False, trans_b=trans_b, target="ascend")
    inputs = workload.gen_inputs()
    tol = 1.6e-2 if dtype is torch.bfloat16 else 1e-2
    torch.testing.assert_close(
        op(*inputs), workload.ref_program(*inputs), rtol=tol, atol=tol
    )


@pytest.mark.smoke
def test_trans_a_is_rejected():
    """The fused Cube path is non-transposed-A only; the op must say so."""
    with pytest.raises(ValueError, match="trans_a=True"):
        GemmBiasFwdOp(trans_a=True, target="ascend")
