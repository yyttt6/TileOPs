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
    PARAMS = [
        (
            "m, n, k, dtype, trans_b",
            [
                pytest.param(
                    1024, 1024, 1024, torch.float16, False,
                    marks=[pytest.mark.smoke], id="smoke-fp16-nn-1k",
                ),
                pytest.param(128, 2112, 7168, torch.bfloat16, True, id="bf16-nt-decode"),
                pytest.param(4096, 2112, 7168, torch.bfloat16, True, id="bf16-nt-prefill"),
            ],
        )
    ]


@GemmBiasFixture.apply
@pytest.mark.parametrize(
    "op_cls, test_cls",
    [
        (GemmBiasFwdOp, GemmBiasTest),
        (GemmBiasReluFwdOp, GemmBiasReluTest),
        (GemmBiasGeluFwdOp, GemmBiasGeluTest),
    ],
    ids=["bias", "bias_relu", "bias_gelu"],
)
def test_gemm_epilogue(op_cls, test_cls, m, n, k, dtype, trans_b):
    workload = test_cls(m, n, k, dtype, trans_a=False, trans_b=trans_b)
    op = op_cls(trans_a=False, trans_b=trans_b)
    inputs = workload.gen_inputs()
    tol = 1.6e-2 if dtype is torch.bfloat16 else 1e-2
    torch.testing.assert_close(
        op(*inputs), workload.ref_program(*inputs), rtol=tol, atol=tol
    )


def test_trans_a_is_rejected():
    """The fused Cube path is non-transposed-A only; the op must say so."""
    with pytest.raises(ValueError, match="trans_a=True"):
        GemmBiasFwdOp(trans_a=True)
