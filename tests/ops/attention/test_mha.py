
import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.ops import MultiHeadAttentionBwdOp, MultiHeadAttentionFwdOp
from workloads.attention.mha import MhaBwdWorkload, MhaFwdWorkload


class MhaBwdTest(MhaBwdWorkload, TestBase):
    pass


class MhaFwdTest(MhaFwdWorkload, TestBase):
    pass


class MhaFwdFixture(FixtureBase):
    PARAMS = [
        (
            "batch, seq_len, heads, dim, causal, dtype, tune",
            [
                pytest.param(
                    1,
                    1024,
                    8,
                    64,
                    False,
                    torch.float16,
                    False,
                    marks=[pytest.mark.smoke, pytest.mark.packaging],
                    id="smoke-fwd-fp16",
                ),
                pytest.param(
                    1,
                    1024,
                    8,
                    64,
                    False,
                    torch.bfloat16,
                    False,
                    marks=pytest.mark.smoke,
                    id="smoke-fwd-bf16",
                ),
                pytest.param(
                    16,
                    2048,
                    16,
                    128,
                    False,
                    torch.float16,
                    False,
                    marks=pytest.mark.full,
                    id="full-fwd-fp16",
                ),
                pytest.param(
                    4,
                    4096,
                    16,
                    128,
                    False,
                    torch.bfloat16,
                    True,
                    marks=pytest.mark.full,
                    id="full-fwd-bf16-tuned",
                ),
            ],
        ),
    ]


class MhaBwdFixture(FixtureBase):
    PARAMS = [
        (
            "batch, seq_len, heads, dim, causal, dtype, tune",
            [
                pytest.param(
                    1,
                    1024,
                    8,
                    64,
                    False,
                    torch.float16,
                    False,
                    marks=pytest.mark.smoke,
                    id="smoke-bwd-fp16",
                ),
                pytest.param(
                    1,
                    1024,
                    8,
                    64,
                    False,
                    torch.bfloat16,
                    False,
                    marks=pytest.mark.smoke,
                    id="smoke-bwd-bf16",
                ),
                pytest.param(
                    16,
                    2048,
                    16,
                    128,
                    False,
                    torch.float16,
                    False,
                    marks=pytest.mark.full,
                    id="full-bwd-fp16-large",
                ),
                pytest.param(
                    4,
                    4096,
                    16,
                    128,
                    False,
                    torch.bfloat16,
                    True,
                    marks=pytest.mark.full,
                    id="full-bwd-bf16-tuned",
                ),
            ],
        ),
    ]


@MhaFwdFixture
def test_mha_fwd(
    batch: int, seq_len: int, heads: int, dim: int, causal: bool, dtype: torch.dtype, tune: bool
) -> None:
    test = MhaFwdTest(batch, heads, seq_len, dim, causal, dtype)
    op = MultiHeadAttentionFwdOp(batch, heads, seq_len, dim, causal, tune=tune)
    test.check(op, *test.gen_inputs(), atol=5e-3, rtol=1e-5)


@pytest.mark.smoke
def test_mha_fwd_dispatches_to_gqa_kernel() -> None:
    op = MultiHeadAttentionFwdOp(1, 8, 128, 64, False)
    assert op._get_kernel((), torch.float16).__class__.__name__.startswith("GQA")


@MhaBwdFixture
def test_mha_bwd(
    batch: int, seq_len: int, heads: int, dim: int, causal: bool, dtype: torch.dtype, tune: bool
) -> None:
    test = MhaBwdTest(batch, heads, seq_len, dim, causal, dtype)
    op = MultiHeadAttentionBwdOp(batch, heads, seq_len, dim, causal, tune=tune)
    test.check(op, *test.gen_inputs(), atol=5e-3, rtol=1e-5)
