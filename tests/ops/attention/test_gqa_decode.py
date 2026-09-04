import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.ops import GroupedQueryAttentionDecodeWithKVCacheFwdOp
from workloads.attention.gqa import (
    GroupedQueryAttentionDecodeWorkload,
)


class GroupedQueryAttentionDecodeTest(GroupedQueryAttentionDecodeWorkload, TestBase):
    pass


class GroupedQueryAttentionDecodeFixture(FixtureBase):
    PARAMS = [
        (
            "batch, heads, heads_kv, seq_len_kv, dim, dtype, tune",
            [
                pytest.param(1, 32, 8, 8192, 128, torch.float16, False, marks=pytest.mark.smoke),
                pytest.param(1, 32, 8, 8192, 128, torch.bfloat16, False, marks=pytest.mark.smoke),
                pytest.param(1, 16, 2, 8192, 128, torch.float16, False, marks=pytest.mark.smoke),
                pytest.param(8, 64, 16, 8192, 128, torch.float16, False, marks=pytest.mark.full),
            ],
        ),
    ]


@GroupedQueryAttentionDecodeFixture
def test_gqa_decode(
    batch: int, heads: int, heads_kv: int, seq_len_kv: int, dim: int, dtype: torch.dtype, tune: bool
) -> None:
    test = GroupedQueryAttentionDecodeTest(batch, heads, heads_kv, seq_len_kv, dim, dtype)
    op = GroupedQueryAttentionDecodeWithKVCacheFwdOp(
        batch, heads, heads_kv, seq_len_kv, dim, tune=tune
    )
    test.check(op, *test.gen_inputs(), atol=1e-2, rtol=1e-2)


@pytest.mark.smoke
@pytest.mark.parametrize(
    "sm_scale, softcap",
    [
        pytest.param(0.25, None, id="custom-sm-scale"),
        pytest.param(None, 2.0, id="softcap"),
    ],
)
def test_gqa_decode_softmax_controls(sm_scale: float | None, softcap: float | None) -> None:
    batch, heads, heads_kv, seq_len_kv, dim = 1, 16, 4, 1024, 64
    dtype = torch.float16
    test = GroupedQueryAttentionDecodeTest(
        batch,
        heads,
        heads_kv,
        seq_len_kv,
        dim,
        dtype,
        sm_scale=sm_scale,
        softcap=softcap,
    )
    op = GroupedQueryAttentionDecodeWithKVCacheFwdOp(
        batch,
        heads,
        heads_kv,
        seq_len_kv,
        dim,
        sm_scale=sm_scale,
        softcap=softcap,
    )
    test.check(op, *test.gen_inputs(), atol=1e-2, rtol=1e-2)


@pytest.mark.smoke
def test_gqa_decode_bs1_runtime_context_switch() -> None:
    """One op instance stays correct across the context range as the cache grows.

    Covers the crossover (1024), a balanced mid split, an aligned split needing >=3 tiles
    per slice, an unaligned length, and the sub-1024 non-split fallback.
    """
    op = GroupedQueryAttentionDecodeWithKVCacheFwdOp(1, 32, 4, 8192, 128)
    kernel = op._get_kernel((), torch.float16)
    assert kernel.__class__.__name__ == "GQADecodeBs1Kernel"
    for real, tier in (
        (6000, "ctx"),
        (3072, "ctx"),
        (2048, "ctx"),
        (1024, "ctx"),
        (512, "no_split"),
    ):
        assert kernel._select_tier(real) == tier
        test = GroupedQueryAttentionDecodeTest(1, 32, 4, real, 128, torch.float16)
        test.check(op, *test.gen_inputs(), atol=1e-2, rtol=1e-2)


@pytest.mark.smoke
def test_gqa_decode_bs1_group4() -> None:
    """The WS kernel generalizes to a query-per-KV-head group other than 8 (here 4)."""
    op = GroupedQueryAttentionDecodeWithKVCacheFwdOp(1, 32, 8, 4096, 128)
    assert op._get_kernel((), torch.float16).__class__.__name__ == "GQADecodeBs1Kernel"
    test = GroupedQueryAttentionDecodeTest(1, 32, 8, 4096, 128, torch.float16)
    test.check(op, *test.gen_inputs(), atol=1e-2, rtol=1e-2)
