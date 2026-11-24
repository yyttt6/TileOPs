"""Tests for multi-head attention paged decode."""

import pytest
import torch

from tests.workload_test_base import FixtureBase, TestBase
from tileops.attention import MHAPagedFwdOp
from workloads.attention.mha import MHADecodePagedWorkload
from workloads.device import run_device, run_device_available
from workloads.numerics import compare_outputs


class MHADecodePagedTest(MHADecodePagedWorkload, TestBase):
    pass


class MHADecodePagedFixture(FixtureBase):
    PARAMS = [
        (
            "batch, heads, seqlen_q, seqlen_kv, dim, page_size, is_causal, dtype, tune",
            [
                pytest.param(
                    1,
                    16,
                    1,
                    512,
                    128,
                    128,
                    False,
                    torch.float16,
                    False,
                    marks=[pytest.mark.smoke, pytest.mark.packaging(family="attention")],
                ),
                # bfloat16 dispatch: the same signature admits it and the paged
                # decode kernels are selected on dtype.
                pytest.param(
                    1,
                    16,
                    1,
                    1024,
                    128,
                    128,
                    False,
                    torch.bfloat16,
                    False,
                    marks=pytest.mark.smoke,
                ),
                pytest.param(
                    1,
                    8,
                    1,
                    1024,
                    64,
                    256,
                    False,
                    torch.float16,
                    False,
                    marks=pytest.mark.full,
                ),
                pytest.param(
                    2,
                    8,
                    1,
                    1024,
                    64,
                    256,
                    False,
                    torch.float16,
                    False,
                    marks=pytest.mark.full,
                ),
                pytest.param(
                    1,
                    8,
                    1,
                    512,
                    64,
                    256,
                    False,
                    torch.float16,
                    False,
                    marks=pytest.mark.full,
                ),
                pytest.param(
                    4,
                    8,
                    1,
                    2048,
                    64,
                    128,
                    False,
                    torch.float16,
                    True,
                    marks=pytest.mark.full,
                ),
                # Enough outputs that the default config runs two splits, then one.
                pytest.param(
                    5,
                    32,
                    1,
                    1024,
                    64,
                    128,
                    False,
                    torch.float16,
                    False,
                    marks=pytest.mark.full,
                ),
                pytest.param(
                    9,
                    32,
                    1,
                    1024,
                    64,
                    128,
                    False,
                    torch.float16,
                    False,
                    marks=pytest.mark.full,
                ),
            ],
        ),
    ]


@MHADecodePagedFixture
def test_mha_decode_paged_op(
    batch: int,
    heads: int,
    seqlen_q: int,
    seqlen_kv: int,
    dim: int,
    page_size: int,
    is_causal: bool,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    test = MHADecodePagedTest(batch, heads, seqlen_q, seqlen_kv, dim, page_size, is_causal, dtype)
    op = MHAPagedFwdOp(page_size=page_size, is_causal=is_causal)
    if tune:
        op.request_tune()
    test.check(op, *test.gen_inputs())


@pytest.mark.parametrize(
    "seqlen_q, is_causal, real_lengths",
    [
        pytest.param(1, False, [1], marks=pytest.mark.smoke),
        # Several queries; a cache shorter than the query block leaves the first ones no key.
        pytest.param(2, True, [65], marks=pytest.mark.smoke),
        pytest.param(4, True, [700, 3], marks=pytest.mark.smoke),
        pytest.param(4, False, [700, 3], marks=pytest.mark.smoke),
        pytest.param(1, False, [37], marks=pytest.mark.full),
        pytest.param(1, False, [700, 1024, 1], marks=pytest.mark.full),
    ],
)
def test_mha_decode_paged_cache_shorter_than_bound(
    seqlen_q: int, is_causal: bool, real_lengths: list
) -> None:
    """Splits, warps and query rows that see no key keep the output finite and exact.

    A cache far shorter than the static bound, or shorter than the causal queries,
    leaves them with no live score. Pool rows no request reads hold NaN.
    """
    _check_partly_filled_cache(seqlen_q, is_causal, real_lengths)


@pytest.mark.sm89
@pytest.mark.smoke
def test_mha_decode_paged_without_wgmma_drops_unread_rows() -> None:
    """SM89 has no TMA or WGMMA, so its paged decode stages the key tile the cache end cuts
    without them; the NaN rows past the cache end must not reach the output."""
    _check_partly_filled_cache(1, False, [700])


def _check_partly_filled_cache(seqlen_q: int, is_causal: bool, real_lengths: list) -> None:
    batch, heads, seqlen_kv, dim, page_size = len(real_lengths), 8, 1024, 64, 256
    test = MHADecodePagedTest(
        batch, heads, seqlen_q, seqlen_kv, dim, page_size, is_causal, torch.float16
    )
    q, k, v, block_table, _full = test.gen_inputs()
    cache_seqlens = torch.tensor(real_lengths, dtype=torch.int32, device=q.device)
    pos = torch.arange(block_table.shape[1] * page_size, device=q.device)
    rows = block_table[:, pos // page_size] * page_size + pos % page_size
    read = torch.zeros(seqlen_kv, dtype=torch.bool, device=q.device)
    read[rows[pos < cache_seqlens[:, None]]] = True
    k[~read] = float("nan")
    v[~read] = float("nan")

    op = MHAPagedFwdOp(page_size=page_size, is_causal=is_causal)
    output = op(q, k, v, block_table, cache_seqlens)

    assert torch.isfinite(output).all(), "output is not finite for a partly filled cache"
    compare_outputs(
        output, test.ref_program(q, k, v, block_table, cache_seqlens), test.verification(q)
    )


@pytest.mark.smoke
def test_mha_decode_paged_table_width_is_independent_of_pool() -> None:
    test = MHADecodePagedTest(2, 8, 1, 1024, 64, 256, False, torch.float16)
    q, k, v, table, lengths = test.gen_inputs()
    lengths.fill_(512)
    op = MHAPagedFwdOp(page_size=256)
    for width in (2, 4):
        block_table = table[:, :width].contiguous()
        output = op(q, k, v, block_table, lengths)
        compare_outputs(
            output, test.ref_program(q, k, v, block_table, lengths), test.verification(q)
        )


@pytest.mark.smoke
@pytest.mark.parametrize(
    "q_len,live_kv",
    [
        pytest.param(129, 512, id="direct-q-tail"),
        pytest.param(64, 4096, id="split-and-merge"),
        pytest.param(129, 256, id="partially-filled-page-table"),
    ],
)
def test_mha_paged_ascend(q_len: int, live_kv: int) -> None:
    if torch.device(run_device()).type != "npu" or not run_device_available():
        pytest.skip("requires Ascend NPU")
    workload = MHADecodePagedWorkload(
        1, 1, q_len, max(512, live_kv), 128, 128, False, torch.bfloat16
    )
    q, k, v, table, lengths = workload.gen_inputs()
    lengths.fill_(live_kv)
    op = MHAPagedFwdOp(page_size=128)
    output = op(q, k, v, table, lengths)
    compare_outputs(output, workload.ref_program(q, k, v, table, lengths), workload.verification(q))


@pytest.mark.smoke
def test_mha_paged_empty_request_and_reused_metadata() -> None:
    """One op consumes new device lengths and page IDs on every invocation."""
    device = run_device()
    test = MHADecodePagedTest(2, 8, 4, 512, 64, 128, True, torch.float16)
    q, k, v, table, lengths = test.gen_inputs()
    op = MHAPagedFwdOp(page_size=128, is_causal=True)
    for first, second in ((0, 129), (65, 2)):
        lengths.copy_(torch.tensor([first, second], dtype=torch.int32, device=device))
        table.copy_(table.flip([1]))
        result = op(q, k, v, table, lengths)
        ref = test.ref_program(q, k, v, table, lengths)
        torch.testing.assert_close(result.float(), ref.float(), rtol=1e-3, atol=1e-3)
        if first == 0:
            assert torch.count_nonzero(result[0]).item() == 0
