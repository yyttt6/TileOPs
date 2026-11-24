"""Paged FP8 operands stay representable through the FP32 logit scaling stage."""

import pytest
import torch

from tileops.attention import GQAPagedFwdOp
from workloads.device import run_device


@pytest.mark.smoke
@pytest.mark.parametrize("per_request", [False, True])
def test_ascend_paged_fp8_compensating_scales(per_request):
    dev = torch.device(run_device())
    if dev.type != "npu":
        pytest.skip("Ascend paged FP8 scale regression")
    # A KV group straddles the 32-head tile.
    b, h, hk, dim, ps = 2, 68, 2, 64, 16
    torch.manual_seed(82)
    q = torch.randn(12, h, dim).to(torch.float8_e4m3fn)
    k = torch.randn(4, ps, hk, dim).to(torch.float8_e4m3fn)
    v = torch.randn(4, ps, hk, dim).to(torch.float8_e4m3fn)
    qscale = torch.full((b, hk), 2.0**18)
    kscale = torch.full((b, hk) if per_request else (1,), 2.0**-18)
    vscale = torch.full_like(kscale, 1.5)
    table = torch.tensor([[2, 0], [3, 1]], dtype=torch.int32)
    lengths = torch.tensor([19, 23], dtype=torch.int32)
    cu = torch.tensor([0, 3, 12], dtype=torch.int32)
    op = GQAPagedFwdOp(is_causal=False, out_dtype=torch.bfloat16)
    result = op(*(x.to(dev) for x in (q, k, v, table, lengths, cu, qscale, kscale, vscale)))
    expected = []
    for n in range(b):
        key = k.float()[table[n].long()].flatten(0, 1)[: lengths[n]].repeat_interleave(h // hk, 1)
        val = (
            v.float()[table[n].long()].flatten(0, 1)[: lengths[n]].repeat_interleave(h // hk, 1)
            * 1.5
        )
        score = torch.einsum("qhd,khd->hqk", q.float()[cu[n] : cu[n + 1]], key) * dim**-0.5
        expected.append(torch.einsum("hqk,khd->qhd", score.softmax(-1), val))
    assert result.dtype == torch.bfloat16
    torch.testing.assert_close(result.cpu().float(), torch.cat(expected), atol=0.008, rtol=0.012)


@pytest.mark.smoke
def test_ascend_paged_ragged_padding_preserves_causal_positions():
    dev = torch.device(run_device())
    if dev.type != "npu":
        pytest.skip("Ascend ragged paged prefill regression")
    torch.manual_seed(391)
    batch, page, width, heads, kh, dim = 3, 48, 4, 8, 2, 64
    q = torch.randn(199, heads, dim, dtype=torch.float16)
    keys = torch.randn(batch * width, page, kh, dim, dtype=torch.float16)
    values = torch.randn_like(keys)
    qd = q.to(dev)
    kd, vd = (torch.empty_like(x, device=dev) for x in (keys, values))
    table = torch.empty(batch, width, dtype=torch.int32, device=dev)
    lengths = torch.empty(batch, dtype=torch.int32, device=dev)
    bounds = torch.empty(batch + 1, dtype=torch.int32, device=dev)
    op = GQAPagedFwdOp(is_causal=True, window_size_left=17)
    for qo, lens, flip in (
        ([0, 139, 190, 199], [101, 133, 0], False),
        ([0, 38, 38, 199], [141, 67, 138], True),
    ):
        order = torch.arange(batch * width, dtype=torch.int32).view(batch, width)
        if flip:
            order = order.flip((0, 1)).contiguous()
        table.copy_(order)
        lengths.copy_(torch.tensor(lens, dtype=torch.int32))
        bounds.copy_(torch.tensor(qo, dtype=torch.int32))
        k, v = keys.clone(), values.clone()
        expected = torch.zeros_like(q)
        for b, lk in enumerate(lens):
            sq = qo[b + 1] - qo[b]
            dead = torch.arange(lk, width * page)
            k[order[b, dead // page].long(), dead % page] = torch.nan
            v[order[b, dead // page].long(), dead % page] = torch.nan
            if not sq or not lk:
                continue
            key = k[order[b].long()].flatten(0, 1)[:lk].float().repeat_interleave(heads // kh, 1)
            val = v[order[b].long()].flatten(0, 1)[:lk].float().repeat_interleave(heads // kh, 1)
            qr = slice(qo[b], qo[b + 1])
            scores = torch.einsum("qhd,khd->hqk", q[qr].float(), key) * dim**-0.5
            pos = torch.arange(sq) + lk - sq
            live = (torch.arange(lk) <= pos[:, None]) & (torch.arange(lk) >= pos[:, None] - 17)
            prob = scores.masked_fill(~live, -torch.inf).softmax(-1).nan_to_num()
            expected[qr] = torch.einsum("hqk,khd->qhd", prob, val).half()
        kd.copy_(k)
        vd.copy_(v)
        actual = op(qd, kd, vd, table, lengths, bounds)
        torch.testing.assert_close(actual.cpu(), expected, rtol=0.012, atol=0.004)
