"""Rectangular prefill gradients using saved natural-log normalization on Ascend."""

import pytest
import torch

from tileops.attention import MLADenseBwdOp, MLAVarlenBwdOp
from workloads.device import run_device


@pytest.mark.smoke
@pytest.mark.parametrize("packed", [False, True])
def test_ascend_mla_backward_rectangular_empty_rows(packed):
    device = torch.device(run_device())
    if device.type != "npu":
        pytest.skip("Ascend MLA backward regression")
    dtype = torch.float16
    heads, dim, dim_v = 2, 64, 32
    torch.manual_seed(31)
    # The first request has more than a whole query tile with no visible key.
    qlens, klens = ((81, 7), (9, 34)) if packed else ((81, 81), (9, 9))
    cq = torch.tensor([0, qlens[0], sum(qlens)], dtype=torch.int32)
    ck = torch.tensor([0, klens[0], sum(klens)], dtype=torch.int32)
    q, k, v = (
        torch.randn(length, heads, width, dtype=dtype).float().requires_grad_()
        for length, width in ((sum(qlens), dim), (sum(klens), dim), (sum(klens), dim_v))
    )
    do = torch.randn(sum(qlens), heads, dim_v, dtype=dtype)
    out, lse = [], []
    scale = 0.13
    for b, (sq, sk) in enumerate(zip(qlens, klens, strict=True)):
        qr, kr = slice(cq[b], cq[b + 1]), slice(ck[b], ck[b + 1])
        logits = torch.einsum("qhd,khd->hqk", q[qr], k[kr]) * scale
        visible = torch.arange(sk)[None, :] <= (torch.arange(sq) + sk - sq)[:, None]
        logits = logits.masked_fill(~visible, -torch.inf)
        lse.append(logits.detach().logsumexp(-1))
        any_key = visible.any(-1)[:, None]
        weights = torch.where(any_key, logits, 0.0).softmax(-1) * any_key
        out.append(torch.einsum("hqk,khd->qhd", weights, v[kr]))
    out = torch.cat(out)
    expected = torch.autograd.grad(out, (q, k, v), do.float())
    args = [x.detach().to(device=device, dtype=dtype) for x in (do, q, k, v, out)]
    normalizer = torch.cat(lse, dim=-1).to(device)
    if packed:
        op = MLAVarlenBwdOp(is_causal=True, sm_scale=scale)
        actual = op(*args, normalizer, cq.to(device), ck.to(device))
    else:
        op = MLADenseBwdOp(is_causal=True, sm_scale=scale)
        args = [x.reshape(2, -1, *x.shape[1:]) for x in args]
        actual = op(*args, normalizer.reshape(heads, 2, -1).transpose(0, 1))
    for got, ref in zip(actual, expected, strict=True):
        torch.testing.assert_close(got.reshape_as(ref).cpu().float(), ref, rtol=0.01, atol=0.005)
    assert torch.count_nonzero(actual[0].reshape_as(q)[: qlens[0] - klens[0]]) == 0
