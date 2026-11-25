"""Gradients across grouped heads, a partial sequence tile, and saved natural-log LSE."""

import pytest
import torch

from tileops.attention import GQADenseBwdOp
from workloads.attention.gqa.dense import dense_gqa_logits
from workloads.device import run_device


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("dim", "dtype", "causal", "cap", "window"),
    [
        (64, torch.bfloat16, True, None, -1),
        (24, torch.float16, False, 2.0, 11),
        (64, torch.bfloat16, True, None, 32),
        (64, torch.float16, True, 3.0, -1),
    ],
)
def test_ascend_gqa_backward_uses_saved_statistics(dim, dtype, causal, cap, window):
    dev = torch.device(run_device())
    if dev.type != "npu":
        pytest.skip("Ascend GQA backward regression")
    b, s, h, hk = 2, 73, 4, 2
    q, k, v = (torch.randn(b, s, n, dim, dtype=dtype).float().requires_grad_() for n in (h, hk, hk))
    do = torch.randn(b, s, h, dim, dtype=dtype)
    options = dict(
        heads=h,
        heads_kv=hk,
        is_causal=causal,
        sm_scale=0.15,
        softcap=cap,
        window_size_left=window,
        window_size_right=-1,
    )
    score = dense_gqa_logits(q, k, **options)
    value = v.repeat_interleave(h // hk, 2).transpose(1, 2)
    out = (score.softmax(-1) @ value).transpose(1, 2)
    ref = torch.autograd.grad(out, (q, k, v), do.float())
    op = GQADenseBwdOp(is_causal=causal, sm_scale=0.15, softcap=cap, window_size_left=window)
    lse = score.logsumexp(-1).detach().to(dev)
    args = [x.detach().to(dev, dtype=dtype) for x in (do, q, k, v, out)]
    result = op(*args, lse)
    for actual, expected in zip(result, ref, strict=True):
        torch.testing.assert_close(actual.cpu().float(), expected, rtol=0.03, atol=0.012)
    # A missing row in the saved forward contributes zero to every gradient.
    lse.fill_(-float("inf"))
    for actual in op(*args, lse):
        torch.testing.assert_close(actual, torch.zeros_like(actual), atol=0, rtol=0)
