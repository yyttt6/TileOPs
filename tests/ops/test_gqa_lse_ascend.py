"""Both results of dense attention over unequal lengths and an empty causal window."""

import pytest
import torch

from tileops.attention import GQADenseWithLSEFwdOp
from workloads.attention.gqa.dense import dense_gqa_logits
from workloads.device import run_device


@pytest.mark.smoke
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    ("sq", "sk", "causal", "window", "cap"),
    [(128, 384, True, -1, None), (71, 157, True, -1, 3.0), (37, 19, False, 0, 3.0)],
)
def test_ascend_dense_lse_mask_and_base(dtype, sq, sk, causal, window, cap):
    device = torch.device(run_device())
    if device.type != "npu":
        pytest.skip("Ascend packed GQA training route")
    logits_options = dict(
        heads=8,
        heads_kv=2,
        is_causal=causal,
        sm_scale=0.13,
        softcap=cap,
        window_size_left=window,
        window_size_right=window,
    )
    q = torch.randn(2, sq, 8, 64, dtype=dtype, device=device)
    k = torch.randn(2, sk, 2, 64, dtype=dtype, device=device)
    v = torch.randn_like(k)
    op = GQADenseWithLSEFwdOp(
        is_causal=causal,
        sm_scale=0.13,
        softcap=cap,
        window_size_left=window,
        window_size_right=window,
    )
    out, lse = op(q, k, v)
    scores = dense_gqa_logits(q.cpu(), k.cpu(), **logits_options)
    expected_lse = scores.logsumexp(dim=-1)
    probs = scores.softmax(-1).nan_to_num()
    expected = (probs @ v.cpu().repeat_interleave(4, dim=2).transpose(1, 2).float()).transpose(1, 2)
    assert lse.dtype == torch.float32 and out.dtype == dtype
    torch.testing.assert_close(lse.cpu(), expected_lse, rtol=1.0e-3, atol=3.0e-3)
    torch.testing.assert_close(out.cpu().float(), expected, rtol=2.0e-2, atol=1.0e-2)
