"""Dense MLA decode with grouped heads and non-contiguous saved cache tensors."""

import pytest
import torch

from tileops.attention import MLADenseDecodeFwdOp
from workloads.device import run_device


@pytest.mark.smoke
def test_ascend_mla_decode_grouped_strided_cache():
    device = torch.device(run_device())
    if device.type != "npu":
        pytest.skip("Ascend dense MLA decode regression")
    torch.manual_seed(17)
    b, h, hk, length, dim, pe = 2, 6, 2, 128, 64, 64
    inputs = [
        torch.randn(*shape, dtype=torch.float16)
        for shape in (
            (b, h, 2 * dim),
            (b, h, 2 * pe),
            (b, length, hk, 2 * dim),
            (b, length, hk, 2 * pe),
        )
    ]
    # Keep the feature stride on the device side of the operator boundary.
    q, qr, k, kr = [x[..., ::2].float() for x in inputs]
    keys = k.repeat_interleave(h // hk, dim=2)
    rope = kr.repeat_interleave(h // hk, dim=2)
    scores = (torch.einsum("bhd,bthd->bht", q, keys) + torch.einsum("bhd,bthd->bht", qr, rope)) * (
        dim + pe
    ) ** -0.5
    expected = torch.einsum("bht,bthd->bhd", scores.softmax(-1), keys)
    args = [v.to(device)[..., ::2] for v in inputs]
    output = MLADenseDecodeFwdOp()(*args)
    torch.testing.assert_close(output.cpu().float(), expected, rtol=0.01, atol=0.002)


@pytest.mark.smoke
def test_ascend_mla_paged_fp8_reuses_live_lengths_and_pages():
    from tileops.attention import MLAPagedFwdOp

    device = torch.device(run_device())
    if device.type != "npu":
        pytest.skip("Ascend paged MLA decode regression")
    torch.manual_seed(71)
    heads, dim, dim_v, page = 64, 576, 512, 64
    q = torch.randn(2, 1, heads, dim, dtype=torch.bfloat16)
    cache = torch.randn(6, page, dim).to(torch.float8_e4m3fn)
    qdev, pool = q.to(device), cache.to(device)
    table = torch.empty(2, 3, dtype=torch.int32, device=device)
    lengths = torch.empty(2, dtype=torch.int32, device=device)
    scale = torch.tensor([0.375], device=device)
    op = MLAPagedFwdOp(dim_v, is_causal=True)
    previous = None
    for pages, sizes in (
        ([[5, 0, 4], [1, 3, 2]], [129, 64]),
        ([[2, 4, 0], [5, 1, 3]], [32, 130]),
    ):
        table.copy_(torch.tensor(pages, dtype=torch.int32))
        lengths.copy_(torch.tensor(sizes, dtype=torch.int32))
        out, lse = op(qdev, pool, table, lengths, scale)
        outputs = []
        normalizers = []
        for i, size in enumerate(sizes):
            rows = cache.float()[pages[i]].reshape(-1, dim)[:size] * 0.375
            logits = q[i, 0].float() @ rows.T * dim**-0.5
            outputs.append(logits.softmax(-1) @ rows[:, :dim_v])
            normalizers.append(logits.logsumexp(-1))
        outputs = torch.stack(outputs)[:, None]
        torch.testing.assert_close(out.cpu().float(), outputs, rtol=0.01, atol=0.004)
        torch.testing.assert_close(
            lse.cpu().squeeze(-1), torch.stack(normalizers), rtol=0, atol=2e-4
        )
        if previous is not None:
            assert (previous - outputs).abs().max() > 0.1
        previous = outputs
