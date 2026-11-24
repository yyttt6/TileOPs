"""Ascend FP8 transpose must preserve raw bytes, and scales remain live on the device."""

import pytest
import torch

from tileops.gemm import BMMFP8FwdOp
from workloads.device import run_device


@pytest.mark.smoke
def test_ascend_bmm_fp8_transpose_preserves_every_bit():
    device = torch.device(run_device())
    if device.type != "npu":
        pytest.skip("Ascend FP8 transpose")
    from tileops._kernels.ascend.gemm.bmm_fp8 import AscendBMMFP8TransposeKernel
    from tileops._kernels.gemm.call_spec import BMMFP8TransposeCall

    batch, rows, cols = 2, 96, 144
    raw = (
        torch.arange(batch * rows * cols, dtype=torch.int32)
        .to(torch.uint8)
        .reshape(batch, rows, cols)
    )
    src = raw.to(device).view(torch.float8_e4m3fn)
    kernel = AscendBMMFP8TransposeKernel(
        BMMFP8TransposeCall(batch=batch, rows=rows, cols=cols, dtype=src.dtype, device=src.device)
    )
    out = kernel(src)
    assert out.is_contiguous()
    torch.testing.assert_close(out.view(torch.uint8).cpu(), raw.transpose(1, 2), rtol=0, atol=0)


@pytest.mark.smoke
def test_ascend_bmm_fp8_changes_scale_between_calls():
    device = torch.device(run_device())
    if device.type != "npu":
        pytest.skip("Ascend FP8 scale and tail regression")
    batch, m, n, k = 3, 145, 144, 96
    a = torch.randn(batch, m, k).to(torch.float8_e4m3fn).to(device)
    b = torch.randn(batch, n, k).to(torch.float8_e4m3fn).to(device)
    scale_a = torch.tensor(0.5, device=device)
    scale_b = torch.tensor(1.5, device=device)
    op = BMMFP8FwdOp(out_dtype=torch.bfloat16, trans_b=True)
    first = op(a, b, scale_a, scale_b)
    expected = (a.cpu().float() @ b.cpu().float().transpose(1, 2)) * 0.75
    torch.testing.assert_close(first.cpu().float(), expected, rtol=0.01, atol=0.02)
    saved = first.cpu().clone()
    scale_b.neg_()
    torch.testing.assert_close(
        op(a, b, scale_a, scale_b).cpu().float(), -saved.float(), rtol=0, atol=0
    )
    torch.testing.assert_close(first.cpu(), saved, rtol=0, atol=0)
