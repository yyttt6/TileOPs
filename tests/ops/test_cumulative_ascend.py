"""Explicit scan output dtype on empty, strided and tail inputs."""

import pytest
import torch

from tileops.reduction import CumprodFwdOp, CumsumFwdOp
from workloads.device import run_device


@pytest.mark.smoke
@pytest.mark.parametrize(
    "op,torch_op", [(CumsumFwdOp, torch.cumsum), (CumprodFwdOp, torch.cumprod)]
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("length", [0, 65])
@pytest.mark.parametrize("strided", [False, True])
def test_ascend_scan_explicit_fp32_output(op, torch_op, dtype, length, strided):
    if str(run_device()).split(":")[0] != "npu":
        pytest.skip("Ascend-specific scan dtype regression")
    width = length * (2 if strided else 1)
    base = (1 + torch.arange(2 * width, device=run_device()).reshape(2, width).float() * 0.0001).to(
        dtype
    )
    inputs = base[:, ::2] if strided else base
    saved = base.cpu().clone()
    expected = torch_op(inputs.cpu(), dim=-1, dtype=torch.float32)
    actual = op(dim=-1, dtype=torch.float32)(inputs)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual.cpu(), expected, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(base.cpu(), saved, atol=0, rtol=0)
