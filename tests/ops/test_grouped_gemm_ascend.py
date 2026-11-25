"""Ascend grouped-GEMM regression for the live expert-row geometry."""

import pytest
import torch

from tileops.moe import ContiguousLayoutSpec, MoEGroupedGEMMFwdOp
from workloads.device import run_device


@pytest.mark.smoke
def test_ascend_grouped_gemm_reloads_offsets_across_shapes():
    device = run_device()
    if torch.device(device).type != "npu":
        pytest.skip("Ascend grouped-GEMM regression")
    # M is excluded from MGroupedGEMMCall equality. Both M and device-side ends
    # may change between calls on the same operator, even with identical weights.
    op = MoEGroupedGEMMFwdOp(
        ContiguousLayoutSpec.tight_physical_psum(), out_dtype=torch.float32
    )
    weights = torch.randn(3, 40, 24, device=device, dtype=torch.float16)
    ends = torch.empty(3, device=device, dtype=torch.int32)
    weights_cpu = weights.cpu().float()
    for boundaries in ([5, 5, 19], [0, 17, 19], [6, 6, 35]):
        data = torch.randn(boundaries[-1], 24, device=device, dtype=torch.float16)
        ends.copy_(torch.tensor(boundaries, dtype=torch.int32, device=device))
        x = data.cpu().float()
        expected = torch.cat(
            [
                x[lo:hi] @ weights_cpu[g].T
                for g, (lo, hi) in enumerate(
                    zip([0, *boundaries[:-1]], boundaries, strict=True)
                )
            ]
        )
        got = op(data, weights, ends)
        torch.testing.assert_close(got.cpu(), expected, rtol=1e-3, atol=1e-3)


@pytest.mark.smoke
def test_ascend_grouped_gemm_returns_empty_output_buffer():
    device = torch.device(run_device())
    if device.type != "npu":
        pytest.skip("requires Ascend")
    op = MoEGroupedGEMMFwdOp(ContiguousLayoutSpec.tight_physical_psum())
    a = torch.empty((3, 32), device=device, dtype=torch.float16)
    weights = a.new_empty((2, 0, 32))
    ends = torch.tensor([1, 3], device=device, dtype=torch.int32)
    out = a.new_empty((3, 0))
    assert op(a, weights, ends, out=out) is out
    assert op(a, weights, ends).shape == out.shape
