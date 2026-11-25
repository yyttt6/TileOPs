"""Exercise the Ascend post-permute output buffer with mixed input/output dtypes."""

import pytest
import torch

from tileops.moe import ContiguousLayoutSpec, MoEPostPermuteFwdOp, RoutingEpilogueSpec
from workloads.device import run_device


@pytest.mark.smoke
def test_post_permute_ascend_reloads_routes_and_preserves_workspace():
    device = run_device()
    if torch.device(device).type != "npu":
        pytest.skip("Ascend post-permute regression")
    # Odd-width rows and repeated source indices cover partial vector stores and gathers.
    tokens, hidden = 2, 67
    experts = torch.randn(5, hidden, device=device, dtype=torch.float16)
    weights = torch.tensor([[0.5, 0.25], [-1.0, 0.75]], device=device)
    indices = torch.tensor([3, 1, 4, 3], device=device, dtype=torch.int32)
    workspace = torch.full((2 + tokens * hidden,), -810.0, device=device, dtype=torch.bfloat16)
    out = workspace[1:-1].view(tokens, hidden)
    op = MoEPostPermuteFwdOp(
        ContiguousLayoutSpec.tight_physical_psum(),
        RoutingEpilogueSpec(routed_scaling_factor=1.5),
        out_dtype=torch.bfloat16,
    )
    for order in ([3, 1, 4, 3], [0, 2, 0, 1]):
        indices.copy_(torch.tensor(order, dtype=torch.int32, device=device))
        expected = (
            (experts.cpu().float()[order].view(tokens, 2, hidden) * weights.cpu().unsqueeze(-1))
            .sum(1)
            .mul(1.5)
            .bfloat16()
        )
        assert op(experts, weights, indices, out=out) is out
        torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
        torch.testing.assert_close(
            workspace.cpu()[[0, -1]], torch.full((2,), -810.0, dtype=torch.bfloat16)
        )


@pytest.mark.smoke
@pytest.mark.parametrize("overlap", [True, False])
def test_ascend_post_permute_shared_allocation(overlap):
    """Overlapping views are refused before writes, adjacent views can share the allocation."""
    device = run_device()
    if torch.device(device).type != "npu":
        pytest.skip("Ascend post-permute regression")
    width = 32
    workspace = torch.arange(3 * width, dtype=torch.float16, device=device)
    experts = workspace[: 2 * width].view(2, width)
    begin = 2 * width - int(overlap)
    out = workspace[begin : begin + width].view(1, width)
    weights = torch.tensor([[0.25, 0.75]], dtype=torch.float32, device=device)
    indices = torch.tensor([1, 0], dtype=torch.int32, device=device)
    op = MoEPostPermuteFwdOp(ContiguousLayoutSpec.tight_physical_psum())
    saved = workspace.cpu().clone()
    if overlap:
        with pytest.raises(ValueError, match="overlap"):
            op(experts, weights, indices, out=out)
        torch.testing.assert_close(workspace.cpu(), saved, rtol=0, atol=0)
    else:
        assert op(experts, weights, indices, out=out) is out
        expected = (saved[width : 2 * width].float() * 0.25 + saved[:width].float() * 0.75).half()
        torch.testing.assert_close(out.cpu()[0], expected, rtol=0, atol=0)
        torch.testing.assert_close(workspace.cpu()[:begin], saved[:begin], rtol=0, atol=0)
