import pytest
import torch

from tileops.moe import ContiguousLayoutSpec, MoEPrePermuteFwdOp
from workloads.device import run_device


@pytest.mark.smoke
def test_large_pre_permute_stable_chunk_boundaries():
    device = run_device()
    if torch.device(device).type != "npu":
        pytest.skip("Ascend large-route regression")
    tokens, top_k, experts, width = 1027, 5, 17, 80
    x = torch.randn(tokens, width, device=device, dtype=torch.bfloat16)
    original = x.cpu().clone()
    n = tokens * top_k
    op = MoEPrePermuteFwdOp(ContiguousLayoutSpec.tight_physical_psum(), experts)
    ids = torch.empty((tokens, top_k), dtype=torch.int32, device=device)
    # Exercise a partial last shard, unused experts, and a changed route map.
    for ref_ids in (torch.arange(n).remainder(5) * 3, torch.full((n,), experts - 1)):
        ids.copy_(ref_ids.int().reshape_as(ids))
        packed, ends, inverse = op(x, ids)
        order = torch.argsort(ref_ids, stable=True)
        expected_inverse = torch.argsort(order).int()
        torch.testing.assert_close(packed.cpu(), original[order // top_k], rtol=0, atol=0)
        torch.testing.assert_close(inverse.cpu(), expected_inverse, rtol=0, atol=0)
        torch.testing.assert_close(
            ends.cpu(),
            torch.bincount(ref_ids, minlength=experts).cumsum(0).int(),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(ids.cpu().flatten(), ref_ids.int(), rtol=0, atol=0)
        torch.testing.assert_close(x.cpu(), original, rtol=0, atol=0)
