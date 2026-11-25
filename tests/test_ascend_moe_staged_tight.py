"""Check each public staged-tight operator against an independent CPU reference."""

import pytest
import torch

from workloads.device import run_device, run_device_available
from workloads.moe import ref_fused_topk, ref_routed_experts


@pytest.mark.smoke
@pytest.mark.parametrize("grouped", [False, True], ids=["softmax", "grouped-sigmoid"])
def test_ascend_staged_tight(grouped):
    device = torch.device(run_device())
    if device.type != "npu" or not run_device_available():
        pytest.skip("requires --tileops-device=npu")
    from tileops import ascend_moe as moe
    from tileops.moe import ContiguousLayoutSpec, RoutingEpilogueSpec

    t, e, k, h, f = 8, 32, 2, 128, 64
    gen = torch.Generator().manual_seed(751)
    logits = torch.randn(t, e, generator=gen)
    bias = torch.randn(e, generator=gen) if grouped else None
    opts = dict(
        top_k=k,
        scoring_func="sigmoid" if grouped else "softmax",
        renormalize=True,
        num_expert_group=4 if grouped else 1,
        topk_group=2 if grouped else 1,
    )
    scale = 2.5 if grouped else 1.0
    layout = ContiguousLayoutSpec.tight_physical_psum()
    dtype = torch.bfloat16
    x = torch.randn(t, h, generator=gen).to(dtype)
    w13 = (torch.randn(e, 2 * f, h, generator=gen) / h**0.5).to(dtype)
    w2 = (torch.randn(e, h, f, generator=gen) / f**0.5).to(dtype)
    dx, dw13, dw2, dg = (v.to(device) for v in (x, w13, w2, logits))
    db = bias.to(device) if grouped else None

    weights, ids = moe.FusedTopKFwdOp(**opts)(dg, db)
    ref_weights, ref_ids = ref_fused_topk(logits, bias, **opts)
    ids_sorted, order = ids.cpu().sort(dim=1)
    expected_sorted, expected_order = ref_ids.sort(dim=1)
    torch.testing.assert_close(ids_sorted.long(), expected_sorted.long())
    torch.testing.assert_close(
        weights.cpu().gather(1, order), ref_weights.gather(1, expected_order), atol=1e-5, rtol=1e-5
    )
    packed, ends, inverse = moe.MoEPrePermuteFwdOp(layout, e)(dx, ids)
    routes = ids.cpu().flatten().long()
    inv = inverse.cpu().long()
    counts = routes.bincount(minlength=e)
    torch.testing.assert_close(ends.cpu().long(), counts.cumsum(0))
    torch.testing.assert_close(inv.sort().values, torch.arange(t * k))
    torch.testing.assert_close(torch.arange(e).repeat_interleave(counts)[inv], routes)
    torch.testing.assert_close(packed.cpu()[inv], x.repeat_interleave(k, dim=0), atol=0, rtol=0)

    expert_y = moe.MoEExpertMLPFwdOp(layout, num_cores=32)(packed, dw13, dw2, ends)
    packed_cpu = packed.cpu()
    expected_mlp = torch.empty_like(packed_cpu)
    start = 0
    for expert, end in enumerate(ends.cpu().tolist()):
        if end > start:
            gate, up = (packed_cpu[start:end].float() @ w13[expert].float().T).chunk(2, dim=-1)
            activated = (torch.nn.functional.silu(gate) * up).to(dtype)
            expected_mlp[start:end] = (activated.float() @ w2[expert].float().T).to(dtype)
        start = end
    torch.testing.assert_close(expert_y.cpu(), expected_mlp, atol=0.02, rtol=0.02)

    post = moe.MoEPostPermuteFwdOp(layout, RoutingEpilogueSpec(routed_scaling_factor=scale))
    expected_post = (
        (expert_y.cpu()[inv].view(t, k, h).float() * weights.cpu()[..., None]).sum(1) * scale
    ).to(dtype)
    torch.testing.assert_close(
        post(expert_y, weights, inverse).cpu(), expected_post, atol=0.02, rtol=0.02
    )
    expected = ref_routed_experts(x, w13, w2, ref_weights, ref_ids, scale=scale)
    experts = moe.FusedMoEExpertsFwdOp(routed_scaling_factor=scale, num_cores=32)
    fused = moe.FusedMoEFwdOp(**opts, routed_scaling_factor=scale, num_cores=32)
    torch.testing.assert_close(
        experts(dx, dw13, dw2, weights, ids).cpu(), expected, atol=0.02, rtol=0.02
    )
    torch.testing.assert_close(fused(dx, dg, dw13, dw2, db).cpu(), expected, atol=0.02, rtol=0.02)


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("experts", "bias_value"),
    [(64, torch.finfo(torch.float32).min), (64, -3.2e38), (66, torch.finfo(torch.float32).min)],
    ids=["review-repro", "below-old-sentinel", "partial-warp"],
)
def test_grouped_topk_with_extreme_finite_bias(experts, bias_value):
    device = torch.device(run_device())
    if device.type != "npu" or not run_device_available():
        pytest.skip("requires --tileops-device=npu")
    from tileops import ascend_moe

    logits = torch.linspace(-2, 2, experts, dtype=torch.float32).repeat(2, 1)
    logits[1].neg_()
    bias = torch.full((experts,), bias_value, dtype=torch.float32, device=device)
    op = ascend_moe.FusedTopKFwdOp(
        top_k=4,
        scoring_func="sigmoid",
        renormalize=True,
        num_expert_group=2,
        topk_group=1,
        routed_scaling_factor=1.5,
    )
    weights, ids = (value.cpu() for value in op(logits.to(device), bias))
    assert ids.dtype == torch.int32 and ids.shape == (2, 4)
    assert weights.dtype == torch.float32 and weights.shape == (2, 4)
    # Adding sigmoid scores to these biases rounds to a tie. Both group sums
    # underflow, so route to the first group and break expert ties by ID.
    torch.testing.assert_close(ids, torch.arange(4, dtype=torch.int32).expand(2, 4))
    assert torch.isfinite(weights).all()
    torch.testing.assert_close(weights.sum(dim=1), torch.full((2,), 1.5))
    expected = logits.sigmoid().gather(1, ids.long())
    expected = expected / expected.sum(dim=1, keepdim=True) * 1.5
    torch.testing.assert_close(weights, expected, atol=1e-6, rtol=1e-5)
