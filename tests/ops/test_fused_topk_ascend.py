"""Ascend fused top-k selection and public-device correctness coverage."""

import dataclasses

import pytest
import torch

from tileops._kernels.ascend.moe.fused_topk import AscendFusedTopKKernel
from tileops._kernels.moe.call_spec import FusedTopKCall, FusedTopKFwdInterface
from tileops.manifest import entry_plan, instantiate, load_adts, load_manifest, load_workloads
from workloads.device import run_device, run_device_available
from workloads.moe import FusedTopKWorkload
from workloads.numerics import compare_outputs

_OP = "FusedTopKFwdOp"
_KERNEL = AscendFusedTopKKernel


@pytest.mark.smoke
def test_ascend_moe_public_registration():
    from tileops._kernels.ascend.moe import AscendMoEGroupedGEMMFP8Kernel
    from tileops.backend._registry import REGISTERED_KERNEL_TYPES

    assert REGISTERED_KERNEL_TYPES[_OP]["ascend"] is _KERNEL
    assert (
        REGISTERED_KERNEL_TYPES["MoEGroupedGEMMFP8FwdOp"]["ascend"] is AscendMoEGroupedGEMMFP8Kernel
    )


def _cases():
    plan = entry_plan(_OP, load_manifest()[_OP], load_adts(), resolve=False)
    return [
        instantiate(plan, row, dtype)
        for row in load_workloads(_OP)
        for dtype in row.get("dtype_cases") or [{}]
    ]


def _request(case):
    (tokens, experts), dtype = case.tensors["gating_output"]
    p = case.params
    return FusedTopKCall(
        num_tokens=tokens,
        num_experts=experts,
        top_k=p["top_k"],
        scoring_func=p["scoring_func"],
        renormalize=p["renormalize"],
        with_correction_bias="correction_bias" in case.tensors,
        num_expert_group=p["num_expert_group"],
        topk_group=p["topk_group"],
        routed_scaling_factor=p["routed_scaling_factor"],
        dtype=getattr(torch, dtype),
    )


@pytest.mark.smoke
def test_every_manifest_case_is_served():
    cases = _cases()
    assert len(cases) == 17
    assert issubclass(_KERNEL, FusedTopKFwdInterface)
    for case in cases:
        assert _KERNEL.refusal(_request(case)) is None


@pytest.mark.smoke
@pytest.mark.parametrize(
    "changes",
    [
        {"num_tokens": 0},
        {"top_k": 0},
        {"top_k": 385},
        {"dtype": torch.float64},
        {"scoring_func": "softmax", "with_correction_bias": True},
        {"num_expert_group": 0},
        {"num_expert_group": 7},
        {"topk_group": 2},
        {"num_experts": 400},
    ],
)
def test_unsupported_contracts_are_explicitly_refused(changes):
    assert _KERNEL.refusal(dataclasses.replace(_request(_cases()[0]), **changes))


@pytest.mark.smoke
@pytest.mark.parametrize("case", _cases(), ids=lambda c: c.case_id)
def test_public_ascend_fused_topk(case):
    if torch.device(run_device()).type != "npu" or not run_device_available():
        pytest.skip("requires an Ascend NPU and --tileops-device=npu")
    workload = FusedTopKWorkload(case)
    inputs = workload.gen_inputs()
    built = _KERNEL(_request(case))
    got = built(*inputs)
    assert got[0].dtype is torch.float32 and got[1].dtype is torch.int32
    compare_outputs(got, workload.ref_program(*inputs), workload.verification(*inputs))


@pytest.mark.smoke
def test_a_huge_finite_bias_keeps_the_weights_unbiased():
    """The bias steers selection only; the weights stay the unbiased scores.

    A bias near the float32 rounding step is the case where recovering the score
    by subtracting the bias back out loses it entirely.
    """
    if torch.device(run_device()).type != "npu" or not run_device_available():
        pytest.skip("requires an Ascend NPU and --tileops-device=npu")
    call = dataclasses.replace(
        _request(_cases()[0]),
        num_tokens=4,
        num_experts=256,
        top_k=8,
        dtype=torch.float32,
        renormalize=True,
        with_correction_bias=True,
        num_expert_group=8,
        topk_group=4,
        routed_scaling_factor=2.5,
    )
    logits = torch.zeros(4, 256, dtype=torch.float32, device=run_device())
    bias = torch.full((256,), 16777216.0, dtype=torch.float32, device=run_device())
    bias[3] = 33554432.0
    weights, ids = _KERNEL(call)(logits, bias)
    assert ids[0, 0].item() == 3
    # Every logit is zero, so every sigmoid score is 0.5 and the renormalized
    # weights are uniform regardless of how large the bias was.
    assert torch.allclose(weights, torch.full_like(weights, 2.5 / 8), atol=1e-6)
