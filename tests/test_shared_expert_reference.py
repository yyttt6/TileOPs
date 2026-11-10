"""Shared-expert references preserve the interface's activation rounding boundary."""

import pytest
import torch

from workloads.moe import FusedMoESharedExpertWorkload, SharedExpertMLPWorkload
from workloads.workload_base import manifest_call


@pytest.mark.smoke
@pytest.mark.parametrize("shared_composite", [False, True], ids=["standalone", "composite"])
@pytest.mark.parametrize(
    "dtype, down_weight, expected_value",
    [
        pytest.param("float16", 0.3125, 0.456787109375, id="float16"),
        pytest.param("bfloat16", 0.5625, 0.8203125, id="bfloat16"),
    ],
)
def test_shared_reference_rounds_activation_before_down(
    shared_composite, dtype, down_weight, expected_value
):
    """A final-only cast changes the result of a single nonzero gated element.

    The gate is 1 and the up value is 2. Rounding 2 * silu(1) to the input
    dtype before multiplying by the down weight gives the exact binary values
    below. Keeping the activation in FP32 gives a different output in both
    dtypes, even after the final cast.
    """
    storage = getattr(torch, dtype)
    hidden = torch.zeros((1, 16), dtype=storage)
    hidden[0, 0] = 1
    gate_up = torch.zeros((32, 16), dtype=storage)
    gate_up[0, 0], gate_up[16, 0] = 1, 2
    down = torch.zeros((16, 16), dtype=storage)
    down[0, 0] = down_weight
    if shared_composite:
        call = manifest_call(
            "FusedMoESharedExpertFwdOp",
            {"D": dtype},
            T=1,
            E=1,
            H=16,
            F=16,
            S=16,
            top_k=1,
            scoring_func="sigmoid",
            renormalize=True,
            num_expert_group=1,
            topk_group=1,
            some=["shared_w_gate_up", "shared_w_down"],
        )
        workload = FusedMoESharedExpertWorkload(call, device="cpu")
        shared, routed = workload.ref_program(
            hidden,
            torch.zeros((1, 1)),
            torch.zeros((1, 32, 16), dtype=storage),
            torch.zeros((1, 16, 16), dtype=storage),
            None,
            gate_up,
            down,
        )
        assert torch.equal(routed, torch.zeros_like(hidden))
    else:
        call = manifest_call("SharedExpertMLPFwdOp", {"D": dtype}, T=1, H=16, S=16)
        workload = SharedExpertMLPWorkload(call, device="cpu")
        shared = workload.ref_program(hidden, gate_up, down)
    expected = torch.zeros_like(hidden)
    expected[0, 0] = expected_value
    assert shared.dtype == storage
    assert torch.equal(shared, expected)
