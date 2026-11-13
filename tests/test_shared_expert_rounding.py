"""Shared activation rounding is observable after a cancelling down projection."""

import pytest
import torch

from workloads.moe import ref_shared_expert

pytestmark = pytest.mark.smoke


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [(torch.float16, -0.0146484375), (torch.bfloat16, -0.015625)],
)
def test_shared_activation_rounds_before_down(dtype, expected):
    hidden = torch.tensor([[1]], dtype=dtype, device="cpu")
    gate_up = torch.tensor([[1], [1.015625], [1], [1]], dtype=dtype, device="cpu")
    down = torch.tensor([[1, -1]], dtype=dtype, device="cpu")

    # FP32 SiLU values are approximately 0.7310586 and 0.7455902. Their
    # token-dtype difference differs from subtracting first and narrowing last.
    actual = ref_shared_expert(hidden, gate_up, down)
    assert torch.equal(actual, torch.tensor([[expected]], dtype=dtype, device="cpu"))
