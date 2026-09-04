"""Focused T069 checks for the MoE routing additions."""

import pytest
import torch

from tileops.backend import TensorSpec
from tileops.kernels.families.moe import (
    build_fused_moe,
    build_fused_moe_experts,
    build_moe_gate_up,
    build_moe_grouped_gemm,
    build_moe_permute_align,
    build_moe_unpermute,
)


def test_permute_align_builder_contract():
    ids = TensorSpec(torch.device("cpu"), torch.int32, (5, 3))
    builder = build_moe_permute_align(ids, total_tokens=5, top_k=3, num_experts=4, block_size=4)
    assert callable(builder)


@pytest.mark.parametrize(
    "builder",
    [build_moe_unpermute, build_moe_grouped_gemm, build_moe_gate_up, build_fused_moe, build_fused_moe_experts],
)
def test_t069_blocked_paths_are_explicit(builder):
    with pytest.raises(NotImplementedError):
        if builder is build_moe_unpermute:
            builder(None, None, None, total_tokens=1, top_k=1, hidden_size=1)
        elif builder is build_moe_grouped_gemm:
            builder(None, None, None, None, numel=1, num_experts=1, n=1, k=1)
        elif builder is build_moe_gate_up:
            builder(None, None, None, None, numel=1, num_experts=1, ffn=1, k=1)
        elif builder is build_fused_moe:
            builder(None, None, None, None, num_tokens=1, num_experts=1, top_k=1, hidden_size=1, ffn_size=1)
        else:
            builder(*([None] * 8), num_tokens=1, num_experts=1, num_experts_local=1, top_k=1, hidden_size=1, ffn_size=1)
