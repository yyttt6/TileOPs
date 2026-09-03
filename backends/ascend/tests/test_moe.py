"""Focused no-pad MoE permute checks for the Ascend backend."""

import pytest
import torch

from tileops.backend import TensorSpec
from tileops_ascend.families.moe import build_moe_permute


def _reference(hidden, ids, experts):
    flat = ids.reshape(-1).tolist()
    counts = [0] * experts
    for expert in flat:
        counts[expert] += 1
    offsets = [0]
    for count in counts:
        offsets.append(offsets[-1] + count)
    cursor = offsets[:-1].copy()
    perm = torch.empty_like(hidden.new_empty((len(flat), hidden.shape[1])))
    fwd = []
    for pair, expert in enumerate(flat):
        slot = cursor[expert]
        cursor[expert] += 1
        fwd.append(slot)
        perm[slot] = hidden[pair // ids.shape[1]]
    return (
        perm,
        torch.tensor(offsets[:-1], dtype=torch.int32),
        torch.tensor(counts, dtype=torch.int32),
        torch.tensor(offsets, dtype=torch.int64),
        torch.tensor(fwd, dtype=torch.int32),
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_moe_permute_nopad(dtype):
    if not hasattr(torch, "npu"):
        pytest.skip("torch_npu is not installed")
    tokens, hidden, top_k, experts = 17, 64, 8, 8
    hidden_cpu = (torch.arange(tokens * hidden, dtype=torch.float32).reshape(tokens, hidden) % 997 / 997).to(dtype)
    ids_cpu = torch.tensor([[(i * 3 + j) % experts for j in range(top_k)] for i in range(tokens)], dtype=torch.int32)
    hidden_npu, ids_npu = hidden_cpu.to("npu"), ids_cpu.to("npu")
    kernel = build_moe_permute(
        TensorSpec.of(hidden_npu),
        TensorSpec.of(ids_npu),
        num_experts=experts,
        num_experts_local=experts,
    )
    actual = kernel(hidden_npu, ids_npu)
    torch.npu.synchronize()
    for got, expected in zip(actual, _reference(hidden_cpu, ids_cpu, experts)):
        torch.testing.assert_close(got.cpu(), expected, atol=0, rtol=0)


def test_moe_permute_rejects_expert_map():
    with pytest.raises(NotImplementedError, match="expert_map"):
        build_moe_permute(
            TensorSpec(torch.device("cpu"), torch.bfloat16, (1, 32)),
            TensorSpec(torch.device("cpu"), torch.int32, (1, 8)),
            TensorSpec(torch.device("cpu"), torch.int32, (8,)),
            num_experts=8,
            num_experts_local=4,
        )
