"""Ascend training state boundaries, including the recurrent fallback."""

import pytest
import torch

from tileops.linear_attention import GLAChunkBwdOp, GLAChunkFwdOp
from workloads.device import run_device
from workloads.linear_attention.ascend_cases import (
    gla_gradient_case,
    gla_training_inputs,
    gla_training_reference,
)
from workloads.linear_attention.gla import chunkwise_verification
from workloads.numerics import compare_outputs


@pytest.mark.smoke
@pytest.mark.parametrize("chunk,keys,values", [(3, 17, 19), (32, 128, 64), (64, 64, 128)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_ascend_gla_training_preserves_fp32_boundaries(chunk, keys, values, dtype):
    if str(run_device()).split(":")[0] != "npu":
        pytest.skip("Ascend training state regression")
    inputs, saved, length = gla_training_inputs(chunk, keys, values, dtype)
    output, final, states = GLAChunkFwdOp(chunk_size=chunk, scale=0.23)(*inputs)
    expected = gla_training_reference(saved, length, chunk, dtype)
    compare_outputs(
        (output.cpu(), final.cpu(), states.cpu()), expected, chunkwise_verification(dtype)
    )
    assert states.dtype == torch.float32
    assert torch.equal(states[:, 0].cpu(), saved[-1])
    assert torch.equal(states[:, -1], final)
    for actual, before in zip(inputs, saved, strict=True):
        assert torch.equal(actual.cpu(), before)


@pytest.mark.smoke
@pytest.mark.parametrize("chunk,keys,values", [(3, 17, 19), (32, 64, 64)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_ascend_gla_initial_gradient_matches_direct_autograd(chunk, keys, values, dtype):
    if str(run_device()).split(":")[0] != "npu":
        pytest.skip("Ascend initial-state gradient regression")
    inputs, before, expected = gla_gradient_case(chunk, keys, values, dtype)
    actual = GLAChunkBwdOp(chunk_size=chunk, scale=0.23, has_initial_state=True)(*inputs)
    compare_outputs(
        tuple((x.cpu() for x in actual)), expected, chunkwise_verification(dtype, backward=True)
    )
    for current, saved in zip(inputs, before, strict=True):
        assert torch.equal(current.cpu(), saved)
