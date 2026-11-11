"""Ascend GLA coverage for K=128 diagonal scores and packed sequence tails."""

import pytest
import torch

from tileops.linear_attention import GLAFwdOp
from workloads.device import run_device
from workloads.linear_attention.ascend_cases import gla_diagonal_inputs, gla_direct_recurrence
from workloads.linear_attention.gla import inference_verification
from workloads.numerics import compare_outputs


@pytest.mark.smoke
@pytest.mark.parametrize("lengths", [(63,), (1, 31, 33, 63)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_ascend_gla_k128_scores_match_direct_recurrence(lengths, dtype):
    if str(run_device()).split(":")[0] != "npu":
        pytest.skip("Ascend-specific chunked GLA regression")
    inputs, before, offsets, offsets_cpu, scale = gla_diagonal_inputs(lengths, dtype)
    output, final = GLAFwdOp(scale=scale)(
        *inputs, cu_seqlens=offsets, cu_seqlens_cpu=offsets_cpu if offsets is not None else None
    )
    expected_output, expected_final = gla_direct_recurrence(*before, lengths, scale)
    compare_outputs(
        (output.cpu(), final.cpu()),
        (expected_output.to(dtype), expected_final),
        inference_verification(dtype),
    )
    for actual, saved in zip(inputs, before, strict=True):
        assert torch.equal(actual.cpu(), saved)
