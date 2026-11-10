"""Regressions for normalized K=128 packing and long chunk recurrence on Ascend."""

import pytest
import torch

from tileops.linear_attention import DeltaNetFwdOp
from workloads.device import run_device
from workloads.linear_attention.ascend_cases import delta_normalized_case
from workloads.linear_attention.deltanet import inference_verification
from workloads.numerics import compare_outputs


@pytest.mark.smoke
@pytest.mark.in_tree_kernels
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("length,initial", [(129, False), (2049, True)])
def test_normalized_delta_state_and_tail(dtype, length, initial):
    if torch.device(run_device()).type != "npu":
        pytest.skip("Ascend normalization and chunk recurrence regression")
    inputs, cpu, expected = delta_normalized_case(dtype, length, initial)
    got = DeltaNetFwdOp(use_qk_l2norm_in_kernel=True)(*inputs)
    compare_outputs(
        tuple((x.cpu() for x in got)), expected, inference_verification(dtype, l2norm=True)
    )
    for tensor, saved in zip(inputs, cpu, strict=True):
        if tensor is not None:
            assert torch.equal(tensor.cpu().view(torch.uint8), saved.view(torch.uint8))
            assert all((output.data_ptr() != tensor.data_ptr() for output in got))
