"""Ascend GDN packed tails, empty sequences, and rectangular recurrent states."""

import pytest
import torch

from tileops.linear_attention import GDNFwdOp
from workloads.device import run_device
from workloads.linear_attention.ascend_cases import gdn_packed_rectangular_case
from workloads.linear_attention.gdn import gated_verification
from workloads.numerics import compare_outputs


@pytest.mark.smoke
@pytest.mark.in_tree_kernels
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("keys,values", [(17, 19), (128, 64)])
def test_ascend_gdn_packed_rectangular_transforms(dtype, keys, values):
    if torch.device(run_device()).type != "npu":
        pytest.skip("Ascend packed recurrence regression")
    inputs, saved, expected = gdn_packed_rectangular_case(dtype, keys, values)
    got = GDNFwdOp(
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        allow_neg_eigval=True,
        use_qk_l2norm_in_kernel=True,
        state_v_first=True,
    )(*inputs)
    compare_outputs(
        tuple((x.cpu() for x in got)),
        expected,
        gated_verification(inputs, l2norm=True, state_v_first=True),
    )
    for tensor, before in zip(inputs, saved, strict=True):
        assert torch.equal(tensor.cpu().view(torch.uint8), before.view(torch.uint8))
        assert all((output.data_ptr() != tensor.data_ptr() for output in got))
