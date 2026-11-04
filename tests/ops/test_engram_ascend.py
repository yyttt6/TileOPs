"""Ascend GateConv selection and actual forward-to-backward saved-tensor coverage."""

import dataclasses

import pytest
import torch

from tileops.kernels.ascend.sequence_modeling import (
    AscendEngramGateConvBwdKernel,
    AscendEngramGateConvFwdKernel,
)
from tileops.kernels.sequence_modeling.engram.call_spec import EngramGateConvCall
from tileops.sequence_modeling import EngramGateConvBwdOp, EngramGateConvFwdOp
from workloads.device import run_device, run_device_available
from workloads.numerics import compare_outputs
from workloads.sequence_modeling.engram import (
    EngramGateConvBwdWorkload,
    EngramGateConvFwdWorkload,
)

_CASES = [
    (1, 32, 256, torch.float16),
    (1, 128, 256, torch.bfloat16),
    (2, 64, 512, torch.float16),
    (3, 96, 768, torch.bfloat16),
    (2, 2, 37, torch.float16),
    (1, 1, 15, torch.bfloat16),
    (2, 1, 1, torch.float16),  # A single valid SIMD lane and no convolution history.
    (1, 1, 4096, torch.bfloat16),  # The supported hidden-width and UB boundary.
]


@pytest.mark.smoke
@pytest.mark.parametrize("M,seq_len,d,dtype", _CASES)
def test_gateconv_ascend_candidates_construct_without_a_device(M, seq_len, d, dtype):
    call = EngramGateConvCall(m=M, seq_len=seq_len, d=d, eps=1e-6, dtype=dtype)
    for kernel in (AscendEngramGateConvFwdKernel, AscendEngramGateConvBwdKernel):
        assert kernel.refusal(call) is None
    # Construction checks the registered implementations against the public interfaces.
    EngramGateConvFwdOp(M, seq_len, d)
    EngramGateConvBwdOp(M, seq_len, d)


@pytest.mark.smoke
@pytest.mark.parametrize(
    "changes", [{"dtype": torch.float32}, {"m": 0}, {"seq_len": 0}, {"d": 4097}]
)
def test_gateconv_ascend_declines_unsupported_calls(changes):
    call = EngramGateConvCall(m=1, seq_len=32, d=256, eps=1e-6, dtype=torch.float16)
    call = dataclasses.replace(call, **changes)
    for kernel in (AscendEngramGateConvFwdKernel, AscendEngramGateConvBwdKernel):
        assert kernel.refusal(call)


@pytest.mark.smoke
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gateconv_ascend_accepts_the_hidden_width_boundary(dtype):
    call = EngramGateConvCall(m=1, seq_len=1, d=4096, eps=1e-6, dtype=dtype)
    for kernel in (AscendEngramGateConvFwdKernel, AscendEngramGateConvBwdKernel):
        assert kernel.refusal(call) is None


@pytest.mark.parametrize(
    "M,seq_len,d,dtype",
    [
        pytest.param(*case, marks=pytest.mark.smoke if i < 2 else pytest.mark.full)
        for i, case in enumerate(_CASES)
    ],
)
def test_gateconv_ascend_forward_backward(M, seq_len, d, dtype):
    if torch.device(run_device()).type != "npu" or not run_device_available():
        pytest.skip("requires an Ascend NPU and --tileops-device=npu")
    forward_workload = EngramGateConvFwdWorkload(M, seq_len, d, dtype)
    inputs = forward_workload.gen_inputs()
    # Exercise both contributions to the shared H/k RMSNorm weight gradient.
    inputs[3].uniform_(0.7, 1.3)
    inputs[4].uniform_(0.7, 1.3)
    snapshots = tuple(t.clone() for t in inputs)
    forward = EngramGateConvFwdOp(M, seq_len, d)
    actual_forward = forward(*inputs)
    compare_outputs(
        actual_forward,
        forward_workload.ref_program(*inputs),
        forward_workload.verification(*inputs),
    )
    dY = torch.randn_like(inputs[0]) * 0.1
    backward_inputs = (dY, *inputs, *actual_forward[1:])
    backward_snapshots = tuple(t.clone() for t in backward_inputs)
    backward_workload = EngramGateConvBwdWorkload(M, seq_len, d, dtype)
    backward = EngramGateConvBwdOp(M, seq_len, d)
    actual_backward = backward(*backward_inputs)
    compare_outputs(
        actual_backward,
        backward_workload.ref_program(*backward_inputs),
        backward_workload.verification(*backward_inputs),
    )
    for actual, repeated in zip(actual_backward, backward(*backward_inputs), strict=True):
        assert torch.equal(actual, repeated)
    for original, snapshot in zip(
        inputs + backward_inputs, snapshots + backward_snapshots, strict=True
    ):
        assert torch.equal(original, snapshot)
