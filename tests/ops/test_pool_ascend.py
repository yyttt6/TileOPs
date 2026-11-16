"""Ascend pooling regressions through the public op interface."""

import functools

import pytest
import torch
from torch.utils._pytree import tree_map

import tileops._ops as ops
from tileops._ops.pool import AvgPool1dFwdOp
from tileops.manifest import load_adts, load_manifest, load_workloads
from tileops.manifest._plan import entry_plan
from tileops.manifest._workload import instantiate
from tileops.pool import MaxPool2dFwdOp, MaxPool2dIndicesFwdOp
from workloads.device import run_device, run_device_available
from workloads.numerics import compare_outputs
from workloads.pool import (
    AdaptiveAvgPool2dCall,
    AdaptiveMaxPool2dCall,
    AvgPoolCall,
    AvgPoolWorkload,
    MaxPoolCall,
    MeanPoolingCallWorkload,
)


def _cases():
    factories = {
        "AvgPool1dFwdOp": AvgPoolCall,
        "AvgPool2dFwdOp": AvgPoolCall,
        "AvgPool3dFwdOp": AvgPoolCall,
        "MaxPool1dFwdOp": MaxPoolCall,
        "MaxPool2dFwdOp": MaxPoolCall,
        "MaxPool3dFwdOp": MaxPoolCall,
        "MaxPool1dIndicesFwdOp": functools.partial(MaxPoolCall, return_indices=True),
        "MaxPool2dIndicesFwdOp": functools.partial(MaxPoolCall, return_indices=True),
        "MaxPool3dIndicesFwdOp": functools.partial(MaxPoolCall, return_indices=True),
        "AdaptiveAvgPool2dFwdOp": AdaptiveAvgPool2dCall,
        "AdaptiveMaxPool2dFwdOp": AdaptiveMaxPool2dCall,
        "AdaptiveMaxPool2dIndicesFwdOp": functools.partial(
            AdaptiveMaxPool2dCall, return_indices=True
        ),
        "MeanPoolingFwdOp": MeanPoolingCallWorkload,
    }
    manifest = load_manifest()
    smoke, full = [], []
    for name, factory in factories.items():
        plan = entry_plan(name, manifest[name], load_adts(), resolve=False)
        first = True
        for row in load_workloads(name):
            for dtype in row["dtype_cases"]:
                call = instantiate(plan, row, dtype)
                case = pytest.param(
                    name,
                    call,
                    factory,
                    id=f"{name}-{call.case_id}",
                    marks=pytest.mark.smoke if first else pytest.mark.full,
                )
                (smoke if first else full).append(case)
                first = False
    return smoke + full


@pytest.mark.parametrize("name,call,factory", list(_cases()))
def test_public_ascend_pool_manifest(name, call, factory):
    if torch.device(run_device()).type != "npu" or not run_device_available():
        pytest.skip("requires an Ascend NPU and --tileops-device=npu")
    workload = factory(call)
    inputs = workload.gen_inputs()
    cpu_inputs = tree_map(lambda x: x.cpu() if isinstance(x, torch.Tensor) else x, inputs)
    expected = workload.ref_program(*cpu_inputs)
    expected = tree_map(
        lambda x: x.to(run_device()) if isinstance(x, torch.Tensor) else x, expected
    )
    actual = getattr(ops, name)(**workload.arguments())(*inputs)
    compare_outputs(actual, expected, workload.verification(*inputs))


@pytest.mark.smoke
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_avg_pool1d_single_element_storage_offset(dtype):
    """A one-element offset must not require aligned packed input loads."""
    if torch.device(run_device()).type != "npu" or not run_device_available():
        pytest.skip("requires an Ascend NPU and --tileops-device=npu")
    workload = AvgPoolWorkload(1, 1, 1, 0, False, True, None, dtype)
    (x,) = workload.gen_inputs(2, 3, 1, storage_offset=1)
    assert x.is_contiguous() and x.storage_offset() == 1
    expected = workload.ref_program(x.cpu()).to(x.device)
    actual = AvgPool1dFwdOp(1, 1)(x)
    compare_outputs(actual, expected, workload.verification(x))


@pytest.mark.smoke
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("offset", [0, 1], ids=["aligned", "offset"])
def test_max_pool2d_alexnet_indices_bulk_writes(dtype, offset):
    """Packed stores retain every value/index, including the final short row."""
    if torch.device(run_device()).type != "npu" or not run_device_available():
        pytest.skip("requires an Ascend NPU and --tileops-device=npu")
    size = 32 * 64 * 55 * 55
    storage = torch.randn(size + offset, device=run_device(), dtype=dtype)
    x = storage[offset:].view(32, 64, 55, 55)
    x[0, 0].fill_(1)
    x[0, 1].zero_()
    x[0, 1, ::2, ::2] = -0.0
    x[0, 2].fill_(float("-inf"))
    x[0, 3, :4, :4] = float("nan")
    x[-1, -1, -1, -1] = float("nan")
    expected_values, expected_indices = torch.nn.functional.max_pool2d(
        x.cpu(), 3, 2, ceil_mode=True, return_indices=True
    )
    values, indices = MaxPool2dIndicesFwdOp(kernel_size=3, stride=2, ceil_mode=True)(x)
    values, indices = values.cpu(), indices.cpu()
    torch.testing.assert_close(values, expected_values, rtol=0, atol=0, equal_nan=True)
    torch.testing.assert_close(indices, expected_indices, rtol=0, atol=0)
    plain = MaxPool2dFwdOp(kernel_size=3, stride=2, ceil_mode=True)(x).cpu()
    torch.testing.assert_close(plain, expected_values, rtol=0, atol=0, equal_nan=True)
    zeros = expected_values == 0
    assert torch.equal(torch.signbit(values[zeros]), torch.signbit(expected_values[zeros]))
