"""Ascend pooling regressions through the public op interface."""

import functools

import pytest
import torch
from torch.utils._pytree import tree_map

import tileops.ops as ops
from tileops.manifest import load_adts, load_manifest, load_workloads
from tileops.manifest.plan import entry_plan
from tileops.manifest.workload import instantiate
from tileops.ops.pool import AvgPool1dFwdOp
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
