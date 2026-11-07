"""Ascend FP8 grouped-GEMM selection and public-device correctness coverage."""

import dataclasses

import pytest
import torch

from tileops.kernels.ascend.moe.moe_grouped_gemm_fp8 import (
    AscendMoEGroupedGemmFP8Kernel,
    tile_m_for,
)
from tileops.kernels.moe.call_spec import MGroupedGemmFP8Call, MGroupedGemmFP8FwdInterface
from tileops.manifest import load_adts, load_manifest, load_workloads
from tileops.manifest.plan import entry_plan
from tileops.manifest.workload import instantiate
from workloads.device import run_device, run_device_available
from workloads.moe import MoEGroupedGemmFP8Workload
from workloads.numerics import compare_outputs

_OP = "MoEGroupedGemmFP8FwdOp"


def _cases():
    plan = entry_plan(_OP, load_manifest()[_OP], load_adts(), resolve=False)
    return [
        instantiate(plan, row, dtype)
        for row in load_workloads(_OP)
        for dtype in row.get("dtype_cases") or [{}]
    ]


def _request(case):
    (groups, max_m, k), ab_dtype = case.tensors["a"]
    (_, n, _), _ = case.tensors["b"]
    (_, _, scale_k), _ = case.tensors["a_scale"]
    (_, scale_n, _), _ = case.tensors["b_scale"]
    return MGroupedGemmFP8Call(
        kind="masked",
        max_m=max_m,
        ab_dtype=getattr(torch, ab_dtype),
        num_groups=groups,
        n=n,
        k=k,
        a_scale_shape=(groups, max_m, scale_k),
        b_scale_shape=(groups, scale_n, scale_k),
    )


@pytest.mark.smoke
def test_every_manifest_case_is_served():
    cases = _cases()
    assert len(cases) == 14
    assert issubclass(AscendMoEGroupedGemmFP8Kernel, MGroupedGemmFP8FwdInterface)
    for case in cases:
        assert AscendMoEGroupedGemmFP8Kernel.refusal(_request(case)) is None


@pytest.mark.smoke
@pytest.mark.parametrize("max_m", [64, 128, 256, 96, 192])
def test_the_m_tile_covers_every_masked_slab(max_m):
    """64-row slabs take a 64-row tile; taller ones a 128-row tile and run one per 128 rows."""
    expected = {64: 64, 128: 128, 256: 128, 96: None, 192: 64}[max_m]
    assert tile_m_for(max_m) == expected


@pytest.mark.smoke
@pytest.mark.parametrize(
    "changes",
    [
        {"kind": "contiguous"},
        {"ab_dtype": torch.bfloat16},
        {"max_m": 96},
        {"k": 7232},
        {"n": 4160},
        {"k": 640},
        {"a_scale_shape": (8, 128, 1)},
        {"b_scale_shape": (8, 1, 56)},
    ],
)
def test_unsupported_contracts_are_explicitly_refused(changes):
    assert AscendMoEGroupedGemmFP8Kernel.refusal(
        dataclasses.replace(_request(_cases()[0]), **changes)
    )


@pytest.mark.smoke
@pytest.mark.parametrize("case", _cases(), ids=lambda c: c.case_id)
def test_public_ascend_grouped_gemm_fp8(case):
    if torch.device(run_device()).type != "npu" or not run_device_available():
        pytest.skip("requires an Ascend NPU and --tileops-device=npu")
    workload = MoEGroupedGemmFP8Workload(case)
    inputs = workload.gen_inputs()
    built = AscendMoEGroupedGemmFP8Kernel(_request(case).on_device(torch.device(run_device())))
    out = built(*inputs)
    assert out.dtype is torch.bfloat16
    assert out.shape == (inputs[0].shape[0], inputs[0].shape[1], inputs[2].shape[1])
    compare_outputs(out, workload.ref_program(*inputs), workload.verification(*inputs))
