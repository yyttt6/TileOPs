"""Ascend NSA coverage: manifest agreement, explicit refusal and the public device path."""

import dataclasses
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tileops.kernels.ascend.attention.nsa_compressed_varlen import (
    AscendNSACompressedFwdVarlenKernel,
)
from tileops.kernels.ascend.attention.nsa_topk_varlen import AscendNSATopKVarlenKernel
from tileops.kernels.ascend.attention.nsa_varlen_unified import AscendNSAFwdVarlenKernel
from tileops.kernels.attention.call_spec import NSACall
from tileops.manifest import load_adts, load_manifest, load_workloads
from tileops.manifest.plan import entry_plan
from tileops.manifest.workload import instantiate
from tileops.ops import NSACompressedVarlenFwdOp, NSATopKVarlenFwdOp, NSAVarlenFwdOp
from workloads.attention.nsa import NSACompressedFwdCall, NSAFwdCall, NSATopKCall
from workloads.device import run_device, run_device_available
from workloads.numerics import compare_outputs

# Each op's Ascend implementation and the number of manifest dtype cases it accepts.
_SPECS = {
    "NSAVarlenFwdOp": (AscendNSAFwdVarlenKernel, 4),
    "NSATopKVarlenFwdOp": (AscendNSATopKVarlenKernel, 10),
    "NSACompressedVarlenFwdOp": (AscendNSACompressedFwdVarlenKernel, 3),
}

_WORKLOADS = {
    "NSAVarlenFwdOp": NSAFwdCall,
    "NSATopKVarlenFwdOp": NSATopKCall,
    "NSACompressedVarlenFwdOp": NSACompressedFwdCall,
}


def _cases(name: str):
    plan = entry_plan(name, load_manifest()[name], load_adts(), resolve=False)
    return [
        instantiate(plan, row, dtype_case)
        for row in load_workloads(name)
        for dtype_case in row["dtype_cases"]
    ]


def _request(name: str, case) -> NSACall:
    """The call the public op assembles from this case, before any tensor exists."""
    indices, params = case.indices, case.params
    dtype = getattr(torch, indices["T"])
    if name == "NSAVarlenFwdOp":
        return NSACall(
            batch=indices["N"],
            c_seq_len=indices["T_q"],
            heads=indices["H"],
            heads_kv=indices["H_kv"],
            dim=indices["D"],
            selected_blocks=indices["SEL"],
            block_size=params["block_size"],
            scale=params["scale"],
            is_causal=params["is_causal"],
            dtype=dtype,
        )
    if name == "NSATopKVarlenFwdOp":
        return NSACall(
            batch=indices["N"],
            c_seq_len=indices["T_q"],
            heads=indices["H"],
            heads_kv=indices["H_kv"],
            dim=indices["D"],
            chunk_num=indices["C"],
            selected_blocks=params["selected_block_num"],
            block_size=params["bs"],
            scale=params["scale"],
            dtype=dtype,
        )
    return NSACall(
        batch=indices["N"],
        c_seq_len=indices["T_q"],
        heads=indices["H"],
        heads_kv=indices["H_kv"],
        dim=indices["DK"],
        dim_v=indices["DV"],
        chunk_num=indices["C"],
        block_size=params["bs"],
        scale=params["scale"],
        dtype=dtype,
    )


def _op(name: str, case):
    params = case.params
    if name == "NSAVarlenFwdOp":
        return NSAVarlenFwdOp(params["is_causal"], params["scale"], params["block_size"])
    if name == "NSATopKVarlenFwdOp":
        return NSATopKVarlenFwdOp(params["scale"], params["selected_block_num"], params["bs"])
    return NSACompressedVarlenFwdOp(params["scale"], params["bs"])


@pytest.mark.smoke
def test_every_manifest_case_has_an_ascend_candidate():
    for name, (kernel, expected) in _SPECS.items():
        cases = _cases(name)
        assert len(cases) == expected
        for case in cases:
            assert kernel.refusal(_request(name, case)) is None, case.case_id
    # Construction validates each registered kernel against its public interface.
    NSAVarlenFwdOp(is_causal=True, scale=0.1, block_size=32)
    NSATopKVarlenFwdOp(scale=1.0, selected_block_num=16, bs=32)
    NSACompressedVarlenFwdOp(scale=0.08838834764831845, bs=32)


@pytest.mark.smoke
@pytest.mark.parametrize(
    "name, changes",
    [
        ("NSAVarlenFwdOp", {"dtype": torch.float32}),
        ("NSAVarlenFwdOp", {"block_size": 64}),
        ("NSAVarlenFwdOp", {"selected_blocks": 2}),
        ("NSAVarlenFwdOp", {"is_causal": False}),
        ("NSATopKVarlenFwdOp", {"dtype": torch.float32}),
        ("NSATopKVarlenFwdOp", {"selected_blocks": 32}),
        ("NSATopKVarlenFwdOp", {"scale": 0.5}),
        ("NSACompressedVarlenFwdOp", {"dtype": torch.float32}),
        ("NSACompressedVarlenFwdOp", {"block_size": 64}),
        ("NSACompressedVarlenFwdOp", {"dim": 64}),
    ],
)
def test_unsupported_contracts_are_explicitly_refused(name: str, changes: dict):
    kernel, _ = _SPECS[name]
    assert kernel.refusal(dataclasses.replace(_request(name, _cases(name)[0]), **changes))


@pytest.mark.smoke
@pytest.mark.parametrize("name", _SPECS)
def test_mixed_npu_indices_are_rejected_before_launch(name: str):
    """Guard against accepting metadata on another NPU before a direct kernel launch."""
    kernel_type, _ = _SPECS[name]
    call = _request(name, _cases(name)[0])
    query = ((call.c_seq_len, call.heads, call.dim), call.dtype)
    offsets = ((call.batch + 1,), torch.int32)
    positions = ((call.c_seq_len, 2), torch.int32)
    if name == "NSAVarlenFwdOp":
        kv = ((call.c_seq_len, call.heads_kv, call.dim), call.dtype)
        arguments = (
            query,
            kv,
            kv,
            ((call.c_seq_len, call.heads_kv, call.selected_blocks), torch.int32),
            ((call.c_seq_len, call.heads_kv), torch.int32),
            offsets,
            positions,
        )
    else:
        arguments = (query, ((call.chunk_num, call.heads_kv, call.dim), call.dtype))
        if name == "NSACompressedVarlenFwdOp":
            arguments += (((call.chunk_num, call.heads_kv, call.dim_v), call.dtype),)
        arguments += (offsets, offsets, positions)

    # Metadata-only stubs exercise validation without building or running device code.
    inputs = []
    for shape, dtype in arguments:
        tensor = Mock(shape=shape, dtype=dtype, device=SimpleNamespace(type="npu", index=0))
        tensor.contiguous.return_value = tensor
        tensor.data_ptr.return_value = 0
        inputs.append(tensor)
    inputs[-1].device = SimpleNamespace(type="npu", index=1)
    kernel = object.__new__(kernel_type)
    kernel.call, kernel.dtype, kernel._compiled = call, call.dtype, Mock()

    with pytest.raises(ValueError, match="every input must be on"):
        kernel(*inputs)
    kernel._compiled.assert_not_called()


@pytest.mark.smoke
@pytest.mark.parametrize(
    "name, case",
    [
        pytest.param(name, case, id=f"{name}-{case.case_id}")
        for name in _SPECS
        for case in _cases(name)
    ],
)
def test_public_ascend_path_matches_the_reference(name: str, case):
    if torch.device(run_device()).type != "npu" or not run_device_available():
        pytest.skip("requires an Ascend NPU and --tileops-device=npu")
    workload = _WORKLOADS[name](case)
    inputs = workload.gen_inputs()
    expected = workload.ref_program(*inputs)
    actual = _op(name, case)(*inputs)
    compare_outputs(actual, expected, workload.verification(*inputs))
