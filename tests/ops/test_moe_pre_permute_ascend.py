"""Ascend pre-permute selection and public-device correctness coverage."""

import dataclasses

import pytest
import torch

from tileops.kernels.ascend.moe.pre_permute import AscendMoEPrePermuteKernel
from tileops.kernels.moe.call_spec import PrePermuteCall
from tileops.manifest import load_adts, load_manifest, load_workloads
from tileops.manifest.plan import entry_plan
from tileops.manifest.workload import instantiate
from tileops.ops.moe import ContiguousLayoutSpec, MoEPrePermuteFwdOp
from workloads.device import run_device, run_device_available
from workloads.moe import MoEPrePermuteWorkload
from workloads.numerics import compare_outputs


def _cases():
    name = "MoEPrePermuteFwdOp"
    plan = entry_plan(name, load_manifest()[name], load_adts(), resolve=False)
    return [
        instantiate(plan, row, dtype)
        for row in load_workloads(name)
        for dtype in row["dtype_cases"]
    ]


def _request(case):
    (tokens, hidden), dtype = case.tensors["hidden_states"]
    (_, top_k), _ = case.tensors["local_expert_ids"]
    return PrePermuteCall(
        layout=ContiguousLayoutSpec.tight_physical_psum(),
        input_dtype=getattr(torch, dtype),
        num_experts=case.params["num_local_experts"],
        num_tokens=tokens,
        hidden_size=hidden,
        top_k=top_k,
        routing_input_kind="local_expert_ids",
    )


@pytest.mark.smoke
def test_all_seven_manifest_cases_have_an_ascend_candidate():
    cases = _cases()
    assert len(cases) == 7
    for case in cases:
        assert AscendMoEPrePermuteKernel.refusal(_request(case)) is None
        # Construction validates the registered kernel against the public interface.
        MoEPrePermuteFwdOp(
            ContiguousLayoutSpec.tight_physical_psum(),
            case.params["num_local_experts"],
        )


@pytest.mark.smoke
@pytest.mark.parametrize(
    "changes",
    [
        {"layout": ContiguousLayoutSpec.aligned_per_row(128)},
        {"layout": ContiguousLayoutSpec.tight_per_row()},
        {"input_dtype": torch.float32},
        {"num_tokens": 1025},
        {"num_experts": 0},
        {"hidden_size": 127},
    ],
)
def test_unsupported_contracts_are_explicitly_refused(changes):
    assert AscendMoEPrePermuteKernel.refusal(dataclasses.replace(_request(_cases()[0]), **changes))


@pytest.mark.smoke
@pytest.mark.parametrize("case", _cases(), ids=lambda c: c.case_id)
@pytest.mark.parametrize("routing", ["manifest", "all-last", "alternating"])
def test_public_ascend_pre_permute(case, routing):
    if torch.device(run_device()).type != "npu" or not run_device_available():
        pytest.skip("requires an Ascend NPU and --tileops-device=npu")
    workload = MoEPrePermuteWorkload(case)
    x, ids = workload.gen_inputs()
    experts = case.params["num_local_experts"]
    if routing == "all-last":
        ids.fill_(experts - 1)
    elif routing == "alternating":
        ids.copy_(
            (torch.arange(ids.numel(), device=ids.device).reshape(ids.shape) % 2) * (experts - 1)
        )
    snapshots = (x.clone(), ids.clone())
    expected = workload.ref_program(x, ids)
    op = MoEPrePermuteFwdOp(ContiguousLayoutSpec.tight_physical_psum(), experts)
    actual = op(x, ids)
    compare_outputs(actual, expected, workload.verification(x, ids))
    for got, ref in zip(actual, expected, strict=True):
        torch.testing.assert_close(got, ref, rtol=0, atol=0)
    for got, ref in zip((x, ids), snapshots, strict=True):
        torch.testing.assert_close(got, ref, rtol=0, atol=0)
