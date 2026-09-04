"""Behavioral contract tests for the staged MoE public boundary."""

import dataclasses

import pytest
import torch

import tileops.ops.moe.staged as staged_module
from tileops.ops.moe import (
    ContiguousLayoutSpec,
    InversePermuteContext,
    MaskedLayoutSpec,
    MaterializedExpertLayout,
    MoeExpertMLPFwdOp,
    MoeGroupedGemmFwdOp,
    MoePostPermuteFwdOp,
    MoePrePermuteFwdOp,
    NoScaleComputeSpec,
    PrePermuteOutput,
    RoutingEpilogueSpec,
)
from tileops.ops.moe.contracts import (
    MaskedMetadata,
    PerRowExpertMetadata,
    PhysicalPsumMetadata,
    routing_epilogue_reference,
)
from workloads.device import DEVICE

pytestmark = pytest.mark.smoke


def _physical_layout(rows: int = 2, experts: int = 2) -> MaterializedExpertLayout:
    ends = torch.arange(1, experts + 1, dtype=torch.int32)
    if experts:
        ends[-1] = rows
    return MaterializedExpertLayout.from_physical_psum(
        ends,
        materialized_rows=rows,
        num_experts=experts,
    )


def test_layout_presets_expose_only_supported_semantics() -> None:
    physical = ContiguousLayoutSpec.tight_physical_psum()
    per_row = ContiguousLayoutSpec.tight_per_row()
    masked = MaskedLayoutSpec(max_m=4)

    assert physical.selection_key == "tight_physical_psum"
    assert per_row.selection_key == "tight_per_row"
    assert masked.selection_key == "masked_predicated"
    assert repr(physical) == "ContiguousLayoutSpec.tight_physical_psum()"
    assert repr(per_row) == "ContiguousLayoutSpec.tight_per_row()"
    with pytest.raises(ValueError, match="non-negative"):
        MaskedLayoutSpec(max_m=-1)

    with pytest.raises(ValueError, match="num_experts must be positive"):
        MoePrePermuteFwdOp(physical, num_experts=0)


def test_materialized_layout_rejects_incompatible_metadata() -> None:
    with pytest.raises(TypeError, match="PhysicalPsumMetadata"):
        MaterializedExpertLayout(
            layout=ContiguousLayoutSpec.tight_physical_psum(),
            metadata=PerRowExpertMetadata(torch.empty(4, dtype=torch.int32)),
            num_experts=2,
            materialized_rows=4,
        )
    with pytest.raises(TypeError, match="MaskedMetadata"):
        MaterializedExpertLayout(
            layout=MaskedLayoutSpec(max_m=4),
            metadata=PhysicalPsumMetadata(torch.empty(2, dtype=torch.int32)),
            num_experts=2,
            materialized_rows=8,
        )


def test_external_materialization_factories_derive_concrete_contracts() -> None:
    physical = MaterializedExpertLayout.from_physical_psum(
        torch.tensor([1, 3], dtype=torch.int32), materialized_rows=3
    )
    assert physical.selection_key == "tight_physical_psum"
    assert physical.num_experts == 2

    per_row = MaterializedExpertLayout.from_per_row_ids(
        torch.tensor([0, 0, 1], dtype=torch.int32), num_experts=2
    )
    assert per_row.selection_key == "tight_per_row"
    assert per_row.materialized_rows == 3

    masked = MaterializedExpertLayout.from_masked_m(
        torch.tensor([2, 1], dtype=torch.int32), max_m=4
    )
    assert masked.selection_key == "masked_predicated"
    assert masked.materialized_rows == 8
    assert masked.max_m == 4


def test_device_value_guards_cover_empty_experts_ordering_and_ranges() -> None:
    psum = PhysicalPsumMetadata(torch.tensor([0, 0, 3], dtype=torch.int32))
    assert psum.device_value_guard(materialized_rows=3).item()
    bad_psum = PhysicalPsumMetadata(torch.tensor([2, 1], dtype=torch.int32))
    assert not bad_psum.device_value_guard(materialized_rows=1).item()

    ids = PerRowExpertMetadata(torch.tensor([0, 0, 1, 1], dtype=torch.int32))
    assert ids.device_value_guard(num_experts=2).item()
    gap = PerRowExpertMetadata(torch.tensor([0, -1, 1], dtype=torch.int32))
    assert not gap.device_value_guard(num_experts=2).item()
    resumed = PerRowExpertMetadata(torch.tensor([0, 1, 0], dtype=torch.int32))
    assert not resumed.device_value_guard(num_experts=2).item()

    masked = MaskedMetadata(torch.tensor([0, 4, 5], dtype=torch.int32))
    assert not masked.device_value_guard(max_m=4).item()


@pytest.mark.parametrize(
    ("ends", "materialized_rows", "expected"),
    [
        pytest.param([], 0, True, id="zero-experts"),
        pytest.param([0, 0, 0], 0, True, id="all-empty"),
        pytest.param([0, 0, 3], 3, True, id="consecutive-empty"),
        pytest.param([0, 2, 1], 1, False, id="decreasing-end"),
        pytest.param([0, 2], 3, False, id="shape-not-authoritative-end"),
    ],
)
def test_physical_psum_guard_covers_empty_and_capacity_edges(
    ends: list[int], materialized_rows: int, expected: bool
) -> None:
    metadata = PhysicalPsumMetadata(torch.tensor(ends, dtype=torch.int32))
    assert metadata.device_value_guard(materialized_rows=materialized_rows).item() is expected


@pytest.mark.skipif(not torch.npu.is_available(), reason="guard test requires CUDA")
def test_device_value_validation_returns_a_device_guard_without_host_readback() -> None:
    metadata = PhysicalPsumMetadata(torch.tensor([0, 2], dtype=torch.int32, device=DEVICE))
    guard = metadata.device_value_guard(materialized_rows=2)
    assert guard.device.type == "cuda"
    assert guard.dtype is torch.bool
    assert guard.shape == ()


def test_pre_output_binds_layout_and_inverse_context_to_one_materialization() -> None:
    layout = _physical_layout()
    context = InversePermuteContext.for_layout(
        torch.tensor([0, 1], dtype=torch.int32),
        layout,
        num_tokens=2,
        top_k=1,
    )
    output = PrePermuteOutput(torch.empty(2, 8), layout, context)
    assert output.expert_layout is layout


def test_pre_output_rejects_context_from_another_materialization() -> None:
    layout = _physical_layout(rows=1, experts=1)
    other = _physical_layout(rows=1, experts=1)
    context = InversePermuteContext.for_layout(
        torch.tensor([0], dtype=torch.int32), other, num_tokens=1, top_k=1
    )
    with pytest.raises(ValueError, match="different materialization"):
        PrePermuteOutput(torch.empty(1, 8), layout, context)


def test_inverse_context_device_guard_accepts_only_local_rows_or_minus_one() -> None:
    layout = MaterializedExpertLayout.from_physical_psum(
        torch.tensor([3], dtype=torch.int32), materialized_rows=3
    )
    context = InversePermuteContext.for_layout(
        torch.tensor([0, -1, 2], dtype=torch.int32), layout, num_tokens=3, top_k=1
    )
    assert context.device_value_guard().item()
    invalid = dataclasses.replace(
        context, inverse_indices=torch.tensor([0, -2, 3], dtype=torch.int32)
    )
    assert not invalid.device_value_guard().item()


def test_routing_epilogue_reference_assigns_each_operation_once() -> None:
    layout = MaterializedExpertLayout.from_physical_psum(
        torch.tensor([3], dtype=torch.int32), materialized_rows=3
    )
    context = InversePermuteContext.for_layout(
        torch.tensor([2, 0, 1, -1], dtype=torch.int32), layout, num_tokens=2, top_k=2
    )
    expert_output = torch.tensor([[1.0], [4.0], [8.0]], dtype=torch.bfloat16)
    weights = torch.tensor([[0.25, 0.5], [0.75, 100.0]], dtype=torch.float32)

    actual = routing_epilogue_reference(
        expert_output, context, weights, RoutingEpilogueSpec(routed_scaling_factor=2.0)
    )

    assert actual.dtype is torch.bfloat16
    torch.testing.assert_close(actual.float(), torch.tensor([[5.0], [6.0]]))


def test_routing_epilogue_handles_empty_local_materialization() -> None:
    layout = MaterializedExpertLayout.from_physical_psum(
        torch.empty(0, dtype=torch.int32), materialized_rows=0
    )
    context = InversePermuteContext.for_layout(
        torch.tensor([-1, -1], dtype=torch.int32), layout, num_tokens=1, top_k=2
    )
    actual = routing_epilogue_reference(
        torch.empty(0, 4, dtype=torch.bfloat16),
        context,
        torch.ones(1, 2, dtype=torch.float32),
        RoutingEpilogueSpec(),
    )
    torch.testing.assert_close(actual, torch.zeros(1, 4, dtype=torch.bfloat16))


def test_compute_and_epilogue_specs_are_minimal_and_frozen() -> None:
    compute = NoScaleComputeSpec()
    epilogue = RoutingEpilogueSpec()
    assert compute.accumulation_dtype is torch.float32
    assert compute.output_dtype is torch.bfloat16
    assert epilogue.accumulation_dtype is torch.float32
    assert epilogue.output_dtype is torch.bfloat16
    with pytest.raises(ValueError, match="finite and positive"):
        RoutingEpilogueSpec(routed_scaling_factor=0.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        epilogue.routed_scaling_factor = 2.0
    with pytest.raises(TypeError, match="NoScaleComputeSpec"):
        MoeGroupedGemmFwdOp(compute=0)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="RoutingEpilogueSpec"):
        MoePostPermuteFwdOp(epilogue=0)  # type: ignore[arg-type]


@pytest.mark.skipif(not torch.npu.is_available(), reason="CallSpec records CUDA architecture")
def test_public_ops_build_complete_calls_before_selection() -> None:
    hidden = torch.empty(2, 8, dtype=torch.bfloat16, device=DEVICE)
    topk_ids = torch.tensor([[0], [1]], dtype=torch.int32, device=DEVICE)
    pre_call = MoePrePermuteFwdOp(
        ContiguousLayoutSpec.tight_physical_psum(), num_experts=2
    ).make_call(hidden, topk_ids)
    assert (
        pre_call.num_experts,
        pre_call.num_tokens,
        pre_call.hidden_size,
        pre_call.top_k,
    ) == (2, 2, 8, 1)

    layout = MaterializedExpertLayout.from_physical_psum(
        torch.tensor([1, 2], dtype=torch.int32, device=DEVICE), materialized_rows=2
    )
    a = torch.empty(2, 8, dtype=torch.bfloat16, device=DEVICE)
    b = torch.empty(2, 4, 8, dtype=torch.bfloat16, device=DEVICE)
    gemm_call = MoeGroupedGemmFwdOp().make_call(a, b, layout)
    assert (gemm_call.materialized_rows, gemm_call.num_experts, gemm_call.n, gemm_call.k) == (
        2,
        2,
        4,
        8,
    )
    assert gemm_call.layout_key == "tight_physical_psum"


@pytest.mark.skipif(not torch.npu.is_available(), reason="CallSpec records CUDA architecture")
def test_call_architecture_comes_from_the_input_device(monkeypatch: pytest.MonkeyPatch) -> None:
    device = torch.device("cuda", torch.npu.current_device())
    observed_indices: list[int | None] = []

    def fake_sm_version(index: int | None = None) -> int:
        observed_indices.append(index)
        return 90

    monkeypatch.setattr(staged_module, "get_sm_version", fake_sm_version)
    layout = MaterializedExpertLayout.from_physical_psum(
        torch.tensor([1], dtype=torch.int32, device=device), materialized_rows=1
    )
    MoeGroupedGemmFwdOp().make_call(
        torch.empty(1, 4, dtype=torch.bfloat16, device=device),
        torch.empty(1, 2, 4, dtype=torch.bfloat16, device=device),
        layout,
    )

    assert observed_indices == [device.index]


@pytest.mark.skipif(not torch.npu.is_available(), reason="CallSpec records CUDA architecture")
def test_staged_wiring_builds_all_family_calls_without_an_executable_candidate() -> None:
    device = torch.device("cuda")
    layout = MaterializedExpertLayout.from_physical_psum(
        torch.tensor([1, 2], dtype=torch.int32, device=device), materialized_rows=2
    )
    expert_input = torch.empty(2, 8, dtype=torch.bfloat16, device=device)
    w_gate_up = torch.empty(2, 12, 8, dtype=torch.bfloat16, device=device)
    w_down = torch.empty(2, 8, 6, dtype=torch.bfloat16, device=device)
    mlp = MoeExpertMLPFwdOp()

    gate_call, down_call = mlp.make_calls(expert_input, w_gate_up, w_down, layout)

    assert gate_call.layout_key == down_call.layout_key == "tight_physical_psum"
    assert (gate_call.k, gate_call.n, down_call.k, down_call.n) == (8, 12, 6, 8)
    assert tuple(mlp.kernel_delegates()) == (mlp.gate_up, mlp.activation_op, mlp.down)

    context = InversePermuteContext.for_layout(
        torch.tensor([0, 1], dtype=torch.int32, device=device),
        layout,
        num_tokens=2,
        top_k=1,
    )
    post_call = MoePostPermuteFwdOp(RoutingEpilogueSpec()).make_call(
        torch.empty(2, 8, dtype=torch.bfloat16, device=device),
        context,
        torch.ones(2, 1, dtype=torch.float32, device=device),
    )
    assert post_call.layout_key == "tight_physical_psum"
    assert (post_call.num_experts, post_call.materialized_rows) == (2, 2)
    assert (post_call.num_tokens, post_call.top_k, post_call.hidden_size) == (2, 1, 8)


@pytest.mark.skipif(not torch.npu.is_available(), reason="CallSpec records CUDA architecture")
def test_post_permute_rejects_wrong_masked_geometry_with_same_row_count() -> None:
    device = torch.device("cuda")
    layout = MaterializedExpertLayout.from_masked_m(
        torch.tensor([2, 1], dtype=torch.int32, device=device), max_m=4
    )
    context = InversePermuteContext.for_layout(
        torch.tensor([0], dtype=torch.int32, device=device),
        layout,
        num_tokens=1,
        top_k=1,
    )
    wrong_geometry = torch.empty(1, 8, 4, dtype=torch.bfloat16, device=device)

    with pytest.raises(ValueError, match="leading dimensions"):
        MoePostPermuteFwdOp().make_call(
            wrong_geometry,
            context,
            torch.ones(1, 1, dtype=torch.float32, device=device),
        )


@pytest.mark.skipif(not torch.npu.is_available(), reason="candidate test uses CUDA calls")
def test_public_op_without_candidates_fails_explicitly() -> None:
    hidden = torch.empty(1, 4, dtype=torch.bfloat16, device=DEVICE)
    topk_ids = torch.zeros(1, 1, dtype=torch.int32, device=DEVICE)
    op = MoePrePermuteFwdOp(ContiguousLayoutSpec.tight_physical_psum(), num_experts=1)
    with pytest.raises(ValueError, match="no implementation serves this call"):
        op(hidden, topk_ids)
