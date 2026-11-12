"""Public indexed expert MLP coverage on Ascend 950."""

import pytest
import torch

from tileops.moe import IndexedExpertMLPFwdOp
from workloads.device import run_device, run_device_available
from workloads.moe import IndexedExpertMLPWorkload
from workloads.numerics import compare_outputs
from workloads.workload_base import manifest_call


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("tokens", "dtype", "ffn", "top_k", "hidden"),
    [
        (1, "float16", 96, 2, 128),
        (1, "bfloat16", 96, 2, 128),
        pytest.param(1, "float16", 1024, 8, 2048, id="direct-persistent-float16"),
        pytest.param(1, "bfloat16", 1024, 8, 2048, id="direct-persistent-bfloat16"),
        (17, "float16", 96, 2, 128),
        (17, "bfloat16", 96, 2, 128),
        pytest.param(33, "bfloat16", 96, 2, 128, id="persistent-buffer-reuse"),
        pytest.param(17, "bfloat16", 128, 2, 128, id="mixed-cube-vector-gate-up"),
        pytest.param(33, "bfloat16", 1024, 8, 2048, id="preloaded-input-reuse"),
        pytest.param(65, "bfloat16", 1024, 8, 2048, id="shard-major-two-shards"),
        pytest.param(33, "float16", 1024, 8, 4096, id="packed-route-tail-float16"),
        pytest.param(33, "bfloat16", 1024, 8, 4096, id="packed-route-tail-bfloat16"),
        pytest.param(65, "float16", 1024, 8, 2048, id="packed-large-group-float16"),
        pytest.param(64, "float16", 256, 8, 8192, id="packed-single-k-tile-down-float16"),
        pytest.param(64, "bfloat16", 256, 8, 8192, id="packed-single-k-tile-down-bfloat16"),
        pytest.param(2, "bfloat16", 256, 8, 32768, id="reduce-ub-capped-hidden"),
    ],
)
def test_indexed_experts_observe_new_routes_on_a_nondefault_stream(
    dtype, tokens, ffn, top_k, hidden
):
    """Reused kernels must read new IDs, including duplicate routes beyond list capacity.

    Ascend Cube gathers multiple rows per expert. Seventeen tokens exercise a second
    group; one token uses direct dispatch. The projection width also has a tile tail.
    More route groups check that a Cube result is consumed before its buffer is reused.
    Aligned widths exercise Vector conversion of the Cube's gate/up result on chip.
    Larger input groups cover L1-to-L0 slices and input reuse across both projection stages.
    Sixty-five tokens cover two gate/up shards with multiple and overflowing route groups.
    Aligned single-token routes cover persistent buffers and column-major down tasks.
    Dense expert routing covers packed inputs, complete overflow lists and 64/128-row groups.
    A 256-wide activation covers a single iteration of the two-version down pipeline.
    A 32768-wide hidden covers the capped reduce tile and its UB working-set bound.
    """
    if torch.device(run_device()).type != "npu" or not run_device_available():
        pytest.skip("requires an Ascend NPU and --tileops-device=npu")
    call = manifest_call(
        "IndexedExpertMLPFwdOp",
        {"D": dtype},
        T=tokens,
        E=8,
        K=top_k,
        H=hidden,
        F=ffn,
        routed_scaling_factor=2.5,
    )
    workload = IndexedExpertMLPWorkload(call)
    args = workload.gen_inputs()
    op = IndexedExpertMLPFwdOp(**call.arguments({}))
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for duplicate in (False, True):
            if duplicate:
                args[-1].fill_(7)
            before = [x.clone() for x in args]
            output = torch.full_like(args[0], float("nan"))
            expected = workload.ref_program(*args)
            assert op(*args, out=output) is output
            # Consume the result on the same stream before any host synchronization.
            actual = output.clone()
            compare_outputs(actual, expected, workload.verification(*args))
            for current, original in zip(args, before, strict=True):
                torch.testing.assert_close(current, original, rtol=0, atol=0)
    stream.synchronize()


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("experts", "hidden", "ffn", "packed"),
    [(8, 4096, 1024, True), (9, 4096, 1024, False), (8, 1024, 256, False)],
    ids=["packed-density-boundary", "grouped-fallback", "small-projection-fallback"],
)
def test_indexed_experts_dispatch_at_the_packing_density_boundary(experts, hidden, ffn, packed):
    """Both sides of the packing threshold observe new IDs and cover empty experts."""
    if torch.device(run_device()).type != "npu" or not run_device_available():
        pytest.skip("requires an Ascend NPU and --tileops-device=npu")
    from tileops._kernels.moe.call_spec import IndexedExpertCall

    call = manifest_call(
        "IndexedExpertMLPFwdOp", {"D": "bfloat16"}, T=32, E=experts, K=8, H=hidden, F=ffn
    )
    workload = IndexedExpertMLPWorkload(call)
    args = workload.gen_inputs()
    op = IndexedExpertMLPFwdOp(**call.arguments({}))
    for duplicate in (False, True):
        if duplicate:
            args[-1].fill_(experts - 1)
        expected = workload.ref_program(*args)
        output = torch.full_like(args[0], float("nan"))
        assert op(*args, out=output) is output
        compare_outputs(output, expected, workload.verification(*args))
    request = IndexedExpertCall(
        num_tokens=32,
        top_k=8,
        num_experts=experts,
        hidden_size=hidden,
        ffn_size=ffn,
        dtype=args[1].dtype,
        device=args[1].device,
    )
    assert (op.key_for("expert_gate_up", request) == "ascend_gate_up_packed") == packed
    assert (op.key_for("expert_down", request) == "ascend_down_packed") == packed


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("dims", "reason"),
    [
        ({"T": 1, "E": 8, "H": 127}, "divisible by 16"),
        ({"T": 33, "E": 1024, "H": 16}, "UB capacity"),
        ({"T": 1, "E": 257, "H": 16, "K": 257}, "reduction UB capacity"),
    ],
    ids=["unaligned-projection", "metadata-capacity", "reduction-capacity"],
)
def test_indexed_experts_refuse_unsupported_shapes(dims, reason):
    """Unsupported DMA alignment and metadata capacity fail before kernel construction."""
    if torch.device(run_device()).type != "npu" or not run_device_available():
        pytest.skip("requires an Ascend NPU and --tileops-device=npu")
    call = manifest_call("IndexedExpertMLPFwdOp", {"D": "float16"}, F=16, **{"K": 2, **dims})
    args = IndexedExpertMLPWorkload(call).gen_inputs()
    with pytest.raises(ValueError, match=reason):
        IndexedExpertMLPFwdOp()(*args)


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("tokens", "experts", "parallel"),
    [(31, 8, False), (32, 8, True), (32, 257, False)],
    ids=["serial-route-boundary", "parallel-route-boundary", "sparse-expert-fallback"],
)
def test_indexed_experts_dispatch_at_the_parallel_statistics_boundary(tokens, experts, parallel):
    """Both statistics implementations consume changed IDs through the public Op."""
    if torch.device(run_device()).type != "npu" or not run_device_available():
        pytest.skip("requires an Ascend NPU and --tileops-device=npu")
    from tileops._kernels.moe.call_spec import IndexedExpertCall

    call = manifest_call(
        "IndexedExpertMLPFwdOp", {"D": "bfloat16"}, T=tokens, E=experts, K=8, H=128, F=128
    )
    workload = IndexedExpertMLPWorkload(call)
    args = workload.gen_inputs()
    op = IndexedExpertMLPFwdOp(**call.arguments({}))
    for duplicate in (False, True):
        if duplicate:
            args[-1].fill_(experts - 1)
        before = [x.clone() for x in args]
        output = torch.full_like(args[0], float("nan"))
        expected = workload.ref_program(*args)
        assert op(*args, out=output) is output
        compare_outputs(output, expected, workload.verification(*args))
        for current, original in zip(args, before, strict=True):
            torch.testing.assert_close(current, original, rtol=0, atol=0)
    request = IndexedExpertCall(
        num_tokens=tokens,
        top_k=8,
        num_experts=experts,
        hidden_size=128,
        ffn_size=128,
        dtype=args[1].dtype,
        device=args[1].device,
    )
    assert (op.key_for("route_stats", request) == "ascend_route_stats_parallel") == parallel


@pytest.mark.smoke
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize(
    ("experts", "hidden", "ffn"),
    [(63, 2048, 1024), (64, 2048, 1024), (64, 2304, 1024), (64, 4096, 768)],
    ids=["grouped-fallback", "vector-weights", "odd-gate-k-tiles", "odd-down-k-tiles"],
)
def test_indexed_experts_stage_weights_at_the_sparse_routing_boundary(dtype, experts, hidden, ffn):
    """Vector weight staging and its grouped fallback handle changed and overflowing IDs."""
    if torch.device(run_device()).type != "npu" or not run_device_available():
        pytest.skip("requires an Ascend NPU and --tileops-device=npu")
    from tileops._kernels.moe.call_spec import IndexedExpertCall

    call = manifest_call(
        "IndexedExpertMLPFwdOp",
        {"D": dtype},
        T=32,
        E=experts,
        K=8,
        H=hidden,
        F=ffn,
        routed_scaling_factor=2.5,
    )
    workload = IndexedExpertMLPWorkload(call)
    args = workload.gen_inputs()
    op = IndexedExpertMLPFwdOp(**call.arguments({}))
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for duplicate in (False, True):
            if duplicate:
                args[-1].fill_(experts - 1)
            before = [x.clone() for x in args]
            output = torch.full_like(args[0], float("nan"))
            expected = workload.ref_program(*args)
            assert op(*args, out=output) is output
            actual = output.clone()
            compare_outputs(actual, expected, workload.verification(*args))
            for current, original in zip(args, before, strict=True):
                torch.testing.assert_close(current, original, rtol=0, atol=0)
    stream.synchronize()
    request = IndexedExpertCall(
        num_tokens=32,
        top_k=8,
        num_experts=experts,
        hidden_size=hidden,
        ffn_size=ffn,
        dtype=args[1].dtype,
        device=args[1].device,
        routed_scaling_factor=2.5,
    )
    assert (op.key_for("expert_gate_up", request) == "ascend_gate_up_vector_weights") == (
        experts == 64 and ffn % 512 == 0
    )
    assert (op.key_for("expert_down", request) == "ascend_down_vector_weights") == (
        experts == 64 and hidden % 512 == 0
    )
