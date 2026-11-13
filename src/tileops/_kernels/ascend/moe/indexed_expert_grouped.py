"""Expert-scheduled projections with complete input groups resident in L1."""

import functools

import tilelang
import tilelang.ascend.language as T

from tileops._kernels.ascend.moe.indexed_expert import _AscendIndexedKernel, _route_epilogue
from tileops._kernels.ascend.moe.indexed_expert_fused import AscendIndexedExpertDownFusedKernel
from tileops._kernels.ascend.moe.indexed_expert_preload import (
    AscendIndexedExpertGateUpPreloadKernel,
)
from tileops._kernels.moe.call_spec import IndexedExpertCall
from tileops.backend._registry import register_kernel_type

__all__ = [
    "AscendIndexedExpertDownGroupedKernel",
    "AscendIndexedExpertDownVectorWeightsKernel",
    "AscendIndexedExpertGateUpGroupedKernel",
    "AscendIndexedExpertGateUpVectorWeightsKernel",
]


@functools.lru_cache(maxsize=64)
def _expert_projection(
    tokens: int,
    top_k: int,
    experts: int,
    n: int,
    k: int,
    dtype: str,
    gated: bool,
    block_n: int,
    block_k: int,
    num_stages: int,
    num_blocks: int,
    n_shards: int,
    l0_k: int = 64,
    job_order: int = 0,
    vector_weights: bool = False,
):
    routes = tokens * top_k
    metadata_size = experts + routes + experts * tokens
    width = 2 * n if gated else n
    cols = 2 * block_n if gated else block_n
    input_rows = tokens if gated else routes
    block_m = 16
    half_m = 8
    tiles = n // block_n

    grid = min(num_blocks, experts * n_shards) if num_blocks else experts * n_shards

    @T.prim_func
    def main(
        a: T.Tensor((input_rows, k), dtype),
        weights: T.Tensor((experts, width, k), dtype),
        ids: T.Tensor((routes,), "int32"),
        metadata: T.Tensor((metadata_size,), "int32"),
        out: T.Tensor((routes, n), dtype),
    ):
        with T.MixedKernel(grid) as (bx, sid):
            a_l1 = T.alloc_l1((block_m, k), dtype)
            b_l1 = T.alloc_l1((cols, block_k), dtype)
            b_ub = T.alloc_shared((cols // 2, block_k), dtype)
            a_l0 = T.alloc_l0a((block_m, l0_k), dtype)
            b_l0 = T.alloc_l0b((cols, l0_k), dtype)
            acc = T.alloc_l0c((block_m, cols), "float32")
            wide = T.alloc_shared((half_m, cols), "float32")
            narrow = T.alloc_shared((half_m, block_n), dtype)
            T.annotate_buffer_versions({b_l1: num_stages, a_l0: 2, b_l0: 2})
            if vector_weights:
                T.annotate_buffer_versions({b_ub: num_stages})
            for pid in T.Persistent([experts * n_shards], grid, bx):
                expert = pid // n_shards if job_order == 0 else pid % experts
                shard = pid % n_shards if job_order == 0 else pid // experts
                for group in T.serial(T.ceildiv(metadata[expert], block_m)):
                    start = group * block_m
                    count = T.min(block_m, metadata[expert] - start)
                    for row in T.serial(count):
                        selected = T.alloc_var("int32", init=0)
                        rank = start + row
                        # The bounded list covers only the first T ranks; recover the rest by rank.
                        if rank < tokens:
                            selected = metadata[experts + routes + expert * tokens + rank]
                        else:
                            for candidate in T.serial(routes):
                                if (
                                    ids[candidate] == expert
                                    and metadata[experts + candidate] == rank
                                ):
                                    selected = candidate
                        source = selected // top_k if gated else selected
                        T.copy(a[source : source + 1, :], a_l1[row : row + 1, :])
                    for nt in T.serial(tiles // n_shards):
                        col = (nt * n_shards + shard) * block_n
                        for ko in T.Pipelined(k // block_k, num_stages=num_stages):
                            off = ko * block_k
                            if vector_weights:
                                weight_col = col + sid * n if gated else col + sid * (block_n // 2)
                                T.copy(
                                    weights[
                                        expert,
                                        weight_col : weight_col + cols // 2,
                                        off : off + block_k,
                                    ],
                                    b_ub,
                                )
                                T.dual_copy(b_ub, b_l1)
                            else:
                                T.copy(
                                    weights[expert, col : col + block_n, off : off + block_k],
                                    b_l1[:block_n, :],
                                )
                                if gated:
                                    T.copy(
                                        weights[
                                            expert, n + col : n + col + block_n, off : off + block_k
                                        ],
                                        b_l1[block_n : 2 * block_n, :],
                                    )
                            for sk in T.serial(block_k // l0_k):
                                T.copy(a_l1[:, off + sk * l0_k : off + (sk + 1) * l0_k], a_l0)
                                T.copy(b_l1[:, sk * l0_k : (sk + 1) * l0_k], b_l0)
                                T.gemm(
                                    a_l0,
                                    b_l0,
                                    acc,
                                    transpose_B=True,
                                    clear_accum=(ko == 0) & (sk == 0),
                                    unit_flag_ctrl=T.Select(
                                        (ko == k // block_k - 1) & (sk == block_k // l0_k - 1), 3, 2
                                    ),
                                )
                        _route_epilogue(
                            acc, wide, narrow, sid, count, half_m, block_n, gated, dtype
                        )
                        for row in T.serial(half_m):
                            mi = sid * half_m + row
                            if mi < count:
                                selected = T.alloc_var("int32", init=0)
                                rank = start + mi
                                if rank < tokens:
                                    selected = metadata[experts + routes + expert * tokens + rank]
                                else:
                                    for candidate in T.serial(routes):
                                        if (
                                            ids[candidate] == expert
                                            and metadata[experts + candidate] == rank
                                        ):
                                            selected = candidate
                                T.copy(narrow[row, :], out[selected, col : col + block_n])

    return tilelang.compile(main, target="ascend", out_idx=[])


def _projection_for_call(
    call: IndexedExpertCall, dtype: str, *, gated: bool, vector_weights: bool = False
):
    n = call.ffn_size if gated else call.hidden_size
    k = call.hidden_size if gated else call.ffn_size
    if gated:
        block_n, block_k, l0_k = (256, 128, 32) if call.num_tokens >= 64 else (128, 256, 64)
        n_shards = 2 if call.num_tokens >= 64 else 4
    else:
        block_n, block_k, l0_k = 256, 256, 64
        n_shards = 4 if call.hidden_size % 1024 == 0 else 2
    return _expert_projection(
        call.num_tokens,
        call.top_k,
        call.num_experts,
        n,
        k,
        dtype,
        gated,
        block_n,
        block_k,
        2,
        0,
        n_shards,
        l0_k,
        1 if gated else 0,
        vector_weights,
    )


class AscendIndexedExpertGateUpGroupedKernel(AscendIndexedExpertGateUpPreloadKernel):
    """Schedule expert groups directly and gather overflow ranks from route metadata."""

    preferred_over = frozenset({"ascend", "ascend_gate_up_fused", "ascend_gate_up_preload"})

    @classmethod
    def refusal(cls, call: IndexedExpertCall) -> str | None:
        reason = super().refusal(call)
        if reason is not None:
            return reason
        if call.num_tokens * call.top_k < call.num_experts:
            return "requires at least one route per expert on average for expert scheduling"
        return None

    def __init__(self, call: IndexedExpertCall):
        _AscendIndexedKernel.__init__(self, call)
        self.kernel = _projection_for_call(call, self.dtype_str, gated=True)


class AscendIndexedExpertDownGroupedKernel(AscendIndexedExpertDownFusedKernel):
    """Reuse L1 activation groups while scheduling down projections by expert."""

    preferred_over = frozenset({"ascend_down", "ascend_down_fused"})

    @classmethod
    def refusal(cls, call: IndexedExpertCall) -> str | None:
        reason = super().refusal(call)
        if reason is not None:
            return reason
        if call.num_tokens < 32 or call.top_k < 8:
            return "requires at least 32 tokens and 8 routes to amortize input preloading"
        if call.num_tokens * call.top_k < call.num_experts:
            return "requires at least one route per expert on average for expert scheduling"
        if call.ffn_size % 256 or call.hidden_size % 512:
            return "requires ffn_size divisible by 256 and hidden_size divisible by 512"
        # Full activation rows plus two weight tiles share the 512 KiB L1 budget.
        if 16 * call.ffn_size * 2 + 2 * 256 * 256 * 2 > 512 * 1024:
            return "preloaded activations and weight pipeline exceed L1 capacity"
        return None

    def __init__(self, call: IndexedExpertCall):
        _AscendIndexedKernel.__init__(self, call)
        self.kernel = _projection_for_call(call, self.dtype_str, gated=False)


def _weight_staging_refusal(call: IndexedExpertCall) -> str | None:
    if call.num_tokens * call.top_k > 4 * call.num_experts:
        return "vector weight staging requires at most four routes per expert on average"
    if call.hidden_size * call.ffn_size < 2 * 1024 * 1024:
        return "vector weight staging requires at least 2M projection elements per expert"
    return None


class AscendIndexedExpertGateUpVectorWeightsKernel(AscendIndexedExpertGateUpGroupedKernel):
    """Stage gate/up weights on both Vector cores before the Cube contraction."""

    preferred_over = frozenset({"ascend_gate_up_expert_group"})

    @classmethod
    def refusal(cls, call: IndexedExpertCall) -> str | None:
        return super().refusal(call) or _weight_staging_refusal(call)

    def __init__(self, call: IndexedExpertCall):
        _AscendIndexedKernel.__init__(self, call)
        self.kernel = _projection_for_call(call, self.dtype_str, gated=True, vector_weights=True)


class AscendIndexedExpertDownVectorWeightsKernel(AscendIndexedExpertDownGroupedKernel):
    """Split down-weight staging between Vector cores while retaining route order."""

    preferred_over = frozenset({"ascend_down_expert_group"})

    @classmethod
    def refusal(cls, call: IndexedExpertCall) -> str | None:
        reason = super().refusal(call) or _weight_staging_refusal(call)
        if reason is not None:
            return reason
        if call.ffn_size < 512:
            return "vector down-weight staging requires two or more K256 tiles"
        return None

    def __init__(self, call: IndexedExpertCall):
        _AscendIndexedKernel.__init__(self, call)
        self.kernel = _projection_for_call(call, self.dtype_str, gated=False, vector_weights=True)


register_kernel_type(
    "IndexedExpertMLPFwdOp", "ascend_gate_up_expert_group", AscendIndexedExpertGateUpGroupedKernel
)
register_kernel_type(
    "IndexedExpertMLPFwdOp", "ascend_down_expert_group", AscendIndexedExpertDownGroupedKernel
)
register_kernel_type(
    "IndexedExpertMLPFwdOp",
    "ascend_gate_up_vector_weights",
    AscendIndexedExpertGateUpVectorWeightsKernel,
)
register_kernel_type(
    "IndexedExpertMLPFwdOp",
    "ascend_down_vector_weights",
    AscendIndexedExpertDownVectorWeightsKernel,
)
