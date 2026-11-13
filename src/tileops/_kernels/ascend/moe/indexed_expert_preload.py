"""Reuse a complete L1 input group across fused gate/up output tiles."""

import functools

import tilelang
import tilelang.ascend.language as T

from tileops._kernels.ascend.moe.indexed_expert import _AscendIndexedKernel, _route_epilogue
from tileops._kernels.ascend.moe.indexed_expert_fused import AscendIndexedExpertGateUpFusedKernel
from tileops._kernels.moe.call_spec import IndexedExpertCall
from tileops.backend._registry import register_kernel_type

__all__ = ["AscendIndexedExpertGateUpPreloadKernel"]


@functools.lru_cache(maxsize=64)
def _preload_projection(
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
):
    routes = tokens * top_k
    grouped = tokens > 1
    metadata_size = experts + routes + experts * tokens if grouped else 1
    width = 2 * n if gated else n
    cols = 2 * block_n if gated else block_n
    input_rows = tokens if gated else routes
    block_m = 16
    half_m = 8
    tiles = n // block_n

    grid = min(num_blocks, routes * n_shards) if num_blocks else routes * n_shards

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
            a_l0 = T.alloc_l0a((block_m, 64), dtype)
            b_l0 = T.alloc_l0b((cols, 64), dtype)
            acc = T.alloc_l0c((block_m, cols), "float32")
            wide = T.alloc_shared((half_m, cols), "float32")
            narrow = T.alloc_shared((half_m, block_n), dtype)
            T.annotate_buffer_versions({b_l1: num_stages, a_l0: 2, b_l0: 2})
            for pid in T.Persistent([routes * n_shards], grid, bx):
                route = pid // n_shards
                shard = pid % n_shards
                expert = ids[route]
                leader = T.alloc_var("int32", init=1)
                count = T.alloc_var("int32", init=1)
                start = T.alloc_var("int32", init=0)
                if grouped:
                    rank = metadata[experts + route]
                    if rank < tokens:
                        start = rank // block_m * block_m
                        count = T.min(block_m, T.min(metadata[expert], tokens) - start)
                        if rank != start:
                            leader = 0
                if leader != 0:
                    for row in T.serial(count):
                        selected = T.alloc_var("int32", init=route)
                        if grouped and count > 1:
                            selected = metadata[experts + routes + expert * tokens + start + row]
                        source = selected // top_k if gated else selected
                        T.copy(a[source : source + 1, :], a_l1[row : row + 1, :])
                    for nt in T.serial(tiles // n_shards):
                        col = (nt * n_shards + shard) * block_n
                        for ko in T.Pipelined(k // block_k, num_stages=num_stages):
                            off = ko * block_k
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
                            for sk in T.serial(block_k // 64):
                                T.copy(a_l1[:, off + sk * 64 : off + (sk + 1) * 64], a_l0)
                                T.copy(b_l1[:, sk * 64 : (sk + 1) * 64], b_l0)
                                T.gemm(
                                    a_l0,
                                    b_l0,
                                    acc,
                                    transpose_B=True,
                                    clear_accum=(ko == 0) & (sk == 0),
                                    unit_flag_ctrl=T.Select(
                                        (ko == k // block_k - 1) & (sk == block_k // 64 - 1), 3, 2
                                    ),
                                )
                        _route_epilogue(
                            acc, wide, narrow, sid, count, half_m, block_n, gated, dtype
                        )
                        for row in T.serial(half_m):
                            mi = sid * half_m + row
                            if mi < count:
                                selected = T.alloc_var("int32", init=route)
                                if grouped and count > 1:
                                    selected = metadata[
                                        experts + routes + expert * tokens + start + mi
                                    ]
                                T.copy(narrow[row, :], out[selected, col : col + block_n])

    return tilelang.compile(main, target="ascend", out_idx=[])


class AscendIndexedExpertGateUpPreloadKernel(AscendIndexedExpertGateUpFusedKernel):
    """Preload routed input rows once per output shard, then reuse them from L1."""

    preferred_over = frozenset({"ascend", "ascend_gate_up_fused"})

    @classmethod
    def refusal(cls, call: IndexedExpertCall) -> str | None:
        reason = super().refusal(call)
        if reason is not None:
            return reason
        if call.num_tokens < 32 or call.top_k < 8:
            return "requires at least 32 tokens and 8 routes to amortize input preloading"
        if call.hidden_size % 256 or call.ffn_size % 512:
            return "requires hidden_size divisible by 256 and ffn_size divisible by 512"
        # Full 16-row input plus two gate/up weight tiles must fit in 512 KiB L1.
        if 16 * call.hidden_size * 2 + 2 * 256 * 256 * 2 > 512 * 1024:
            return "preloaded input and weight pipeline exceed L1 capacity"
        return None

    def __init__(self, call: IndexedExpertCall):
        _AscendIndexedKernel.__init__(self, call)
        # More routes already supply parallelism; use fewer N shards for larger T.
        n_shards = 2 if call.num_tokens >= 64 else 4
        self.kernel = _preload_projection(
            call.num_tokens,
            call.top_k,
            call.num_experts,
            call.ffn_size,
            call.hidden_size,
            self.dtype_str,
            True,
            128,
            256,
            2,
            0,
            n_shards,
        )


register_kernel_type(
    "IndexedExpertMLPFwdOp", "ascend_gate_up_preload", AscendIndexedExpertGateUpPreloadKernel
)
