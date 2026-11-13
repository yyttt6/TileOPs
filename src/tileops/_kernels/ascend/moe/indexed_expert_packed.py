"""Pack expert input rows once, then share weight tiles across larger route groups."""

import functools
from typing import ClassVar

import tilelang
import tilelang.ascend.language as T
import torch

from tileops._kernels.ascend.moe.indexed_expert import _AscendIndexedKernel, _route_epilogue
from tileops._kernels.moe.call_spec import (
    IndexedExpertCall,
    IndexedExpertDownFwdInterface,
    IndexedExpertGateUpFwdInterface,
)
from tileops.backend._registry import register_kernel_type

__all__ = ["AscendIndexedExpertDownPackedKernel", "AscendIndexedExpertGateUpPackedKernel"]


@functools.lru_cache(maxsize=64)
def _pack_inputs(tokens, top_k, experts, k, dtype, gated):
    routes = tokens * top_k
    input_rows = tokens if gated else routes
    old_size = experts + routes + experts * tokens
    size = experts + routes

    @T.prim_func
    def main(
        a: T.Tensor((input_rows, k), dtype),
        ids: T.Tensor((routes,), "int32"),
        metadata: T.Tensor((old_size,), "int32"),
        packed: T.Tensor((routes, k), dtype),
        expanded: T.Tensor((size,), "int32"),
    ):
        with T.Kernel(routes) as route:
            row = T.alloc_shared((k,), dtype)
            expert = ids[route]
            rank = metadata[experts + route]
            base = T.alloc_var("int32", init=0)
            for e in T.serial(experts):
                if e < expert:
                    base = base + metadata[e]
            source = route // top_k if gated else route
            T.copy(a[source, :], row)
            T.copy(row, packed[base + rank, :])
            with T.SimtVF(threads=128):
                for _j in T.Parallel(1):
                    expanded[experts + base + rank] = route
                    if route < experts:
                        expanded[route] = metadata[route]

    return tilelang.compile(main, target="ascend", out_idx=[])


@functools.lru_cache(maxsize=64)
def _packed_projection(
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
    group_rows: int = 64,
):
    routes = tokens * top_k
    metadata_size = experts + routes
    width = 2 * n if gated else n
    cols = 2 * block_n if gated else block_n
    input_rows = routes
    block_m = group_rows
    half_m = group_rows // 2
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
            a_l1 = T.alloc_l1((block_m, block_k), dtype)
            b_l1 = T.alloc_l1((cols, block_k), dtype)
            a_l0 = T.alloc_l0a((block_m, l0_k), dtype)
            b_l0 = T.alloc_l0b((cols, l0_k), dtype)
            acc = T.alloc_l0c((block_m, cols), "float32")
            wide = T.alloc_shared((half_m, cols), "float32")
            narrow = T.alloc_shared((half_m, block_n), dtype)
            T.annotate_buffer_versions({a_l1: num_stages, b_l1: num_stages, a_l0: 2, b_l0: 2})
            for pid in T.Persistent([experts * n_shards], grid, bx):
                expert = pid // n_shards if job_order == 0 else pid % experts
                shard = pid % n_shards if job_order == 0 else pid // experts
                base = T.alloc_var("int32", init=0)
                for e in T.serial(experts):
                    if e < expert:
                        base = base + metadata[e]
                for group in T.serial(T.ceildiv(metadata[expert], block_m)):
                    start = group * block_m
                    count = T.min(block_m, metadata[expert] - start)
                    for nt in T.serial(tiles // n_shards):
                        col = (nt * n_shards + shard) * block_n
                        for ko in T.Pipelined(k // block_k, num_stages=num_stages):
                            off = ko * block_k
                            T.copy(
                                a[base + start : base + start + count, off : off + block_k],
                                a_l1[:count, :],
                            )
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
                                T.copy(a_l1[:, sk * l0_k : (sk + 1) * l0_k], a_l0)
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
                                rank = start + mi
                                selected = metadata[experts + base + rank]
                                T.copy(narrow[row, :], out[selected, col : col + block_n])

    return tilelang.compile(main, target="ascend", out_idx=[])


class _AscendIndexedPackedKernel(_AscendIndexedKernel):
    gated: ClassVar[bool]

    @classmethod
    def refusal(cls, call: IndexedExpertCall) -> str | None:
        reason = super().refusal(call)
        if reason is not None:
            return reason
        if call.num_tokens < 32 or call.top_k < 8:
            return "requires at least 32 tokens and 8 routes to amortize packing"
        routes = call.num_tokens * call.top_k
        if routes < 32 * call.num_experts:
            return "requires at least 32 routes per expert on average to amortize packing"
        if call.hidden_size * call.ffn_size < 2 * 1024 * 1024:
            return "requires at least 2M projection elements per expert to amortize packing"
        if call.hidden_size * call.ffn_size * routes < 128 * 1024 * 1024 * call.num_experts:
            return (
                "requires at least 128M routed projection elements per expert to amortize packing"
            )
        if call.hidden_size % 256 or call.ffn_size % 256:
            return "requires hidden_size and ffn_size divisible by 256"
        if max(call.hidden_size, call.ffn_size) > 32768:
            return "input row exceeds the 64 KiB packing buffer capacity"
        return None

    def __init__(self, call: IndexedExpertCall):
        super().__init__(call)
        n = call.ffn_size if self.gated else call.hidden_size
        k = call.hidden_size if self.gated else call.ffn_size
        routes = call.num_tokens * call.top_k
        # Streaming both operands makes 64/128 rows fit independently of full K.
        # L0C peaks at 128 * 256 FP32 elements; both L1 pipelines fit in 384 KiB.
        group_rows = 128 if routes > 64 * call.num_experts else 64
        block_n = 128 if self.gated else 256
        self.pack = _pack_inputs(
            call.num_tokens, call.top_k, call.num_experts, k, self.dtype_str, self.gated
        )
        self.kernel = _packed_projection(
            call.num_tokens,
            call.top_k,
            call.num_experts,
            n,
            k,
            self.dtype_str,
            self.gated,
            block_n,
            256,
            2,
            min(256, call.num_experts * (n // block_n)),
            n // block_n,
            64,
            1 if self.gated else 0,
            group_rows,
        )

    def forward(
        self,
        a: torch.Tensor,
        weights: torch.Tensor,
        expert_ids: torch.Tensor,
        route_metadata: torch.Tensor | None = None,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._require_devices(
            a=a, weights=weights, expert_ids=expert_ids, route_metadata=route_metadata, out=out
        )
        if route_metadata is None:
            raise ValueError("route_metadata is required for grouped dispatch")
        c = self.call
        routes = c.num_tokens * c.top_k
        n = c.ffn_size if self.gated else c.hidden_size
        k = c.hidden_size if self.gated else c.ffn_size
        packed = a.new_empty((routes, k))
        # Private counts and a compact inverse permutation cover every route, including
        # ranks beyond the public metadata's bounded expert lists.
        expanded = torch.empty(
            c.num_experts + routes,
            dtype=torch.int32,
            device=a.device,
        )
        if out is None:
            out = a.new_empty((c.num_tokens, c.top_k, n))
        ids = expert_ids.view(-1)
        self.pack(a.view(-1, k), ids, route_metadata, packed, expanded)
        self.kernel(packed, weights, ids, expanded, out.view(-1, n))
        return out


class AscendIndexedExpertGateUpPackedKernel(
    _AscendIndexedPackedKernel, IndexedExpertGateUpFwdInterface
):
    """Pack token routes once and share gate/up weight tiles across 64/128 rows."""

    gated = True
    preferred_over = frozenset(
        {"ascend", "ascend_gate_up_fused", "ascend_gate_up_preload", "ascend_gate_up_expert_group"}
    )


class AscendIndexedExpertDownPackedKernel(
    _AscendIndexedPackedKernel, IndexedExpertDownFwdInterface
):
    """Pack activation routes once and scatter low-precision down results in route order."""

    gated = False
    preferred_over = frozenset({"ascend_down", "ascend_down_fused", "ascend_down_expert_group"})


register_kernel_type(
    "IndexedExpertMLPFwdOp", "ascend_gate_up_packed", AscendIndexedExpertGateUpPackedKernel
)
register_kernel_type(
    "IndexedExpertMLPFwdOp", "ascend_down_packed", AscendIndexedExpertDownPackedKernel
)
