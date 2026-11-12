"""Mixed Cube/Vector expert projections with FP32 epilogues kept on chip."""

import functools

import tilelang
import tilelang.ascend.language as T
import torch

from tileops._kernels.ascend.moe.indexed_expert import (
    _AscendIndexedKernel,
    _calibrated_device_blocks,
    _route_epilogue,
)
from tileops._kernels.moe.call_spec import (
    IndexedExpertCall,
    IndexedExpertDownFwdInterface,
    IndexedExpertGateUpFwdInterface,
)
from tileops.backend._registry import register_kernel_type

__all__ = ["AscendIndexedExpertDownFusedKernel", "AscendIndexedExpertGateUpFusedKernel"]


@functools.lru_cache(maxsize=64)
def _fused_projection(
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
    job_order: int = 0,
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
    grid = min(num_blocks, routes * tiles) if num_blocks else routes * tiles

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
            acc = T.alloc_l0c((block_m, cols), "float32")
            wide = T.alloc_shared((half_m, cols), "float32")
            narrow = T.alloc_shared((half_m, block_n), dtype)
            T.annotate_buffer_versions({a_l1: num_stages, b_l1: num_stages})
            for pid in T.Persistent([routes * tiles], grid, bx):
                route = pid // tiles if job_order == 0 else pid % routes
                col = (pid % tiles if job_order == 0 else pid // routes) * block_n
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
                    for ko in T.Pipelined(k // block_k, num_stages=num_stages):
                        off = ko * block_k
                        for row in T.serial(count):
                            selected = T.alloc_var("int32", init=route)
                            if grouped and count > 1:
                                selected = metadata[
                                    experts + routes + expert * tokens + start + row
                                ]
                            source = selected // top_k if gated else selected
                            T.copy(
                                a[source : source + 1, off : off + block_k], a_l1[row : row + 1, :]
                            )
                        T.copy(
                            weights[expert, col : col + block_n, off : off + block_k],
                            b_l1[:block_n, :],
                        )
                        if gated:
                            T.copy(
                                weights[expert, n + col : n + col + block_n, off : off + block_k],
                                b_l1[block_n : 2 * block_n, :],
                            )
                        T.gemm(
                            a_l1,
                            b_l1,
                            acc,
                            transpose_B=True,
                            clear_accum=ko == 0,
                            unit_flag_ctrl=T.Select(ko == k // block_k - 1, 3, 2),
                        )
                    _route_epilogue(
                        acc, wide, narrow, sid, count, half_m, block_n, gated, dtype
                    )
                    for row in T.serial(half_m):
                        mi = sid * half_m + row
                        if mi < count:
                            selected = T.alloc_var("int32", init=route)
                            if grouped and count > 1:
                                selected = metadata[experts + routes + expert * tokens + start + mi]
                            T.copy(narrow[row, :], out[selected, col : col + block_n])

    return tilelang.compile(main, target="ascend", out_idx=[])


def _projection_for_call(call: IndexedExpertCall, dtype: str, *, gated: bool):
    n = call.ffn_size if gated else call.hidden_size
    k = call.hidden_size if gated else call.ffn_size
    # Pair gate/up columns in each Cube tile; the Vector epilogue sees both in FP32.
    n_cap = 128 if gated and call.num_tokens == 1 else 256
    k_cap = 128 if gated and call.grouped_dispatch else 256
    block_n = next(b for b in (256, 128, 64) if b <= n_cap and n % b == 0)
    block_k = next(b for b in (256, 128, 64, 32, 16) if b <= k_cap and k % b == 0)
    # Enough direct tiles amortize persistent buffers; down tiles interleave routes.
    direct_persistent = (
        call.num_tokens == 1 and call.top_k >= 8 and call.top_k * (n // block_n) >= 56
    )
    if direct_persistent:
        # The 28/56 grids filled the 28-Cube/56-Vector 950PR; scale with the device
        # so a 32-Cube/64-Vector 950DT does not idle a quarter of its cores.
        cube, vector = _calibrated_device_blocks()
        num_blocks = (cube or 28) if gated else (vector or 56)
    else:
        num_blocks = 64 if not gated and call.grouped_dispatch else 0
    job_order = 1 if direct_persistent and not gated else 0
    return _fused_projection(
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
        num_blocks,
        job_order,
    )


class AscendIndexedExpertGateUpFusedKernel(_AscendIndexedKernel, IndexedExpertGateUpFwdInterface):
    """Cube gate/up with Vector SiLU, one cast, and direct route-order stores."""

    preferred_over = frozenset({"ascend"})

    @classmethod
    def refusal(cls, call: IndexedExpertCall) -> str | None:
        reason = super().refusal(call)
        if reason is not None:
            return reason
        if call.ffn_size % 64:
            return "requires ffn_size divisible by 64 for the SIMD epilogue"
        return None

    def __init__(self, call: IndexedExpertCall):
        super().__init__(call)
        self.kernel = _projection_for_call(call, self.dtype_str, gated=True)

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
        c = self.call
        if c.grouped_dispatch and route_metadata is None:
            raise ValueError("route_metadata is required for grouped dispatch")
        if out is None:
            out = a.new_empty((c.num_tokens, c.top_k, c.ffn_size))
        metadata = route_metadata if route_metadata is not None else expert_ids.view(-1)[:1]
        self.kernel(a, weights, expert_ids.view(-1), metadata, out.view(-1, c.ffn_size))
        return out


class AscendIndexedExpertDownFusedKernel(_AscendIndexedKernel, IndexedExpertDownFwdInterface):
    """Cube down projection with Vector narrowing and direct route-order stores."""

    preferred_over = frozenset({"ascend_down"})

    @classmethod
    def refusal(cls, call: IndexedExpertCall) -> str | None:
        reason = super().refusal(call)
        if reason is not None:
            return reason
        if call.hidden_size % 64:
            return "requires hidden_size divisible by 64 for the SIMD epilogue"
        return None

    def __init__(self, call: IndexedExpertCall):
        super().__init__(call)
        self.kernel = _projection_for_call(call, self.dtype_str, gated=False)

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
        c = self.call
        if c.grouped_dispatch and route_metadata is None:
            raise ValueError("route_metadata is required for grouped dispatch")
        if out is None:
            out = a.new_empty((c.num_tokens, c.top_k, c.hidden_size))
        metadata = route_metadata if route_metadata is not None else expert_ids.view(-1)[:1]
        self.kernel(
            a.view(-1, c.ffn_size),
            weights,
            expert_ids.view(-1),
            metadata,
            out.view(-1, c.hidden_size),
        )
        return out


register_kernel_type(
    "IndexedExpertMLPFwdOp", "ascend_gate_up_fused", AscendIndexedExpertGateUpFusedKernel
)
register_kernel_type(
    "IndexedExpertMLPFwdOp", "ascend_down_fused", AscendIndexedExpertDownFusedKernel
)
