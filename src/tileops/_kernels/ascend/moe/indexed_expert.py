"""Route-indexed expert projections and reduction for Ascend 950."""

import functools
from typing import ClassVar

import tilelang
import tilelang.ascend.language as T
import torch

from tileops._kernels.kernel_base import Kernel
from tileops._kernels.moe.call_spec import (
    IndexedExpertCall,
    IndexedExpertDownFwdInterface,
    IndexedExpertGateUpFwdInterface,
    IndexedRouteStatsFwdInterface,
    IndexedWeightedReduceFwdInterface,
)
from tileops.backend._registry import register_kernel_type

__all__ = [
    "AscendIndexedExpertDownKernel",
    "AscendIndexedExpertGateUpKernel",
    "AscendIndexedRouteStatsKernel",
    "AscendIndexedRouteStatsParallelKernel",
    "AscendIndexedWeightedReduceKernel",
]

# Projection programs, epilogues and allocations share this scratch row stride.
_ROUTE_TILE_ROWS = 16


@functools.lru_cache(maxsize=1)
def _calibrated_device_blocks() -> tuple[int | None, int | None]:
    """Cube and Vector core counts of the current device; ``(None, None)`` if unknown.

    Grid sizes that filled the 28-Cube/56-Vector Ascend950PR development card
    scale with these counts so the 32-Cube/64-Vector Ascend950DT keeps full waves.
    """
    try:
        from tilelang.carver.arch.driver import get_num_cube_cores, get_num_vector_cores

        return get_num_cube_cores(device_id=0), get_num_vector_cores(device_id=0)
    except Exception:
        return None, None


@T.macro
def _route_epilogue(
    acc, wide, narrow, sid, count, half_m: int, block_n: int, gated: bool, dtype: str
) -> None:
    """Shared Vector epilogue: FP32 dual_copy from L0C, activation, low-precision pack.

    Only this activation part is identical across the fused/preload/grouped/packed
    projection kernels; each caller keeps its own route-order store loop.
    """
    # dav-950 dual-destination FIX stores must retain FP32.
    T.dual_copy(acc, wide, unit_flag_ctrl=3)
    with T.SimdVF():
        full = T.simd.pset(32)
        for row in T.serial(half_m):
            if row + sid * half_m < count:
                for chunk in T.serial(block_n // 64):
                    value = T.simd.vld(wide[row, chunk * 64])
                    if gated:
                        up = T.simd.vld(wide[row, block_n + chunk * 64])
                        neg = T.simd.vmuls(value, -1.0, full)
                        exp = T.simd.vexp(neg, full)
                        den = T.simd.vadds(exp, 1.0, full)
                        silu = T.simd.vdiv(value, den, full)
                        value = T.simd.vmul(silu, up, full)
                    low = T.simd.vcvt(value, dtype, full, sat=False)
                    T.simd.vsts(narrow[row, chunk * 64], low, full, dist="PK_B32")


@functools.lru_cache(maxsize=32)
def _route_stats(tokens: int, top_k: int, experts: int):
    routes = tokens * top_k
    size = experts + routes + experts * tokens
    padded = (size + 63) // 64 * 64

    @T.prim_func
    def main(ids: T.Tensor((routes,), "int32"), out: T.Tensor((size,), "int32")):
        with T.Kernel(1):
            ids_ub = T.alloc_shared(((routes + 63) // 64 * 64,), "int32")
            meta = T.alloc_shared((padded,), "int32")
            T.copy(ids, ids_ub[:routes])
            with T.SimtVF(threads=128):
                for i in T.Parallel(padded):
                    meta[i] = 0
            # Stable ranks satisfy the ABI, which does not prescribe atomic arrival order.
            for route in T.serial(routes):
                expert = ids_ub[route]
                rank = meta[expert]
                meta[experts + route] = rank
                if rank < tokens:
                    meta[experts + routes + expert * tokens + rank] = route
                meta[expert] = rank + 1
            T.copy(meta[:size], out)

    return tilelang.compile(main, target="ascend", out_idx=[])


@functools.lru_cache(maxsize=64)
def _projection_program(
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
    num_blocks: int = 0,
):
    routes = tokens * top_k
    grouped = tokens > 1
    size = experts + routes + experts * tokens if grouped else 1
    width = 2 * n if gated else n
    input_rows = tokens if gated else routes
    out_dtype = "float32"
    # Sixteen gathered rows share each expert weight tile on Cube.
    block_m = _ROUTE_TILE_ROWS
    n_tiles = (width + block_n - 1) // block_n
    grid = min(num_blocks, routes * n_tiles) if num_blocks else routes * n_tiles

    @T.prim_func
    def main(
        a: T.Tensor((input_rows, k), dtype),
        weights: T.Tensor((experts, width, k), dtype),
        ids: T.Tensor((routes,), "int32"),
        metadata: T.Tensor((size,), "int32"),
        out: T.Tensor((routes, block_m, width), out_dtype),
    ):
        with T.Kernel(grid) as bx:
            a_l1 = T.alloc_l1((block_m, block_k), dtype)
            b_l1 = T.alloc_l1((block_n, block_k), dtype)
            acc = T.alloc_l0c((block_m, block_n), "float32")
            T.annotate_buffer_versions({a_l1: num_stages, b_l1: num_stages})
            # Persistent down tiles share buffers; the GEMM resets accumulation per tile.
            for pid in T.Persistent([routes * n_tiles], grid, bx):
                route = pid // n_tiles
                col = pid % n_tiles * block_n
                actual_n = T.min(block_n, width - col)
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
                    for ko in T.Pipelined(T.ceildiv(k, block_k), num_stages=num_stages):
                        offset = ko * block_k
                        actual_k = T.min(block_k, k - offset)
                        for row in T.serial(count):
                            selected = T.alloc_var("int32", init=route)
                            if grouped and count > 1:
                                selected = metadata[
                                    experts + routes + expert * tokens + start + row
                                ]
                            source = selected // top_k if gated else selected
                            T.copy(
                                a[source : source + 1, offset : offset + actual_k],
                                a_l1[row : row + 1, :actual_k],
                            )
                        T.copy(
                            weights[expert, col : col + actual_n, offset : offset + actual_k],
                            b_l1[:actual_n, :actual_k],
                        )
                        T.gemm(
                            a_l1,
                            b_l1,
                            acc,
                            transpose_B=True,
                            clear_accum=ko == 0,
                        )
                    # L0C stores require a zero-origin matrix; Vector scatters rows later.
                    T.copy(acc[:count, :actual_n], out[route, :count, col : col + actual_n])

    return main


@functools.lru_cache(maxsize=64)
def _projection(
    tokens: int,
    top_k: int,
    experts: int,
    n: int,
    k: int,
    dtype: str,
    gated: bool,
    block_n_cap: int = 256,
    block_k_cap: int = 256,
    num_stages: int = 2,
    num_blocks: int = 0,
):
    # Wider tiles and two L1 versions overlap DMA with Cube; divisors retain alignment.
    width = 2 * n if gated else n
    block_n = next(b for b in (256, 128, 64, 32, 16) if b <= block_n_cap and width % b == 0)
    block_k = next(b for b in (512, 256, 128, 64, 32, 16) if b <= block_k_cap and k % b == 0)
    program = _projection_program(
        tokens, top_k, experts, n, k, dtype, gated, block_n, block_k, num_stages, num_blocks
    )
    return tilelang.compile(program, target="ascend", out_idx=[])


@functools.lru_cache(maxsize=64)
def _projection_epilogue(tokens: int, top_k: int, experts: int, n: int, dtype: str, gated: bool):
    routes = tokens * top_k
    grouped = tokens > 1
    size = experts + routes + experts * tokens if grouped else 1
    block_m = _ROUTE_TILE_ROWS
    width = 2 * n if gated else n
    block = 256
    tiles = (n + block - 1) // block

    @T.prim_func
    def main(
        x: T.Tensor((routes, block_m, width), "float32"),
        ids: T.Tensor((routes,), "int32"),
        metadata: T.Tensor((size,), "int32"),
        out: T.Tensor((routes, n), dtype),
    ):
        with T.Kernel(routes * tiles) as bid:
            route = bid // tiles
            col = bid % tiles * block
            length = T.min(block, n - col)
            leader = T.alloc_var("int32", init=route)
            row = T.alloc_var("int32", init=0)
            if grouped:
                rank = metadata[experts + route]
                if rank < tokens:
                    expert = ids[route]
                    start = rank // block_m * block_m
                    leader = metadata[experts + routes + expert * tokens + start]
                    row = rank % block_m
            gate = T.alloc_shared((block,), "float32")
            up = T.alloc_shared((block,), "float32")
            result = T.alloc_shared((block,), dtype)
            T.copy(x[leader, row, col : col + length], gate[:length])
            if gated:
                T.copy(x[leader, row, n + col : n + col + length], up[:length])
            with T.SimtVF(threads=128):
                for i in T.Parallel(block):
                    if gated:
                        result[i] = T.Cast(dtype, gate[i] / (1.0 + T.exp(-gate[i])) * up[i])
                    else:
                        result[i] = T.Cast(dtype, gate[i])
            T.copy(result[:length], out[route, col : col + length])

    return tilelang.compile(main, target="ascend", out_idx=[])


@functools.lru_cache(maxsize=32)
def _weighted_reduce(tokens: int, top_k: int, hidden: int, dtype: str, scale: float):
    # One token is a single 2D gather: many small blocks hide its fixed cost better.
    # Larger route batches amortize per-block setup, so take the widest tile the UB
    # working set (route rows plus the result) allows instead of a fixed 256 columns.
    if tokens == 1:
        block = min(256, hidden)
    else:
        cap = 160 * 1024 // (2 * top_k + 2) // 64 * 64
        block = min(hidden, max(64, cap))
    tiles = (hidden + block - 1) // block

    @T.prim_func
    def main(
        x: T.Tensor((tokens, top_k, hidden), dtype),
        weights: T.Tensor((tokens, top_k), "float32"),
        out: T.Tensor((tokens, hidden), dtype),
    ):
        with T.Kernel(tokens * tiles) as bid:
            token = bid // tiles
            col = bid % tiles * block
            length = T.min(block, hidden - col)
            rows = T.alloc_shared((top_k, block), dtype)
            probs = T.alloc_shared(((top_k + 7) // 8 * 8,), "float32")
            result = T.alloc_shared((block,), dtype)
            T.copy(x[token, :, col : col + length], rows[:, :length])
            T.copy(weights[token, :], probs[:top_k])
            with T.SimtVF(threads=128):
                acc = T.alloc_fragment((block,), "float32")
                for i in T.Parallel(block):
                    acc[i] = 0.0
                for slot in T.serial(top_k):
                    for i in T.Parallel(block):
                        acc[i] = acc[i] + T.Cast("float32", rows[slot, i]) * probs[slot]
                for i in T.Parallel(block):
                    result[i] = T.Cast(dtype, acc[i] * scale)
            T.copy(result[:length], out[token, col : col + length])

    return tilelang.compile(main, target="ascend", out_idx=[])


class _AscendIndexedKernel(Kernel):
    devices: ClassVar[frozenset[str]] = frozenset({"npu"})

    @classmethod
    def refusal(cls, call: IndexedExpertCall) -> str | None:
        if call.dtype not in (torch.float16, torch.bfloat16):
            return "requires float16 or bfloat16"
        if min(call.num_tokens, call.top_k, call.num_experts, call.hidden_size, call.ffn_size) <= 0:
            return "requires positive token, route, expert and projection dimensions"
        if call.hidden_size % 16 or call.ffn_size % 16:
            return "requires hidden_size and ffn_size divisible by 16"
        size = IndexedRouteStatsFwdInterface.output_size(call)
        if size > 32768:
            return "route metadata exceeds the 32768-element UB capacity"
        routes = call.num_tokens * call.top_k
        # Counts/ranks/lists and IDs coexist in UB during the statistics pass.
        if (size + 63) // 64 * 64 + (routes + 63) // 64 * 64 > 49152:
            return "route metadata and IDs exceed the combined UB capacity"
        # The reduction keeps K rows of 256 low-precision elements in UB.
        if call.top_k > 256:
            return "top_k exceeds the 256-route reduction UB capacity"
        return None

    def __init__(self, call: IndexedExpertCall):
        reason = self.refusal(call)
        if reason is not None:
            raise ValueError(reason)
        super().__init__(device_index=call.device.index if call.device is not None else None)
        self.call = call
        self.dtype = call.dtype


class AscendIndexedRouteStatsKernel(_AscendIndexedKernel, IndexedRouteStatsFwdInterface):
    """Stable per-expert counts, ranks and bounded route lists on device."""

    def __init__(self, call: IndexedExpertCall):
        super().__init__(call)
        self.kernel = _route_stats(call.num_tokens, call.top_k, call.num_experts)

    def forward(self, expert_ids: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        self._require_devices(expert_ids=expert_ids, out=out)
        self.kernel(expert_ids.view(-1), out)
        return out


@functools.lru_cache(maxsize=64)
def _parallel_route_stats(tokens: int, top_k: int, experts: int, num_blocks: int = 112):
    routes = tokens * top_k
    size = experts + routes + experts * tokens
    padded_routes = (routes + 63) // 64 * 64
    padded_tokens = (tokens + 63) // 64 * 64
    blocks = min(experts, num_blocks) if num_blocks else experts

    @T.prim_func
    def main(ids: T.Tensor((routes,), "int32"), out: T.Tensor((size,), "int32")):
        with T.Kernel(blocks) as bx:
            ids_ub = T.alloc_shared((padded_routes,), "int32")
            ranks = T.alloc_shared((padded_routes,), "int32")
            route_list = T.alloc_shared((padded_tokens,), "int32")
            count = T.alloc_shared((64,), "int32")
            T.copy(ids, ids_ub[:routes])
            T.ascend_set_flag("MTE2_V", 0)
            T.ascend_wait_flag("MTE2_V", 0)
            for step in T.serial(T.ceildiv(experts, blocks)):
                expert = step * blocks + bx
                if expert < experts:
                    with T.SimdVF():
                        full = T.simd.pset(32)
                        zero = T.simd.vdup(0, "int32", full)
                        for part in T.serial(padded_tokens // 64):
                            T.simd.vsts(route_list[part * 64], zero, full)
                        # Finish zeroing before selected lanes scatter into the list.
                        T.simd.mem_bar("VST_VLD")
                        offset = T.simd.alloc_var("int32")
                        offset = T.simd.vdup(0, "int32", full)
                        for chunk in T.serial(padded_routes // 64):
                            valid = T.simd.update_mask(T.min(64, routes - chunk * 64), 32)
                            values = T.simd.vld(ids_ub[chunk * 64])
                            selected = T.simd.vcmps(values, expert, valid, "eq")
                            prefix = T.simd.vusqz(selected, "int32")
                            rank = T.simd.vadd(prefix, offset, full)
                            T.simd.vsts(ranks[chunk * 64], rank, selected)
                            listed = T.simd.vcmps(rank, tokens, selected, "lt")
                            route = T.simd.vci(chunk * 64, "int32")
                            T.simd.vscatter(
                                route,
                                T.access_ptr(route_list[0], "w", extent=padded_tokens),
                                T.reinterpret("uint32x64", rank),
                                listed,
                            )
                            ones = T.simd.vdup(1, "int32", selected)
                            partial = T.simd.vcadd(ones, full)
                            offset = T.simd.vadd(offset, T.simd.vdupv(partial, full), full)
                        T.simd.vsts(count[0], offset, full, dist="ONEPT_B32")
                        T.simd.mem_bar("VST_VLD")
                    # SIMD stores must finish before SIMT reads the scratch buffers.
                    T.ascend_set_flag("V_S", 0)
                    T.ascend_wait_flag("V_S", 0)
                    with T.SimtVF(threads=128):
                        for i in T.Parallel(routes):
                            if ids_ub[i] == expert:
                                out[experts + i] = ranks[i]
                            if i < tokens:
                                out[experts + routes + expert * tokens + i] = route_list[i]
                            if i == 0:
                                out[expert] = count[0]
                    # Complete SIMT reads before reusing scratch for the next expert.
                    T.ascend_set_flag("V_S", 1)
                    T.ascend_wait_flag("V_S", 1)

    return tilelang.compile(
        main,
        target="ascend",
        out_idx=[],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE: False,
            tilelang.PassConfigKey.TL_DISABLE_SHARED_MEMORY_REUSE: True,
            tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
        },
    )


class AscendIndexedRouteStatsParallelKernel(AscendIndexedRouteStatsKernel):
    """Distribute experts across Vector blocks without changing the metadata ABI."""

    preferred_over = frozenset({"ascend_route_stats"})

    @classmethod
    def refusal(cls, call: IndexedExpertCall) -> str | None:
        reason = super().refusal(call)
        if reason is not None:
            return reason
        routes = call.num_tokens * call.top_k
        if call.num_tokens <= 1 or routes < 256:
            return "parallel route statistics requires at least 256 grouped routes"
        if call.num_experts > routes:
            return "parallel route statistics requires no more experts than routes"
        return None

    def __init__(self, call: IndexedExpertCall):
        _AscendIndexedKernel.__init__(self, call)
        # Two route-statistics waves per Vector core matched the 950PR; scale with
        # the device so a 64-Vector 950DT keeps full waves instead of a stale tail.
        _, vector = _calibrated_device_blocks()
        num_blocks = 2 * (vector or 56)
        self.kernel = _parallel_route_stats(
            call.num_tokens, call.top_k, call.num_experts, num_blocks
        )


register_kernel_type(
    "IndexedExpertMLPFwdOp", "ascend_route_stats_parallel", AscendIndexedRouteStatsParallelKernel
)


class AscendIndexedExpertGateUpKernel(_AscendIndexedKernel, IndexedExpertGateUpFwdInterface):
    """Cube projection to FP32, then Vector SiLU and multiplication before the cast."""

    def __init__(self, call: IndexedExpertCall):
        super().__init__(call)
        self.kernel = _projection(
            call.num_tokens,
            call.top_k,
            call.num_experts,
            call.ffn_size,
            call.hidden_size,
            self.dtype_str,
            True,
        )
        self.activate = _projection_epilogue(
            call.num_tokens, call.top_k, call.num_experts, call.ffn_size, self.dtype_str, True
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
        c = self.call
        if c.grouped_dispatch and route_metadata is None:
            raise ValueError("route_metadata is required for grouped dispatch")
        projected = a.new_empty(
            (c.num_tokens * c.top_k, _ROUTE_TILE_ROWS, 2 * c.ffn_size), dtype=torch.float32
        )
        if out is None:
            out = a.new_empty((c.num_tokens, c.top_k, c.ffn_size))
        metadata = route_metadata if route_metadata is not None else expert_ids.view(-1)[:1]
        self.kernel(a, weights, expert_ids.view(-1), metadata, projected)
        self.activate(projected, expert_ids.view(-1), metadata, out.view(-1, c.ffn_size))
        return out


class AscendIndexedExpertDownKernel(_AscendIndexedKernel, IndexedExpertDownFwdInterface):
    """Cube down projection, scattered to route order and rounded to storage dtype."""

    def __init__(self, call: IndexedExpertCall):
        super().__init__(call)
        self.kernel = _projection(
            call.num_tokens,
            call.top_k,
            call.num_experts,
            call.hidden_size,
            call.ffn_size,
            self.dtype_str,
            False,
            num_blocks=64 if call.grouped_dispatch else 0,
        )

        self.scatter = _projection_epilogue(
            call.num_tokens, call.top_k, call.num_experts, call.hidden_size, self.dtype_str, False
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
        c = self.call
        if c.grouped_dispatch and route_metadata is None:
            raise ValueError("route_metadata is required for grouped dispatch")
        projected = a.new_empty(
            (c.num_tokens * c.top_k, _ROUTE_TILE_ROWS, c.hidden_size), dtype=torch.float32
        )
        if out is None:
            out = a.new_empty((c.num_tokens, c.top_k, c.hidden_size))
        metadata = route_metadata if route_metadata is not None else expert_ids.view(-1)[:1]
        self.kernel(a.view(-1, c.ffn_size), weights, expert_ids.view(-1), metadata, projected)
        self.scatter(projected, expert_ids.view(-1), metadata, out.view(-1, c.hidden_size))
        return out


class AscendIndexedWeightedReduceKernel(_AscendIndexedKernel, IndexedWeightedReduceFwdInterface):
    """Apply route weights in FP32, sum in slot order, scale and store in place."""

    def __init__(self, call: IndexedExpertCall):
        super().__init__(call)
        self.kernel = _weighted_reduce(
            call.num_tokens,
            call.top_k,
            call.hidden_size,
            self.dtype_str,
            call.routed_scaling_factor,
        )

    def forward(
        self, expert_output: torch.Tensor, topk_weights: torch.Tensor, output: torch.Tensor
    ) -> torch.Tensor:
        self._require_devices(expert_output=expert_output, topk_weights=topk_weights, output=output)
        self.kernel(expert_output, topk_weights, output)
        return output


# The benchmark discovers an op by the "ascend" key; each other stage needs its own key.
register_kernel_type("IndexedExpertMLPFwdOp", "ascend", AscendIndexedExpertGateUpKernel)
register_kernel_type("IndexedExpertMLPFwdOp", "ascend_route_stats", AscendIndexedRouteStatsKernel)
register_kernel_type("IndexedExpertMLPFwdOp", "ascend_down", AscendIndexedExpertDownKernel)
register_kernel_type("IndexedExpertMLPFwdOp", "ascend_reduce", AscendIndexedWeightedReduceKernel)
