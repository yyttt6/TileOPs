"""Dense shared-expert projections and their rounded gated activation on Ascend 950."""

import functools
from typing import ClassVar

import tilelang
import tilelang.ascend.language as T
import torch
from tilelang.carver.arch.driver import get_num_cube_cores

from tileops._kernels.kernel_base import Entry, Kernel
from tileops._kernels.moe.call_spec import SharedExpertMLPCall, SharedExpertMLPFwdInterface
from tileops.backend._registry import register_kernel_type

__all__ = ["AscendSharedExpertMLPKernel"]


@T.macro
def _simd_activation(wide, activated, half_m, gate_n, dtype):
    """Round Cube projections, then evaluate the gated activation in FP32."""
    with T.SimdVF():
        mask = T.simd.pset(32)
        one = T.simd.vdup(1.0, "float32", mask)
        for row in T.serial(half_m):
            for chunk in T.Unroll((gate_n + 63) // 64, explicit=True):
                # Overlap the final 64-lane window for N=96. All transfers stay
                # within the row, and repeated lanes receive the same value.
                j = T.min(chunk * 64, gate_n - 64)
                g_raw = T.simd.vld(wide[row, j])
                u_raw = T.simd.vld(wide[row, gate_n + j])
                g_low = T.simd.vcvt(g_raw, dtype, mask, round="ROUND_R", sat=False)
                u_low = T.simd.vcvt(u_raw, dtype, mask, round="ROUND_R", sat=False)
                g = T.simd.vcvt(g_low, "float32", mask, part=0)
                u = T.simd.vcvt(u_low, "float32", mask, part=0)
                neg = T.simd.vmuls(g, -1.0, mask)
                # Native exp/div match the installed SIMT path's precision.
                ex = T.simd.vexp(neg, mask, precision="ftz_true")
                den = T.simd.vadds(ex, 1.0, mask)
                sigmoid = T.simd.vdiv(one, den, mask, precision="ftz_true")
                silu = T.simd.vmul(g, sigmoid, mask)
                value = T.simd.vmul(silu, u, mask)
                low = T.simd.vcvt(value, dtype, mask, round="ROUND_R", sat=False)
                T.simd.vsts(activated[row, j], low, mask, dist="PK_B32")


def _projection_config(m: int, n: int, k: int, cube_cores: int) -> dict:
    """Select bounded L1/Cube tiles from the profiled occupancy and DMA regimes."""
    padded_m = (m + 15) // 16 * 16
    block_m = next(b for b in (128, 64, 32, 16) if padded_m % b == 0)
    block_n = next(b for b in (128, 64, 32, 16) if n % b == 0)
    block_k = next(b for b in (256, 128, 64, 32, 16) if k % b == 0)
    stages = 2
    group_m = 1
    if m >= 512 and m % 256 == 0 and n % 128 == 0 and k % 256 == 0:
        # A 256-row tile halves logical weight reloads relative to 128 rows.
        # At N=256, the two A/B versions occupy 512 KiB L1 and C uses 256 KiB L0C.
        block_m = 256
        block_n = 256 if n % 256 == 0 else 128
        group_m = next(g for g in (4, 2, 1) if (m // block_m) % g == 0)
    elif (
        m == 32
        and n % 256 == 0
        and k >= 4096
        and k % 256 == 0
        and cube_cores // 2 <= n // 256 < cube_cores
    ):
        # A half-wave of wider tiles needs a third DMA version to hide latency.
        block_n, stages = 256, 3
    return dict(
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        stages=stages,
        num_blocks=cube_cores,
        group_m=group_m,
    )


def _nt_gemm_program(
    m: int,
    n: int,
    k: int,
    dtype: str,
    block_m: int,
    block_n: int,
    block_k: int,
    stages: int,
    num_blocks: int,
    group_m: int = 1,
):
    """Compute ``a @ weights.T`` without transposing the weights in global memory."""
    m_tiles = (m + block_m - 1) // block_m
    n_tiles = n // block_n
    # Persistent work bounds the launch while retaining full waves on 950 SKUs.
    grid = min(num_blocks, m_tiles * n_tiles)

    @T.prim_func
    def main(
        a: T.Tensor((m, k), dtype),
        weights: T.Tensor((n, k), dtype),
        out: T.Tensor((m, n), dtype),
    ):
        with T.Kernel(grid) as bx:
            a_l1 = T.alloc_l1((block_m, block_k), dtype)
            b_l1 = T.alloc_l1((block_n, block_k), dtype)
            acc = T.alloc_l0c((block_m, block_n), "float32")
            T.annotate_buffer_versions({a_l1: stages, b_l1: stages})
            for task in T.Persistent([m_tiles * n_tiles], grid, bx):
                # Neighboring M tiles consume the same weight tile close in time.
                # Logical reads are unchanged; the order permits reuse in L2.
                row = (task // (n_tiles * group_m) * group_m + task % group_m) * block_m
                col = task // group_m % n_tiles * block_n
                rows = T.min(block_m, m - row)
                for ko in T.Pipelined(k // block_k, num_stages=stages):
                    off = ko * block_k
                    T.copy(a[row : row + rows, off : off + block_k], a_l1[:rows, :])
                    T.copy(weights[col : col + block_n, off : off + block_k], b_l1)
                    T.gemm(
                        a_l1,
                        b_l1,
                        acc,
                        transpose_B=True,
                        clear_accum=ko == 0,
                    )
                # FIX rounds the FP32 accumulation to the input storage dtype.
                T.copy(acc[:rows, :], out[row : row + rows, col : col + block_n])

    return main


@functools.lru_cache(maxsize=128)
def _nt_gemm_kernel(
    m: int,
    n: int,
    k: int,
    dtype: str,
    block_m: int = 0,
    block_n: int = 0,
    block_k: int = 0,
    stages: int = 2,
    num_blocks: int = 32,
    group_m: int = 1,
):
    # L1/FIX transfers pack a partial M region to a multiple of 16. Its packed
    # extent must match the static Cube tile, including the final token block.
    padded_m = (m + 15) // 16 * 16
    block_m = block_m or next(b for b in (128, 64, 32, 16) if padded_m % b == 0)
    block_n = block_n or next(b for b in (128, 64, 32, 16) if n % b == 0)
    block_k = block_k or next(b for b in (128, 64, 32, 16) if k % b == 0)
    if padded_m % block_m or n % block_n or k % block_k:
        raise ValueError("GEMM tiles must divide packed M, N and K")
    if ((m + block_m - 1) // block_m) % group_m:
        raise ValueError("M scheduling group must divide the number of M tiles")
    program = _nt_gemm_program(
        m, n, k, dtype, block_m, block_n, block_k, stages, num_blocks, group_m
    )
    return tilelang.compile(program, target="ascend", out_idx=[2])


@functools.lru_cache(maxsize=64)
def _silu_mul_kernel(
    m: int, n: int, dtype: str, block_n: int = 0, stages: int = 1, num_blocks: int = 64
):
    """Keep gate, sigmoid and up in FP32 until the activation's final store."""
    block_n = block_n or next(b for b in (256, 128, 64, 32, 16) if n % b == 0)
    col_tiles = n // block_n
    grid = min(num_blocks, m * col_tiles)

    @T.prim_func
    def main(gate_up: T.Tensor((m, 2 * n), dtype), out: T.Tensor((m, n), dtype)):
        with T.Kernel(grid) as bx:
            gate = T.alloc_shared((block_n,), dtype)
            up = T.alloc_shared((block_n,), dtype)
            result = T.alloc_shared((block_n,), dtype)
            T.annotate_buffer_versions({gate: stages, up: stages, result: stages})
            for task in T.Persistent([m * col_tiles], grid, bx, num_stages=stages):
                row = task // col_tiles
                col = task % col_tiles * block_n
                T.copy(gate_up[row, col : col + block_n], gate)
                T.copy(gate_up[row, n + col : n + col + block_n], up)
                with T.SimtVF(threads=128 if block_n <= 256 else min(1024, block_n)):
                    for i in T.Parallel(block_n):
                        g = T.cast(gate[i], "float32")
                        u = T.cast(up[i], "float32")
                        sigmoid = 1.0 / (1.0 + T.exp(-g))
                        result[i] = T.cast(g * sigmoid * u, dtype)
                T.copy(result, out[row, col : col + block_n])

    return tilelang.compile(main, target="ascend", out_idx=[1])


@functools.lru_cache(maxsize=64)
def _fused_gate_up_kernel(
    m: int,
    s: int,
    k: int,
    dtype: str,
    block_m: int,
    block_n: int,
    block_k: int,
    stages: int,
    num_blocks: int,
    group_m: int = 1,
):
    """Pair gate/up columns and activate on Vector after rounding the Cube result."""
    if m != 32 or block_m != 32 or group_m != 1:
        raise ValueError("the fused gate/up epilogue requires one 32-token M tile")
    gate_n = block_n // 2
    half_m = block_m // 2
    n_tiles = s // gate_n
    grid = min(num_blocks, n_tiles)

    @T.prim_func
    def main(
        a: T.Tensor((m, k), dtype),
        weights: T.Tensor((2 * s, k), dtype),
        out: T.Tensor((m, s), dtype),
    ):
        with T.MixedKernel(grid) as (bx, sid):
            a_l1 = T.alloc_l1((block_m, block_k), dtype)
            b_l1 = T.alloc_l1((block_n, block_k), dtype)
            acc = T.alloc_l0c((block_m, block_n), "float32")
            # Each Vector core consumes half the rows. The widest epilogue uses
            # 16 KiB FP32 scratch and 4 KiB output, within the compiler's UB limit.
            wide = T.alloc_shared((half_m, block_n), "float32")
            activated = T.alloc_shared((half_m, gate_n), dtype)
            T.annotate_buffer_versions({a_l1: stages, b_l1: stages})
            for task in T.Persistent([n_tiles], grid, bx):
                col = task * gate_n
                for ko in T.Pipelined(k // block_k, num_stages=stages):
                    off = ko * block_k
                    T.copy(a[:, off : off + block_k], a_l1)
                    T.copy(weights[col : col + gate_n, off : off + block_k], b_l1[:gate_n, :])
                    T.copy(
                        weights[s + col : s + col + gate_n, off : off + block_k],
                        b_l1[gate_n:, :],
                    )
                    T.gemm(
                        a_l1,
                        b_l1,
                        acc,
                        transpose_B=True,
                        clear_accum=ko == 0,
                        unit_flag_ctrl=T.Select(ko == k // block_k - 1, 3, 2),
                    )
                # The 950 dual-destination FIX transfer retains FP32. Explicit
                # narrow/widen casts preserve the original projection rounding.
                T.dual_copy(acc, wide, unit_flag_ctrl=3)
                if gate_n >= 64:
                    _simd_activation(wide, activated, half_m, gate_n, dtype)
                else:
                    # Narrow tiles cannot hold a full 64-lane SIMD load/store.
                    with T.SimtVF(threads=min(1024, half_m * gate_n)):
                        for row, j in T.Parallel(half_m, gate_n):
                            g = T.cast(T.cast(wide[row, j], dtype), "float32")
                            u = T.cast(T.cast(wide[row, gate_n + j], dtype), "float32")
                            sigmoid = 1.0 / (1.0 + T.exp(-g))
                            activated[row, j] = T.cast(g * sigmoid * u, dtype)
                T.copy(
                    activated,
                    out[sid * half_m : (sid + 1) * half_m, col : col + gate_n],
                )

    return tilelang.compile(main, target="ascend", out_idx=[2])


@functools.lru_cache(maxsize=64)
def _fused_prefill_gate_up_kernel(
    m: int,
    s: int,
    k: int,
    dtype: str,
    num_blocks: int,
    gate_n: int = 128,
    mad_m: int = 64,
):
    """Keep separate zero-origin L0C matrices for a bounded prefill epilogue.

    A token count that does not fill the last 256-row tile is padded in the
    intermediate, not in the input: the final A load reads only the live rows and
    the epilogue writes whole static tiles into ``[m_tiles * 256, s]``. Vector and
    down consumers only see the first ``m`` rows, so padded lanes never reach the
    result and the compact-FIX extent always equals the Cube tile.
    """
    block_m, block_k = 256, 128
    mad_k = 64
    columns = 2 * gate_n
    m_slices = block_m // mad_m
    half_m = mad_m // 2
    m_tiles = (m + block_m - 1) // block_m
    padded_m = m_tiles * block_m
    tail = m % block_m != 0
    n_tiles = s // gate_n
    group_m = next(g for g in (4, 2, 1) if m_tiles % g == 0)
    grid = min(num_blocks, m_tiles * n_tiles)
    stages = 4

    @T.prim_func
    def main(
        a: T.Tensor((m, k), dtype),
        weights: T.Tensor((2 * s, k), dtype),
        out: T.Tensor((padded_m, s), dtype),
    ):
        with T.MixedKernel(grid) as (bx, sid):
            a_l1 = T.alloc_l1((block_m, block_k), dtype)
            b_l1 = T.alloc_l1((columns, block_k), dtype)
            a_l0 = T.alloc_l0a((mad_m, mad_k), dtype)
            b_l0 = T.alloc_l0b((columns, mad_k), dtype)
            acc = T.alloc_l0c((m_slices, mad_m, columns), "float32")
            wide = T.alloc_shared((half_m, columns), "float32")
            activated = T.alloc_shared((half_m, gate_n), dtype)
            # Four L1 versions use at most 512 KiB, and the B L0 ring at most
            # 64 KiB. The 128x192 C matrix uses 60 KiB of Vector scratch;
            # the default 64x256 matrix uses 40 KiB.
            T.annotate_buffer_versions({a_l1: stages, b_l1: stages, a_l0: 2, b_l0: 2})
            for task in T.Persistent([m_tiles * n_tiles], grid, bx):
                row = (task // (n_tiles * group_m) * group_m + task % group_m) * block_m
                col = task // group_m % n_tiles * gate_n
                rows = T.min(block_m, m - row) if tail else block_m
                for ko in T.Pipelined(k // block_k, num_stages=stages):
                    off = ko * block_k
                    if tail:
                        T.copy(a[row : row + rows, off : off + block_k], a_l1[:rows, :])
                    else:
                        T.copy(a[row : row + block_m, off : off + block_k], a_l1)
                    T.copy(weights[col : col + gate_n, off : off + block_k], b_l1[:gate_n, :])
                    T.copy(
                        weights[s + col : s + col + gate_n, off : off + block_k],
                        b_l1[gate_n:, :],
                    )
                    with T.Stage(0):
                        for mi in T.Unroll(m_slices, explicit=True):
                            for ki in T.serial(
                                block_k // mad_k,
                                annotations={"multi_buffer_eligible": [a_l0, b_l0]},
                            ):
                                T.copy(
                                    a_l1[
                                        mi * mad_m : (mi + 1) * mad_m, ki * mad_k : (ki + 1) * mad_k
                                    ],
                                    a_l0,
                                )
                                # Two K slices fill the entire B L0 ring. Later
                                # M tiles read those versions without reloading B.
                                if mi == 0:
                                    T.copy(b_l1[:, ki * mad_k : (ki + 1) * mad_k], b_l0)
                                T.gemm(
                                    a_l0,
                                    b_l0,
                                    acc[mi, :, :],
                                    transpose_B=True,
                                    clear_accum=(ko == 0) & (ki == 0),
                                    unit_flag_ctrl=T.Select(
                                        (ko == k // block_k - 1) & (ki == block_k // mad_k - 1),
                                        3,
                                        2,
                                    ),
                                )
                for mi in T.serial(m_slices):
                    # The trailing matrix dimensions start at zero: no unsupported
                    # compact FIX slice of the original 256-row matrix is needed.
                    T.dual_copy(acc[mi, :, :], wide, unit_flag_ctrl=3)
                    _simd_activation(wide, activated, half_m, gate_n, dtype)
                    T.copy(
                        activated,
                        out[
                            row + mi * mad_m + sid * half_m : row + mi * mad_m + (sid + 1) * half_m,
                            col : col + gate_n,
                        ],
                    )

    return tilelang.compile(main, target="ascend", out_idx=[2])


@functools.lru_cache(maxsize=64)
def _down_ring_kernel(
    m: int,
    n: int,
    k: int,
    dtype: str,
    num_blocks: int,
    block_m: int = 128,
    block_n: int = 256,
    block_k: int = 128,
    stages: int = 3,
):
    """Overlap tile generations using two physical L0C slots and FIX writeback."""
    m_tiles, n_tiles = m // block_m, n // block_n
    group_m = next(g for g in (4, 2, 1) if m_tiles % g == 0)
    grid = min(num_blocks, m_tiles * n_tiles)

    @T.prim_func
    def main(
        a: T.Tensor((m, k), dtype),
        weights: T.Tensor((n, k), dtype),
        out: T.Tensor((m, n), dtype),
    ):
        with T.Kernel(grid) as bx:
            a_l1 = T.alloc_l1((block_m, block_k), dtype)
            b_l1 = T.alloc_l1((block_n, block_k), dtype)
            # Explicit slots are required: the accumulator's K-loop recurrence
            # prevents the compiler from applying annotate_buffer_versions to it.
            acc = T.alloc_l0c((2, block_m, block_n), "float32")
            T.annotate_manual_multi_buffer(acc)
            T.annotate_buffer_versions({a_l1: stages, b_l1: stages})
            for task in T.Persistent([m_tiles * n_tiles], grid, bx, num_stages=2):
                slot = task // grid % 2
                row = (task // (n_tiles * group_m) * group_m + task % group_m) * block_m
                col = task // group_m % n_tiles * block_n
                for ko in T.Pipelined(k // block_k, num_stages=stages):
                    off = ko * block_k
                    T.copy(a[row : row + block_m, off : off + block_k], a_l1)
                    T.copy(weights[col : col + block_n, off : off + block_k], b_l1)
                    T.gemm(
                        a_l1,
                        b_l1,
                        acc[slot, :, :],
                        transpose_B=True,
                        clear_accum=ko == 0,
                    )
                T.copy(acc[slot, :, :], out[row : row + block_m, col : col + block_n])

    return tilelang.compile(main, target="ascend", out_idx=[2])


class AscendSharedExpertMLPKernel(Kernel, SharedExpertMLPFwdInterface):
    """Two FP32-accumulating Cube GEMMs with a dtype-rounded gated activation."""

    devices: ClassVar[frozenset[str]] = frozenset({"npu"})

    @classmethod
    def refusal(cls, call: SharedExpertMLPCall) -> str | None:
        if call.dtype not in (torch.float16, torch.bfloat16):
            return "requires float16 or bfloat16 inputs"
        if call.num_tokens < 0 or call.hidden_size <= 0 or call.ffn_size <= 0:
            return "requires nonnegative tokens and positive hidden/FFN sizes"
        if call.hidden_size % 16 or call.ffn_size % 16:
            return "requires hidden_size and ffn_size divisible by 16 for aligned Cube DMA"
        return super().refusal(call)

    @classmethod
    def entry_for(cls, call: SharedExpertMLPCall) -> Entry:
        return call, lambda: cls(call)

    def __init__(self, call: SharedExpertMLPCall):
        index = call.device.index if call.device is not None else None
        super().__init__(device_index=index)
        self.call = call
        # Fused prefill writes whole 256-row tiles; the last tile's padding is
        # trimmed before the down projection consumes the activation.
        self._activation_trim = 0
        # Empty tokens return a fresh empty output without compiling a zero-grid program.
        if call.num_tokens == 0:
            return
        dtype = str(call.dtype).split(".")[-1]
        cube_cores = int(get_num_cube_cores(device_id=index or 0) or 32)
        config = {
            "gate_up": _projection_config(
                call.num_tokens, 2 * call.ffn_size, call.hidden_size, cube_cores
            ),
            "down": _projection_config(
                call.num_tokens, call.hidden_size, call.ffn_size, cube_cores
            ),
        }
        # The fused prefill epilogue writes whole 256-row tiles. Use it when the
        # packed fallback would split into narrow 16/32/64-row tiles, or when the
        # padded 256-row tail wastes at most one 16-row DMA. A cheap fallback plan
        # with 128-row tiles and little padding stays ahead for the other counts.
        packed_m = (call.num_tokens + 15) // 16 * 16
        fused_m = (call.num_tokens + 255) // 256 * 256
        prefill_fused = (
            call.num_tokens >= 96
            and call.hidden_size >= 4096
            and call.hidden_size % 128 == 0
            and call.ffn_size >= 1024
            and call.ffn_size % 128 == 0
            and (packed_m % 128 or fused_m - call.num_tokens <= 16)
        )
        down_ring = (
            call.num_tokens >= 4096
            and call.num_tokens % 256 == 0
            and call.hidden_size >= 4096
            and call.hidden_size % 256 == 0
            and call.ffn_size <= 4096
            and call.ffn_size % 128 == 0
        )
        small_down_ring = (
            call.num_tokens == 32
            and call.hidden_size == 5120
            and call.ffn_size == 1536
            and call.dtype == torch.bfloat16
        )
        if prefill_fused:
            gate_n, mad_m = 128, 64
            m_tiles = (call.num_tokens + 255) // 256
            if call.num_tokens == 512 and call.ffn_size % 96 == 0:
                waves_128 = (2 * (call.ffn_size // 128) + cube_cores - 1) // cube_cores
                waves_96 = (2 * (call.ffn_size // 96) + cube_cores - 1) // cube_cores
                # A narrower tile can reduce the idle part of the final Cube
                # wave. Pair it with a larger M contraction to retain throughput.
                if waves_96 * 192 < waves_128 * 256:
                    gate_n, mad_m = 96, 128
            config["gate_up"] = dict(
                block_m=256,
                block_n=2 * gate_n,
                block_k=128,
                stages=4,
                mad_m=mad_m,
                mad_k=64,
                num_blocks=cube_cores,
                group_m=next(g for g in (4, 2, 1) if m_tiles % g == 0),
            )
            # The tail tile writes its padded rows into the intermediate; only the
            # first m tokens are consumed by the down projection.
            self._activation_trim = call.num_tokens if call.num_tokens % 256 else 0
            self._gate_up = _fused_prefill_gate_up_kernel(
                call.num_tokens,
                call.ffn_size,
                call.hidden_size,
                dtype,
                cube_cores,
                gate_n,
                mad_m,
            )
            self._activate = None
            config["activation"] = {"fused": True}
        elif call.num_tokens == 32:
            self._gate_up = _fused_gate_up_kernel(
                call.num_tokens, call.ffn_size, call.hidden_size, dtype, **config["gate_up"]
            )
            self._activate = None
            config["activation"] = {"fused": True}
        else:
            self._gate_up = _nt_gemm_kernel(
                call.num_tokens,
                2 * call.ffn_size,
                call.hidden_size,
                dtype,
                **config["gate_up"],
            )
            self._activate = self._activation_kernel(dtype, cube_cores, config)
        if small_down_ring:
            # Two 16 KiB accumulators let the second short projection tile
            # start before the first FIX writeback finishes. Paired full-MLP
            # measurements retain this plan only for the profiled GLM shape.
            self._down = _down_ring_kernel(
                call.num_tokens,
                call.hidden_size,
                call.ffn_size,
                dtype,
                cube_cores,
                block_m=32,
                block_n=128,
                block_k=256,
                stages=2,
            )
            config["down"].update(tile_stages=2, acc_versions=2)
        elif down_ring:
            config["down"] = dict(
                block_m=128,
                block_n=256,
                block_k=128,
                stages=3,
                num_blocks=cube_cores,
                tile_stages=2,
                acc_versions=2,
                group_m=next(g for g in (4, 2, 1) if (call.num_tokens // 128) % g == 0),
            )
            self._down = _down_ring_kernel(
                call.num_tokens, call.hidden_size, call.ffn_size, dtype, cube_cores
            )
        else:
            self._down = _nt_gemm_kernel(
                call.num_tokens, call.hidden_size, call.ffn_size, dtype, **config["down"]
            )
        # Publish once all stages are configured; Kernel freezes nested config dictionaries.
        self.config = config

    def _activation_kernel(self, dtype: str, cube_cores: int, config: dict):
        call = self.call
        # Wide UB tiles amortize transfers and synchronization in prefill. At most
        # 24 KiB of storage is live across the two versions of all three buffers.
        widths = (
            (2048, 1024, 512, 256, 128, 64, 32, 16)
            if call.num_tokens >= 512
            else (256, 128, 64, 32, 16)
        )
        activation_block = next(b for b in widths if call.ffn_size % b == 0)
        config["activation"] = dict(
            block_n=activation_block,
            stages=2 if call.num_tokens >= 512 else 1,
            num_blocks=2 * cube_cores,
        )
        return _silu_mul_kernel(call.num_tokens, call.ffn_size, dtype, **config["activation"])

    def forward(
        self, hidden: torch.Tensor, w_gate_up: torch.Tensor, w_down: torch.Tensor
    ) -> torch.Tensor:
        if self.call.num_tokens == 0:
            return torch.empty_like(hidden)
        # Each launch uses the caller's current stream; temporaries remain live through
        # their consumer launches and no cross-block barrier or host synchronization is used.
        gate_up = self._gate_up(hidden, w_gate_up)
        activated = gate_up if self._activate is None else self._activate(gate_up)
        if self._activation_trim:
            activated = activated[: self._activation_trim]
        return self._down(activated, w_down)


register_kernel_type("SharedExpertMLPFwdOp", "ascend", AscendSharedExpertMLPKernel)
