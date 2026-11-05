"""Ascend scans and recurrent updates for State-Space Dual (SSD) models."""

import functools
import math

import tilelang
import tilelang.ascend.language as T

from tileops.backend import register_kernel_type
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.mamba.call_spec import (
    SSDChunkCumsumFwdInterface,
    SSDDecodeFwdInterface,
    SSDStatePassingFwdInterface,
)

__all__ = ["AscendSSDChunkCumsumKernel", "AscendSSDRecurrentKernel", "AscendSSDStatePassingKernel"]


@functools.lru_cache(maxsize=64)
def _cumsum(batch, length, heads, chunk, dtype, bias, softplus, lower, upper):
    chunks = length // chunk
    # Sixteen adjacent heads make the source gather contiguous while bounding UB storage.
    tile_h = min(16, heads)
    staged_inputs = heads % tile_h == 0 and heads % 8 == 0
    groups = (heads + tile_h - 1) // tile_h
    blocks = min(64, batch * chunks * groups)
    stages = (chunk - 1).bit_length()

    @T.prim_func
    def main(
        X: T.Tensor((batch, length, heads), "float32"),
        A: T.Tensor((heads,), "float32"),
        Bias: T.Tensor((heads,), "float32"),
        D: T.Tensor((batch, heads, chunks, chunk), dtype),
        Y: T.Tensor((batch, heads, chunks, chunk), "float32"),
    ):
        with T.Kernel(blocks) as bx:
            values = T.alloc_shared((tile_h, chunk), "float32")
            scratch = T.alloc_shared((tile_h, chunk), "float32")
            dt = T.alloc_shared((tile_h, chunk), dtype)
            raw = T.alloc_shared((chunk, tile_h), "float32")
            scales = T.alloc_shared((tile_h,), "float32")
            biases = T.alloc_shared((tile_h,), "float32")
            for step in T.serial(T.ceildiv(batch * chunks * groups, blocks)):
                work = step * blocks + bx
                if work < batch * chunks * groups:
                    b = work // (chunks * groups)
                    c = work // groups % chunks
                    h0 = work % groups * tile_h
                    # Stage regular head tiles once; avoid three scalar GM reads
                    # per element. Irregular head tails retain the guarded gather.
                    if staged_inputs:
                        T.copy(X[b, c * chunk : (c + 1) * chunk, h0 : h0 + tile_h], raw)
                        T.copy(A[h0 : h0 + tile_h], scales)
                        if bias:
                            T.copy(Bias[h0 : h0 + tile_h], biases)
                    with T.SimtVF(threads=256):
                        for t, h in T.Parallel(chunk, tile_h):
                            if h0 + h < heads:
                                v = raw[t, h] if staged_inputs else X[b, c * chunk + t, h0 + h]
                                if bias:
                                    v = v + (biases[h] if staged_inputs else Bias[h0 + h])
                                if softplus:
                                    v = T.if_then_else(v > 20, v, T.log(1 + T.exp(T.min(v, 20))))
                                v = T.max(v, lower)
                                if math.isfinite(upper):
                                    v = T.min(v, upper)
                                dt[h, t] = T.cast(v, dtype)
                                values[h, t] = v * (scales[h] if staged_inputs else A[h0 + h])
                    if chunk % 64 == 0:
                        with T.SimdVF():
                            mask = T.simd.pset(32)
                            index = T.simd.vci(T.int32(0), "int32")
                            zero_i = T.simd.vdup(T.int32(0), "int32")
                            zero = T.simd.vdup(T.float32(0), "float32")
                            for h in T.serial(tile_h):
                                carry = T.simd.alloc_local(1, "float32")
                                carry[0] = zero
                                for segment in T.serial(chunk // 64):
                                    prefix = T.simd.alloc_local(1, "float32")
                                    prefix[0] = T.simd.vld(values[h, segment * 64])
                                    for shift in T.unroll(6):
                                        active = T.simd.vcmps(
                                            index, T.int32(1 << shift), mask, op="ge"
                                        )
                                        earlier = T.simd.vmax(
                                            T.simd.vsub(
                                                index,
                                                T.simd.vdup(T.int32(1 << shift), "int32"),
                                                mask,
                                            ),
                                            zero_i,
                                            mask,
                                        )
                                        contribution = T.simd.vsel(
                                            T.simd.vselr(prefix[0], earlier), zero, active
                                        )
                                        prefix[0] = T.simd.vadd(prefix[0], contribution, mask)
                                    prefix[0] = T.simd.vadd(prefix[0], carry[0], mask)
                                    T.simd.vsts(values[h, segment * 64], prefix[0], mask)
                                    carry[0] = T.simd.vselr(
                                        prefix[0], T.simd.vdup(T.int32(63), "int32")
                                    )
                    else:
                        for level in T.unroll(stages):
                            with T.SimtVF(threads=256):
                                for h, t in T.Parallel(tile_h, chunk):
                                    if level % 2 == 0:
                                        scratch[h, t] = values[h, t] + T.if_then_else(
                                            t >= (1 << level),
                                            values[h, T.max(0, t - (1 << level))],
                                            0,
                                        )
                                    else:
                                        values[h, t] = scratch[h, t] + T.if_then_else(
                                            t >= (1 << level),
                                            scratch[h, T.max(0, t - (1 << level))],
                                            0,
                                        )
                    valid_heads = T.min(tile_h, heads - h0)
                    T.copy(dt[:valid_heads, :], D[b, h0 : h0 + valid_heads, c, :])
                    if chunk % 64 == 0 or stages % 2 == 0:
                        T.copy(values[:valid_heads, :], Y[b, h0 : h0 + valid_heads, c, :])
                    else:
                        T.copy(scratch[:valid_heads, :], Y[b, h0 : h0 + valid_heads, c, :])

    return tilelang.compile(main, target="ascend", out_idx=[3, 4])


@functools.lru_cache(maxsize=64)
def _passing(batch, chunks, heads, width, dtype, initial):
    # One lane owns an independent recurrent state coordinate across all chunks.
    tile = 256
    groups = (heads * width + tile - 1) // tile
    blocks = min(64, batch * groups)

    @T.prim_func
    def main(
        X: T.Tensor((batch, chunks, heads * width), dtype),
        A: T.Tensor((batch, heads, chunks), "float32"),
        Init: T.Tensor((batch, heads * width), "float32"),
        Y: T.Tensor((batch, chunks, heads * width), "float32"),
        Final: T.Tensor((batch, heads * width), "float32"),
    ):
        with T.Kernel(blocks) as bx, T.SimtVF(threads=256):
            for step in T.serial(T.ceildiv(batch * groups, blocks)):
                work = step * blocks + bx
                if work < batch * groups:
                    b = work // groups
                    for j in T.Parallel(tile):
                        col = work % groups * tile + j
                        if col < heads * width:
                            state = T.alloc_local((1,), "float32")
                            state[0] = 0
                            if initial:
                                state[0] = Init[b, col]
                            for c in T.serial(chunks):
                                Y[b, c, col] = state[0]
                                state[0] = T.exp(A[b, col // width, c]) * state[0] + T.cast(
                                    X[b, c, col], "float32"
                                )
                            Final[b, col] = state[0]

    return tilelang.compile(main, target="ascend", out_idx=[3, 4])


@functools.lru_cache(maxsize=64)
def _recurrent(batch, heads, width, states, groups, dtype):
    # A warp reduces one output coordinate; each lane owns disjoint mutable state elements.
    rows = batch * heads * width
    blocks = min(64, (rows + 7) // 8)

    @T.prim_func
    def main(
        A: T.Tensor((heads, width, states), "float32"),
        Dt: T.Tensor((batch, heads, width), "float32"),
        X: T.Tensor((batch, heads, width), dtype),
        B: T.Tensor((batch, groups, states), dtype),
        C: T.Tensor((batch, groups, states), dtype),
        State: T.Tensor((batch, heads, width, states), "float32"),
        Y: T.Tensor((batch, heads, width), "float32"),
    ):
        with T.Kernel(blocks) as bx, T.SimtVF(threads=256):
            tx = T.get_thread_binding()
            lane = tx % 32
            for step in T.serial(T.ceildiv(rows, blocks * 8)):
                row = (step * blocks + bx) * 8 + tx // 32
                if row < rows:
                    b = row // (heads * width)
                    h = row // width % heads
                    p = row % width
                    g = h // (heads // groups)
                    accum = T.alloc_local((1,), "float32")
                    accum[0] = 0
                    for k in T.serial(T.ceildiv(states, 32)):
                        n = k * 32 + lane
                        if n < states:
                            dt = Dt[b, h, p]
                            v = T.exp(dt * A[h, p, n]) * State[b, h, p, n] + dt * T.cast(
                                X[b, h, p], "float32"
                            ) * T.cast(B[b, g, n], "float32")
                            State[b, h, p, n] = v
                            accum[0] = accum[0] + v * T.cast(C[b, g, n], "float32")
                    for shift in T.unroll(5):
                        accum[0] = (
                            accum[0]
                            + T.call_extern(  # lint: extern-target allow: CANN shuffle has no wrapper in the pinned Ascend API.
                                "float32", "asc_shfl_xor", accum[0], T.int32(1) << shift
                            )
                        )
                    if lane == 0:
                        Y[b, h, p] = accum[0]

    return tilelang.compile(main, target="ascend", out_idx=[6])


@functools.lru_cache(maxsize=64)
def _recurrent_simd(batch, heads, width, states, groups, dtype):
    # Sixteen output coordinates share DMA setup and each group's B/C vectors.
    tile_p = 16
    tiles = width // tile_p
    blocks = min(64, batch * heads * tiles)

    @T.prim_func
    def main(
        A: T.Tensor((heads, width, states), "float32"),
        Dt: T.Tensor((batch, heads, width), "float32"),
        X: T.Tensor((batch, heads, width), dtype),
        B: T.Tensor((batch, groups, states), dtype),
        C: T.Tensor((batch, groups, states), dtype),
        State: T.Tensor((batch, heads, width, states), "float32"),
        Y: T.Tensor((batch, heads, width), "float32"),
    ):
        with T.Kernel(blocks) as bx:
            a = T.alloc_shared((tile_p, states), "float32")
            state = T.alloc_shared((tile_p, states), "float32")
            dt = T.alloc_shared((tile_p,), "float32")
            x = T.alloc_shared((tile_p,), dtype)
            bv = T.alloc_shared((states,), dtype)
            cv = T.alloc_shared((states,), dtype)
            out = T.alloc_shared((tile_p,), "float32")
            for step in T.serial(T.ceildiv(batch * heads * tiles, blocks)):
                work = step * blocks + bx
                if work < batch * heads * tiles:
                    b = work // (heads * tiles)
                    h = work // tiles % heads
                    p0 = work % tiles * tile_p
                    g = h // (heads // groups)
                    T.copy(A[h, p0 : p0 + tile_p, :], a)
                    T.copy(State[b, h, p0 : p0 + tile_p, :], state)
                    T.copy(Dt[b, h, p0 : p0 + tile_p], dt)
                    T.copy(X[b, h, p0 : p0 + tile_p], x)
                    T.copy(B[b, g, :], bv)
                    T.copy(C[b, g, :], cv)
                    with T.SimdVF():
                        full = T.simd.pset(32)
                        for p in T.serial(tile_p):
                            delta = T.simd.vld(dt[p], dist="BRC_B32")
                            if dtype == "float32":
                                value = T.simd.vld(x[p], dist="BRC_B32")
                            else:
                                value = T.simd.vcvt(
                                    T.simd.vld(x[p], dist="BRC_B16"), "float32", part=0
                                )
                            drive = T.simd.vmul(delta, value, full)
                            accum = T.simd.alloc_local(1, "float32")
                            accum[0] = T.simd.vdup(T.float32(0), "float32")
                            for segment in T.serial(states // 64):
                                n = segment * 64
                                if dtype == "float32":
                                    input_b = T.simd.vld(bv[n])
                                    input_c = T.simd.vld(cv[n])
                                else:
                                    input_b = T.simd.vcvt(
                                        T.simd.vld(bv[n], dist="UNPK_B16"), "float32", part=0
                                    )
                                    input_c = T.simd.vcvt(
                                        T.simd.vld(cv[n], dist="UNPK_B16"), "float32", part=0
                                    )
                                decay = T.simd.vexp(T.simd.vmul(delta, T.simd.vld(a[p, n]), full))
                                updated = T.simd.vadd(
                                    T.simd.vmul(decay, T.simd.vld(state[p, n]), full),
                                    T.simd.vmul(drive, input_b, full),
                                    full,
                                )
                                T.simd.vsts(state[p, n], updated, full)
                                accum[0] = T.simd.vadd(
                                    accum[0], T.simd.vmul(updated, input_c, full), full
                                )
                            T.simd.vsts(
                                out[p], T.simd.vcadd(accum[0], full), full, dist="ONEPT_B32"
                            )
                    T.copy(state, State[b, h, p0 : p0 + tile_p, :])
                    T.copy(out, Y[b, h, p0 : p0 + tile_p])

    return tilelang.compile(main, target="ascend", out_idx=[6])


class AscendSSDChunkCumsumKernel(Kernel, SSDChunkCumsumFwdInterface):
    """Transform step sizes and compute chunk-local decay prefix sums."""

    devices = frozenset({"npu"})

    def __init__(self, call):
        super().__init__(device_index=call.device.index)
        self._run = _cumsum(
            call.batch,
            call.seq_len,
            call.n_heads,
            call.chunk_len,
            self.dtype_to_str(call.out_dtype),
            call.has_dt_bias,
            call.dt_softplus,
            call.dt_min,
            call.dt_max,
        )

    def forward(self, dt, A, dt_bias=None):
        return self._run(dt, A, A if dt_bias is None else dt_bias.contiguous())


class AscendSSDStatePassingKernel(Kernel, SSDStatePassingFwdInterface):
    """Carry chunk-end states through a serial scan of independent coordinates."""

    devices = frozenset({"npu"})

    def __init__(self, call):
        super().__init__(device_index=call.device.index)
        self._shape = (call.batch, call.num_chunks, call.n_heads, call.d_state)
        self._run = _passing(
            call.batch,
            call.num_chunks,
            call.n_heads,
            call.d_state,
            self.dtype_to_str(call.dtype),
            call.has_initial_states,
        )

    def forward(self, states, dA_chunk_cumsum, initial_states):
        batch, chunks, heads, width = self._shape
        prev, final = self._run(
            states.view(batch, chunks, -1), dA_chunk_cumsum, initial_states.view(batch, -1)
        )
        return prev.view(self._shape), final.view(batch, heads, width)


class AscendSSDRecurrentKernel(Kernel, SSDDecodeFwdInterface):
    """Fuse recurrent state updates with the output projection."""

    devices = frozenset({"npu"})

    def __init__(self, call):
        super().__init__(device_index=call.device.index)
        # Bound the two FP32 tiles in UB; retain the streaming path for larger states.
        recurrent = (
            _recurrent_simd
            if call.d_state <= 512 and call.d_state % 64 == 0 and call.d_head % 16 == 0
            else _recurrent
        )
        self._run = recurrent(
            call.batch,
            call.n_heads,
            call.d_head,
            call.d_state,
            call.n_groups,
            self.dtype_to_str(call.dtype),
        )

    def forward(self, A, dt, x, B_in, C_in, state):
        return self._run(A, dt, x, B_in, C_in, state)


register_kernel_type("SSDChunkCumsumFwdOp", "ascend", AscendSSDChunkCumsumKernel)
register_kernel_type("SSDStatePassingFwdOp", "ascend", AscendSSDStatePassingKernel)
register_kernel_type("SSDRecurrentFwdOp", "ascend", AscendSSDRecurrentKernel)
