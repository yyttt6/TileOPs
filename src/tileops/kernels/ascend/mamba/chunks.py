"""Cube contractions and vector preprocessing for State-Space Dual (SSD) chunks."""

import functools

import tilelang
import tilelang.ascend.language as T
import torch

from tileops.backend import register_kernel_type
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.mamba.call_spec import (
    SSDChunkCouplingFwdInterface,
    SSDChunkScanFwdInterface,
    SSDChunkStateFwdInterface,
)

__all__ = ["AscendSSDChunkCouplingKernel", "AscendSSDChunkScanKernel", "AscendSSDChunkStateKernel"]


@functools.lru_cache(maxsize=128)
def _contract(
    batch, heads, groups, m, n, k, dtype, out_dtype, grouped_a=False, grouped_b=False, causal=False
):
    # A 64x64 output tile fits the accumulator while retaining enough independent Cube work.
    bm, bn, bk = 64, 64, 128
    mt, nt = (m + bm - 1) // bm, (n + bn - 1) // bn
    tiles = mt * (mt + 1) // 2 if causal else mt * nt
    blocks = min(32, batch * tiles)
    abatch = batch // heads * groups if grouped_a else batch
    bbatch = batch // heads * groups if grouped_b else batch

    def triangle_row(index):
        # Build an integer lookup expression: Cube scalar code has no sqrt intrinsic.
        row = T.int32(0)
        for candidate in range(1, mt):
            row = T.if_then_else(index >= candidate * (candidate + 1) // 2, candidate, row)
        return row

    @T.prim_func
    def main(
        A: T.Tensor((abatch, m, k), dtype),
        B: T.Tensor((bbatch, n, k), dtype),
        Y: T.Tensor((batch, m, n), out_dtype),
    ):
        with T.Kernel(blocks) as bx:
            a = T.alloc_l1((bm, bk), dtype)
            b = T.alloc_l1((bn, bk), dtype)
            a0 = T.alloc_l0a((bm, bk), dtype)
            b0 = T.alloc_l0b((bn, bk), dtype)
            acc = T.alloc_l0c((bm, bn), "float32")
            T.annotate_buffer_versions({a: 2, b: 2, a0: 2, b0: 2})
            for step in T.serial(T.ceildiv(batch * tiles, blocks)):
                work = step * blocks + bx
                if work < batch * tiles:
                    item = work // tiles
                    ai = (
                        item // heads * groups + item % heads // (heads // groups)
                        if grouped_a
                        else item
                    )
                    bi = (
                        item // heads * groups + item % heads // (heads // groups)
                        if grouped_b
                        else item
                    )
                    if causal:
                        # Enumerate lower-triangle tiles densely so every Cube core works.
                        tile_id = work % tiles
                        tile_row = triangle_row(tile_id)
                        row = tile_row * bm
                        col = (tile_id - tile_row * (tile_row + 1) // 2) * bn
                    else:
                        row, col = work // nt % mt * bm, work % nt * bn
                    mr, nr = T.min(bm, m - row), T.min(bn, n - col)
                    for ki in T.Pipelined(T.ceildiv(k, bk), num_stages=2):
                        kr = T.min(bk, k - ki * bk)
                        T.copy(A[ai, row : row + mr, ki * bk : ki * bk + kr], a[:mr, :kr])
                        T.copy(B[bi, col : col + nr, ki * bk : ki * bk + kr], b[:nr, :kr])
                        T.copy(a[:mr, :kr], a0[:mr, :kr])
                        T.copy(b[:nr, :kr], b0[:nr, :kr])
                        T.gemm(
                            a0[:mr, :kr],
                            b0[:nr, :kr],
                            acc[:mr, :nr],
                            transpose_B=True,
                            clear_accum=ki == 0,
                        )
                    T.copy(acc[:mr, :nr], Y[item, row : row + mr, col : col + nr])

    return tilelang.compile(main, target="ascend", out_idx=[2])


@functools.lru_cache(maxsize=64)
def _causal_mask(batch, chunk, dtype):
    total = batch * chunk * chunk
    # DMA contiguous tiles and mask/cast in UB; scalar GM reads dominated this epilogue.
    tile = min(8192, total)
    blocks = min(64, (total + tile - 1) // tile)

    @T.prim_func
    def main(X: T.Tensor((total,), "float32"), Y: T.Tensor((total,), dtype)):
        with T.Kernel(blocks) as bx:
            raw = T.alloc_shared((tile,), "float32")
            out = T.alloc_shared((tile,), dtype)
            for step in T.serial(T.ceildiv(total, blocks * tile)):
                start = (step * blocks + bx) * tile
                if start < total:
                    count = T.min(tile, total - start)
                    T.copy(X[start : start + count], raw[:count])
                    if chunk % 64 == 0:
                        with T.SimdVF():
                            full = T.simd.pset(32)
                            zero = T.simd.vdup(T.float32(0), "float32")
                            for segment in T.serial(tile // 64):
                                first = start + segment * 64
                                columns = T.simd.vci(T.cast(first % chunk, "int32"), "int32")
                                lower = T.simd.vcmps(
                                    columns, T.cast(first // chunk % chunk, "int32"), full, op="le"
                                )
                                value = T.simd.vsel(T.simd.vld(raw[segment * 64]), zero, lower)
                                if dtype == "float32":
                                    T.simd.vsts(out[segment * 64], value, full)
                                else:
                                    narrowed = T.simd.vcvt(value, dtype, full, sat=False)
                                    T.simd.vsts(out[segment * 64], narrowed, full, dist="PK_B32")
                    else:
                        with T.SimtVF(threads=256):
                            for j in T.Parallel(tile, coalesced_width=T.int32(1)):
                                i = start + j
                                if i < total:
                                    out[j] = T.if_then_else(
                                        i % chunk <= i // chunk % chunk,
                                        T.cast(raw[j], dtype),
                                        T.cast(0, dtype),
                                    )
                    T.copy(out[:count], Y[start : start + count])

    return tilelang.compile(main, target="ascend", out_idx=[1])


@functools.lru_cache(maxsize=64)
def _weighted_state(batch, chunks, chunk, heads, width, dtype, dt_dtype, segmented):
    total = batch * chunks * heads * width * chunk
    tile = 2048
    blocks = min(64, (total + tile - 1) // tile)

    @T.prim_func
    def main(
        X: T.Tensor((batch, chunks * chunk, heads, width), dtype),
        Dt: T.Tensor((batch, heads, chunks, chunk), dt_dtype),
        A: T.Tensor((batch, heads, chunks, chunk), "float32"),
        Seq: T.Tensor((batch, chunks * chunk), "int32"),
        Y: T.Tensor((total,), dtype),
    ):
        with T.Kernel(blocks) as bx:
            out = T.alloc_shared((tile,), dtype)
            for step in T.serial(T.ceildiv(total, blocks * tile)):
                start = (step * blocks + bx) * tile
                if start < total:
                    with T.SimtVF(threads=256):
                        for j in T.Parallel(tile, coalesced_width=T.int32(1)):
                            i = start + j
                            if i < total:
                                t, p = i % chunk, i // chunk % width
                                h, c = (
                                    i // (chunk * width) % heads,
                                    i // (chunk * width * heads) % chunks,
                                )
                                b = i // (chunk * width * heads * chunks)
                                weight = T.exp(
                                    T.min(A[b, h, c, chunk - 1] - A[b, h, c, t], 0)
                                ) * T.cast(Dt[b, h, c, t], "float32")
                                if segmented:
                                    last = Seq[b, c * chunk + chunk - 1]
                                    weight = T.if_then_else(
                                        last >= 0 and Seq[b, c * chunk + t] == last, weight, 0
                                    )
                                out[j] = T.cast(
                                    weight * T.cast(X[b, c * chunk + t, h, p], "float32"), dtype
                                )
                    count = T.min(tile, total - start)
                    T.copy(out[:count], Y[start : start + count])

    return tilelang.compile(main, target="ascend", out_idx=[4])


@functools.lru_cache(maxsize=64)
def _weighted_coupling(batch, chunks, chunk, heads, groups, dtype):
    total = batch * chunks * heads * chunk * chunk
    tile = 2048
    blocks = min(64, (total + tile - 1) // tile)

    @T.prim_func
    def main(
        CB: T.Tensor((batch, chunks, groups, chunk, chunk), dtype),
        A: T.Tensor((batch, heads, chunks, chunk), "float32"),
        Dt: T.Tensor((batch, heads, chunks, chunk), dtype),
        Y: T.Tensor((total,), dtype),
    ):
        with T.Kernel(blocks) as bx:
            out = T.alloc_shared((tile,), dtype)
            for step in T.serial(T.ceildiv(total, blocks * tile)):
                start = (step * blocks + bx) * tile
                if start < total:
                    with T.SimtVF(threads=256):
                        for j in T.Parallel(tile, coalesced_width=T.int32(1)):
                            i = start + j
                            if i < total:
                                s, t = i % chunk, i // chunk % chunk
                                h, c = (
                                    i // (chunk * chunk) % heads,
                                    i // (chunk * chunk * heads) % chunks,
                                )
                                b = i // (chunk * chunk * heads * chunks)
                                value = T.alloc_local((1,), "float32")
                                value[0] = T.float32(0)
                                if s <= t:
                                    value[0] = (
                                        T.cast(CB[b, c, h // (heads // groups), t, s], "float32")
                                        * T.exp(A[b, h, c, t] - A[b, h, c, s])
                                        * T.cast(Dt[b, h, c, s], "float32")
                                    )
                                out[j] = T.cast(value[0], dtype)
                    count = T.min(tile, total - start)
                    T.copy(out[:count], Y[start : start + count])

    return tilelang.compile(main, target="ascend", out_idx=[3])


@functools.lru_cache(maxsize=64)
def _combine(batch, chunks, chunk, heads, width):
    total = batch * chunks * chunk * heads * width
    tile = 2048
    blocks = min(64, (total + tile - 1) // tile)

    @T.prim_func
    def main(
        History: T.Tensor((batch, chunks, heads, chunk, width), "float32"),
        Intra: T.Tensor((batch, chunks, heads, chunk, width), "float32"),
        A: T.Tensor((batch, heads, chunks, chunk), "float32"),
        Y: T.Tensor((total,), "float32"),
    ):
        with T.Kernel(blocks) as bx:
            out = T.alloc_shared((tile,), "float32")
            for step in T.serial(T.ceildiv(total, blocks * tile)):
                start = (step * blocks + bx) * tile
                if start < total:
                    with T.SimtVF(threads=256):
                        for j in T.Parallel(tile):
                            i = start + j
                            if i < total:
                                p, h, t = (
                                    i % width,
                                    i // width % heads,
                                    i // (width * heads) % chunk,
                                )
                                c, b = (
                                    i // (width * heads * chunk) % chunks,
                                    i // (width * heads * chunk * chunks),
                                )
                                out[j] = (
                                    Intra[b, c, h, t, p]
                                    + T.exp(A[b, h, c, t]) * History[b, c, h, t, p]
                                )
                    count = T.min(tile, total - start)
                    T.copy(out[:count], Y[start : start + count])

    return tilelang.compile(main, target="ascend", out_idx=[3])


class AscendSSDChunkCouplingKernel(Kernel, SSDChunkCouplingFwdInterface):
    """Contract grouped C and B tensors and mask their strictly upper triangle."""

    devices = frozenset({"npu"})

    def __init__(self, call):
        super().__init__(device_index=call.device.index)
        self._shape = (
            call.batch,
            call.seq_len // call.chunk_len,
            call.chunk_len,
            call.n_groups,
            call.d_state,
        )
        b, c, q, g, n = self._shape
        dtype = self.dtype_to_str(call.dtype)
        # BF16 output rounding is sensitive to half-way sums; use the FP32 contraction
        # path of the independent reference rather than accumulating BF16 Cube products.
        self._promote = call.dtype == torch.bfloat16
        self._dot = _contract(
            b * c * g, g, g, q, q, n, "float32" if self._promote else dtype, "float32", causal=True
        )
        self._mask = _causal_mask(b * c * g, q, dtype)

    def forward(self, C_mat, B_mat):
        b, c, q, g, n = self._shape
        cp = C_mat.view(self._shape).permute(0, 1, 3, 2, 4).contiguous().view(b * c * g, q, n)
        bp = B_mat.view(self._shape).permute(0, 1, 3, 2, 4).contiguous().view(b * c * g, q, n)
        if self._promote:
            cp, bp = cp.float(), bp.float()
        raw = self._dot(cp, bp)
        return self._mask(raw.view(-1)).view(b, c, g, q, q)


class AscendSSDChunkStateKernel(Kernel, SSDChunkStateFwdInterface):
    """Form scaled input tiles and contract the chunk-end states on Cube cores."""

    devices = frozenset({"npu"})

    def __init__(self, call):
        super().__init__(device_index=call.device.index)
        self._shape = (
            call.batch,
            call.num_chunks,
            call.chunk_len,
            call.n_heads,
            call.d_head,
            call.n_groups,
            call.d_state,
        )
        b, c, q, h, p, g, n = self._shape
        dtype = self.dtype_to_str(call.dtype)
        self._weight = _weighted_state(
            b, c, q, h, p, dtype, self.dtype_to_str(call.dt_dtype), call.has_seq_idx
        )
        self._dot = _contract(b * c * h, h, g, p, n, q, dtype, "float32", grouped_b=True)

    def forward(self, x, Bmat, dt, dA_cumsum, seq_idx):
        b, c, q, h, p, g, n = self._shape
        weighted = self._weight(x, dt, dA_cumsum, seq_idx).view(b * c * h, p, q)
        bp = Bmat.view(b, c, q, g, n).permute(0, 1, 3, 4, 2).contiguous().view(b * c * g, n, q)
        return self._dot(weighted, bp).view(b, c, h, p, n)


class AscendSSDChunkScanKernel(Kernel, SSDChunkScanFwdInterface):
    """Combine history and causal intra-chunk contractions."""

    devices = frozenset({"npu"})

    def __init__(self, call):
        super().__init__(device_index=call.device.index)
        self._shape = (
            call.batch,
            call.num_chunks,
            call.chunk_len,
            call.n_heads,
            call.d_head,
            call.n_groups,
            call.d_state,
        )
        b, c, q, h, p, g, n = self._shape
        self._dtype = call.dtype
        dtype = self.dtype_to_str(call.dtype)
        self._weight = _weighted_coupling(b, c, q, h, g, dtype)
        self._intra = _contract(b * c * h, h, g, q, p, q, dtype, "float32")
        self._history = _contract(b * c * h, h, g, q, p, n, dtype, "float32", grouped_a=True)
        self._combine = _combine(b, c, q, h, p)

    def forward(self, x, cb, dA_cumsum, C, prev_states, dt):
        b, c, q, h, p, g, n = self._shape
        weighted = self._weight(cb, dA_cumsum, dt).view(b * c * h, q, q)
        xp = x.view(b, c, q, h, p).permute(0, 1, 3, 4, 2).contiguous().view(b * c * h, p, q)
        intra = self._intra(weighted, xp).view(b, c, h, q, p)
        cp = C.view(b, c, q, g, n).permute(0, 1, 3, 2, 4).contiguous().view(b * c * g, q, n)
        history = self._history(cp, prev_states.to(self._dtype).view(b * c * h, p, n)).view(
            b, c, h, q, p
        )
        return self._combine(history, intra, dA_cumsum).view(b, c * q, h, p)


register_kernel_type("SSDChunkCouplingFwdOp", "ascend", AscendSSDChunkCouplingKernel)
register_kernel_type("SSDChunkStateFwdOp", "ascend", AscendSSDChunkStateKernel)
register_kernel_type("SSDChunkScanFwdOp", "ascend", AscendSSDChunkScanKernel)
