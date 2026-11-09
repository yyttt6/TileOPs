"""FP32 weighted SSD chunk contractions with bounded preprocessing tiles."""

import functools

import tilelang
import tilelang.ascend.language as T
import torch
import torch.nn.functional as F

from tileops._kernels.kernel_base import Kernel
from tileops._kernels.mamba.call_spec import SSDChunkStateFwdInterface
from tileops.backend._registry import register_kernel_type


@T.macro
def _weights(delta, steps, same, weights, has_seq: bool):
    with T.SimdVF():
        mask = T.simd.pset(32)
        for part in T.serial(2):
            raw = T.simd.vld(delta[part * 64])
            capped = T.simd.vmins(raw, T.float32(0), mask)
            capped = T.simd.vsel(raw, capped, T.simd.vcmp(raw, raw, mask, op="ne"))
            decay = T.simd.vexp(capped, mask, precision="ftz_false")
            value = T.simd.vmul(decay, T.simd.vld(steps[part * 64]), mask)
            if has_seq:
                value = T.simd.vmul(value, T.simd.vld(same[part * 64]), mask)
            T.simd.vsts(weights[part * 64], value, mask, extent=64)
        T.simd.mem_bar("VST_VLD")


@functools.lru_cache(maxsize=128)
def build(b, c, q, h, p, n, g, dtype, dt_dtype, has_seq):
    groups = b * c * h
    qp = (q + 127) // 128 * 128
    pp = (p + 63) // 64 * 64
    np = (n + 63) // 64 * 64
    nt = max((n + 15) // 16, pp // 16)
    qt = qp // 128
    tasks = groups * nt * qt
    blocks = min(64, tasks)
    dma = n % 16 == 0
    xdma = p % 16 == 0

    @T.prim_func
    def weighted(
        X: T.Tensor((b, c * q, h, p), dtype),
        B: T.Tensor((b, c * q, g, n), dtype),
        DT: T.Tensor((b, h, c, q), dt_dtype),
        DA: T.Tensor((b, h, c, q), "float32"),
        Seq: T.Tensor((b, c * q), "int32"),
        W: T.Tensor((groups, np, qp), "float32"),
        Packed: T.Tensor((groups, pp, qp), "float32"),
    ):
        with T.Kernel(blocks) as bx:
            data = T.alloc_shared((128, 16), dtype)
            xdata = T.alloc_shared((128, 16), dtype)
            out = T.alloc_shared((16, 128), "float32")
            weights = T.alloc_shared((128,), "float32")
            deltas = T.alloc_shared((128,), "float32")
            steps = T.alloc_shared((128,), "float32")
            masks = T.alloc_shared((128,), "float32")
            for step in T.serial(T.ceildiv(tasks, blocks)):
                task = step * blocks + bx
                if task < tasks:
                    group = task // (nt * qt)
                    bi = group // (c * h)
                    ci = group // h % c
                    hi = group % h
                    gi = hi // (h // g)
                    ni = task // qt % nt * 16
                    qi = task % qt * 128
                    count = T.min(128, q - qi)
                    if dma:  # noqa: SIM102
                        if ni < n:
                            T.copy(
                                B[bi, ci * q + qi : ci * q + qi + count, gi, ni : ni + 16],
                                data[:count, :],
                            )
                    if xdma:  # noqa: SIM102
                        if ni < p:
                            T.copy(
                                X[bi, ci * q + qi : ci * q + qi + count, hi, ni : ni + 16],
                                xdata[:count, :],
                            )
                    with T.SimtVF(threads=256):
                        for j in T.Parallel(128):
                            delta = T.alloc_var("float32")
                            dt_value = T.alloc_var("float32")
                            same = T.alloc_var("float32")
                            delta = T.float32(0)
                            dt_value = T.float32(0)
                            same = T.float32(1)
                            if j < count:
                                delta = DA[bi, hi, ci, q - 1] - DA[bi, hi, ci, qi + j]
                                dt_value = T.cast(DT[bi, hi, ci, qi + j], "float32")
                                if has_seq:
                                    end = Seq[bi, ci * q + q - 1]
                                    same = T.cast(
                                        (end >= 0) and (Seq[bi, ci * q + qi + j] == end), "float32"
                                    )
                            deltas[j] = delta
                            steps[j] = dt_value
                            masks[j] = same
                    _weights(deltas, steps, masks, weights, has_seq)
                    with T.SimtVF(threads=256):
                        for i, j in T.Parallel(16, 128):
                            value = T.alloc_var("float32")
                            value = T.float32(0)
                            if ni + i < n and j < count:
                                if dma:
                                    value = weights[j] * T.cast(data[j, i], "float32")
                                else:
                                    value = weights[j] * T.cast(
                                        B[bi, ci * q + qi + j, gi, ni + i], "float32"
                                    )
                            out[i, j] = value
                    if ni < np:
                        T.copy(out, W[group, ni : ni + 16, qi : qi + 128])
                    if ni < pp:
                        with T.SimtVF(threads=256):
                            for i, j in T.Parallel(16, 128):
                                value = T.alloc_var("float32")
                                value = T.float32(0)
                                if ni + i < p and j < count:
                                    if xdma:
                                        value = T.cast(xdata[j, i], "float32")
                                    else:
                                        value = T.cast(
                                            X[bi, ci * q + qi + j, hi, ni + i], "float32"
                                        )
                                out[i, j] = value
                        T.copy(out, Packed[group, ni : ni + 16, qi : qi + 128])
            # The remaining N padding is initialized separately by the wrapper.

    tiles_p = pp // 64
    tiles_n = np // 64
    total = groups * tiles_p * tiles_n
    cubes = min(32, total)

    @T.prim_func
    def contract(
        X: T.Tensor((groups, pp, qp), "float32"),
        W: T.Tensor((groups, np, qp), "float32"),
        O: T.Tensor((groups, pp, np), "float32"),
    ):
        with T.Kernel(cubes) as bx:
            a = T.alloc_l1((64, 128), "float32")
            z = T.alloc_l1((64, 128), "float32")
            acc = T.alloc_l0c((64, 64), "float32")
            T.annotate_buffer_versions({a: 2, z: 2})
            for step in T.serial(T.ceildiv(total, cubes)):
                tile = step * cubes + bx
                if tile < total:
                    group = tile // (tiles_p * tiles_n)
                    pi = tile // tiles_n % tiles_p * 64
                    ni = tile % tiles_n * 64
                    for ki in T.Pipelined(qp // 128, num_stages=2):
                        T.copy(X[group, pi : pi + 64, ki * 128 : ki * 128 + 128], a)
                        T.copy(W[group, ni : ni + 64, ki * 128 : ki * 128 + 128], z)
                        T.gemm(a, z, acc, transpose_B=True, clear_accum=ki == 0)
                    T.copy(acc, O[group, pi : pi + 64, ni : ni + 64])

    return tilelang.compile(weighted, target="ascend"), tilelang.compile(
        contract, target="ascend", out_idx=-1
    )


@T.macro
def _weights_unfused(delta, steps, same, weights, has_seq: bool):
    with T.SimdVF():
        mask = T.simd.pset(32)
        for part in T.serial(2):
            raw = T.simd.vld(delta[part * 64])
            capped = T.simd.vmins(raw, T.float32(0), mask)
            capped = T.simd.vsel(raw, capped, T.simd.vcmp(raw, raw, mask, op="ne"))
            decay = T.simd.vexp(capped, mask, precision="ftz_false")
            value = T.simd.vmul(decay, T.simd.vld(steps[part * 64]), mask)
            if has_seq:
                value = T.simd.vmul(value, T.simd.vld(same[part * 64]), mask)
            T.simd.vsts(weights[part * 64], value, mask, extent=64)
        T.simd.mem_bar("VST_VLD")


@functools.lru_cache(maxsize=128)
def build_unfused(b, c, q, h, p, n, g, dtype, dt_dtype, has_seq):
    groups = b * c * h
    qp = (q + 127) // 128 * 128
    pp = (p + 63) // 64 * 64
    np = (n + 63) // 64 * 64
    nt = (n + 15) // 16
    qt = qp // 128
    tasks = groups * nt * qt
    blocks = min(64, tasks)
    dma = n % 16 == 0

    @T.prim_func
    def weighted(
        B: T.Tensor((b, c * q, g, n), dtype),
        DT: T.Tensor((b, h, c, q), dt_dtype),
        DA: T.Tensor((b, h, c, q), "float32"),
        Seq: T.Tensor((b, c * q), "int32"),
        W: T.Tensor((groups, np, qp), "float32"),
    ):
        with T.Kernel(blocks) as bx:
            data = T.alloc_shared((128, 16), dtype)
            out = T.alloc_shared((16, 128), "float32")
            weights = T.alloc_shared((128,), "float32")
            deltas = T.alloc_shared((128,), "float32")
            steps = T.alloc_shared((128,), "float32")
            masks = T.alloc_shared((128,), "float32")
            for step in T.serial(T.ceildiv(tasks, blocks)):
                task = step * blocks + bx
                if task < tasks:
                    group = task // (nt * qt)
                    bi = group // (c * h)
                    ci = group // h % c
                    hi = group % h
                    gi = hi // (h // g)
                    ni = task // qt % nt * 16
                    qi = task % qt * 128
                    count = T.min(128, q - qi)
                    if dma:
                        T.copy(
                            B[bi, ci * q + qi : ci * q + qi + count, gi, ni : ni + 16],
                            data[:count, :],
                        )
                    with T.SimtVF(threads=256):
                        for j in T.Parallel(128):
                            delta = T.alloc_var("float32")
                            dt_value = T.alloc_var("float32")
                            same = T.alloc_var("float32")
                            delta = T.float32(0)
                            dt_value = T.float32(0)
                            same = T.float32(1)
                            if j < count:
                                delta = DA[bi, hi, ci, q - 1] - DA[bi, hi, ci, qi + j]
                                dt_value = T.cast(DT[bi, hi, ci, qi + j], "float32")
                                if has_seq:
                                    end = Seq[bi, ci * q + q - 1]
                                    same = T.cast(
                                        (end >= 0) and (Seq[bi, ci * q + qi + j] == end), "float32"
                                    )
                            deltas[j] = delta
                            steps[j] = dt_value
                            masks[j] = same
                    _weights_unfused(deltas, steps, masks, weights, has_seq)
                    with T.SimtVF(threads=256):
                        for i, j in T.Parallel(16, 128):
                            value = T.alloc_var("float32")
                            value = T.float32(0)
                            if ni + i < n and j < count:
                                if dma:
                                    value = weights[j] * T.cast(data[j, i], "float32")
                                else:
                                    value = weights[j] * T.cast(
                                        B[bi, ci * q + qi + j, gi, ni + i], "float32"
                                    )
                            out[i, j] = value
                    T.copy(out, W[group, ni : ni + 16, qi : qi + 128])
            # The remaining N padding is initialized separately by the wrapper.

    tiles_p = pp // 64
    tiles_n = np // 64
    total = groups * tiles_p * tiles_n
    cubes = min(32, total)

    @T.prim_func
    def contract(
        X: T.Tensor((groups, pp, qp), "float32"),
        W: T.Tensor((groups, np, qp), "float32"),
        O: T.Tensor((groups, pp, np), "float32"),
    ):
        with T.Kernel(cubes) as bx:
            a = T.alloc_l1((64, 128), "float32")
            z = T.alloc_l1((64, 128), "float32")
            acc = T.alloc_l0c((64, 64), "float32")
            T.annotate_buffer_versions({a: 2, z: 2})
            for step in T.serial(T.ceildiv(total, cubes)):
                tile = step * cubes + bx
                if tile < total:
                    group = tile // (tiles_p * tiles_n)
                    pi = tile // tiles_n % tiles_p * 64
                    ni = tile % tiles_n * 64
                    for ki in T.Pipelined(qp // 128, num_stages=2):
                        T.copy(X[group, pi : pi + 64, ki * 128 : ki * 128 + 128], a)
                        T.copy(W[group, ni : ni + 64, ki * 128 : ki * 128 + 128], z)
                        T.gemm(a, z, acc, transpose_B=True, clear_accum=ki == 0)
                    T.copy(acc, O[group, pi : pi + 64, ni : ni + 64])

    return tilelang.compile(weighted, target="ascend"), tilelang.compile(
        contract, target="ascend", out_idx=-1
    )


class AscendSSDChunkStateKernel(Kernel, SSDChunkStateFwdInterface):
    devices = frozenset({"npu"})

    @classmethod
    def entry_for(cls, call):
        return call, lambda: cls(call)

    def __init__(self, call):
        super().__init__()
        self.call = call
        self.fused = call.batch * call.num_chunks * call.n_heads < 8192
        builder = build if self.fused else build_unfused
        self.weighted, self.compiled = builder(
            call.batch,
            call.num_chunks,
            call.chunk_len,
            call.n_heads,
            call.d_head,
            call.d_state,
            call.n_groups,
            str(call.dtype).split(".")[-1],
            str(call.dt_dtype or call.dtype).split(".")[-1],
            call.has_seq_idx,
        )

    def forward(self, x, B, dt, dA_cumsum, seq_idx):
        c = self.call
        b, nc, q, h, p, n = c.batch, c.num_chunks, c.chunk_len, c.n_heads, c.d_head, c.d_state
        qp = (q + 127) // 128 * 128
        pp = (p + 63) // 64 * 64
        np = (n + 63) // 64 * 64
        if self.fused:
            packed = torch.empty((b * nc * h, pp, qp), device=x.device, dtype=torch.float32)
        else:
            packed = x.float().reshape(b, nc, q, h, p).permute(0, 1, 3, 4, 2).contiguous()
            if qp != q or pp != p:
                packed = F.pad(packed, (0, qp - q, 0, pp - p))
            packed = packed.reshape(b * nc * h, pp, qp)
        # For aligned N every W cell is written by weighted(), including Q
        # padding. Unaligned N retains zero initialization for unwritten rows.
        allocate = torch.empty if n % 64 == 0 else torch.zeros
        w = allocate((b * nc * h, np, qp), device=x.device, dtype=torch.float32)
        if self.fused:
            self.weighted(x, B, dt, dA_cumsum, seq_idx, w, packed)
        else:
            self.weighted(B, dt, dA_cumsum, seq_idx, w)
        result = self.compiled(packed, w)
        return result[:, :p, :n].contiguous().reshape(b, nc, h, p, n)


register_kernel_type("SSDChunkStateFwdOp", "ascend", AscendSSDChunkStateKernel)
