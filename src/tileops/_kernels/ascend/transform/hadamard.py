"""FP32 radix-two butterfly passes followed by the public base-matrix product."""

import functools
import math

import tilelang
import tilelang.ascend.language as T
import torch

from tileops._kernels.kernel_base import Kernel
from tileops._kernels.transform.call_spec import HADAMARD_BASES, HadamardTransformFwdInterface


@functools.lru_cache(maxsize=128)
def build_local(total, m, dtype):
    tile = 8192
    levels = min(13, m.bit_length() - 1)
    tiles = (total + tile - 1) // tile
    blocks = min(64, tiles)

    @T.prim_func
    def main(X: T.Tensor((total,), dtype), Y: T.Tensor((total,), "float32")):
        with T.Kernel(blocks) as bx:
            a = T.alloc_shared((tile,), "float32")
            b = T.alloc_shared((tile,), "float32")
            for step in T.serial(T.ceildiv(tiles, blocks)):
                offset = (step * blocks + bx) * tile
                with T.SimtVF(threads=256):
                    for i in T.Parallel(tile):
                        a[i] = T.float32(0)
                        if offset + i < total:
                            a[i] = T.cast(X[offset + i], "float32")
                for level in T.unroll(levels, explicit=True):
                    with T.SimtVF(threads=256):
                        for i in T.Parallel(tile):
                            lo = (i // (2 * (1 << level))) * (2 * (1 << level)) + i % (1 << level)
                            if level % 2 == 0:
                                if i % (2 * (1 << level)) < (1 << level):
                                    b[i] = a[lo] + a[lo + (1 << level)]
                                else:
                                    b[i] = a[lo] - a[lo + (1 << level)]
                            else:
                                if i % (2 * (1 << level)) < (1 << level):
                                    a[i] = b[lo] + b[lo + (1 << level)]
                                else:
                                    a[i] = b[lo] - b[lo + (1 << level)]
                with T.SimtVF(threads=256):
                    for i in T.Parallel(tile):
                        if offset + i < total:
                            if levels % 2 == 0:
                                Y[offset + i] = a[i]
                            else:
                                Y[offset + i] = b[i]

    return tilelang.compile(main, target="ascend", out_idx=[1])


@functools.lru_cache(maxsize=128)
def build_global(total, h):
    blocks = min(64, (total + 255) // 256)

    @T.prim_func
    def main(X: T.Tensor((total,), "float32"), Y: T.Tensor((total,), "float32")):
        with T.Kernel(blocks) as bx:  # noqa: SIM117
            with T.SimtVF(threads=256):
                for step in T.serial(T.ceildiv(total, blocks * 256)):
                    for lane in T.Parallel(256):
                        i = (step * blocks + bx) * 256 + lane
                        if i < total:
                            lo = (i // (2 * h)) * (2 * h) + i % h
                            if i % (2 * h) < h:
                                Y[i] = X[lo] + X[lo + h]
                            else:
                                Y[i] = X[lo] - X[lo + h]

    return tilelang.compile(main, target="ascend", out_idx=[1])


@functools.lru_cache(maxsize=128)
def build_base_shared(total, n, base):
    m = n // base
    rows = total // n
    blocks = min(64, rows)
    norm = math.sqrt(n)

    @T.prim_func
    def main(
        X: T.Tensor((total,), "float32"),
        B: T.Tensor((base, base), "float32"),
        Y: T.Tensor((total,), "float32"),
    ):
        with T.Kernel(blocks) as bx:
            values = T.alloc_shared((n,), "float32")
            output = T.alloc_shared((n,), "float32")
            coefficients = T.alloc_shared((base, base), "float32")
            T.copy(B, coefficients)
            for step in T.serial(T.ceildiv(rows, blocks)):
                row = step * blocks + bx
                if row < rows:
                    T.copy(X[row * n : row * n + n], values)
                    with T.SimtVF(threads=256):
                        for i in T.Parallel(n):
                            col = i // m
                            k = i % m
                            acc = T.alloc_var("float32")
                            acc = T.float32(0)
                            for a in T.unroll(base, explicit=True):
                                acc = acc + values[a * m + k] * coefficients[a, col]
                            output[i] = acc / T.float32(norm)
                    T.copy(output, Y[row * n : row * n + n])

    return tilelang.compile(main, target="ascend", out_idx=[2])


@functools.lru_cache(maxsize=128)
def build_base(total, n, base):
    if base > 1 and n <= 8192 and n % 8 == 0:
        return build_base_shared(total, n, base)
    m = n // base
    blocks = min(64, (total + 255) // 256)
    norm = math.sqrt(n)

    @T.prim_func
    def main(
        X: T.Tensor((total,), "float32"),
        B: T.Tensor((base, base), "float32"),
        Y: T.Tensor((total,), "float32"),
    ):
        with T.Kernel(blocks) as bx:  # noqa: SIM117
            with T.SimtVF(threads=256):
                for step in T.serial(T.ceildiv(total, blocks * 256)):
                    for lane in T.Parallel(256):
                        i = (step * blocks + bx) * 256 + lane
                        if i < total:
                            row = i // n
                            col = (i % n) // m
                            k = i % m
                            acc = T.alloc_var("float32")
                            if base == 1:
                                acc = X[i]
                            else:
                                acc = T.float32(0)
                                for a in T.serial(base):
                                    acc = acc + X[row * n + a * m + k] * B[a, col]
                            Y[i] = acc / T.float32(norm)

    return tilelang.compile(main, target="ascend", out_idx=[2])


@functools.lru_cache(maxsize=128)
def build_fused(total, n, dtype):
    """Keep the first nine butterfly levels in warp registers and fuse the output."""
    parts = min(16, n // 32)
    threads = min(1024, 8192 // parts)
    tile = threads * parts
    levels = n.bit_length() - 1
    register_levels = 5 + parts.bit_length() - 1
    chunk = parts * 32
    tiles = total // tile
    blocks = min(64, tiles)
    norm = math.sqrt(n)

    @T.prim_func
    def main(X: T.Tensor((total,), dtype), Y: T.Tensor((total,), dtype)):
        with T.Kernel(blocks) as bx:
            inp = T.alloc_shared((tile,), dtype)
            out = T.alloc_shared((tile,), dtype)
            a = T.alloc_shared((tile,), "float32")
            b = T.alloc_shared((tile,), "float32")
            for step in T.serial(T.ceildiv(tiles, blocks)):
                job = step * blocks + bx
                if job < tiles:
                    offset = job * tile
                    T.copy(X[offset : offset + tile], inp)
                    with T.SimtVF(threads=threads):
                        tid = T.get_thread_binding()
                        lane = tid % 32
                        warp = tid // 32
                        v = T.alloc_local((parts,), "float32")
                        for part in T.unroll(parts, explicit=True):
                            v[part] = T.cast(inp[warp * chunk + part * 32 + lane], "float32")
                            for level in T.unroll(5, explicit=True):
                                other = T.call_extern(  # lint: extern-target allow: Ascend native SIMD lowering
                                    "float32", "asc_shfl_xor", v[part], T.int32(1 << level)
                                )
                                v[part] = T.if_then_else(
                                    lane % (2 << level) < (1 << level),
                                    v[part] + other,
                                    other - v[part],
                                )
                        for level in T.unroll(register_levels - 5, explicit=True):
                            for pair in T.unroll(parts // 2, explicit=True):
                                lo = (pair // (1 << level)) * (2 << level) + pair % (1 << level)
                                left = v[lo]
                                right = v[lo + (1 << level)]
                                v[lo] = left + right
                                v[lo + (1 << level)] = left - right
                        for part in T.unroll(parts, explicit=True):
                            a[warp * chunk + part * 32 + lane] = v[part]
                    for stage in T.unroll(max(0, levels - register_levels), explicit=True):
                        with T.SimtVF(threads=256):
                            for i in T.Parallel(tile // 2):
                                lo = (i // (chunk << stage)) * (2 * chunk << stage) + i % (
                                    chunk << stage
                                )
                                if stage % 2 == 0:
                                    b[lo] = a[lo] + a[lo + (chunk << stage)]
                                    b[lo + (chunk << stage)] = a[lo] - a[lo + (chunk << stage)]
                                else:
                                    a[lo] = b[lo] + b[lo + (chunk << stage)]
                                    a[lo + (chunk << stage)] = b[lo] - b[lo + (chunk << stage)]
                    with T.SimtVF(threads=256):
                        for i in T.Parallel(tile):
                            if max(0, levels - register_levels) % 2 == 0:
                                out[i] = T.cast(a[i] / T.float32(norm), dtype)
                            else:
                                out[i] = T.cast(b[i] / T.float32(norm), dtype)
                    T.copy(out, Y[offset : offset + tile])

    return tilelang.compile(main, target="ascend", out_idx=[1])


class AscendHadamardKernel(Kernel, HadamardTransformFwdInterface):
    devices = frozenset({"npu"})

    @classmethod
    def entry_for(cls, call):
        return call, lambda: cls(call)

    def __init__(self, call):
        super().__init__()
        self.call = call

    def forward(self, x):
        c = self.call
        if x.numel() == 0:
            return x.clone()
        total = x.numel()
        n = c.n
        base = c.base_order
        m = n // base
        if base == 1 and 32 <= n <= 8192 and total % 8192 == 0 and x.data_ptr() % 32 == 0:
            self.fused = build_fused(total, n, str(x.dtype).removeprefix("torch."))
            return self.fused(x.reshape(-1)).reshape(x.shape)
        self.local = build_local(total, m, str(x.dtype).removeprefix("torch."))
        v = self.local(x.reshape(-1))
        self.passes = []
        h = 8192
        while h < m:
            kernel = build_global(total, h)
            self.passes.append(kernel)
            v = kernel(v)
            h *= 2
        if base == 1:
            matrix = torch.empty((1, 1), dtype=torch.float32, device=x.device)
        else:
            matrix = torch.tensor(
                [[1.0 if sign == "+" else -1.0 for sign in row] for row in HADAMARD_BASES[base]],
                dtype=torch.float32,
                device=x.device,
            )
        self.mix = build_base(total, n, base)
        return self.mix(v, matrix).reshape(x.shape).to(x.dtype)
