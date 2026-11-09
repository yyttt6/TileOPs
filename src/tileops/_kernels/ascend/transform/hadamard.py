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
                            for a in T.serial(base):
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
