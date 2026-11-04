"""Ascend comparison kernels with packed bool output and broadcast reuse.


Import this module to register the six implementations. The public BinaryOp
normalizes input contiguity; this backend preserves broadcasting and returns a
fresh bool tensor. Native FP16/BF16 SIMD comparisons process 128 values at a
time; FP32 uses 64 lanes. Integer/bool fallback packs four output bytes into int32.
"""

import functools
import math
from typing import ClassVar

import torch

from tileops.backend import register_kernel_type
from tileops.kernels.elementwise.call_spec import BinaryPredicateFwdInterface, BroadcastCall
from tileops.kernels.kernel_base import Kernel

try:
    import tilelang
    import tilelang.ascend.language as T
except ImportError as exc:
    _IMPORT_ERROR = exc
    T = None
else:
    _IMPORT_ERROR = None


def _predicate(a, b, kind, dtype):
    if dtype in ("float16", "bfloat16"):
        a, b = T.cast(a, "float32"), T.cast(b, "float32")
    elif dtype == "bool":
        a, b = a != 0, b != 0
    if kind == "eq":
        return a == b
    if kind == "ne":
        return a != b
    if kind == "gt":
        return a > b
    if kind == "ge":
        return a >= b
    if kind == "lt":
        return a < b
    return a <= b


if T is not None:

    @T.macro
    def _native16_body(a, b, c, width: int, begin, period: int, kind: str):
        with T.SimdVF():
            mask = T.simd.pset(16)
            one = T.simd.vdup(T.int16(1), "int16", mask)
            zero = T.simd.vdup(T.int16(0), "int16", mask)
            for i in range(width // 128):
                av = T.simd.vld(a[i * 128])
                if period:  # noqa: SIM108 - preserve validated TileLang DSL structure
                    bv = T.simd.vld(b[(begin + i * 128) % period])
                else:
                    bv = T.simd.vld(b[i * 128])
                pred = T.simd.vcmp(av, bv, mask, kind)
                value = T.simd.vsel(one, zero, pred)
                T.simd.vsts(c[i * 128], value, mask, dist="PK_B16")
            if width % 128:
                last = width // 128 * 128
                tail_mask = T.simd.pset(32)
                av = T.simd.vcvt(T.simd.vld(a[last], dist="UNPK_B16"), "float32", part=0)
                if period:
                    bv = T.simd.vcvt(
                        T.simd.vld(b[(begin + last) % period], dist="UNPK_B16"), "float32", part=0
                    )
                else:
                    bv = T.simd.vcvt(T.simd.vld(b[last], dist="UNPK_B16"), "float32", part=0)
                pred = T.simd.vcmp(av, bv, tail_mask, kind)
                one32 = T.simd.vdup(T.int32(1), "int32", tail_mask)
                zero32 = T.simd.vdup(T.int32(0), "int32", tail_mask)
                value = T.simd.vsel(one32, zero32, pred)
                T.simd.vsts(c[last], value, tail_mask, dist="PK4_B32")

    @T.macro
    def _native16_channel(a, b, c, channels: int, repeat: int, kind: str, b_begin):
        with T.SimdVF():
            mask = T.simd.pset(16)
            tail_mask = T.simd.pset(32)
            one = T.simd.vdup(T.int16(1), "int16", mask)
            zero = T.simd.vdup(T.int16(0), "int16", mask)
            for ch in range(channels):
                bv = T.simd.vld(b[b_begin + ch], dist="BRC_B16")
                for i in range(repeat // 128):
                    index = ch * repeat + i * 128
                    av = T.simd.vld(a[index])
                    pred = T.simd.vcmp(av, bv, mask, kind)
                    value = T.simd.vsel(one, zero, pred)
                    T.simd.vsts(c[index], value, mask, dist="PK_B16")
                if repeat % 128:
                    index = ch * repeat + repeat // 128 * 128
                    av = T.simd.vcvt(T.simd.vld(a[index], dist="UNPK_B16"), "float32", part=0)
                    bv32 = T.simd.vcvt(bv, "float32", part=0)
                    pred = T.simd.vcmp(av, bv32, tail_mask, kind)
                    one32 = T.simd.vdup(T.int32(1), "int32", tail_mask)
                    zero32 = T.simd.vdup(T.int32(0), "int32", tail_mask)
                    value = T.simd.vsel(one32, zero32, pred)
                    T.simd.vsts(c[index], value, tail_mask, dist="PK4_B32")

    @T.macro
    def _simd_body(a, b, c, width: int, begin, period: int, repeat: int, kind: str, dtype: str):
        with T.SimdVF():
            mask = T.simd.pset(32)
            one = T.simd.vdup(T.int32(1), "int32", mask)
            zero = T.simd.vdup(T.int32(0), "int32", mask)
            for i in range(width // 64):
                if dtype == "float32":
                    av = T.simd.vld(a[i * 64])
                else:
                    av = T.simd.vcvt(T.simd.vld(a[i * 64], dist="UNPK_B16"), "float32", part=0)
                if period and repeat > 1:
                    if dtype == "float32":
                        bv = T.simd.vld(b[((begin + i * 64) // repeat) % period], dist="BRC_B32")
                    else:
                        bv = T.simd.vcvt(
                            T.simd.vld(b[((begin + i * 64) // repeat) % period], dist="BRC_B16"),
                            "float32",
                            part=0,
                        )
                elif period:
                    # SIMD bias periods and tile offsets are multiples of 64.
                    # Reduce in vector units to avoid wide integer division.
                    if dtype == "float32":
                        bv = T.simd.vld(b[((begin // 64 + i) % (period // 64)) * 64])
                    else:
                        bv = T.simd.vcvt(
                            T.simd.vld(
                                b[((begin // 64 + i) % (period // 64)) * 64], dist="UNPK_B16"
                            ),
                            "float32",
                            part=0,
                        )
                else:
                    if dtype == "float32":
                        bv = T.simd.vld(b[i * 64])
                    else:
                        bv = T.simd.vcvt(T.simd.vld(b[i * 64], dist="UNPK_B16"), "float32", part=0)
                pred = T.simd.vcmp(av, bv, mask, kind)
                value = T.simd.vsel(one, zero, pred)
                T.simd.vsts(c[i * 64], value, mask, dist="PK4_B32")

    @functools.lru_cache(maxsize=512)
    def build_simd(
        n,
        dtype,
        kind,
        period=0,
        repeat=1,
        blocks=64,
        tile=8192,
        stages=2,
        threads=1024,
        store_policy=4,
    ):
        full, tail = divmod(n, tile)
        grid = min(blocks, full) if full else 1
        b_size, b_width, b_tail = period or n, period or tile, period or tail
        assert n % 64 == 0 and tile % 64 == 0
        assert not period or (repeat == 1 and period % 64 == 0) or repeat % 64 == 0

        @T.prim_func
        def main(
            A: T.Tensor((n,), dtype), B: T.Tensor((b_size,), dtype), C: T.Tensor((n,), "uint8")
        ):
            with T.Kernel(grid) as bx:
                if full:
                    a = T.alloc_shared((tile,), dtype)
                    b = T.alloc_shared((b_width,), dtype)
                    c = T.alloc_shared((tile,), "uint8")
                    T.annotate_buffer_versions({a: stages, b: stages, c: stages})
                    if period:
                        T.copy(B[0:period], b)
                    for step in T.Pipelined((full + grid - 1) // grid, num_stages=stages):
                        block = step * grid + bx
                        if block < full:
                            begin = block * tile
                            T.copy(A[begin : begin + tile], a)
                            if not period:
                                T.copy(B[begin : begin + tile], b)
                            if (
                                dtype != "float32"
                                and (not period or period % 128 == 0)
                                and repeat == 1
                            ):
                                _native16_body(a, b, c, tile, begin, period, kind)
                            else:
                                _simd_body(a, b, c, tile, begin, period, repeat, kind, dtype)
                            T.copy(c, C[begin : begin + tile], l2_cache_ctrl=store_policy)
                if tail:  # noqa: SIM102 - preserve validated TileLang DSL structure
                    if bx == 0:
                        at = T.alloc_shared((tail,), dtype)
                        bt = T.alloc_shared((b_tail,), dtype)
                        ct = T.alloc_shared((tail,), "uint8")
                        T.copy(A[full * tile : n], at)
                        if period:
                            T.copy(B[0:period], bt)
                        else:
                            T.copy(B[full * tile : n], bt)
                        if dtype != "float32" and (not period or period % 128 == 0) and repeat == 1:
                            _native16_body(at, bt, ct, tail, full * tile, period, kind)
                        else:
                            _simd_body(at, bt, ct, tail, full * tile, period, repeat, kind, dtype)
                        T.copy(ct, C[full * tile : n], l2_cache_ctrl=store_policy)

        return tilelang.compile(main, target="ascend", out_idx=-1)

    @T.macro
    def _channel_body(a, b, c, channels: int, repeat: int, kind: str, dtype: str, b_begin):
        with T.SimdVF():
            mask = T.simd.pset(32)
            one = T.simd.vdup(T.int32(1), "int32", mask)
            zero = T.simd.vdup(T.int32(0), "int32", mask)
            for ch in range(channels):
                if dtype == "float32":
                    bv = T.simd.vld(b[b_begin + ch], dist="BRC_B32")
                else:
                    bv = T.simd.vcvt(T.simd.vld(b[b_begin + ch], dist="BRC_B16"), "float32", part=0)
                for i in range(repeat // 64):
                    index = ch * repeat + i * 64
                    if dtype == "float32":
                        av = T.simd.vld(a[index])
                    else:
                        av = T.simd.vcvt(T.simd.vld(a[index], dist="UNPK_B16"), "float32", part=0)
                    pred = T.simd.vcmp(av, bv, mask, kind)
                    value = T.simd.vsel(one, zero, pred)
                    T.simd.vsts(c[index], value, mask, dist="PK4_B32")

    @functools.lru_cache(maxsize=512)
    def build_channel(n, dtype, kind, period, repeat, blocks=72, tile=8, stages=2, threads=1024):
        channels = tile
        width = channels * repeat
        count = n // width
        grid = min(blocks, count)
        assert repeat % 64 == 0 and period % channels == 0 and n % width == 0

        @T.prim_func
        def main(
            A: T.Tensor((n,), dtype), B: T.Tensor((period,), dtype), C: T.Tensor((n,), "uint8")
        ):
            with T.Kernel(grid) as bx:
                a = T.alloc_shared((width,), dtype)
                b = T.alloc_shared((period,), dtype)
                c = T.alloc_shared((width,), "uint8")
                T.annotate_buffer_versions({a: stages, c: stages})
                T.copy(B, b)
                for step in T.Pipelined((count + grid - 1) // grid, num_stages=stages):
                    block = step * grid + bx
                    if block < count:
                        offset = block * width
                        T.copy(A[offset : offset + width], a)
                        if dtype != "float32":
                            _native16_channel(
                                a, b, c, channels, repeat, kind, (block * channels) % period
                            )
                        else:
                            _channel_body(
                                a, b, c, channels, repeat, kind, dtype, (block * channels) % period
                            )
                        T.copy(c, C[offset : offset + width], l2_cache_ctrl="NORMAL_FV")

        return tilelang.compile(main, target="ascend", out_idx=-1)

    @T.macro
    def _packed_body(
        a, b, c, width: int, begin, period: int, repeat: int, kind: str, dtype: str, threads: int
    ):
        with T.SimtVF(threads=threads):
            for i in T.Parallel(width // 4):
                j0 = 4 * i
                j1 = 4 * i + 1
                j2 = 4 * i + 2
                j3 = 4 * i + 3
                if period:
                    k0 = ((begin + j0) // repeat) % period
                    k1 = ((begin + j1) // repeat) % period
                    k2 = ((begin + j2) // repeat) % period
                    k3 = ((begin + j3) // repeat) % period
                else:
                    k0, k1, k2, k3 = j0, j1, j2, j3
                p0 = T.cast(_predicate(a[j0], b[k0], kind, dtype), "int32")
                p1 = T.cast(_predicate(a[j1], b[k1], kind, dtype), "int32")
                p2 = T.cast(_predicate(a[j2], b[k2], kind, dtype), "int32")
                p3 = T.cast(_predicate(a[j3], b[k3], kind, dtype), "int32")
                c[i] = T.bitwise_or(
                    T.bitwise_or(p0, T.shift_left(p1, 8)),
                    T.bitwise_or(T.shift_left(p2, 16), T.shift_left(p3, 24)),
                )

    @functools.lru_cache(maxsize=512)
    def build(n, dtype, kind, period=0, repeat=1, blocks=64, tile=8192, stages=2, threads=1024):
        """Return a kernel; tuning parameters are explicit and source-bound."""
        storage = "uint8" if dtype == "bool" else dtype
        full, tail = divmod(n, tile)
        grid = min(blocks, full) if full else 1
        b_size = period or n
        b_width = period or tile
        b_tail = period or tail
        tail_threads = min(threads, max(32, ((tail // 4 + 31) // 32) * 32))
        assert n % 4 == 0 and tile % 4 == 0

        @T.prim_func
        def main(
            A: T.Tensor((n,), storage),
            B: T.Tensor((b_size,), storage),
            C: T.Tensor((n // 4,), "int32"),
        ):
            with T.Kernel(grid) as bx:
                if full:
                    a = T.alloc_shared((tile,), storage)
                    b = T.alloc_shared((b_width,), storage)
                    c = T.alloc_shared((tile // 4,), "int32")
                    T.annotate_buffer_versions({a: stages, b: stages, c: stages})
                    if period:
                        T.copy(B[0:period], b)
                    for step in T.Pipelined((full + grid - 1) // grid, num_stages=stages):
                        block = step * grid + bx
                        if block < full:
                            begin = block * tile
                            T.copy(A[begin : begin + tile], a)
                            if not period:
                                T.copy(B[begin : begin + tile], b)
                            _packed_body(a, b, c, tile, begin, period, repeat, kind, dtype, threads)
                            T.copy(c, C[begin // 4 : begin // 4 + tile // 4])
                if tail:  # noqa: SIM102 - preserve validated TileLang DSL structure
                    if bx == 0:
                        at = T.alloc_shared((tail,), storage)
                        bt = T.alloc_shared((b_tail,), storage)
                        ct = T.alloc_shared((tail // 4,), "int32")
                        T.copy(A[full * tile : n], at)
                        if period:
                            T.copy(B[0:period], bt)
                        else:
                            T.copy(B[full * tile : n], bt)
                        _packed_body(
                            at, bt, ct, tail, full * tile, period, repeat, kind, dtype, tail_threads
                        )
                        T.copy(ct, C[full * tile // 4 : n // 4])

        return tilelang.compile(main, target="ascend", out_idx=-1)


def _layout(a_shape, b_shape):
    shape = tuple(torch.broadcast_shapes(a_shape, b_shape))
    n = math.prod(shape)
    a_count, b_count = math.prod(a_shape), math.prod(b_shape)
    if a_count == b_count == n:
        return shape, 0, 1, False, False
    swapped = b_count == n and a_count != n
    small = a_shape if swapped else b_shape
    large_count = b_count if swapped else a_count
    padded = (1,) * (len(shape) - len(small)) + tuple(small)
    nontrivial = [i for i, value in enumerate(padded) if value > 1]
    if large_count == n and nontrivial:
        lo, hi = nontrivial[0], nontrivial[-1]
        if all(padded[i] == shape[i] for i in range(lo, hi + 1)):
            period = math.prod(padded[lo : hi + 1])
            if period <= 4096:
                return shape, period, math.prod(shape[hi + 1 :]), swapped, False
    if large_count == n and not nontrivial:
        return shape, 1, 1, swapped, False
    return shape, 0, 1, False, True


class _AscendComparison(Kernel, BinaryPredicateFwdInterface):
    devices: ClassVar[frozenset[str]] = frozenset({"npu"})
    kind = "eq"

    @classmethod
    def refusal(cls, call: BroadcastCall):
        return None

    @classmethod
    def entry_for(cls, call: BroadcastCall):
        return call, lambda: cls(call.a_shape, call.b_shape, call.dtype)

    def __init__(self, a_shape, b_shape, dtype):
        if _IMPORT_ERROR is not None:
            raise ImportError("Ascend comparison requires tilelang.ascend") from _IMPORT_ERROR
        super().__init__()
        self.shape, self.period, self.repeat, self.swapped, self.expand = _layout(a_shape, b_shape)
        self.n = math.prod(self.shape)
        self.dtype = dtype
        self.kind_native = (
            {"gt": "lt", "lt": "gt", "ge": "le", "le": "ge"}.get(self.kind, self.kind)
            if self.swapped
            else self.kind
        )
        self.padded = (self.n + 3) // 4 * 4
        # Ragged outputs use explicit zero padding; the output view is cropped.
        if self.padded != self.n:
            self.period, self.repeat, self.expand, self.swapped = 0, 1, True, False
            self.kind_native = self.kind
        dtype_name = str(dtype).removeprefix("torch.")
        floating = dtype in (torch.float16, torch.bfloat16, torch.float32)
        channels = 4 if dtype == torch.float32 else 8
        item_bytes = 4 if dtype == torch.float32 else 2
        channel_bytes = channels * self.repeat * (item_bytes + 1) * 3 + self.period * item_bytes
        if not self.n:
            self.compiled = None
        elif (
            floating
            and self.n % 64 == 0
            and self.repeat > 1
            and self.repeat % 64 == 0
            and self.period % 8 == 0
            and channel_bytes <= 248 * 1024
        ):
            self.compiled = build_channel(
                self.n, dtype_name, self.kind_native, self.period, self.repeat, 72, channels, 3
            )
        elif (
            floating
            and self.n % 64 == 0
            and (not self.period or (self.repeat == 1 and self.period % 64 == 0))
        ):
            tile = 8192 if dtype == torch.float32 else 16384
            blocks, stages = (64, 2) if dtype == torch.float32 else (64, 3)
            # Tail buffers coexist with the full tile in generated device code.
            # Two versions leave room for every possible SIMD tail.
            if self.n % tile:
                stages = 2
            self.compiled = build_simd(
                self.n,
                dtype_name,
                self.kind_native,
                self.period,
                self.repeat,
                blocks,
                tile,
                stages,
                store_policy=0,
            )
        else:
            # SIMT reserves 32 KiB of UB; int64 needs smaller full tiles.
            tile = 4096 if dtype == torch.int64 else 8192
            self.compiled = build(
                self.padded, dtype_name, self.kind_native, self.period, self.repeat, tile=tile
            )

    def forward(self, input, other):
        if not self.n:
            return torch.empty(self.shape, device=input.device, dtype=torch.bool)
        if self.expand:
            a = input.expand(self.shape).contiguous().reshape(-1)
            b = other.expand(self.shape).contiguous().reshape(-1)
        else:
            a, b = (other, input) if self.swapped else (input, other)
            a, b = a.reshape(-1), b.reshape(-1)
        if self.padded != self.n:
            a = torch.nn.functional.pad(a, (0, self.padded - self.n))
            b = torch.nn.functional.pad(b, (0, self.padded - self.n))
        if self.dtype == torch.bool:
            a, b = a.view(torch.uint8), b.view(torch.uint8)
        return self.compiled(a, b).view(torch.uint8).view(torch.bool)[: self.n].reshape(self.shape)


class AscendEqKernel(_AscendComparison):
    kind = "eq"


class AscendNeKernel(_AscendComparison):
    kind = "ne"


class AscendGtKernel(_AscendComparison):
    kind = "gt"


class AscendGeKernel(_AscendComparison):
    kind = "ge"


class AscendLtKernel(_AscendComparison):
    kind = "lt"


class AscendLeKernel(_AscendComparison):
    kind = "le"


register_kernel_type("EqFwdOp", "ascend", AscendEqKernel)
register_kernel_type("NeFwdOp", "ascend", AscendNeKernel)
register_kernel_type("GtFwdOp", "ascend", AscendGtKernel)
register_kernel_type("GeFwdOp", "ascend", AscendGeKernel)
register_kernel_type("LtFwdOp", "ascend", AscendLtKernel)
register_kernel_type("LeFwdOp", "ascend", AscendLeKernel)
