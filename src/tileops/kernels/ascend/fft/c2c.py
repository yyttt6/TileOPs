"""Split-FP32 FFT with local radix-8 fusion and paired global DIT stages."""

import functools
import math

import tilelang
import tilelang.ascend.language as T
import torch
from tilelang.carver.arch.driver import get_num_vector_cores

import tileops.backend
from tileops.kernels.fft import FFTC2CCall, FFTC2CFwdInterface
from tileops.kernels.kernel_base import Entry, Kernel


def _fft_prim_func(batch: int, n: int, vector_cores: int):
    dtype = "float32"
    if not isinstance(batch, int) or not isinstance(n, int):
        raise TypeError("batch and n must be static Python integers")
    if batch <= 0 or n <= 1 or (n & (n - 1)) or n > 65536:
        raise ValueError("batch must be positive, n a power of two in [2,65536]")
    if vector_cores <= 0:
        raise ValueError("vector_cores must be positive")
    tile = min(n, 4096)
    stages = n.bit_length() - 1
    local_stages = tile.bit_length() - 1
    pack = max(1, 4096 // n) if n <= tile else 1
    # One-wave row balancing within a 5120-element input-buffer budget.
    # This geometry rule is shared by all inputs; no complete-kernel dispatch.
    balanced_pack = (batch + vector_cores - 1) // vector_cores
    if n <= tile and pack < balanced_pack and balanced_pack * tile <= 5120:
        pack = balanced_pack
    packets = (batch + pack - 1) // pack
    blocks = min(packets, vector_cores)
    iterations = (packets + blocks - 1) // blocks
    # Whole VF iterations keep cross-thread barriers out of partial-lane branches.
    butterfly_groups = pack * tile // (8 if local_stages >= 3 else 2)
    threads = min(256, butterfly_groups & -butterfly_groups)
    cs = [(-1) ** k * math.pi ** (2 * k) / math.factorial(2 * k) for k in range(1, 7)]
    ss = [(-1) ** k * math.pi ** (2 * k + 1) / math.factorial(2 * k + 1) for k in range(1, 7)]
    pi_low = math.pi - float.fromhex("0x1.921fb60000000p+1")

    @T.macro
    def butterfly(ar_buf, ai_buf, br_buf, bi_buf, ia, ib, wr, wi):
        ar = ar_buf[ia]
        ai = ai_buf[ia]
        br = br_buf[ib]
        bi = bi_buf[ib]
        tr = T.call_extern(  # lint: extern-target allow: Ascend has no scalar FMA intrinsic; preserve FP32 rounding.
            dtype, "fmaf", wr, br, -wi * bi
        )
        ti = T.call_extern(  # lint: extern-target allow: Ascend has no scalar FMA intrinsic; preserve FP32 rounding.
            dtype, "fmaf", wr, bi, wi * br
        )
        ar_buf[ia] = ar + tr
        ai_buf[ia] = ai + ti
        br_buf[ib] = ar - tr
        bi_buf[ib] = ai - ti

    @T.macro
    def fused_first_triple(r, im, p0):
        # The first radix-8 group has only unit/quadrant and +/-45 degree roots.
        x0r = T.alloc_var(dtype)
        x0i = T.alloc_var(dtype)
        x1r = T.alloc_var(dtype)
        x1i = T.alloc_var(dtype)
        x2r = T.alloc_var(dtype)
        x2i = T.alloc_var(dtype)
        x3r = T.alloc_var(dtype)
        x3i = T.alloc_var(dtype)
        x4r = T.alloc_var(dtype)
        x4i = T.alloc_var(dtype)
        x5r = T.alloc_var(dtype)
        x5i = T.alloc_var(dtype)
        x6r = T.alloc_var(dtype)
        x6i = T.alloc_var(dtype)
        x7r = T.alloc_var(dtype)
        x7i = T.alloc_var(dtype)
        tr = T.alloc_var(dtype)
        ti = T.alloc_var(dtype)
        x0r = r[p0 + 0]
        x0i = im[p0 + 0]
        x1r = r[p0 + 1]
        x1i = im[p0 + 1]
        x2r = r[p0 + 2]
        x2i = im[p0 + 2]
        x3r = r[p0 + 3]
        x3i = im[p0 + 3]
        x4r = r[p0 + 4]
        x4i = im[p0 + 4]
        x5r = r[p0 + 5]
        x5i = im[p0 + 5]
        x6r = r[p0 + 6]
        x6i = im[p0 + 6]
        x7r = r[p0 + 7]
        x7i = im[p0 + 7]
        tr = x1r
        ti = x1i
        x1r = x0r - tr
        x1i = x0i - ti
        x0r = x0r + tr
        x0i = x0i + ti
        tr = x3r
        ti = x3i
        x3r = x2r - tr
        x3i = x2i - ti
        x2r = x2r + tr
        x2i = x2i + ti
        tr = x5r
        ti = x5i
        x5r = x4r - tr
        x5i = x4i - ti
        x4r = x4r + tr
        x4i = x4i + ti
        tr = x7r
        ti = x7i
        x7r = x6r - tr
        x7i = x6i - ti
        x6r = x6r + tr
        x6i = x6i + ti
        tr = x2r
        ti = x2i
        x2r = x0r - tr
        x2i = x0i - ti
        x0r = x0r + tr
        x0i = x0i + ti
        tr = x3i
        ti = -x3r
        x3r = x1r - tr
        x3i = x1i - ti
        x1r = x1r + tr
        x1i = x1i + ti
        tr = x6r
        ti = x6i
        x6r = x4r - tr
        x6i = x4i - ti
        x4r = x4r + tr
        x4i = x4i + ti
        tr = x7i
        ti = -x7r
        x7r = x5r - tr
        x7i = x5i - ti
        x5r = x5r + tr
        x5i = x5i + ti
        tr = x4r
        ti = x4i
        x4r = x0r - tr
        x4i = x0i - ti
        x0r = x0r + tr
        x0i = x0i + ti
        tr = T.call_extern(  # lint: extern-target allow: Ascend has no scalar FMA intrinsic; preserve FP32 rounding.
            dtype, "fmaf", T.float32(math.sqrt(0.5)), x5r, T.float32(math.sqrt(0.5)) * x5i
        )
        ti = T.call_extern(  # lint: extern-target allow: Ascend has no scalar FMA intrinsic; preserve FP32 rounding.
            dtype, "fmaf", T.float32(math.sqrt(0.5)), x5i, (-T.float32(math.sqrt(0.5))) * x5r
        )
        x5r = x1r - tr
        x5i = x1i - ti
        x1r = x1r + tr
        x1i = x1i + ti
        tr = x6i
        ti = -x6r
        x6r = x2r - tr
        x6i = x2i - ti
        x2r = x2r + tr
        x2i = x2i + ti
        tr = T.call_extern(  # lint: extern-target allow: Ascend has no scalar FMA intrinsic; preserve FP32 rounding.
            dtype, "fmaf", -T.float32(math.sqrt(0.5)), x7r, T.float32(math.sqrt(0.5)) * x7i
        )
        ti = T.call_extern(  # lint: extern-target allow: Ascend has no scalar FMA intrinsic; preserve FP32 rounding.
            dtype, "fmaf", -T.float32(math.sqrt(0.5)), x7i, (-T.float32(math.sqrt(0.5))) * x7r
        )
        x7r = x3r - tr
        x7i = x3i - ti
        x3r = x3r + tr
        x3i = x3i + ti
        r[p0 + 0] = x0r
        im[p0 + 0] = x0i
        r[p0 + 1] = x1r
        im[p0 + 1] = x1i
        r[p0 + 2] = x2r
        im[p0 + 2] = x2i
        r[p0 + 3] = x3r
        im[p0 + 3] = x3i
        r[p0 + 4] = x4r
        im[p0 + 4] = x4i
        r[p0 + 5] = x5r
        im[p0 + 5] = x5i
        r[p0 + 6] = x6r
        im[p0 + 6] = x6i
        r[p0 + 7] = x7r
        im[p0 + 7] = x7i

    @T.macro
    def fused_triple(r, im, wreal, wimag, p0, half):
        # Snapshot all eight points before any shared-memory write.
        x0r = T.alloc_var(dtype)
        x0i = T.alloc_var(dtype)
        x1r = T.alloc_var(dtype)
        x1i = T.alloc_var(dtype)
        x2r = T.alloc_var(dtype)
        x2i = T.alloc_var(dtype)
        x3r = T.alloc_var(dtype)
        x3i = T.alloc_var(dtype)
        x4r = T.alloc_var(dtype)
        x4i = T.alloc_var(dtype)
        x5r = T.alloc_var(dtype)
        x5i = T.alloc_var(dtype)
        x6r = T.alloc_var(dtype)
        x6i = T.alloc_var(dtype)
        x7r = T.alloc_var(dtype)
        x7i = T.alloc_var(dtype)
        tr = T.alloc_var(dtype)
        ti = T.alloc_var(dtype)
        j = p0 % half
        x0r = r[p0 + 0 * half]
        x0i = im[p0 + 0 * half]
        x1r = r[p0 + 1 * half]
        x1i = im[p0 + 1 * half]
        x2r = r[p0 + 2 * half]
        x2i = im[p0 + 2 * half]
        x3r = r[p0 + 3 * half]
        x3i = im[p0 + 3 * half]
        x4r = r[p0 + 4 * half]
        x4i = im[p0 + 4 * half]
        x5r = r[p0 + 5 * half]
        x5i = im[p0 + 5 * half]
        x6r = r[p0 + 6 * half]
        x6i = im[p0 + 6 * half]
        x7r = r[p0 + 7 * half]
        x7i = im[p0 + 7 * half]
        tr = T.call_extern(  # lint: extern-target allow: Ascend has no scalar FMA intrinsic; preserve FP32 rounding.
            dtype,
            "fmaf",
            wreal[(j + 0 * half) * (tile // (2 * half))],
            x1r,
            -wimag[(j + 0 * half) * (tile // (2 * half))] * x1i,
        )
        ti = T.call_extern(  # lint: extern-target allow: Ascend has no scalar FMA intrinsic; preserve FP32 rounding.
            dtype,
            "fmaf",
            wreal[(j + 0 * half) * (tile // (2 * half))],
            x1i,
            wimag[(j + 0 * half) * (tile // (2 * half))] * x1r,
        )
        x1r = x0r - tr
        x1i = x0i - ti
        x0r = x0r + tr
        x0i = x0i + ti
        tr = T.call_extern(  # lint: extern-target allow: Ascend has no scalar FMA intrinsic; preserve FP32 rounding.
            dtype,
            "fmaf",
            wreal[(j + 0 * half) * (tile // (2 * half))],
            x3r,
            -wimag[(j + 0 * half) * (tile // (2 * half))] * x3i,
        )
        ti = T.call_extern(  # lint: extern-target allow: Ascend has no scalar FMA intrinsic; preserve FP32 rounding.
            dtype,
            "fmaf",
            wreal[(j + 0 * half) * (tile // (2 * half))],
            x3i,
            wimag[(j + 0 * half) * (tile // (2 * half))] * x3r,
        )
        x3r = x2r - tr
        x3i = x2i - ti
        x2r = x2r + tr
        x2i = x2i + ti
        tr = T.call_extern(  # lint: extern-target allow: Ascend has no scalar FMA intrinsic; preserve FP32 rounding.
            dtype,
            "fmaf",
            wreal[(j + 0 * half) * (tile // (2 * half))],
            x5r,
            -wimag[(j + 0 * half) * (tile // (2 * half))] * x5i,
        )
        ti = T.call_extern(  # lint: extern-target allow: Ascend has no scalar FMA intrinsic; preserve FP32 rounding.
            dtype,
            "fmaf",
            wreal[(j + 0 * half) * (tile // (2 * half))],
            x5i,
            wimag[(j + 0 * half) * (tile // (2 * half))] * x5r,
        )
        x5r = x4r - tr
        x5i = x4i - ti
        x4r = x4r + tr
        x4i = x4i + ti
        tr = T.call_extern(  # lint: extern-target allow: Ascend has no scalar FMA intrinsic; preserve FP32 rounding.
            dtype,
            "fmaf",
            wreal[(j + 0 * half) * (tile // (2 * half))],
            x7r,
            -wimag[(j + 0 * half) * (tile // (2 * half))] * x7i,
        )
        ti = T.call_extern(  # lint: extern-target allow: Ascend has no scalar FMA intrinsic; preserve FP32 rounding.
            dtype,
            "fmaf",
            wreal[(j + 0 * half) * (tile // (2 * half))],
            x7i,
            wimag[(j + 0 * half) * (tile // (2 * half))] * x7r,
        )
        x7r = x6r - tr
        x7i = x6i - ti
        x6r = x6r + tr
        x6i = x6i + ti
        tr = T.call_extern(  # lint: extern-target allow: Ascend has no scalar FMA intrinsic; preserve FP32 rounding.
            dtype,
            "fmaf",
            wreal[(j + 0 * half) * (tile // (4 * half))],
            x2r,
            -wimag[(j + 0 * half) * (tile // (4 * half))] * x2i,
        )
        ti = T.call_extern(  # lint: extern-target allow: Ascend has no scalar FMA intrinsic; preserve FP32 rounding.
            dtype,
            "fmaf",
            wreal[(j + 0 * half) * (tile // (4 * half))],
            x2i,
            wimag[(j + 0 * half) * (tile // (4 * half))] * x2r,
        )
        x2r = x0r - tr
        x2i = x0i - ti
        x0r = x0r + tr
        x0i = x0i + ti
        tr = T.call_extern(  # lint: extern-target allow: Ascend has no scalar FMA intrinsic; preserve FP32 rounding.
            dtype,
            "fmaf",
            wreal[(j + 1 * half) * (tile // (4 * half))],
            x3r,
            -wimag[(j + 1 * half) * (tile // (4 * half))] * x3i,
        )
        ti = T.call_extern(  # lint: extern-target allow: Ascend has no scalar FMA intrinsic; preserve FP32 rounding.
            dtype,
            "fmaf",
            wreal[(j + 1 * half) * (tile // (4 * half))],
            x3i,
            wimag[(j + 1 * half) * (tile // (4 * half))] * x3r,
        )
        x3r = x1r - tr
        x3i = x1i - ti
        x1r = x1r + tr
        x1i = x1i + ti
        tr = T.call_extern(  # lint: extern-target allow: Ascend has no scalar FMA intrinsic; preserve FP32 rounding.
            dtype,
            "fmaf",
            wreal[(j + 0 * half) * (tile // (4 * half))],
            x6r,
            -wimag[(j + 0 * half) * (tile // (4 * half))] * x6i,
        )
        ti = T.call_extern(  # lint: extern-target allow: Ascend has no scalar FMA intrinsic; preserve FP32 rounding.
            dtype,
            "fmaf",
            wreal[(j + 0 * half) * (tile // (4 * half))],
            x6i,
            wimag[(j + 0 * half) * (tile // (4 * half))] * x6r,
        )
        x6r = x4r - tr
        x6i = x4i - ti
        x4r = x4r + tr
        x4i = x4i + ti
        tr = T.call_extern(  # lint: extern-target allow: Ascend has no scalar FMA intrinsic; preserve FP32 rounding.
            dtype,
            "fmaf",
            wreal[(j + 1 * half) * (tile // (4 * half))],
            x7r,
            -wimag[(j + 1 * half) * (tile // (4 * half))] * x7i,
        )
        ti = T.call_extern(  # lint: extern-target allow: Ascend has no scalar FMA intrinsic; preserve FP32 rounding.
            dtype,
            "fmaf",
            wreal[(j + 1 * half) * (tile // (4 * half))],
            x7i,
            wimag[(j + 1 * half) * (tile // (4 * half))] * x7r,
        )
        x7r = x5r - tr
        x7i = x5i - ti
        x5r = x5r + tr
        x5i = x5i + ti
        tr = T.call_extern(  # lint: extern-target allow: Ascend has no scalar FMA intrinsic; preserve FP32 rounding.
            dtype,
            "fmaf",
            wreal[(j + 0 * half) * (tile // (8 * half))],
            x4r,
            -wimag[(j + 0 * half) * (tile // (8 * half))] * x4i,
        )
        ti = T.call_extern(  # lint: extern-target allow: Ascend has no scalar FMA intrinsic; preserve FP32 rounding.
            dtype,
            "fmaf",
            wreal[(j + 0 * half) * (tile // (8 * half))],
            x4i,
            wimag[(j + 0 * half) * (tile // (8 * half))] * x4r,
        )
        x4r = x0r - tr
        x4i = x0i - ti
        x0r = x0r + tr
        x0i = x0i + ti
        tr = T.call_extern(  # lint: extern-target allow: Ascend has no scalar FMA intrinsic; preserve FP32 rounding.
            dtype,
            "fmaf",
            wreal[(j + 1 * half) * (tile // (8 * half))],
            x5r,
            -wimag[(j + 1 * half) * (tile // (8 * half))] * x5i,
        )
        ti = T.call_extern(  # lint: extern-target allow: Ascend has no scalar FMA intrinsic; preserve FP32 rounding.
            dtype,
            "fmaf",
            wreal[(j + 1 * half) * (tile // (8 * half))],
            x5i,
            wimag[(j + 1 * half) * (tile // (8 * half))] * x5r,
        )
        x5r = x1r - tr
        x5i = x1i - ti
        x1r = x1r + tr
        x1i = x1i + ti
        tr = T.call_extern(  # lint: extern-target allow: Ascend has no scalar FMA intrinsic; preserve FP32 rounding.
            dtype,
            "fmaf",
            wreal[(j + 2 * half) * (tile // (8 * half))],
            x6r,
            -wimag[(j + 2 * half) * (tile // (8 * half))] * x6i,
        )
        ti = T.call_extern(  # lint: extern-target allow: Ascend has no scalar FMA intrinsic; preserve FP32 rounding.
            dtype,
            "fmaf",
            wreal[(j + 2 * half) * (tile // (8 * half))],
            x6i,
            wimag[(j + 2 * half) * (tile // (8 * half))] * x6r,
        )
        x6r = x2r - tr
        x6i = x2i - ti
        x2r = x2r + tr
        x2i = x2i + ti
        tr = T.call_extern(  # lint: extern-target allow: Ascend has no scalar FMA intrinsic; preserve FP32 rounding.
            dtype,
            "fmaf",
            wreal[(j + 3 * half) * (tile // (8 * half))],
            x7r,
            -wimag[(j + 3 * half) * (tile // (8 * half))] * x7i,
        )
        ti = T.call_extern(  # lint: extern-target allow: Ascend has no scalar FMA intrinsic; preserve FP32 rounding.
            dtype,
            "fmaf",
            wreal[(j + 3 * half) * (tile // (8 * half))],
            x7i,
            wimag[(j + 3 * half) * (tile // (8 * half))] * x7r,
        )
        x7r = x3r - tr
        x7i = x3i - ti
        x3r = x3r + tr
        x3i = x3i + ti
        r[p0 + 0 * half] = x0r
        im[p0 + 0 * half] = x0i
        r[p0 + 1 * half] = x1r
        im[p0 + 1 * half] = x1i
        r[p0 + 2 * half] = x2r
        im[p0 + 2 * half] = x2i
        r[p0 + 3 * half] = x3r
        im[p0 + 3 * half] = x3i
        r[p0 + 4 * half] = x4r
        im[p0 + 4 * half] = x4i
        r[p0 + 5 * half] = x5r
        im[p0 + 5 * half] = x5i
        r[p0 + 6 * half] = x6r
        im[p0 + 6 * half] = x6i
        r[p0 + 7 * half] = x7r
        im[p0 + 7 * half] = x7i

    @T.macro
    def fill_roots(wr_buf, wi_buf, half, offset, count, base):
        with T.SimtVF(threads=256):
            for k in T.Parallel(count):
                j = offset + k
                q = (2 * j + half // 2) // half
                t = T.cast(j, dtype) / T.cast(half, dtype) - T.cast(q, dtype) * T.float32(0.5)
                z = t * t
                cp0 = T.call_extern(  # lint: extern-target allow: Ascend has no scalar FMA intrinsic; preserve FP32 rounding.
                    dtype, "fmaf", T.float32(cs[5]), z, T.float32(cs[4])
                )
                cp1 = T.call_extern(  # lint: extern-target allow: Ascend has no scalar FMA intrinsic; preserve FP32 rounding.
                    dtype, "fmaf", cp0, z, T.float32(cs[3])
                )
                cp2 = T.call_extern(  # lint: extern-target allow: Ascend has no scalar FMA intrinsic; preserve FP32 rounding.
                    dtype, "fmaf", cp1, z, T.float32(cs[2])
                )
                cp3 = T.call_extern(  # lint: extern-target allow: Ascend has no scalar FMA intrinsic; preserve FP32 rounding.
                    dtype, "fmaf", cp2, z, T.float32(cs[1])
                )
                cp4 = T.call_extern(  # lint: extern-target allow: Ascend has no scalar FMA intrinsic; preserve FP32 rounding.
                    dtype, "fmaf", cp3, z, T.float32(cs[0])
                )
                c = T.call_extern(  # lint: extern-target allow: Ascend has no scalar FMA intrinsic; preserve FP32 rounding.
                    dtype, "fmaf", cp4, z, T.float32(1)
                )
                sp0 = T.call_extern(  # lint: extern-target allow: Ascend has no scalar FMA intrinsic; preserve FP32 rounding.
                    dtype, "fmaf", T.float32(ss[5]), z, T.float32(ss[4])
                )
                sp1 = T.call_extern(  # lint: extern-target allow: Ascend has no scalar FMA intrinsic; preserve FP32 rounding.
                    dtype, "fmaf", sp0, z, T.float32(ss[3])
                )
                sp2 = T.call_extern(  # lint: extern-target allow: Ascend has no scalar FMA intrinsic; preserve FP32 rounding.
                    dtype, "fmaf", sp1, z, T.float32(ss[2])
                )
                sp3 = T.call_extern(  # lint: extern-target allow: Ascend has no scalar FMA intrinsic; preserve FP32 rounding.
                    dtype, "fmaf", sp2, z, T.float32(ss[1])
                )
                sp4 = T.call_extern(  # lint: extern-target allow: Ascend has no scalar FMA intrinsic; preserve FP32 rounding.
                    dtype, "fmaf", sp3, z, T.float32(ss[0])
                )
                correction = T.call_extern(  # lint: extern-target allow: Ascend has no scalar FMA intrinsic; preserve FP32 rounding.
                    dtype, "fmaf", sp4, z, T.float32(pi_low)
                )
                s = T.call_extern(  # lint: extern-target allow: Ascend has no scalar FMA intrinsic; preserve FP32 rounding.
                    dtype, "fmaf", t, T.float32(math.pi), t * correction
                )
                wr_buf[base + k] = T.if_then_else(q == 0, c, T.if_then_else(q == 1, -s, -c))
                wi_buf[base + k] = T.if_then_else(q == 0, -s, T.if_then_else(q == 1, -c, s))

    @T.macro
    def global_pair(R, I, Wr, Wi):
        # One thread owns the same lane in four distinct loaded UB banks.
        # Keep every operand and first-stage intermediate private before stores.
        for i in T.Parallel(tile):
            ar, ai = T.alloc_var(dtype), T.alloc_var(dtype)
            br, bi = T.alloc_var(dtype), T.alloc_var(dtype)
            cr, ci = T.alloc_var(dtype), T.alloc_var(dtype)
            dr, di = T.alloc_var(dtype), T.alloc_var(dtype)
            t0r, t0i = T.alloc_var(dtype), T.alloc_var(dtype)
            t1r, t1i = T.alloc_var(dtype), T.alloc_var(dtype)
            v0r, v0i = T.alloc_var(dtype), T.alloc_var(dtype)
            v1r, v1i = T.alloc_var(dtype), T.alloc_var(dtype)
            v2r, v2i = T.alloc_var(dtype), T.alloc_var(dtype)
            v3r, v3i = T.alloc_var(dtype), T.alloc_var(dtype)
            u0r, u0i = T.alloc_var(dtype), T.alloc_var(dtype)
            u1r, u1i = T.alloc_var(dtype), T.alloc_var(dtype)
            w2r, w2i = T.alloc_var(dtype), T.alloc_var(dtype)
            ar, ai = R[0 * tile + i], I[0 * tile + i]
            br, bi = R[1 * tile + i], I[1 * tile + i]
            cr, ci = R[2 * tile + i], I[2 * tile + i]
            dr, di = R[3 * tile + i], I[3 * tile + i]
            t0r = T.call_pure_extern(  # lint: extern-target allow: Ascend has no scalar FMA intrinsic; preserve FP32 rounding.
                dtype, "fmaf", br, Wr[i], -(bi * Wi[i])
            )
            t0i = T.call_pure_extern(  # lint: extern-target allow: Ascend has no scalar FMA intrinsic; preserve FP32 rounding.
                dtype, "fmaf", br, Wi[i], bi * Wr[i]
            )
            t1r = T.call_pure_extern(  # lint: extern-target allow: Ascend has no scalar FMA intrinsic; preserve FP32 rounding.
                dtype, "fmaf", dr, Wr[i], -(di * Wi[i])
            )
            t1i = T.call_pure_extern(  # lint: extern-target allow: Ascend has no scalar FMA intrinsic; preserve FP32 rounding.
                dtype, "fmaf", dr, Wi[i], di * Wr[i]
            )
            v0r, v0i = ar + t0r, ai + t0i
            v1r, v1i = ar - t0r, ai - t0i
            v2r, v2i = cr + t1r, ci + t1i
            v3r, v3i = cr - t1r, ci - t1i
            u0r = T.call_pure_extern(  # lint: extern-target allow: Ascend has no scalar FMA intrinsic; preserve FP32 rounding.
                dtype, "fmaf", v2r, Wr[tile + i], -(v2i * Wi[tile + i])
            )
            u0i = T.call_pure_extern(  # lint: extern-target allow: Ascend has no scalar FMA intrinsic; preserve FP32 rounding.
                dtype, "fmaf", v2r, Wi[tile + i], v2i * Wr[tile + i]
            )
            # W(offset+half; 2*half) = -i * W(offset; 2*half).
            w2r, w2i = Wi[tile + i], -Wr[tile + i]
            u1r = T.call_pure_extern(  # lint: extern-target allow: Ascend has no scalar FMA intrinsic; preserve FP32 rounding.
                dtype, "fmaf", v3r, w2r, -(v3i * w2i)
            )
            u1i = T.call_pure_extern(  # lint: extern-target allow: Ascend has no scalar FMA intrinsic; preserve FP32 rounding.
                dtype, "fmaf", v3r, w2i, v3i * w2r
            )
            R[0 * tile + i], I[0 * tile + i] = v0r + u0r, v0i + u0i
            R[2 * tile + i], I[2 * tile + i] = v0r - u0r, v0i - u0i
            R[1 * tile + i], I[1 * tile + i] = v1r + u1r, v1i + u1i
            R[3 * tile + i], I[3 * tile + i] = v1r - u1r, v1i - u1i
        T.sync_threads()

    @T.prim_func
    def fft(
        real: T.Tensor((batch, n), dtype),
        imag: T.Tensor((batch, n), dtype),
        out_real: T.Tensor((batch, n), dtype),
        out_imag: T.Tensor((batch, n), dtype),
    ):
        with T.Kernel(blocks) as block:
            a_real = T.alloc_shared(((4 if n > tile else pack) * tile,), dtype)
            a_imag = T.alloc_shared(((4 if n > tile else pack) * tile,), dtype)
            wr_table = T.alloc_shared((2 * tile if n > tile else tile // 2,), dtype)
            wi_table = T.alloc_shared((2 * tile if n > tile else tile // 2,), dtype)
            if n <= tile:
                fill_roots(wr_table, wi_table, tile // 2, 0, tile // 2, 0)
            for iteration in T.serial(iterations):
                packet = block + iteration * blocks
                first_row = packet * pack
                if packet < packets:
                    if n > tile:
                        fill_roots(wr_table, wi_table, tile // 2, 0, tile // 2, 0)
                    for tile_index in T.serial(n // tile):
                        start = tile_index * tile
                        with T.SimtVF(threads=threads):
                            for k in T.Parallel(pack * tile):
                                packed_row = k // tile
                                row = first_row + packed_row
                                pos = start + k % tile
                                v0 = T.cast(pos, "uint32")
                                v1 = ((v0 >> 1) & 0x55555555) | ((v0 & 0x55555555) << 1)
                                v2 = ((v1 >> 2) & 0x33333333) | ((v1 & 0x33333333) << 2)
                                v3 = ((v2 >> 4) & 0x0F0F0F0F) | ((v2 & 0x0F0F0F0F) << 4)
                                v4 = ((v3 >> 8) & 0x00FF00FF) | ((v3 & 0x00FF00FF) << 8)
                                v5 = (v4 >> 16) | (v4 << 16)
                                source = T.cast(v5 >> (32 - stages), "int32")
                                a_real[k] = T.if_then_else(
                                    row < batch, real[row, source], T.float32(0)
                                )
                                a_imag[k] = T.if_then_else(
                                    row < batch, imag[row, source], T.float32(0)
                                )
                            # Omit absent stages before tracing their index arithmetic.
                            if local_stages >= 3:
                                for k in T.Parallel(pack * tile // 8):
                                    row_base = (k // (tile // 8)) * tile
                                    group = k % (tile // 8)
                                    fused_first_triple(a_real, a_imag, row_base + 8 * group)
                            if local_stages >= 6:
                                for triple_stage in T.unroll(1, local_stages // 3):
                                    half = 1 << (3 * triple_stage)
                                    for k in T.Parallel(pack * tile // 8):
                                        row_base = (k // (tile // 8)) * tile
                                        group = k % (tile // 8)
                                        j = group % half
                                        low = row_base + (group // half) * (8 * half) + j
                                        fused_triple(a_real, a_imag, wr_table, wi_table, low, half)
                            if local_stages % 3:
                                for tail_stage in T.unroll((local_stages // 3) * 3, local_stages):
                                    half = 1 << tail_stage
                                    for k in T.Parallel(pack * tile // 2):
                                        row_base = (k // (tile // 2)) * tile
                                        group = k % (tile // 2)
                                        j = group % half
                                        low = row_base + (group // half) * (2 * half) + j
                                        twiddle = j * (tile // (2 * half))
                                        butterfly(
                                            a_real,
                                            a_imag,
                                            a_real,
                                            a_imag,
                                            low,
                                            low + half,
                                            wr_table[twiddle],
                                            wi_table[twiddle],
                                        )
                            if n <= tile:
                                for k in T.Parallel(pack * tile):
                                    row = first_row + k // tile
                                    if row < batch:
                                        out_real[row, k % tile] = a_real[k]
                                        out_imag[row, k % tile] = a_imag[k]
                        if n > tile:
                            T.copy(a_real[0:tile], out_real[first_row, start : start + tile])
                            T.copy(a_imag[0:tile], out_imag[first_row, start : start + tile])
                            T.ascend_pipe_barrier("PIPE_ALL")
                    if n > tile:
                        # Fuse pairs of global stages: each 4*half group is one
                        # disjoint write epoch. No cross-block communication occurs.
                        for global_pair_stage in T.serial((stages - local_stages) // 2):
                            half = T.shift_left(tile, 2 * global_pair_stage)
                            for chunk in T.serial(half // tile):
                                offset = chunk * tile
                                fill_roots(wr_table, wi_table, half, offset, tile, 0)
                                fill_roots(wr_table, wi_table, 2 * half, offset, tile, tile)
                                for group in T.serial(n // (4 * half)):
                                    a = group * 4 * half + offset
                                    b, c, d = a + half, a + 2 * half, a + 3 * half
                                    T.copy(
                                        out_real[first_row, a : a + tile],
                                        a_real[0 * tile : (0 + 1) * tile],
                                    )
                                    T.copy(
                                        out_imag[first_row, a : a + tile],
                                        a_imag[0 * tile : (0 + 1) * tile],
                                    )
                                    T.copy(
                                        out_real[first_row, b : b + tile],
                                        a_real[1 * tile : (1 + 1) * tile],
                                    )
                                    T.copy(
                                        out_imag[first_row, b : b + tile],
                                        a_imag[1 * tile : (1 + 1) * tile],
                                    )
                                    T.copy(
                                        out_real[first_row, c : c + tile],
                                        a_real[2 * tile : (2 + 1) * tile],
                                    )
                                    T.copy(
                                        out_imag[first_row, c : c + tile],
                                        a_imag[2 * tile : (2 + 1) * tile],
                                    )
                                    T.copy(
                                        out_real[first_row, d : d + tile],
                                        a_real[3 * tile : (3 + 1) * tile],
                                    )
                                    T.copy(
                                        out_imag[first_row, d : d + tile],
                                        a_imag[3 * tile : (3 + 1) * tile],
                                    )
                                    with T.SimtVF(threads=threads):
                                        global_pair(a_real, a_imag, wr_table, wi_table)
                                    T.copy(
                                        a_real[0 * tile : (0 + 1) * tile],
                                        out_real[first_row, a : a + tile],
                                    )
                                    T.copy(
                                        a_imag[0 * tile : (0 + 1) * tile],
                                        out_imag[first_row, a : a + tile],
                                    )
                                    T.copy(
                                        a_real[1 * tile : (1 + 1) * tile],
                                        out_real[first_row, b : b + tile],
                                    )
                                    T.copy(
                                        a_imag[1 * tile : (1 + 1) * tile],
                                        out_imag[first_row, b : b + tile],
                                    )
                                    T.copy(
                                        a_real[2 * tile : (2 + 1) * tile],
                                        out_real[first_row, c : c + tile],
                                    )
                                    T.copy(
                                        a_imag[2 * tile : (2 + 1) * tile],
                                        out_imag[first_row, c : c + tile],
                                    )
                                    T.copy(
                                        a_real[3 * tile : (3 + 1) * tile],
                                        out_real[first_row, d : d + tile],
                                    )
                                    T.copy(
                                        a_imag[3 * tile : (3 + 1) * tile],
                                        out_imag[first_row, d : d + tile],
                                    )
                        for stage in T.serial(
                            2 * ((stages - local_stages) // 2), stages - local_stages
                        ):
                            half = T.shift_left(tile, stage)
                            for chunk in T.serial(half // tile):
                                offset = chunk * tile
                                fill_roots(wr_table, wi_table, half, offset, tile, 0)
                                for group in T.serial(n // (2 * half)):
                                    a = group * 2 * half + offset
                                    b = a + half
                                    T.copy(
                                        out_real[first_row, a : a + tile],
                                        a_real[0 * tile : (0 + 1) * tile],
                                    )
                                    T.copy(
                                        out_imag[first_row, a : a + tile],
                                        a_imag[0 * tile : (0 + 1) * tile],
                                    )
                                    T.copy(
                                        out_real[first_row, b : b + tile],
                                        a_real[1 * tile : (1 + 1) * tile],
                                    )
                                    T.copy(
                                        out_imag[first_row, b : b + tile],
                                        a_imag[1 * tile : (1 + 1) * tile],
                                    )
                                    with T.SimtVF(threads=threads):
                                        for i in T.Parallel(tile):
                                            ar, ai = T.alloc_var(dtype), T.alloc_var(dtype)
                                            br, bi = T.alloc_var(dtype), T.alloc_var(dtype)
                                            tr, ti = T.alloc_var(dtype), T.alloc_var(dtype)
                                            ar, ai = a_real[i], a_imag[i]
                                            br, bi = a_real[tile + i], a_imag[tile + i]
                                            tr = T.call_pure_extern(  # lint: extern-target allow: Ascend has no scalar FMA intrinsic; preserve FP32 rounding.
                                                dtype, "fmaf", br, wr_table[i], -(bi * wi_table[i])
                                            )
                                            ti = T.call_pure_extern(  # lint: extern-target allow: Ascend has no scalar FMA intrinsic; preserve FP32 rounding.
                                                dtype, "fmaf", br, wi_table[i], bi * wr_table[i]
                                            )
                                            a_real[i], a_imag[i] = ar + tr, ai + ti
                                            a_real[tile + i], a_imag[tile + i] = ar - tr, ai - ti
                                        T.sync_threads()
                                    T.copy(
                                        a_real[0 * tile : (0 + 1) * tile],
                                        out_real[first_row, a : a + tile],
                                    )
                                    T.copy(
                                        a_imag[0 * tile : (0 + 1) * tile],
                                        out_imag[first_row, a : a + tile],
                                    )
                                    T.copy(
                                        a_real[1 * tile : (1 + 1) * tile],
                                        out_real[first_row, b : b + tile],
                                    )
                                    T.copy(
                                        a_imag[1 * tile : (1 + 1) * tile],
                                        out_imag[first_row, b : b + tile],
                                    )

    return fft


@functools.lru_cache(maxsize=64)
def _make_fft_kernel(batch: int, n: int, vector_cores: int, device_index: int):
    # Include the device in the cache key because compiled handles can be device-local.
    return tilelang.compile(_fft_prim_func(batch, n, vector_cores), target="ascend", out_idx=[2, 3])


class AscendFFTC2CKernel(Kernel, FFTC2CFwdInterface):
    """Forward complex64 FFT with one arithmetic flow on Ascend vector cores.

    FP64/complex128 is unsupported by this implementation and refused before
    compilation. Length one is the public op's out-of-place identity.
    """

    devices = frozenset({"npu"})

    @classmethod
    def refusal(cls, call: FFTC2CCall) -> str | None:
        if call.dtype != torch.complex64:
            return "FP64/complex128 is unsupported by this Ascend kernel; use complex64"
        if call.n < 2 or call.n > 65536 or call.n & (call.n - 1):
            return "requires a power-of-two length in [2, 65536]"
        return super().refusal(call)

    @classmethod
    def entry_for(cls, call: FFTC2CCall) -> Entry:
        return call, lambda: cls(call.n, call.dtype, call.device.index)

    def __init__(self, n: int, dtype: torch.dtype, device_index: int | None) -> None:
        super().__init__(device_index=device_index)
        self.n = n
        self.dtype = dtype
        self.vector_cores = int(get_num_vector_cores(device_id=device_index))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._require_devices(x=x)
        batch = x.numel() // self.n
        if batch == 0:
            return x.clone(memory_format=torch.contiguous_format)
        # Dtype views only reinterpret storage. All FFT arithmetic stays on the NPU.
        dense = x.resolve_conj().contiguous()
        components = dense.view(torch.float32).reshape(batch, self.n, 2)
        real = components[..., 0].contiguous()
        imag = components[..., 1].contiguous()
        device_index = x.device.index
        kernel = _make_fft_kernel(batch, self.n, self.vector_cores, device_index)
        out_real, out_imag = kernel(real, imag)
        interleaved = torch.stack((out_real, out_imag), dim=-1)
        return interleaved.view(torch.complex64).reshape(x.shape)


tileops.backend.register_kernel_type("FFTC2CFwdOp", "ascend", AscendFFTC2CKernel)
