"""Full-feature L1 convolution with K-block-major streamed weights."""

import functools
import math

import tilelang
import tilelang.ascend.language as T
from tilelang.layout import make_ascend_major_k_layout

from tileops._csrc import csrc_include


@functools.lru_cache(maxsize=128)
def _streamed_pack(co, ci, kh, kw, dtype):
    k = ci * kh * kw
    bk = 64
    assert ci % 16 == 0 and k % bk == 0
    kt = k // bk
    tile = 1 << max(0, min(8, 8192 // k).bit_length() - 1)
    cores = min(64, (co + tile - 1) // tile)

    @T.prim_func
    def main(
        W: T.Tensor((co, k), dtype),
        Unused: T.Tensor((co,), dtype),
        P: T.Tensor((kt, co, bk), dtype),
    ):
        with T.Kernel(cores) as core:
            raw = T.alloc_shared((tile, k), dtype)
            result = T.alloc_shared((kt, tile, bk), dtype)
            for step in T.serial(T.ceildiv(co, cores * tile)):
                start = (step * cores + core) * tile
                if start < co:
                    T.copy(W[start : start + tile, :], raw)
                    with T.SimtVF(threads=256):
                        for z, i, j in T.Parallel(kt, tile, bk):
                            kk = z * bk + j
                            ic = kk // (kh * kw * 16) * 16 + kk % 16
                            tap = kk // 16 % (kh * kw)
                            result[z, i, j] = raw[i, ic * kh * kw + tap]
                    T.copy(result, P[:, start : start + tile, :])

    return tilelang.compile(
        main,
        target="ascend",
        out_idx=-1,
        compile_flags=csrc_include("ascend_convolution.h"),
    )


@functools.lru_cache(maxsize=128)
def _streamed_contract(batch, ci, co, size, out, window, stride, pad, dilation, dtype, has_bias):
    assert size[0] == out[0] == window[0] == 1 and ci % 16 == 0 and co % 16 == 0
    _, h, w = size
    _, oh, ow = out
    _, kh, kw = window
    _, sh, sw = stride
    _, ph, pw = pad
    _, dh, dw = dilation
    packed = ci * kh * kw
    bk = 64
    assert packed % bk == 0
    kt = packed // bk
    bm = min(256, co)
    plane = oh * ow
    while bm > 16 and batch * ((co + bm - 1) // bm) * ((plane + 15) // 16) < 32:
        bm //= 2
    assert co % bm == 0
    mt = co // bm
    wanted = max((plane + 255) // 256, (32 + batch * mt - 1) // (batch * mt))
    wave = 32 // math.gcd(32, batch * mt)
    wanted = (wanted + wave - 1) // wave * wave
    bn = min(256, max(16, (plane + wanted * 16 - 1) // (wanted * 16) * 16))
    nt = (plane + bn - 1) // bn
    tiles = batch * mt * nt
    cores = min(32, tiles)
    fw = w
    fh = ((ow - 1 + bn + ow - 1) // ow - 1) * sh + (kh - 1) * dh + 1
    align = 16 // math.gcd(w, 16)
    fh = (fh + align - 1) // align * align
    left = pw
    right = max(0, (ow - 1) * sw + (kw - 1) * dw + 1 - w - pw)
    assert fh * fw * ci * 2 + 2 * bm * bk * 2 + bm * 2 <= 512 * 1024
    quant = 16 if dtype == "bfloat16" else 1
    convert = 0 if dtype == "bfloat16" else 1

    @T.prim_func
    def main(
        X: T.Tensor((batch, ci, h * w), dtype),
        W: T.Tensor((kt, co, bk), dtype),
        Bias: T.Tensor((co,), dtype),
        Y: T.Tensor((batch, co, plane), dtype),
    ):
        with T.Kernel(cores) as core:
            xl = T.alloc_l1((fh * fw, ci), dtype)
            wl = T.alloc_l1((2, bm, bk), dtype)
            bl = T.alloc_l1((bm,), dtype)
            a = T.alloc_l0a((2, bn, bk), dtype)
            b = T.alloc_l0b((2, bm, bk), dtype)
            acc = T.alloc_l0c((bn, bm), "float32")
            T.annotate_layout(
                {xl: make_ascend_major_k_layout(xl), b: make_ascend_major_k_layout(b)}
            )
            T.annotate_buffer_versions({xl: 1, wl: 1, bl: 1, a: 1, b: 1, acc: 1})
            with T.Cube():
                for step in T.serial(T.ceildiv(tiles, cores)):
                    index = step * cores + core
                    if index < tiles:
                        ni = index % nt * bn
                        mi = index // nt % mt * bm
                        bi = index // (nt * mt)
                        base = ni // ow * sh - ph
                        begin = T.max(0, base)
                        count = T.max(0, T.min(h, base + fh) - begin)
                        if count < fh:
                            T.fill(xl, T.cast(0, dtype))
                            T.ascend_pipe_barrier("PIPE_MTE2")
                        if count > 0:
                            T.copy(
                                X[bi, :, begin * w : (begin + count) * w],
                                xl[(begin - base) * w : (begin - base + count) * w, :],
                                transpose=True,
                            )
                        if has_bias:
                            T.evaluate(
                                T.call_extern(
                                    "void",
                                    "tileops::conv_bias_gm",
                                    T.access_ptr(bl, "w"),
                                    T.access_ptr(Bias[mi], "r"),
                                    bm,
                                )
                            )
                        T.ascend_set_flag("MTE2_MTE1", 3)
                        T.ascend_wait_flag("MTE2_MTE1", 3)
                        if has_bias:
                            T.evaluate(
                                T.call_extern(
                                    "void",
                                    "tileops::conv_bias_bt",
                                    T.access_ptr(bl, "r"),
                                    bm,
                                    convert,
                                )
                            )
                        T.ascend_set_flag("MTE1_MTE2", 0)
                        T.ascend_set_flag("MTE1_MTE2", 1)
                        T.ascend_set_flag("M_MTE1", 0)
                        T.ascend_set_flag("M_MTE1", 1)
                        for t in T.serial(kt + 2):
                            if t < kt:
                                slot = t % 2
                                T.ascend_wait_flag("MTE1_MTE2", slot)
                                T.copy(W[t, mi : mi + bm, :], wl[slot, :, :])
                                T.ascend_set_flag("MTE2_MTE1", slot)
                            if t > 0 and t <= kt:
                                prev = t - 1
                                slot = prev % 2
                                T.ascend_wait_flag("MTE2_MTE1", slot)
                                T.ascend_wait_flag("M_MTE1", slot)
                                T.copy(wl[slot, :, :], b[slot, :, :])
                                T.evaluate(
                                    T.call_extern(
                                        "void",
                                        "tileops::conv_resident_img",
                                        T.access_ptr(a[slot, 0, 0], "w"),
                                        T.access_ptr(xl, "r"),
                                        fh,
                                        fw,
                                        left,
                                        right,
                                        bk,
                                        bn,
                                        ni % ow,
                                        prev * bk,
                                        sh,
                                        sw,
                                        kh,
                                        kw,
                                        dh,
                                        dw,
                                        ci,
                                    )
                                )
                                T.ascend_set_flag("MTE1_MTE2", slot)
                                T.ascend_set_flag("MTE1_M", slot)
                            if t > 1:
                                slot = (t - 2) % 2
                                T.ascend_wait_flag("MTE1_M", slot)
                                if has_bias and t == 2:
                                    T.evaluate(
                                        T.call_extern(
                                            "void",
                                            "tileops::conv_bias_mad",
                                            T.access_ptr(acc, "rw"),
                                            T.access_ptr(a[slot, 0, 0], "r"),
                                            T.access_ptr(b[slot, 0, 0], "r"),
                                            bn,
                                            bk,
                                            bm,
                                            T.uint64(0),
                                        )
                                    )
                                else:
                                    T.gemm(
                                        a[slot, :, :],
                                        b[slot, :, :],
                                        acc,
                                        transpose_B=True,
                                        clear_accum=t == 2,
                                    )
                                T.ascend_set_flag("M_MTE1", slot)
                        T.ascend_wait_flag("MTE1_MTE2", 0)
                        T.ascend_wait_flag("MTE1_MTE2", 1)
                        T.ascend_wait_flag("M_MTE1", 0)
                        T.ascend_wait_flag("M_MTE1", 1)
                        T.ascend_set_flag("M_FIX", 0)
                        T.ascend_wait_flag("M_FIX", 0)
                        T.evaluate(
                            T.call_extern(
                                "void",
                                "tileops::conv_native_store",
                                T.access_ptr(Y[bi, mi, ni], "w"),
                                T.access_ptr(acc, "r"),
                                bm,
                                T.min(bn, plane - ni),
                                plane,
                                bn,
                                quant,
                            )
                        )
                        T.ascend_set_flag("FIX_M", 0)
                        T.ascend_wait_flag("FIX_M", 0)

    return tilelang.compile(
        main,
        target="ascend",
        out_idx=-1,
        compile_flags=csrc_include("ascend_convolution.h"),
    )


def _streamed_supported(batch, ci, co, groups, size, out, window, stride, pad, dilation):
    if not (
        groups == 1
        and size[0] == out[0] == 1
        and window == (1, 3, 3)
        and ci >= 128
        and ci % 16 == 0
        and co >= 256
        and co % 256 == 0
        and out[1] * out[2] >= 8192
        and ci * 9 % 64 == 0
    ):
        return False
    mt = co // 256
    plane = out[1] * out[2]
    wanted = max((plane + 255) // 256, (32 + batch * mt - 1) // (batch * mt))
    wave = 32 // math.gcd(32, batch * mt)
    wanted = (wanted + wave - 1) // wave * wave
    bn = min(256, max(16, (plane + wanted * 16 - 1) // (wanted * 16) * 16))
    fh = ((out[2] - 1 + bn + out[2] - 1) // out[2] - 1) * stride[1] + 2 * dilation[1] + 1
    align = 16 // math.gcd(size[2], 16)
    fh = (fh + align - 1) // align * align
    return fh * size[2] * ci * 2 + 2 * 256 * 64 * 2 + 256 * 2 <= 512 * 1024
