"""Full-channel downsampling with fused SIMD weight packing."""

import functools
import math

import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S
from tilelang.layout import make_ascend_major_k_layout

from tileops._csrc import csrc_include


@functools.lru_cache(maxsize=128)
def _small_downsample_contract(
    batch, ci, co, groups, bundle, size, out, window, stride, pad, dilation, dtype, has_bias=False
):
    assert batch == groups == bundle == 1 and ci == co == 128 and (window == (1, 3, 3))
    (_, h, w) = size
    (_, oh, ow) = out
    (_, kh, kw) = window
    (_, sh, sw) = stride
    (_, ph, pw) = pad
    (_, dh, dw) = dilation
    packed = ci * kh * kw
    bk = 576
    bm = 32
    bn = 48
    plane = oh * ow
    mt = co // bm
    nt = (plane + bn - 1) // bn
    parts = min(nt, 32 // mt)
    cores = mt * parts
    fw = w
    fh = ((ow - 1 + bn + ow - 1) // ow - 1) * sh + (kh - 1) * dh + 1
    align = 16 // math.gcd(w, 16)
    fh = (fh + align - 1) // align * align
    left = pw
    right = max(0, (ow - 1) * sw + (kw - 1) * dw + 1 - w - pw)
    assert fh * fw * ci * 2 + bm * packed * 2 + bm * 2 <= 512 * 1024
    assert (2 * bm + 4) * packed <= 192 * 1024
    convert = 1 if dtype == "float16" else 0
    quant = 16 if dtype == "bfloat16" else 1

    @T.prim_func
    def main(
        X: T.Tensor((batch, ci, h * w), dtype),
        W: T.Tensor((co, packed), dtype),
        Bias: T.Tensor((co,), dtype),
        Y: T.Tensor((batch, co, plane), dtype),
    ):
        with T.MixedKernel(cores, sids=2) as (core, sid):
            xl = T.alloc_l1((fh * fw, ci), dtype)
            wl = T.alloc_l1((bm, packed), dtype)
            bl = T.alloc_l1((bm,), dtype)
            a = T.alloc_l0a((1, bn, bk), dtype)
            b = T.alloc_l0b((1, bm, bk), dtype)
            acc = T.alloc_l0c((bn, bm), "float32")
            T.annotate_layout(
                {xl: make_ascend_major_k_layout(xl), b: make_ascend_major_k_layout(b)}
            )
            T.annotate_buffer_versions({xl: 1, wl: 1, a: 1, b: 1, acc: 1, bl: 1})
            pitch = (packed + 127) // 128 * 128
            raw = T.alloc_shared((bm // 2, packed), dtype)
            reordered = T.alloc_shared((packed // 16, bm // 2 + 1, 16), dtype)
            index = T.alloc_shared((pitch,), "uint16")
            mi = core // parts % mt * bm
            bi = core // (parts * mt)
            with T.Cube():
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
                    T.ascend_set_flag("MTE2_MTE1", 2)
                    T.ascend_wait_flag("MTE2_MTE1", 2)
                    T.evaluate(
                        T.call_extern(
                            "void", "tileops::conv_bias_bt", T.access_ptr(bl, "r"), bm, convert
                        )
                    )
                ni = core % parts * bn
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
                T.ascend_set_flag("MTE2_MTE1", 3)
            with T.SimtVF(threads=256):
                for j in T.Parallel(pitch, coalesced_width=T.int32(1)):
                    channel = j // (kh * kw * 16) * 16 + j % 16
                    tap = j // 16 % (kh * kw)
                    index[j] = T.cast(
                        T.if_then_else(j < packed, channel * kh * kw + tap, 0), "uint16"
                    )
            T.dual_copy(W[mi : mi + bm, :], raw)
            with T.SimdVF():
                for part in T.serial(pitch // 128):
                    remaining = T.cast(T.min(128, packed - part * 128), "uint32")
                    mask = S.update_mask(remaining, 16)
                    offsets = S.vld(index[part * 128])
                    dst = S.make_ubuf_ptr(reordered[part * 8, 0, 0], dtype)
                    for row in T.serial(bm // 2):
                        value = S.vgather2(raw[row, 0], offsets, mask)
                        dst = S.vsstb(value, dst, T.int32(bm // 2 + 1 << 16 | 1), mask, update=True)
            T.evaluate(
                T.call_intrin(
                    "void",
                    tilelang.tvm.ir.Op.get("tl.ascend_nd2nz_post_copy"),
                    T.access_ptr(wl[sid * (bm // 2), 0], "w"),
                    T.access_ptr(reordered, "r"),
                    bm // 2,
                    packed,
                    bm,
                    "half",
                )
            )
            with T.Cube():
                for step in T.serial(T.ceildiv(nt, parts)):
                    ni = (step * parts + core % parts) * bn
                    if ni < plane:
                        if step > 0:
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
                            T.ascend_set_flag("MTE2_MTE1", 3)
                        T.ascend_wait_flag("MTE2_MTE1", 3)
                        T.ascend_set_flag("M_MTE1", 0)
                        for t in T.serial(packed // bk):
                            slot = T.int32(0)
                            T.ascend_wait_flag("M_MTE1", slot)
                            T.copy(wl[:, t * bk : t * bk + bk], b[slot, :, :])
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
                                    t * bk,
                                    sh,
                                    sw,
                                    kh,
                                    kw,
                                    dh,
                                    dw,
                                    ci,
                                )
                            )
                            T.ascend_set_flag("MTE1_M", slot)
                            slot = T.int32(0)
                            T.ascend_wait_flag("MTE1_M", slot)
                            if has_bias and t == 0:
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
                                    clear_accum=t == 0,
                                )
                            T.ascend_set_flag("M_MTE1", slot)
                        T.ascend_wait_flag("M_MTE1", 0)
                        T.ascend_set_flag("M_FIX", 0)
                        T.ascend_wait_flag("M_FIX", 0)
                        T.evaluate(
                            T.call_extern(
                                "void",
                                "tileops::conv_native_store",
                                T.access_ptr(Y[bi, mi, ni], "w"),
                                T.access_ptr(acc, "r"),
                                T.min(bm, co - mi),
                                T.min(bn, plane - ni),
                                plane,
                                bn,
                                quant,
                            )
                        )
                        T.ascend_set_flag("FIX_M", 0)
                        T.ascend_wait_flag("FIX_M", 0)
                        T.ascend_set_flag("MTE1_MTE2", 3)
                        T.ascend_wait_flag("MTE1_MTE2", 3)

    return tilelang.compile(
        main, target="ascend", out_idx=-1, compile_flags=csrc_include("ascend_convolution.h")
    )


def _small_downsample_supported(batch, ci, co, groups, size, out, window, stride, dilation):
    if not (
        batch == groups == 1
        and ci == co == 128
        and size[0] == out[0] == 1
        and window == (1, 3, 3)
        and stride == (1, 2, 2)
        and dilation == (1, 1, 1)
        and size[2] % 4 == 0
        and 64 <= out[1] * out[2] <= 512
    ):
        return False
    fh = ((out[2] - 1 + 48 + out[2] - 1) // out[2] - 1) * 2 + 3
    align = 16 // math.gcd(size[2], 16)
    fh = (fh + align - 1) // align * align
    return fh * size[2] * ci * 2 + 32 * ci * 9 * 2 + 32 * 2 <= 512 * 1024
