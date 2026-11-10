"""Channel-blocked downsampling with fused SIMD weight packing."""

import functools
import math

import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S
from tilelang.layout import make_ascend_major_k_layout

from tileops._csrc import csrc_include


@functools.lru_cache(maxsize=128)
def _chunked_downsample_contract(
    batch, ci, co, groups, bundle, size, out, window, stride, pad, dilation, dtype, has_bias=False
):
    bm = 64
    bn = 112
    assert groups == bundle == 1 and ci == 128 and co == 256 and window == (1, 3, 3)
    assert (
        batch == 1
        and size[0] == out[0] == window[0] == 1
        and ci % 16 == 0
        and co % bm == 0
        and bm % 32 == 0
    )
    _, h, w = size
    _, oh, ow = out
    _, kh, kw = window
    _, sh, sw = stride
    _, ph, pw = pad
    _, dh, dw = dilation
    packed = ci * kh * kw
    bk = kh * kw * 32
    cb = ci // 32
    chunk_m = 32
    assert bk * bm * 2 <= 65536 and bk * bn * 2 <= 65536
    plane = oh * ow
    mt = co // bm
    nt = (plane + bn - 1) // bn
    parts = min(nt, 32 // mt)
    cores = mt * parts
    fh = ((ow - 1 + bn + ow - 1) // ow - 1) * sh + (kh - 1) * dh + 1
    align = 16 // math.gcd(w, 16)
    fh = (fh + align - 1) // align * align
    fw = w
    left = pw
    right = max(0, (ow - 1) * sw + (kw - 1) * dw + 1 - w - pw)
    assert 2 * fh * fw * 32 * 2 + bm * packed * 2 + bm * 2 <= 512 * 1024
    assert chunk_m * packed * 3 + 2 * ((packed + 127) // 128 * 128) <= 192 * 1024
    quant = 16 if dtype == "bfloat16" else 1
    convert = 0 if dtype == "bfloat16" else 1

    @T.prim_func
    def main(
        X: T.Tensor((batch, ci, h * w), dtype),
        W: T.Tensor((co, packed), dtype),
        Bias: T.Tensor((co,), dtype),
        Y: T.Tensor((batch, co, plane), dtype),
    ):
        with T.MixedKernel(cores, sids=2) as (core, sid):
            xl = T.alloc_l1((2, fh * fw, 32), dtype)
            wl = T.alloc_l1((bm, packed), dtype)
            bl = T.alloc_l1((bm,), dtype)
            a = T.alloc_l0a((bn, bk), dtype)
            b = T.alloc_l0b((bm, bk), dtype)
            acc = T.alloc_l0c((bn, bm), "float32")
            T.annotate_layout(
                {xl: make_ascend_major_k_layout(xl), b: make_ascend_major_k_layout(b)}
            )
            T.annotate_buffer_versions({xl: 1, wl: 1, bl: 1, a: 1, b: 1, acc: 1})
            pitch = (packed + 127) // 128 * 128
            raw = T.alloc_shared((chunk_m // 2, packed), dtype)
            reordered = T.alloc_shared((packed // 16, chunk_m // 2 + 1, 16), dtype)
            index = T.alloc_shared((pitch,), "uint16")
            mi = core // parts * bm
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
                    T.fill(xl[0, :, :], T.cast(0, dtype))
                    T.ascend_pipe_barrier("PIPE_MTE2")
                if count > 0:
                    T.copy(
                        X[0, :32, begin * w : (begin + count) * w],
                        xl[0, (begin - base) * w : (begin - base + count) * w, :],
                        transpose=True,
                    )
                T.ascend_set_flag("MTE2_MTE1", 0)
            with T.SimtVF(threads=256):
                for j in T.Parallel(pitch, coalesced_width=T.int32(1)):
                    channel = j // (kh * kw * 16) * 16 + j % 16
                    tap = j // 16 % (kh * kw)
                    index[j] = T.cast(
                        T.if_then_else(j < packed, channel * kh * kw + tap, 0), "uint16"
                    )
            for chunk in T.serial(bm // chunk_m):
                T.dual_copy(W[mi + chunk * chunk_m : mi + (chunk + 1) * chunk_m, :], raw)
                with T.SimdVF():
                    for part in T.serial(pitch // 128):
                        mask = S.update_mask(T.cast(T.min(128, packed - part * 128), "uint32"), 16)
                        offsets = S.vld(index[part * 128])
                        dst = S.make_ubuf_ptr(reordered[part * 8, 0, 0], dtype)
                        for row in T.serial(chunk_m // 2):
                            value = S.vgather2(raw[row, 0], offsets, mask)
                            dst = S.vsstb(
                                value,
                                dst,
                                T.int32(((chunk_m // 2 + 1) << 16) | 1),
                                mask,
                                update=True,
                            )
                T.evaluate(
                    T.call_intrin(
                        "void",
                        tilelang.tvm.ir.Op.get("tl.ascend_nd2nz_post_copy"),
                        T.access_ptr(wl[chunk * chunk_m + sid * (chunk_m // 2), 0], "w"),
                        T.access_ptr(reordered, "r"),
                        chunk_m // 2,
                        packed,
                        bm,
                        "half",
                    )
                )
            with T.Cube():
                for step in T.serial(T.ceildiv(nt, parts)):
                    ni = (step * parts + core % parts) * bn
                    if ni < plane:
                        base = ni // ow * sh - ph
                        begin = T.max(0, base)
                        count = T.max(0, T.min(h, base + fh) - begin)
                        if step > 0:
                            if count < fh:
                                T.fill(xl[0, :, :], T.cast(0, dtype))
                                T.ascend_pipe_barrier("PIPE_MTE2")
                            if count > 0:
                                T.copy(
                                    X[0, :32, begin * w : (begin + count) * w],
                                    xl[0, (begin - base) * w : (begin - base + count) * w, :],
                                    transpose=True,
                                )
                            T.ascend_set_flag("MTE2_MTE1", 0)
                        T.ascend_set_flag("MTE1_MTE2", 1)
                        T.ascend_set_flag("M_MTE1", 2)
                        for t in T.serial(1, cb + 1):
                            if t < cb:
                                slot = t % 2
                                T.ascend_wait_flag("MTE1_MTE2", slot)
                                if count < fh:
                                    T.fill(xl[slot, :, :], T.cast(0, dtype))
                                    T.ascend_pipe_barrier("PIPE_MTE2")
                                if count > 0:
                                    T.copy(
                                        X[0, t * 32 : t * 32 + 32, begin * w : (begin + count) * w],
                                        xl[
                                            slot, (begin - base) * w : (begin - base + count) * w, :
                                        ],
                                        transpose=True,
                                    )
                                T.ascend_set_flag("MTE2_MTE1", slot)
                            prev = t - 1
                            slot = prev % 2
                            T.ascend_wait_flag("MTE2_MTE1", slot)
                            T.ascend_wait_flag("M_MTE1", 2)
                            T.copy(wl[:, prev * bk : prev * bk + bk], b)
                            T.evaluate(
                                T.call_extern(
                                    "void",
                                    "tileops::conv_native_load3d",
                                    T.access_ptr(a, "w"),
                                    T.access_ptr(xl[slot, 0, 0], "r"),
                                    fh,
                                    fw,
                                    left,
                                    right,
                                    bk,
                                    bn,
                                    ni % ow,
                                    sh,
                                    sw,
                                    kh,
                                    kw,
                                    dh,
                                    dw,
                                    32,
                                )
                            )
                            T.ascend_set_flag("MTE1_MTE2", slot)
                            T.ascend_set_flag("MTE1_M", 2)
                            T.ascend_wait_flag("MTE1_M", 2)
                            if has_bias and prev == 0:
                                T.evaluate(
                                    T.call_extern(
                                        "void",
                                        "tileops::conv_bias_mad",
                                        T.access_ptr(acc, "rw"),
                                        T.access_ptr(a, "r"),
                                        T.access_ptr(b, "r"),
                                        bn,
                                        bk,
                                        bm,
                                        T.uint64(0),
                                    )
                                )
                            else:
                                T.gemm(a, b, acc, transpose_B=True, clear_accum=prev == 0)
                            T.ascend_set_flag("M_MTE1", 2)
                        T.ascend_wait_flag("MTE1_MTE2", 0)
                        T.ascend_wait_flag("MTE1_MTE2", 1)
                        T.ascend_wait_flag("M_MTE1", 2)
                        T.ascend_set_flag("M_FIX", 0)
                        T.ascend_wait_flag("M_FIX", 0)
                        T.evaluate(
                            T.call_extern(
                                "void",
                                "tileops::conv_native_store",
                                T.access_ptr(Y[0, mi, ni], "w"),
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
        main, target="ascend", out_idx=-1, compile_flags=csrc_include("ascend_convolution.h")
    )


def _chunked_downsample_supported(batch, ci, co, groups, size, out, window, stride, dilation):
    if not (
        batch == groups == 1
        and ci == 128
        and co == 256
        and size[0] == out[0] == 1
        and window == (1, 3, 3)
        and stride == (1, 2, 2)
        and dilation == (1, 1, 1)
        and size[2] % 4 == 0
        and 512 <= out[1] * out[2] <= 2048
    ):
        return False
    fh = ((out[2] - 1 + 112 + out[2] - 1) // out[2] - 1) * 2 + 3
    align = 16 // math.gcd(size[2], 16)
    fh = (fh + align - 1) // align * align
    return 2 * fh * size[2] * 32 * 2 + 64 * ci * 9 * 2 + 64 * 2 <= 512 * 1024
