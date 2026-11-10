"""Full-channel L1 residency for small-channel spatial convolutions."""

import functools
import math

import tilelang
import tilelang.ascend.language as T
from tilelang.layout import make_ascend_major_k_layout

from tileops._csrc import csrc_include


def _feature_supported(batch, ci, co, groups, size, out, window, stride, dilation):
    if not (
        groups == 1
        and size[0] == out[0] == 1
        and window == (1, 5, 5)
        and ci in (16, 32, 64)
        and co <= 128
        and co % 16 == 0
        and batch <= 32
    ):
        return False
    plane = out[1] * out[2]
    wanted = max((plane + 191) // 192, (32 + batch - 1) // batch)
    bn = min(192, max(16, (plane + wanted * 16 - 1) // (wanted * 16) * 16))
    fh = ((out[2] - 1 + bn + out[2] - 1) // out[2] - 1) * stride[1] + 4 * dilation[1] + 1
    align = 16 // math.gcd(size[2], 16)
    fh = (fh + align - 1) // align * align
    return fh * size[2] * ci * 2 + co * ci * 25 * 2 + co * 2 <= 512 * 1024


@functools.lru_cache(maxsize=128)
def _feature_contract(
    batch,
    ci,
    co,
    groups,
    bundle,
    size,
    out,
    window,
    stride,
    pad,
    dilation,
    dtype,
    has_bias=False,
):
    assert (
        groups == bundle == 1
        and size[0] == out[0] == window[0] == 1
        and ci % 16 == 0
        and co % 16 == 0
    )
    _, h, w = size
    _, oh, ow = out
    _, kh, kw = window
    _, sh, sw = stride
    _, ph, pw = pad
    _, dh, dw = dilation
    packed = ci * kh * kw
    # Choose a divisor to avoid a partially initialized final L0 fractal.
    bk = 48 if kh == 1 and kw == 3 else 80 if kh == kw == 5 else kh * kw * 16
    assert packed % bk == 0
    bm = min(256 if kh == 1 else 128, co, 1 << ((16384 // bk).bit_length() - 1))
    bn = min(256, (16384 // bk) // 16 * 16)
    plane = oh * ow
    mt = (co + bm - 1) // bm
    wanted = max((plane + bn - 1) // bn, (32 + batch * mt - 1) // (batch * mt))
    bn = min(bn, max(16, (plane + wanted * 16 - 1) // (wanted * 16) * 16))
    nt = (plane + bn - 1) // bn
    assert batch * mt <= 32
    parts = min(nt, 32 // (batch * mt))
    cores = batch * mt * parts
    one_d = h == oh == kh == 1
    if one_d:
        fh = 1
        fw = ((bn - 1) * sw + (kw - 1) * dw + 1 + 15) // 16 * 16
        left = right = 0
    else:
        fw = w
        fh = ((ow - 1 + bn + ow - 1) // ow - 1) * sh + (kh - 1) * dh + 1
        align = 16 // math.gcd(w, 16)
        fh = (fh + align - 1) // align * align
        left = pw
        right = max(0, (ow - 1) * sw + (kw - 1) * dw + 1 - w - pw)
    assert fh * fw * ci * 2 + bm * packed * 2 <= 512 * 1024
    assert bm * bk * 4 <= 65536 and bn * bk * 4 <= 65536
    assert co % bm == 0
    convert = 1 if dtype == "float16" else 0
    quant = 16 if dtype == "bfloat16" else 1

    @T.prim_func
    def main(
        X: T.Tensor((batch, ci, h * w), dtype),
        W: T.Tensor((co, packed), dtype),
        Bias: T.Tensor((co,), dtype),
        Y: T.Tensor((batch, co, plane), dtype),
    ):
        with T.Kernel(cores) as core:
            xl = T.alloc_l1((fh * fw, ci), dtype)
            wl = T.alloc_l1((bm, packed), dtype)
            bl = T.alloc_l1((bm,), dtype)
            a = T.alloc_l0a((2, bn, bk), dtype)
            b = T.alloc_l0b((2, bm, bk), dtype)
            acc = T.alloc_l0c((bn, bm), "float32")
            T.annotate_layout(
                {xl: make_ascend_major_k_layout(xl), b: make_ascend_major_k_layout(b)}
            )
            T.annotate_buffer_versions({xl: 1, wl: 1, a: 1, b: 1, acc: 1, bl: 1})
            with T.Cube():
                mi = core // parts % mt * bm
                bi = core // (parts * mt)
                if mi + bm > co:
                    T.fill(wl, T.cast(0, dtype))
                    T.ascend_pipe_barrier("PIPE_MTE2")
                T.copy(W[mi : mi + T.min(bm, co - mi), :], wl[: T.min(bm, co - mi), :])
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
                            "void",
                            "tileops::conv_bias_bt",
                            T.access_ptr(bl, "r"),
                            bm,
                            convert,
                        )
                    )
                for step in T.serial(T.ceildiv(nt, parts)):
                    ni = (step * parts + core % parts) * bn
                    if ni < plane:
                        if one_d:
                            base = ni * sw - pw
                            begin = T.max(0, base)
                            count = T.max(0, T.min(w, base + fw) - begin)
                            if count < fw:
                                T.fill(xl, T.cast(0, dtype))
                                T.ascend_pipe_barrier("PIPE_MTE2")
                            if count > 0:
                                T.copy(
                                    X[bi, :, begin : begin + count],
                                    xl[begin - base : begin - base + count, :],
                                    transpose=True,
                                )
                        else:
                            base = ni // ow * sh - ph
                            begin = T.max(0, base)
                            count = T.max(0, T.min(h, base + fh) - begin)
                            if count < fh:
                                T.fill(xl, T.cast(0, dtype))
                                T.ascend_pipe_barrier("PIPE_MTE2")
                            if count > 0:
                                T.copy(
                                    X[bi, :, begin * w : (begin + count) * w],
                                    xl[
                                        (begin - base) * w : (begin - base + count) * w,
                                        :,
                                    ],
                                    transpose=True,
                                )
                        T.ascend_set_flag("MTE2_MTE1", 3)
                        T.ascend_wait_flag("MTE2_MTE1", 3)
                        T.ascend_set_flag("M_MTE1", 0)
                        T.ascend_set_flag("M_MTE1", 1)
                        for t in T.serial(packed // bk + 1):
                            if t < packed // bk:
                                slot = t % 2
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
                                        0 if one_d else ni % ow,
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
                            if t > 0:
                                slot = (t - 1) % 2
                                T.ascend_wait_flag("MTE1_M", slot)
                                if has_bias and t == 1:
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
                                        clear_accum=t == 1,
                                    )
                                T.ascend_set_flag("M_MTE1", slot)
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
        main,
        target="ascend",
        out_idx=-1,
        compile_flags=csrc_include("ascend_convolution.h"),
    )
