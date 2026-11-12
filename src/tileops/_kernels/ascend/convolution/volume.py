"""Fused weight reordering with overlapped full-channel feature loads."""

import functools
import math

import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S
from tilelang.layout import make_ascend_major_k_layout

from tileops._csrc import csrc_include


@functools.lru_cache(maxsize=128)
def _volume_fused_contract(
    batch, ci, co, groups, bundle, size, out, window, stride, pad, dilation, dtype, has_bias=False
):
    assert batch == groups == bundle == 1 and ci % 16 == 0 and co % 16 == 0 and co <= 128
    d, h, w = size
    od, oh, ow = out
    kd, kh, kw = window
    sd, sh, sw = stride
    pd, ph, pw = pad
    dd, dh, dw = dilation
    assert kh == kw == 3 and ci <= 64
    per_depth = ci * kh * kw
    bk = 32
    assert per_depth % bk == 0
    bm = co
    bn = min(512, (16384 // bk) // 16 * 16)
    packed = kd * per_depth
    assert bm * bk * 4 <= 65536
    plane = oh * ow
    nt = (plane + bn - 1) // bn
    tiles = od * nt
    cores = min(32, tiles)
    steps = (tiles + cores - 1) // cores
    fw = w
    fh = ((ow - 1 + bn + ow - 1) // ow - 1) * sh + (kh - 1) * dh + 1
    align = 16 // math.gcd(w, 16)
    fh = (fh + align - 1) // align * align
    left = pw
    right = max(0, (ow - 1) * sw + (kw - 1) * dw + 1 - w - pw)
    assert 2 * kd * fh * fw * ci * 2 + bm * packed * 2 + bm * 2 <= 512 * 1024
    quant = 16 if dtype == "bfloat16" else 1
    convert = 0 if dtype == "bfloat16" else 1

    @T.prim_func
    def main(
        X: T.Tensor((batch, ci, d * h * w), dtype),
        W: T.Tensor((co, packed), dtype),
        Bias: T.Tensor((co,), dtype),
        Y: T.Tensor((batch, co, od * plane), dtype),
    ):
        with T.MixedKernel(
            cores,
            sids=2,
        ) as (core, sid):
            xl = T.alloc_l1((2, kd, fh * fw, ci), dtype)
            wl = T.alloc_l1((bm, packed), dtype)
            bl = T.alloc_l1((bm,), dtype)
            a = T.alloc_l0a((2, bn, bk), dtype)
            b = T.alloc_l0b((2, bm, bk), dtype)
            acc = T.alloc_l0c((bn, bm), "float32")
            T.annotate_layout(
                {xl: make_ascend_major_k_layout(xl), b: make_ascend_major_k_layout(b)}
            )
            T.annotate_buffer_versions({xl: 1, wl: 1, bl: 1, a: 1, b: 1, acc: 1})
            pitch = (packed + 127) // 128 * 128
            raw = T.alloc_shared((bm // 2, packed), dtype)
            reordered = T.alloc_shared((bm // 2, packed), dtype)
            gather_index = T.alloc_shared((pitch,), "uint16")
            with T.SimtVF(threads=256):
                for j in T.Parallel(pitch, coalesced_width=T.int32(1)):
                    iz = j // per_depth
                    channel = j // (kh * kw * 16) % (ci // 16) * 16 + j % 16
                    tap = j // 16 % (kh * kw)
                    gather_index[j] = T.cast(
                        T.if_then_else(j < packed, channel * kd * kh * kw + iz * kh * kw + tap, 0),
                        "uint16",
                    )
            with T.Cube():
                index = core
                slot = T.int32(0)
                ni = index % nt * bn
                oz = index // nt
                for kz in T.serial(kd):
                    iz = oz * sd - pd + kz * dd
                    base = ni // ow * sh - ph
                    begin = T.max(0, base)
                    count = T.max(0, T.min(h, base + fh) - begin)
                    if iz < 0 or iz >= d or count < fh:
                        T.fill(xl[slot, kz, :, :], T.cast(0, dtype))
                        T.ascend_pipe_barrier("PIPE_MTE2")
                    if iz >= 0 and iz < d and count > 0:
                        T.copy(
                            X[0, :, iz * h * w + begin * w : iz * h * w + (begin + count) * w],
                            xl[slot, kz, (begin - base) * w : (begin - base + count) * w, :],
                            transpose=True,
                        )
                T.ascend_set_flag("MTE2_MTE1", slot)
            T.dual_copy(W, raw)
            with T.SimdVF():
                for part in T.serial(pitch // 128):
                    remaining = T.cast(T.min(128, packed - part * 128), "uint32")
                    mask = S.update_mask(remaining, 16)
                    offsets = S.vld(gather_index[part * 128])
                    for row in T.serial(bm // 2):
                        value = S.vgather2(raw[row, 0], offsets, mask)
                        S.vsts(reordered[row, part * 128], value, mask, "NORM_B16")
            T.dual_copy(reordered, wl)
            with T.Cube():
                if has_bias:
                    T.evaluate(
                        T.call_extern(
                            "void",
                            "tileops::conv_bias_gm",
                            T.access_ptr(bl, "w"),
                            T.access_ptr(Bias, "r"),
                            bm,
                        )
                    )
                    T.ascend_set_flag("MTE2_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 3)
                    T.evaluate(
                        T.call_extern(
                            "void", "tileops::conv_bias_bt", T.access_ptr(bl, "r"), bm, convert
                        )
                    )
                T.ascend_set_flag("MTE1_MTE2", 1)
                for step in T.serial(steps + 1):
                    if step > 0 and step < steps:
                        index = step * cores + core
                        if index < tiles:
                            slot = step % 2
                            ni = index % nt * bn
                            oz = index // nt
                            T.ascend_wait_flag("MTE1_MTE2", slot)
                            for kz in T.serial(kd):
                                iz = oz * sd - pd + kz * dd
                                base = ni // ow * sh - ph
                                begin = T.max(0, base)
                                count = T.max(0, T.min(h, base + fh) - begin)
                                if iz < 0 or iz >= d or count < fh:
                                    T.fill(xl[slot, kz, :, :], T.cast(0, dtype))
                                    T.ascend_pipe_barrier("PIPE_MTE2")
                                if iz >= 0 and iz < d and count > 0:
                                    T.copy(
                                        X[
                                            0,
                                            :,
                                            iz * h * w + begin * w : iz * h * w
                                            + (begin + count) * w,
                                        ],
                                        xl[
                                            slot,
                                            kz,
                                            (begin - base) * w : (begin - base + count) * w,
                                            :,
                                        ],
                                        transpose=True,
                                    )
                            T.ascend_set_flag("MTE2_MTE1", slot)
                    if step > 0:
                        index = (step - 1) * cores + core
                        if index < tiles:
                            prev = (step - 1) % 2
                            ni = index % nt * bn
                            oz = index // nt
                            T.ascend_wait_flag("MTE2_MTE1", prev)
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
                                            T.access_ptr(
                                                xl[prev, t // (per_depth // bk), 0, 0], "r"
                                            ),
                                            fh,
                                            fw,
                                            left,
                                            right,
                                            bk,
                                            bn,
                                            ni % ow,
                                            t % (per_depth // bk) * bk,
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
                            T.ascend_set_flag("MTE1_MTE2", prev)
                            T.ascend_set_flag("M_FIX", 0)
                            T.ascend_wait_flag("M_FIX", 0)
                            T.evaluate(
                                T.call_extern(
                                    "void",
                                    "tileops::conv_native_store",
                                    T.access_ptr(Y[0, 0, oz * plane + ni], "w"),
                                    T.access_ptr(acc, "r"),
                                    co,
                                    T.min(bn, plane - ni),
                                    od * plane,
                                    bn,
                                    quant,
                                )
                            )
                            T.ascend_set_flag("FIX_M", 0)
                            T.ascend_wait_flag("FIX_M", 0)
                T.ascend_wait_flag("MTE1_MTE2", 0)
                T.ascend_wait_flag("MTE1_MTE2", 1)

    return tilelang.compile(
        main, target="ascend", out_idx=-1, compile_flags=csrc_include("ascend_convolution.h")
    )


def _volume_fused_supported(batch, ci, co, groups, size, out, window, stride, dilation):
    if not (
        batch == groups == 1
        and ci == 32
        and co == 64
        and window == (3, 3, 3)
        and stride == dilation == (1, 1, 1)
        and out[0] >= 8
        and out[1] * out[2] >= 4096
    ):
        return False
    bn = 512
    fh = ((out[2] - 1 + bn + out[2] - 1) // out[2] - 1) + 3
    align = 16 // math.gcd(size[2], 16)
    fh = (fh + align - 1) // align * align
    return 2 * 3 * fh * size[2] * ci * 2 + co * ci * 27 * 2 + co * 2 <= 512 * 1024
