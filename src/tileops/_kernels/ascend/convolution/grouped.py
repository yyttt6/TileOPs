"""Depth-folded grouped convolutions with four-plane accumulation."""

import functools

import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S
from tilelang.layout import make_ascend_major_k_layout

from tileops._csrc import csrc_include


@functools.lru_cache(maxsize=128)
def _grouped_folded_contract(
    batch, ci, co, groups, bundle, size, out, window, stride, pad, dilation, dtype, has_bias=False
):
    assert not has_bias and bundle == 1
    assert _grouped_folded_supported(batch, ci, co, groups, size, out, window, stride, dilation)
    assert pad == (1, 1, 1)
    (depth, height, width) = size
    wpitch = 32
    dp = depth + 2
    plane = height * width
    fh = (height + 2 + 31) // 32 * 32
    positions = fh * width
    half = positions // 2
    bm = 16
    bn = 192
    bk = 80
    nt = (plane + bn - 1) // bn
    quant = 16 if dtype == "bfloat16" else 1

    @T.prim_func
    def main(
        X: T.Tensor((batch, ci, depth * height * width), dtype),
        W: T.Tensor((co, 54), dtype),
        UnusedBias: T.Tensor((co,), dtype),
        Y: T.Tensor((batch, co, depth * plane), dtype),
    ):
        with T.Kernel(groups) as core:
            volume = T.reshape(X, (batch, ci, depth, height, width))
            raww = T.alloc_shared((4, 64), dtype)
            wp = T.alloc_shared((bk // 16, bm // 2 + 1, 16), dtype)
            folded = T.alloc_shared((depth * half, 8), dtype)
            xl = T.alloc_l1((depth * positions, 8), dtype)
            wl = T.alloc_l1((bm, bk), dtype)
            a = T.alloc_l0a((2, bn, bk), dtype)
            b = T.alloc_l0b((bm, bk), dtype)
            acc = T.alloc_l0c((4 * plane, bm), "float32")
            T.annotate_layout({b: make_ascend_major_k_layout(b)})
            T.annotate_buffer_versions(
                {raww: 1, wp: 1, wl: 1, a: 1, b: 1, acc: 1, xl: 1, folded: 1}
            )
            with T.Vector() as sid:
                T.copy(W[core * 4 : core * 4 + 4, :], raww[:, :54])
                T.ascend_set_flag("MTE2_V", 7)
                T.ascend_wait_flag("MTE2_V", 7)
                with T.SimdVF():
                    full = S.pset(16)
                    zero = S.vdup(T.cast(0, dtype), dtype, full)
                    lanes = T.reinterpret("uint16x128", S.vci(T.int16(0), "int16"))
                    rows = S.vshrs(lanes, T.uint16(4), full)
                    low = S.vand(lanes, S.vdup(T.uint16(15), "uint16"), full)
                    channels = S.vand(lanes, S.vdup(T.uint16(7), "uint16"), full)
                    base = S.vadd(
                        S.vadd(
                            S.vmuls(rows, T.uint16(64), full),
                            S.vmuls(channels, T.uint16(9), full),
                            full,
                        ),
                        S.vshrs(low, T.uint16(3), full),
                        full,
                    )
                    valid_rows = S.vcmps(
                        S.vadds(rows, T.uint16(sid * 8), full), T.uint16(4), full, op="lt"
                    )
                    valid_channels = S.vcmps(channels, T.uint16(6), full, op="lt")
                    valid_rc = S.pand(valid_rows, valid_channels, full)
                    zero_index = S.vdup(T.uint16(0), "uint16", full)
                    for cb in T.serial(5):
                        valid_k = S.vcmps(
                            S.vadds(low, T.uint16(cb * 16), full), T.uint16(72), full, op="lt"
                        )
                        valid = S.pand(valid_rc, valid_k, full)
                        offsets = S.vsel(S.vadds(base, T.uint16(cb * 2), full), zero_index, valid)
                        values = S.vgather2(raww[0, 0], offsets, full)
                        values = S.vsel(values, zero, valid)
                        S.vsts(wp[cb, 0, 0], values, full, "NORM_B16")
                T.ascend_set_flag("V_MTE3", 7)
                T.ascend_wait_flag("V_MTE3", 7)
                T.evaluate(
                    T.call_intrin(
                        "void",
                        tilelang.tvm.ir.Op.get("tl.ascend_nd2nz_post_copy"),
                        T.access_ptr(wl[sid * (bm // 2), 0], "w"),
                        T.access_ptr(wp, "r"),
                        bm // 2,
                        bk,
                        bm,
                        "half",
                    )
                )
                T.ascend_set_flag("MTE3_V", 7)
                T.ascend_wait_flag("MTE3_V", 7)
                rawx = T.alloc_shared((2, dp, fh, wpitch), dtype)
                with T.SimdVF():
                    zero_mask = S.pset(16)
                    zero = S.vdup(T.cast(0, dtype), dtype, zero_mask)
                    for block in T.serial(2 * dp * fh * wpitch // 128):
                        S.vsts(rawx[0, 0, 0, block * 128], zero, zero_mask, "NORM_B16")
                T.ascend_set_flag("V_MTE2", 7)
                T.ascend_wait_flag("V_MTE2", 7)
                for ic in T.serial(2):
                    for iz in T.serial(depth):
                        T.copy(
                            volume[0, core * 2 + ic, iz, :, :],
                            rawx[ic, iz + 1, 1 : height + 1, :width],
                        )
                T.ascend_set_flag("MTE2_V", 7)
                T.ascend_wait_flag("MTE2_V", 7)
                with T.SimdVF():
                    full = S.pset(16)
                    zero = S.vdup(T.cast(0, dtype), dtype, full)
                    for local_z in T.serial(depth // 2):
                        oz = sid * (depth // 2) + local_z
                        for chunk in T.serial(positions // 128):
                            pos = T.reinterpret("uint16x128", S.vci(T.int16(chunk * 128), "int16"))
                            quarter = S.vshrs(pos, T.uint16(2), full)
                            row = S.vshrs(
                                S.vmuls(S.vadds(quarter, T.uint16(1), full), T.uint16(73), full),
                                T.uint16(9),
                                full,
                            )
                            offsets = S.vadd(pos, S.vmuls(row, T.uint16(4), full), full)
                            v0 = S.vgather2(rawx[0, oz + 0, 0, 0], offsets, full)
                            v1 = S.vgather2(rawx[0, oz + 1, 0, 0], offsets, full)
                            v2 = S.vgather2(rawx[0, oz + 2, 0, 0], offsets, full)
                            v3 = S.vgather2(rawx[1, oz + 0, 0, 0], offsets, full)
                            v4 = S.vgather2(rawx[1, oz + 1, 0, 0], offsets, full)
                            v5 = S.vgather2(rawx[1, oz + 2, 0, 0], offsets, full)
                            (p0, p1) = S.vintlv(v0, v4)
                            (p2, p3) = S.vintlv(v1, v5)
                            (p4, p5) = S.vintlv(v2, zero)
                            (p6, p7) = S.vintlv(v3, zero)
                            (q0, q1) = S.vintlv(p0, p4)
                            (q2, q3) = S.vintlv(p2, p6)
                            (q4, q5) = S.vintlv(p1, p5)
                            (q6, q7) = S.vintlv(p3, p7)
                            (o0, o1) = S.vintlv(q0, q2)
                            (o2, o3) = S.vintlv(q1, q3)
                            (o4, o5) = S.vintlv(q4, q6)
                            (o6, o7) = S.vintlv(q5, q7)
                            S.vsts(
                                folded[local_z * positions + chunk * 128 + 0, 0],
                                o0,
                                full,
                                "NORM_B16",
                            )
                            S.vsts(
                                folded[local_z * positions + chunk * 128 + 16, 0],
                                o1,
                                full,
                                "NORM_B16",
                            )
                            S.vsts(
                                folded[local_z * positions + chunk * 128 + 32, 0],
                                o2,
                                full,
                                "NORM_B16",
                            )
                            S.vsts(
                                folded[local_z * positions + chunk * 128 + 48, 0],
                                o3,
                                full,
                                "NORM_B16",
                            )
                            S.vsts(
                                folded[local_z * positions + chunk * 128 + 64, 0],
                                o4,
                                full,
                                "NORM_B16",
                            )
                            S.vsts(
                                folded[local_z * positions + chunk * 128 + 80, 0],
                                o5,
                                full,
                                "NORM_B16",
                            )
                            S.vsts(
                                folded[local_z * positions + chunk * 128 + 96, 0],
                                o6,
                                full,
                                "NORM_B16",
                            )
                            S.vsts(
                                folded[local_z * positions + chunk * 128 + 112, 0],
                                o7,
                                full,
                                "NORM_B16",
                            )
                T.ascend_set_flag("V_MTE3", 7)
                T.ascend_wait_flag("V_MTE3", 7)
                T.dual_copy(folded, xl)
                T.ascend_set_flag("MTE3_V", 7)
                T.ascend_wait_flag("MTE3_V", 7)
            with T.Cube():
                T.copy(wl, b)
                T.ascend_set_flag("M_MTE1", 0)
                T.ascend_set_flag("M_MTE1", 1)
                for oz in T.serial(depth):
                    for ni_idx in T.serial(nt + 1):
                        if ni_idx < nt:
                            slot = ni_idx % 2
                            T.ascend_wait_flag("M_MTE1", slot)
                            T.evaluate(
                                T.call_extern(
                                    "void",
                                    "tileops::conv_native_load3d",
                                    T.access_ptr(a[slot, 0, 0], "w"),
                                    T.access_ptr(xl[oz * positions, 0], "r"),
                                    fh,
                                    width,
                                    1,
                                    1,
                                    72,
                                    T.min(bn, plane - ni_idx * bn),
                                    ni_idx * bn,
                                    1,
                                    1,
                                    3,
                                    3,
                                    1,
                                    1,
                                    8,
                                )
                            )
                            T.ascend_set_flag("MTE1_M", slot)
                        if ni_idx > 0:
                            prev = ni_idx - 1
                            slot = prev % 2
                            T.ascend_wait_flag("MTE1_M", slot)
                            T.evaluate(
                                T.call_extern(
                                    "void",
                                    "tileops::conv_folded_mad",
                                    T.access_ptr(acc[oz % 4 * plane + prev * bn, 0], "rw"),
                                    T.access_ptr(a[slot, 0, 0], "r"),
                                    T.access_ptr(b, "r"),
                                    T.min(bn, plane - prev * bn),
                                    bk,
                                    bm,
                                )
                            )
                            T.ascend_set_flag("M_MTE1", slot)
                    if oz % 4 == 3:
                        T.ascend_set_flag("M_FIX", 0)
                        T.ascend_wait_flag("M_FIX", 0)
                        T.evaluate(
                            T.call_extern(
                                "void",
                                "tileops::conv_native_store",
                                T.access_ptr(Y[0, core * 4, (oz - 3) * plane], "w"),
                                T.access_ptr(acc, "r"),
                                4,
                                4 * plane,
                                depth * plane,
                                4 * plane,
                                quant,
                            )
                        )
                        T.ascend_set_flag("FIX_M", 0)
                        T.ascend_wait_flag("FIX_M", 0)
                T.ascend_wait_flag("M_MTE1", 0)
                T.ascend_wait_flag("M_MTE1", 1)

    return tilelang.compile(
        main, target="ascend", out_idx=-1, compile_flags=csrc_include("ascend_convolution.h")
    )


def _grouped_folded_supported(batch, ci, co, groups, size, out, window, stride, dilation):
    if not (
        batch == 1
        and 8 <= groups <= 32
        and (ci == 2 * groups)
        and (co == 4 * groups)
        and (size == out)
        and (window == (3, 3, 3))
        and (stride == dilation == (1, 1, 1))
        and (4 <= size[0] <= 12)
        and (size[0] % 4 == 0)
        and (16 <= size[1] <= 36)
        and (size[1] % 4 == 0)
        and (size[2] == 28)
    ):
        return False
    (depth, height, width) = size
    fh = (height + 2 + 31) // 32 * 32
    positions = fh * width
    plane = height * width
    raw_bytes = 2 * (depth + 2) * fh * 32 * 2
    folded_bytes = depth // 2 * positions * 8 * 2
    return (
        fh <= 64
        and plane % 16 == 0
        and (4 * plane <= 4095)
        and (raw_bytes + folded_bytes + 4096 <= 192 * 1024)
        and (depth * positions * 8 * 2 + 16 * 80 * 2 <= 512 * 1024)
        and (4 * plane * 16 * 4 <= 256 * 1024)
    )
