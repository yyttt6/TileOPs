"""Double-buffered L0 path for long one-dimensional convolution windows."""

import functools
import math

import tilelang
import tilelang.ascend.language as T
from tilelang.layout import make_ascend_major_k_layout

from tileops._csrc import csrc_include


@functools.lru_cache(maxsize=128)
def _pipelined_contract(
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
    depth, height, width = size
    od, oh, ow = out
    kd, kh, kw = window
    sd, sh, sw = stride
    pd, ph, pw = pad
    dd, dh, dw = dilation
    cing = ci // groups * bundle
    m = co // groups * bundle
    logical_groups = groups // bundle
    cb = (cing + 15) // 16
    bk = kh * kw * 16
    max_bm = min(256, (16384 // bk) // 16 * 16)
    bm = min(1 << (max_bm.bit_length() - 1), max(16, tilelang.next_power_of_2(m)))
    max_bn = min(256, (16384 // bk) // 16 * 16)
    spatial = oh * ow
    base_tiles = batch * logical_groups * od * ((m + bm - 1) // bm)
    wanted_nt = max((spatial + max_bn - 1) // max_bn, (32 + base_tiles - 1) // base_tiles)
    bn = min(max_bn, max(16, (spatial + wanted_nt * 16 - 1) // (wanted_nt * 16) * 16))
    packed = kd * cb * bk
    total_packed = packed
    assert not has_bias or m % bm == 0
    one_d = height == oh == kh == 1
    if one_d:
        frameh = 1
        framew = ((bn - 1) * sw + (kw - 1) * dw + 1 + 15) // 16 * 16
        left = right = 0
    else:
        framew = width
        frameh = ((ow - 1 + bn + ow - 1) // ow - 1) * sh + (kh - 1) * dh + 1
        alignment = 16 // math.gcd(width, 16)
        frameh = (frameh + alignment - 1) // alignment * alignment
        left = pw
        right = max(0, (ow - 1) * sw + (kw - 1) * dw + 1 - width - pw)
    spatial = oh * ow
    mt = (m + bm - 1) // bm
    nt = (spatial + bn - 1) // bn
    tiles = batch * logical_groups * od * mt * nt
    cores = min(32, tiles)
    assert bn > 0 and bm > 0
    assert 4 * max(bn, bm) * bk <= 65536
    assert 2 * (frameh * framew * 16 * 2 + bm * bk * 2) <= 512 * 1024
    w_rows = (co + ((-m) % bm) + 15) // 16 * 16
    bias_bytes = bm * 2
    global_weights = (
        logical_groups == 1 or m % 16 == 0
    ) and 2 * frameh * framew * 32 + w_rows * packed * 2 + bias_bytes <= 512 * 1024
    tile_weights = global_weights or (
        2 * frameh * framew * 32 + bm * packed * 2 + bias_bytes <= 512 * 1024
    )
    weight_rows = w_rows if global_weights else bm
    convert = 0 if dtype == "bfloat16" else 1
    quant = 16 if dtype == "bfloat16" else 1

    @T.prim_func
    def main(
        X: T.Tensor((batch, ci, depth * height * width), dtype),
        W: T.Tensor((co, total_packed), dtype),
        Bias: T.Tensor((co,), dtype),
        Y: T.Tensor((batch, logical_groups, m, od * spatial), dtype),
    ):
        with T.Kernel(
            cores,
        ) as core:
            xl = T.alloc_l1((2, frameh * framew, 16), dtype)
            wl = T.alloc_l1((2, bm, bk), dtype)
            a = T.alloc_l0a((2, bn, bk), dtype)
            b = T.alloc_l0b((2, bm, bk), dtype)
            acc = T.alloc_l0c((bn, bm), "float32")
            T.annotate_layout(
                {xl: make_ascend_major_k_layout(xl), b: make_ascend_major_k_layout(b)}
            )
            T.annotate_buffer_versions({xl: 1, wl: 1, a: 1, b: 1, acc: 1})
            whole_w = T.alloc_l1(
                (weight_rows if tile_weights else 16, packed if tile_weights else 16),
                dtype,
            )
            T.annotate_buffer_versions({whole_w: 1})
            bias_l1 = T.alloc_l1((bm,), dtype)
            T.annotate_buffer_versions({bias_l1: 1})
            with T.Cube():
                if global_weights:
                    if weight_rows > co:
                        T.fill(whole_w, T.cast(0, dtype))
                        T.ascend_pipe_barrier("PIPE_MTE2")
                    T.copy(W[:, :packed], whole_w[:co, :])
                for step in T.serial(T.ceildiv(tiles, cores)):
                    index = step * cores + core
                    if index < tiles:
                        ni = index % nt * bn
                        mi = index // nt % mt * bm
                        oz = index // (nt * mt) % od
                        g = index // (nt * mt * od) % logical_groups
                        bi = index // (nt * mt * od * logical_groups)
                        if has_bias:
                            T.evaluate(
                                T.call_extern(
                                    "void",
                                    "tileops::conv_bias_gm",
                                    T.access_ptr(bias_l1, "w"),
                                    T.access_ptr(Bias[g * m + mi], "r"),
                                    bm,
                                )
                            )
                            T.ascend_set_flag("MTE2_MTE1", 3)
                            T.ascend_wait_flag("MTE2_MTE1", 3)
                            T.evaluate(
                                T.call_extern(
                                    "void",
                                    "tileops::conv_bias_bt",
                                    T.access_ptr(bias_l1, "r"),
                                    bm,
                                    convert,
                                )
                            )
                        if tile_weights and not global_weights:
                            if mi + bm > m:
                                T.fill(whole_w, T.cast(0, dtype))
                                T.ascend_pipe_barrier("PIPE_MTE2")
                            T.copy(
                                W[g * m + mi : g * m + mi + T.min(bm, m - mi), :packed],
                                whole_w[: T.min(bm, m - mi), :],
                            )
                        T.ascend_set_flag("MTE1_MTE2", 0)
                        T.ascend_set_flag("MTE1_MTE2", 1)
                        T.ascend_set_flag("M_MTE1", 0)
                        T.ascend_set_flag("M_MTE1", 1)
                        for kt in T.serial(kd * cb + 2):
                            if kt < kd * cb:
                                slot = kt % 2
                                kz = kt // cb
                                cblock = kt % cb
                                iz = oz * sd - pd + kz * dd
                                channels = T.min(16, cing - cblock * 16)
                                T.ascend_wait_flag("MTE1_MTE2", slot)
                                if not tile_weights and m % bm != 0 and mi + bm > m:
                                    T.fill(wl[slot, :, :], T.cast(0, dtype))
                                if one_d:
                                    base = ni * sw - pw
                                    begin = T.max(0, base)
                                    count = T.max(0, T.min(width, base + framew) - begin)
                                    if iz < 0 or iz >= depth or count < framew:
                                        T.fill(xl[slot, :, :], T.cast(0, dtype))
                                    if (
                                        iz < 0
                                        or iz >= depth
                                        or count < framew
                                        or (not tile_weights and mi + bm > m)
                                    ):
                                        T.ascend_pipe_barrier("PIPE_MTE2")
                                    if iz >= 0 and iz < depth and count > 0:
                                        T.copy(
                                            X[
                                                bi,
                                                g * cing + cblock * 16 : g * cing
                                                + cblock * 16
                                                + channels,
                                                iz * width + begin : iz * width + begin + count,
                                            ],
                                            xl[
                                                slot,
                                                begin - base : begin - base + count,
                                                :channels,
                                            ],
                                            transpose=True,
                                        )
                                else:
                                    base = ni // ow * sh - ph
                                    begin = T.max(0, base)
                                    count = T.max(0, T.min(height, base + frameh) - begin)
                                    if iz < 0 or iz >= depth or count < frameh:
                                        T.fill(xl[slot, :, :], T.cast(0, dtype))
                                    if (
                                        iz < 0
                                        or iz >= depth
                                        or count < frameh
                                        or (not tile_weights and mi + bm > m)
                                    ):
                                        T.ascend_pipe_barrier("PIPE_MTE2")
                                    if iz >= 0 and iz < depth and count > 0:
                                        T.copy(
                                            X[
                                                bi,
                                                g * cing + cblock * 16 : g * cing
                                                + cblock * 16
                                                + channels,
                                                iz * height * width + begin * width : iz
                                                * height
                                                * width
                                                + (begin + count) * width,
                                            ],
                                            xl[
                                                slot,
                                                (begin - base) * width : (begin - base + count)
                                                * width,
                                                :channels,
                                            ],
                                            transpose=True,
                                        )
                                wk = (kz * cb + cblock) * bk
                                if not tile_weights:
                                    T.copy(
                                        W[
                                            g * m + mi : g * m + mi + T.min(bm, m - mi),
                                            wk : wk + bk,
                                        ],
                                        wl[slot, : T.min(bm, m - mi), :],
                                    )
                                T.ascend_set_flag("MTE2_MTE1", slot)
                            if kt > 0 and kt <= kd * cb:
                                previous = kt - 1
                                previous_slot = previous % 2
                                T.ascend_wait_flag("MTE2_MTE1", previous_slot)
                                T.ascend_wait_flag("M_MTE1", previous_slot)
                                if global_weights:
                                    T.copy(
                                        whole_w[
                                            g * m + mi : g * m + mi + bm,
                                            previous * bk : previous * bk + bk,
                                        ],
                                        b[previous_slot, :, :],
                                    )
                                elif tile_weights:
                                    T.copy(
                                        whole_w[:, previous * bk : previous * bk + bk],
                                        b[previous_slot, :, :],
                                    )
                                else:
                                    T.copy(wl[previous_slot, :, :], b[previous_slot, :, :])
                                T.evaluate(
                                    T.call_extern(
                                        "void",
                                        "tileops::conv_native_load3d",
                                        T.access_ptr(a[previous_slot, 0, 0], "w"),
                                        T.access_ptr(
                                            xl[previous_slot, 0, 0],
                                            "r",
                                            extent=frameh * framew * 16,
                                        ),
                                        frameh,
                                        framew,
                                        left,
                                        right,
                                        bk,
                                        bn,
                                        0 if one_d else ni % ow,
                                        sh,
                                        sw,
                                        kh,
                                        kw,
                                        dh,
                                        dw,
                                        16,
                                    )
                                )
                                T.ascend_set_flag("MTE1_MTE2", previous_slot)
                                T.ascend_set_flag("MTE1_M", previous_slot)
                            if kt > 1:
                                compute = kt - 2
                                compute_slot = compute % 2
                                T.ascend_wait_flag("MTE1_M", compute_slot)
                                if has_bias and compute == 0:
                                    T.evaluate(
                                        T.call_extern(
                                            "void",
                                            "tileops::conv_bias_mad",
                                            T.access_ptr(acc, "rw"),
                                            T.access_ptr(a[compute_slot, 0, 0], "r"),
                                            T.access_ptr(b[compute_slot, 0, 0], "r"),
                                            bn,
                                            bk,
                                            bm,
                                            T.uint64(0),
                                        )
                                    )
                                else:
                                    T.gemm(
                                        a[compute_slot, :, :],
                                        b[compute_slot, :, :],
                                        acc,
                                        transpose_B=True,
                                        clear_accum=compute == 0,
                                    )
                                T.ascend_set_flag("M_MTE1", compute_slot)
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
                                T.access_ptr(Y[bi, g, mi, oz * spatial + ni], "w"),
                                T.access_ptr(acc, "r"),
                                T.min(bm, m - mi),
                                T.min(bn, spatial - ni),
                                od * spatial,
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
