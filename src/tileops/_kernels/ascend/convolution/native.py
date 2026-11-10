"""Hardware img2col convolution with pipelined loads and channel-first output."""

import functools
import math

import tilelang
import tilelang.ascend.language as T
from tilelang.layout import make_ascend_major_k_layout

from tileops._csrc import csrc_include


def _native_c0(ci, groups, size, out, window, pad):
    """Small-C0 transfers require four-channel input and aligned destination rows."""
    one_d = size[1] == out[1] == window[1] == 1
    aligned = pad[2] % 4 == 0 if one_d else size[2] % 4 == 0
    return 4 if ci // groups <= 4 and aligned else 16


def _native_bundle(ci, co, groups, size, out, window, stride, pad, dilation):
    """Return a supported group bundle, or let the caller use explicit unfolding."""
    _, height, width = size
    _, oh, ow = out
    kd, kh, kw = window
    _, sh, sw = stride
    _, _, pw = pad
    _, dh, dw = dilation
    c0 = _native_c0(ci, groups, size, out, window, pad)
    bk = (kh * kw * c0 + 15) // 16 * 16
    if bk > 2048 or max(kh, kw, dh, dw) > 255 or max(sh, sw) > 63:
        return None
    cing, cog = ci // groups, co // groups
    # Block-diagonal packing would propagate nonfinite values between groups.
    bundle = 1
    bn = min(512 if c0 == 4 else 256, (32768 // bk) // 16 * 16)
    max_bm = min(256, (32768 // bk) // 16 * 16)
    bm = min(1 << (max_bm.bit_length() - 1), max(16, tilelang.next_power_of_2(cog * bundle)))
    one_d = height == oh == kh == 1
    if one_d:
        frameh = 1
        framew = ((bn - 1) * sw + (kw - 1) * dw + 1 + 15) // 16 * 16
    else:
        framew = width
        frameh = ((ow - 1 + bn + ow - 1) // ow - 1) * sh + (kh - 1) * dh + 1
        alignment = 16 // math.gcd(width, 16)
        frameh = (frameh + alignment - 1) // alignment * alignment
        right = max(0, (ow - 1) * sw + (kw - 1) * dw + 1 - width - pw)
        if not 0 <= pw <= 255 or right > 255 or ow > 32768:
            return None
    packed = kd * ((cing * bundle + c0 - 1) // c0) * bk
    original = cing * kd * kh * kw
    if (
        max(frameh, framew) > 65535
        or 2 * (frameh * framew * c0 * 2 + bm * bk * 2) + bn * 32 > 512 * 1024
    ):
        return None
    if original + packed + 32 > 49152:
        return None
    return bundle


@functools.lru_cache(None)
def _pack_weights(co, ci, kd, kh, kw, groups, bundle, dtype, has_bias=False, c0=16):
    cing = ci // groups
    cog = co // groups
    cb = (cing * bundle + c0 - 1) // c0
    bk = (kh * kw * c0 + 15) // 16 * 16
    original = cing * kd * kh * kw
    packed = kd * cb * bk
    total_packed = packed + 16 if has_bias else packed
    rawpitch = (original + 15) // 16 * 16
    tile = max(1, min(8, max(1, co // 64), 8192 // max(rawpitch, packed)))
    cores = min(64, (co + tile - 1) // tile)

    @T.prim_func
    def main(
        W: T.Tensor((co, original), dtype),
        Bias: T.Tensor((co,), dtype),
        P: T.Tensor((co, total_packed), dtype),
    ):
        with T.Kernel(cores) as core:
            raw = T.alloc_shared((tile, rawpitch), dtype)
            result = T.alloc_shared((tile, total_packed), dtype)
            bias_ub = T.alloc_shared((tile,), dtype)
            for step in T.serial(T.ceildiv(co, cores * tile)):
                start = (step * cores + core) * tile
                if start < co:
                    T.copy(W[start : start + tile, :], raw[:, :original])
                    if has_bias:
                        T.copy(Bias[start : start + tile], bias_ub)
                    with T.SimtVF(threads=256):
                        for i, j in T.Parallel(tile, total_packed):
                            bundled_ci = j // bk % cb * c0 + j % bk % c0
                            source_ci = bundled_ci - ((start + i) // cog % bundle) * cing
                            kz = j // (cb * bk)
                            tap = j % bk // c0
                            if j < packed and tap < kh * kw and source_ci >= 0 and source_ci < cing:
                                result[i, j] = raw[i, source_ci * kd * kh * kw + kz * kh * kw + tap]
                            elif has_bias and j == packed:
                                result[i, j] = bias_ub[i]
                            else:
                                result[i, j] = T.cast(0, dtype)
                    T.copy(result, P[start : start + tile, :])

    return tilelang.compile(main, target="ascend", out_idx=-1)


@functools.lru_cache(None)
def _native_contract(
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
    bias_table=False,
    block_m=0,
    block_n=0,
    reuse_weights=False,
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
    c0 = _native_c0(ci, groups, size, out, window, pad)
    cb = (ci // groups + c0 - 1) // c0
    bk = (kh * kw * c0 + 15) // 16 * 16
    max_bm = min(256, (32768 // bk) // 16 * 16)
    bm = block_m or min(1 << (max_bm.bit_length() - 1), max(16, tilelang.next_power_of_2(m)))
    max_bn = min(512 if c0 == 4 else 256, (32768 // bk) // 16 * 16, 65536 // bm)
    spatial = oh * ow
    base_tiles = batch * logical_groups * od * ((m + bm - 1) // bm)
    wanted_nt = max((spatial + max_bn - 1) // max_bn, (32 + base_tiles - 1) // base_tiles)
    bn = block_n or min(max_bn, max(16, (spatial + wanted_nt * 16 - 1) // (wanted_nt * 16) * 16))
    packed = kd * cb * bk
    total_packed = packed + 16 if has_bias and not bias_table else packed
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
    if reuse_weights:
        assert batch == groups == bundle == kd == od == 1 and co % bm == 0 and mt <= 32
    parts = min(nt, 32 // mt) if reuse_weights else 1
    cores = mt * parts if reuse_weights else min(32, tiles)
    assert 0 < bm <= max_bm and bm % 16 == 0
    assert 0 < bn <= max_bn and bn % 16 == 0
    assert 2 * (frameh * framew * c0 * 2 + bm * bk * 2) <= 512 * 1024
    bias_rows = (co + bm + 15) // 16 * 16
    preload = bias_table or (
        has_bias
        and (logical_groups == 1 or m % 16 == 0)
        and 2 * (frameh * framew * c0 * 2 + bm * bk * 2) + (bias_rows + bn) * 32 <= 512 * 1024
    )
    w_rows = (co + ((-m) % bm) + 15) // 16 * 16
    bias_bytes = (
        bm * 2 if bias_table else (bias_rows + bn) * 32 if preload else bn * 32 if has_bias else 0
    )
    assert not bias_table or (has_bias and m % bm == 0)
    convert = 0 if dtype == "bfloat16" else 1
    global_weights = (
        (logical_groups == 1 or m % 16 == 0)
        and (not has_bias or preload)
        and 2 * frameh * framew * c0 * 2 + w_rows * packed * 2 + bias_bytes <= 512 * 1024
    )
    tile_weights = global_weights or (
        (not has_bias or preload)
        and 2 * frameh * framew * c0 * 2 + bm * packed * 2 + bias_bytes <= 512 * 1024
    )
    assert not reuse_weights or tile_weights
    group_weights = (
        logical_groups > 1
        and logical_groups == cores
        and mt == 1
        and not has_bias
        and tile_weights
        and not global_weights
    )
    weight_rows = w_rows if global_weights else bm
    one_bits = 0x3F803F80 if dtype == "bfloat16" else 0x3C003C00
    quant = 16 if dtype == "bfloat16" else 1

    @T.prim_func
    def main(
        X: T.Tensor((batch, ci, depth * height * width), dtype),
        W: T.Tensor((co, total_packed), dtype),
        Bias: T.Tensor((co,), dtype),
        Y: T.Tensor((batch, logical_groups, m, od * spatial), dtype),
    ):
        with T.Kernel(cores) as core:
            xl = T.alloc_l1((2, frameh * framew, c0), dtype)
            wl = T.alloc_l1((2, bm, bk), dtype)
            a = T.alloc_l0a((bn, bk), dtype)
            b = T.alloc_l0b((bm, bk), dtype)
            acc = T.alloc_l0c((bn, bm), "float32")
            T.annotate_layout({b: make_ascend_major_k_layout(b)})
            if c0 == 16:
                T.annotate_layout({xl: make_ascend_major_k_layout(xl)})
            T.annotate_buffer_versions({xl: 1, wl: 1, a: 1, b: 1, acc: 1})
            whole_w = T.alloc_l1(
                (weight_rows if tile_weights else 16, packed if tile_weights else 16), dtype
            )
            T.annotate_buffer_versions({whole_w: 1})
            ones = T.alloc_l1((bn, 16), dtype)
            bias_l1 = T.alloc_l1((bias_rows if preload else 16, 16), dtype)
            T.annotate_buffer_versions({ones: 1, bias_l1: 1})
            bt_l1 = T.alloc_l1((bm,), dtype)
            T.annotate_buffer_versions({bt_l1: 1})
            with T.Cube():
                if group_weights:
                    T.fill(whole_w, T.cast(0, dtype))
                    T.ascend_pipe_barrier("PIPE_MTE2")
                    T.copy(W[core * m : core * m + m, :packed], whole_w[:m, :])
                if global_weights:
                    if weight_rows > co:
                        T.fill(whole_w, T.cast(0, dtype))
                        T.ascend_pipe_barrier("PIPE_MTE2")
                    T.copy(W[:, :packed], whole_w[:co, :])
                if preload and not bias_table:
                    T.fill(bias_l1, T.cast(0, dtype))
                    T.ascend_pipe_barrier("PIPE_MTE2")
                    T.copy(W[:, packed : packed + 16], bias_l1[:co, :])
                if has_bias and not bias_table:
                    T.evaluate(
                        T.call_extern(
                            "void",
                            "tileops::conv_native_fill",
                            T.access_ptr(ones, "w"),
                            bn,
                            one_bits,
                        )
                    )
                    T.ascend_set_flag("MTE2_MTE1", 0)
                    T.ascend_wait_flag("MTE2_MTE1", 0)
                for step in T.serial(T.ceildiv(tiles, cores)):
                    index = (
                        (core // parts * nt + step * parts + core % parts)
                        if reuse_weights
                        else step * cores + core
                    )
                    if (step * parts + core % parts < nt) if reuse_weights else (index < tiles):
                        if group_weights:
                            ni = step % nt * bn
                            mi = T.int32(0)
                            oz = step // nt % od
                            g = core
                            bi = step // (nt * od)
                        else:
                            ni = index % nt * bn
                            mi = index // nt % mt * bm
                            oz = index // (nt * mt) % od
                            g = index // (nt * mt * od) % logical_groups
                            bi = index // (nt * mt * od * logical_groups)
                        if bias_table and (not reuse_weights or step == 0):
                            T.evaluate(
                                T.call_extern(
                                    "void",
                                    "tileops::conv_bias_gm",
                                    T.access_ptr(bt_l1, "w"),
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
                                    T.access_ptr(bt_l1, "r"),
                                    bm,
                                    convert,
                                )
                            )
                        if (
                            tile_weights
                            and not global_weights
                            and not group_weights
                            and (not reuse_weights or step == 0)
                        ):
                            if mi + bm > m:
                                T.fill(whole_w, T.cast(0, dtype))
                                T.ascend_pipe_barrier("PIPE_MTE2")
                            T.copy(
                                W[g * m + mi : g * m + mi + T.min(bm, m - mi), :packed],
                                whole_w[: T.min(bm, m - mi), :],
                            )
                        T.ascend_set_flag("MTE1_MTE2", 0)
                        T.ascend_set_flag("MTE1_MTE2", 1)
                        T.ascend_set_flag("M_MTE1", 2)
                        for kt in T.serial(kd * cb + 1):
                            if kt < kd * cb:
                                slot = kt % 2
                                kz = kt // cb
                                cblock = kt % cb
                                iz = oz * sd - pd + kz * dd
                                channels = T.min(c0, cing - cblock * c0)
                                T.ascend_wait_flag("MTE1_MTE2", slot)
                                if not tile_weights and m % bm != 0 and mi + bm > m:
                                    T.fill(wl[slot, :, :], T.cast(0, dtype))
                                if one_d:
                                    base = ni * sw - pw
                                    begin = T.max(0, base)
                                    count = T.max(0, T.min(width, base + framew) - begin)
                                    if iz < 0 or iz >= depth or count < framew:
                                        if c0 == 4:
                                            T.evaluate(
                                                T.call_extern(
                                                    "void",
                                                    "tileops::conv_native_fill",
                                                    T.access_ptr(xl[slot, 0, 0], "w"),
                                                    frameh * framew // 4,
                                                    0,
                                                )
                                            )
                                        else:
                                            T.fill(xl[slot, :, :], T.cast(0, dtype))
                                    if (
                                        iz < 0
                                        or iz >= depth
                                        or count < framew
                                        or (not tile_weights and mi + bm > m)
                                    ):
                                        T.ascend_pipe_barrier("PIPE_MTE2")
                                    if iz >= 0 and iz < depth and count > 0:
                                        if c0 == 4:
                                            T.evaluate(
                                                T.call_extern(
                                                    "void",
                                                    "tileops::conv_native_input_small",
                                                    T.access_ptr(xl[slot, begin - base, 0], "w"),
                                                    T.access_ptr(
                                                        X[bi, g * cing, iz * width + begin], "r"
                                                    ),
                                                    depth * height * width * 2,
                                                    count,
                                                    cing,
                                                )
                                            )
                                        else:
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
                                        if c0 == 4:
                                            T.evaluate(
                                                T.call_extern(
                                                    "void",
                                                    "tileops::conv_native_fill",
                                                    T.access_ptr(xl[slot, 0, 0], "w"),
                                                    frameh * framew // 4,
                                                    0,
                                                )
                                            )
                                        else:
                                            T.fill(xl[slot, :, :], T.cast(0, dtype))
                                    if (
                                        iz < 0
                                        or iz >= depth
                                        or count < frameh
                                        or (not tile_weights and mi + bm > m)
                                    ):
                                        T.ascend_pipe_barrier("PIPE_MTE2")
                                    if iz >= 0 and iz < depth and count > 0:
                                        if c0 == 4:
                                            T.evaluate(
                                                T.call_extern(
                                                    "void",
                                                    "tileops::conv_native_input_small",
                                                    T.access_ptr(
                                                        xl[slot, (begin - base) * width, 0], "w"
                                                    ),
                                                    T.access_ptr(
                                                        X[
                                                            bi,
                                                            g * cing,
                                                            iz * height * width + begin * width,
                                                        ],
                                                        "r",
                                                    ),
                                                    depth * height * width * 2,
                                                    count * width,
                                                    cing,
                                                )
                                            )
                                        else:
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
                            if kt > 0:
                                previous = kt - 1
                                previous_slot = previous % 2
                                T.ascend_wait_flag("MTE2_MTE1", previous_slot)
                                T.ascend_wait_flag("M_MTE1", 2)
                                if global_weights:
                                    T.copy(
                                        whole_w[
                                            g * m + mi : g * m + mi + bm,
                                            previous * bk : previous * bk + bk,
                                        ],
                                        b,
                                    )
                                elif tile_weights:
                                    T.copy(whole_w[:, previous * bk : previous * bk + bk], b)
                                else:
                                    T.copy(wl[previous_slot, :, :], b)
                                T.evaluate(
                                    T.call_extern(
                                        "void",
                                        "tileops::conv_native_load3d",
                                        T.access_ptr(a, "w"),
                                        T.access_ptr(
                                            xl[previous_slot, 0, 0],
                                            "r",
                                            extent=frameh * framew * c0,
                                        ),
                                        frameh,
                                        framew,
                                        left,
                                        right,
                                        kh * kw * c0,
                                        bn,
                                        0 if one_d else ni % ow,
                                        sh,
                                        sw,
                                        kh,
                                        kw,
                                        dh,
                                        dw,
                                        c0,
                                    )
                                )
                                T.ascend_set_flag("MTE1_MTE2", previous_slot)
                                T.ascend_set_flag("MTE1_M", 2)
                                T.ascend_wait_flag("MTE1_M", 2)
                                if bias_table and previous == 0:
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
                                    T.gemm(a, b, acc, transpose_B=True, clear_accum=previous == 0)
                                T.ascend_set_flag("M_MTE1", 2)
                        T.ascend_wait_flag("MTE1_MTE2", 0)
                        T.ascend_wait_flag("MTE1_MTE2", 1)
                        T.ascend_wait_flag("M_MTE1", 2)
                        if has_bias and not bias_table:
                            if preload and not bias_table:
                                T.copy(ones, a[:, :16])
                                T.copy(bias_l1[g * m + mi : g * m + mi + bm, :], b[:, :16])
                            else:
                                T.fill(wl[0, :, :], T.cast(0, dtype))
                                T.ascend_pipe_barrier("PIPE_MTE2")
                                T.copy(
                                    W[
                                        g * m + mi : g * m + mi + T.min(bm, m - mi),
                                        packed : packed + 16,
                                    ],
                                    wl[0, : T.min(bm, m - mi), :16],
                                )
                                T.ascend_set_flag("MTE2_MTE1", 0)
                                T.ascend_wait_flag("MTE2_MTE1", 0)
                                T.copy(ones, a[:, :16])
                                T.copy(wl[0, :, :16], b[:, :16])
                            T.ascend_set_flag("MTE1_M", 0)
                            T.ascend_wait_flag("MTE1_M", 0)
                            T.gemm(a[:, :16], b[:, :16], acc, transpose_B=True, clear_accum=False)
                            T.ascend_set_flag("M_MTE1", 0)
                            T.ascend_wait_flag("M_MTE1", 0)
                            T.ascend_set_flag("MTE1_MTE2", 0)
                            T.ascend_wait_flag("MTE1_MTE2", 0)
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
        main, target="ascend", out_idx=-1, compile_flags=csrc_include("ascend_convolution.h")
    )


def _native_tuning(batch, ci, co, groups, size, out, window, stride, dilation, has_bias):
    """Keep a fixed output-channel tile resident for wide downsampling windows."""
    if not (batch == groups == 1 and size[0] == out[0] == window[0] == 1):
        return None
    plane = out[1] * out[2]
    reuse = (
        ci == 128
        and co == 256
        and window == (1, 5, 5)
        and stride == (1, 2, 2)
        and dilation == (1, 1, 1)
        and plane >= 256
    )
    dilated = (
        not has_bias
        and ci >= 1024
        and ci % 128 == 0
        and co >= 128
        and co % 64 == 0
        and ci * 9 * 6 <= 192 * 1024
        and window == (1, 3, 3)
        and stride == (1, 1, 1)
        and min(dilation[1:]) >= 8
        and 512 <= plane <= 2048
    )
    if not (reuse or dilated):
        return None
    bm, bn = 64, 64 if reuse else 128
    fh = (
        ((out[2] - 1 + bn + out[2] - 1) // out[2] - 1) * stride[1]
        + (window[1] - 1) * dilation[1]
        + 1
    )
    align = 16 // math.gcd(size[2], 16)
    fh = (fh + align - 1) // align * align
    weight_bytes = bm * ci * 25 * 2 + bm * 2 if reuse else 2 * bm * 144 * 2
    if 2 * fh * size[2] * 16 * 2 + weight_bytes > 512 * 1024:
        return None
    return bm, bn, reuse
