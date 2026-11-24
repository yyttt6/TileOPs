"""Head-grouped flash attention over variable-length requests on Ascend."""

import functools
import math

import tilelang
import tilelang.ascend.language as T
import torch

from tileops._kernels.ascend.attention.gqa_dense_prefill import pass_configs, prefill_query_major
from tileops._kernels.attention.call_spec import GQAVarlenFwdInterface
from tileops._kernels.kernel_base import Kernel
from tileops.backend._registry import register_kernel_type


@functools.lru_cache(maxsize=64)
def _layout(batch, group, bm, bn, omit_aligned=False):
    @T.prim_func
    def main(
        cq: T.Tensor((batch + 1,), "int32"),
        ck: T.Tensor((batch + 1,), "int32"),
        off: T.Tensor((batch + 1, 2), "int32"),
    ):
        with T.Kernel(1):  # noqa: SIM117 - distinct TileLang vector scope
            with T.SimtVF(threads=32):
                for _ in T.Parallel(1):
                    q = T.alloc_local((1,), "int32")
                    k = T.alloc_local((1,), "int32")
                    q[0] = 0
                    k[0] = 0
                    for b in T.serial(batch):
                        off[b, 0] = q[0]
                        off[b, 1] = k[0]
                        sq = cq[b + 1] - cq[b]
                        sk = ck[b + 1] - ck[b]
                        if not omit_aligned or not (
                            sq > 0 and sk >= sq and sq % 128 == 0 and sk % 128 == 0
                        ):
                            q[0] += T.ceildiv(sq * group, bm) * bm
                            k[0] += T.ceildiv(sk, bn) * bn
                    off[batch, 0] = q[0]
                    off[batch, 1] = k[0]

    return tilelang.compile(main, target="ascend", out_idx=[2])


@functools.lru_cache(maxsize=128)
def _reorder_plain(batch, nh, kh, dim, row_tile, dtype, which, pack):
    """Transfer contiguous head groups with strided DMA, padding each request independently."""
    group = nh // kh
    width = group * dim
    assert row_tile % group == 0
    chunk = row_tile // group
    while chunk > 1 and chunk * width > 8192:
        chunk //= 2
    span = chunk * group
    total = T.dynamic("tokens")
    storage = T.dynamic("packed_elements")

    @T.prim_func
    def main(
        tokens: T.Tensor((total, nh * dim), dtype),
        padded: T.Tensor((kh, storage), dtype),
        cu: T.Tensor((batch + 1,), "int32"),
        off: T.Tensor((batch + 1, 2), "int32"),
    ):
        with T.Kernel(64) as core:
            block = T.alloc_shared((chunk, width), dtype)
            flat = T.reshape(block, (chunk * width,))
            units = off[batch, which] // span
            for task in T.serial(core, units * kh, 64):
                head = task % kh
                ri = task // kh * span
                b = T.alloc_var("int32", init=0)
                hi = T.alloc_var("int32", init=batch)
                while b < hi:
                    mid = (b + hi) // 2
                    if off[mid + 1, which] <= ri:
                        b = mid + 1
                    else:
                        hi = mid
                token = cu[b] + (ri - off[b, which]) // group
                limit = cu[b + 1]
                if pack:
                    if token + chunk <= limit:
                        T.copy(
                            tokens[token : token + chunk, head * width : (head + 1) * width], block
                        )
                    else:
                        with T.SimtVF(threads=128):
                            for row, col in T.Parallel(chunk, width):
                                if token + row < limit:
                                    block[row, col] = tokens[token + row, head * width + col]
                                else:
                                    block[row, col] = 0
                    T.copy(flat, padded[head, ri * dim : ri * dim + chunk * width])
                else:
                    T.copy(padded[head, ri * dim : ri * dim + chunk * width], flat)
                    if token + chunk <= limit:
                        T.copy(
                            block, tokens[token : token + chunk, head * width : (head + 1) * width]
                        )
                    else:
                        with T.SimtVF(threads=128):
                            for row, col in T.Parallel(chunk, width):
                                if token + row < limit:
                                    tokens[token + row, head * width + col] = block[row, col]

    return tilelang.compile(main, target="ascend", out_idx=[])


@functools.lru_cache(maxsize=64)
def _pack_e4m3(batch, nh, kh, dim, row_tile, dtype, which):
    """Dequantize 64 E4M3 values per vector after the grouped DMA load."""
    group = nh // kh
    width = group * dim
    chunk = row_tile // group
    while chunk > 1 and chunk * width > 4096:
        chunk //= 2
    span = chunk * group
    tokens = T.dynamic("tokens")
    storage = T.dynamic("storage")
    fp8 = "float8_e4m3fn"

    @T.prim_func
    def main(
        x: T.Tensor((tokens, nh * dim), fp8),
        cu: T.Tensor((batch + 1,), "int32"),
        off: T.Tensor((batch + 1, 2), "int32"),
        scales: T.Tensor((batch, kh), "float32"),
        packed: T.Tensor((kh, storage), dtype),
    ):
        with T.Kernel(64) as core:
            raw = T.alloc_shared((chunk, width), fp8)
            flat = T.reshape(raw, (chunk * width,))
            words = T.view(raw, (chunk * width // 4,), dtype="uint32")
            result = T.alloc_shared((chunk * width,), dtype)
            scale = T.alloc_shared((8,), "float32")
            units = off[batch, which] // span
            for job in T.serial(core, units * kh, 64):
                head = job % kh
                ri = job // kh * span
                b = T.alloc_var("int32", init=0)
                hi = T.alloc_var("int32", init=batch)
                while b < hi:
                    mid = (b + hi) // 2
                    if off[mid + 1, which] <= ri:
                        b = mid + 1
                    else:
                        hi = mid
                pos = cu[b] + (ri - off[b, which]) // group
                end = cu[b + 1]
                T.copy(scales[b, head : head + 1], scale[:1])
                if pos + chunk <= end:
                    T.copy(x[pos : pos + chunk, head * width : (head + 1) * width], raw)
                else:
                    # The tail occupies a contiguous number of token rows. Transfer only
                    # live bytes and clear the full UB tile in native 32-bit lanes.
                    T.fill(words, T.uint32(0))
                    if pos < end:
                        nr = end - pos
                        T.copy(x[pos:end, head * width : (head + 1) * width], raw[:nr, :])
                with T.SimdVF():
                    mask = T.simd.pset(32)
                    factor = T.simd.vld(scale[0], dist="BRC_B32")
                    for part in T.serial(chunk * width // 64):
                        values = T.simd.vcvt(
                            T.simd.vld(flat[part * 64], dist="UNPK4_B8"), "float32"
                        )
                        scaled = T.simd.vcvt(
                            T.simd.vmul(values, factor, mask),
                            dtype,
                            round="ROUND_R",
                            sat=False,
                            part=0,
                        )
                        T.simd.vsts(result[part * 64], scaled, mask, dist="PK_B32")
                T.copy(result, packed[head, ri * dim : ri * dim + chunk * width])

    return tilelang.compile(main, target="ascend", out_idx=[])


@functools.lru_cache(maxsize=64)
def _pack_rotary(batch, nh, kh, dim, row_tile, dtype, which, rot, interleaved):
    """Load token groups and their rotary tables once, then rotate in on-chip memory."""
    group = nh // kh
    width = group * dim
    chunk = row_tile // group
    while chunk > 1 and chunk * width > 4096:
        chunk //= 2
    span = chunk * group
    total = T.dynamic("tokens")
    storage = T.dynamic("storage")
    positions = T.dynamic("positions")

    @T.prim_func
    def main(
        x: T.Tensor((total, nh * dim), dtype),
        cuq: T.Tensor((batch + 1,), "int32"),
        cuk: T.Tensor((batch + 1,), "int32"),
        off: T.Tensor((batch + 1, 2), "int32"),
        cos: T.Tensor((positions, rot // 2), dtype),
        sin: T.Tensor((positions, rot // 2), dtype),
        packed: T.Tensor((kh, storage), dtype),
    ):
        with T.Kernel(64) as core:
            raw = T.alloc_shared((chunk, width), dtype)
            ct = T.alloc_shared((chunk, rot // 2), dtype)
            st = T.alloc_shared((chunk, rot // 2), dtype)
            out = T.alloc_shared((chunk, width), dtype)
            flat = T.reshape(out, (chunk * width,))
            units = off[batch, which] // span
            for job in T.serial(core, units * kh, 64):
                head = job % kh
                ri = job // kh * span
                b = T.alloc_var("int32", init=0)
                hi = T.alloc_var("int32", init=batch)
                while b < hi:
                    mid = (b + hi) // 2
                    if off[mid + 1, which] <= ri:
                        b = mid + 1
                    else:
                        hi = mid
                start = (ri - off[b, which]) // group
                qlen = cuq[b + 1] - cuq[b]
                klen = cuk[b + 1] - cuk[b]
                length = qlen if which == 0 else klen
                origin = cuq[b] if which == 0 else cuk[b]
                token = origin + start
                pos = start + (klen - qlen if which == 0 else 0)
                if start + chunk <= length and pos >= 0:
                    T.copy(x[token : token + chunk, head * width : (head + 1) * width], raw)
                    T.copy(cos[pos : pos + chunk, :], ct)
                    T.copy(sin[pos : pos + chunk, :], st)
                else:
                    with T.SimtVF(threads=128):
                        for rr, dd in T.Parallel(chunk, width):
                            if start + rr < length:
                                raw[rr, dd] = x[token + rr, head * width + dd]
                            else:
                                raw[rr, dd] = 0
                        for rr, cc in T.Parallel(chunk, rot // 2):
                            if start + rr < length:
                                p = pos + rr
                                at = T.if_then_else(p < 0, p + positions, p)
                                ct[rr, cc] = cos[at, cc]
                                st[rr, cc] = sin[at, cc]
                            else:
                                ct[rr, cc] = 1
                                st[rr, cc] = 0
                with T.SimtVF(threads=256):
                    for rr, dd in T.Parallel(chunk, width):
                        col = dd % dim
                        if col < rot:
                            pair = col // 2 if interleaved else col % (rot // 2)
                            other = (col ^ 1) if interleaved else (col + rot // 2) % rot
                            first = col % 2 == 0 if interleaved else col < rot // 2
                            value = T.cast(raw[rr, dd], "float32")
                            partner = T.cast(raw[rr, dd - col + other], "float32")
                            out[rr, dd] = T.cast(
                                value * T.cast(ct[rr, pair], "float32")
                                + T.if_then_else(first, -partner, partner)
                                * T.cast(st[rr, pair], "float32"),
                                dtype,
                            )
                        else:
                            out[rr, dd] = raw[rr, dd]
                T.copy(flat, packed[head, ri * dim : ri * dim + chunk * width])

    return tilelang.compile(main, target="ascend", out_idx=[])


@functools.lru_cache(maxsize=128)
def _pack(batch, heads, kh, dim, row_tile, dtype, fp8, kind, rot, interleaved):
    is_q = kind == "q"
    group = heads // kh
    total = T.dynamic("tokens")
    capacity = T.dynamic("packed_rows")
    nh = heads if is_q else kh
    which = 0 if is_q else 1
    step = 16
    dp = (dim + 15) // 16 * 16
    P = T.dynamic("positions")
    rawtype = "uint8" if fp8 else dtype

    def decode(raw):
        if fp8:
            u = T.cast(raw, "uint32")
            e = (u >> 3) & 15
            m = u & 7
            mag = T.if_then_else(
                e == 0,
                T.cast(m, "float32") * 0.001953125,
                T.reinterpret(((e + 120) << 23) | (m << 20), "float32"),
            )
            v = T.if_then_else(
                (u & 127) == 127,
                T.reinterpret(T.uint32(0x7FC00000), "float32"),
                T.if_then_else((u & 128) != 0, -mag, mag),
            )
        else:
            v = T.cast(raw, "float32")
        return v

    @T.prim_func
    def main(
        x: T.Tensor((total, nh, dim), rawtype),
        cq: T.Tensor((batch + 1,), "int32"),
        ck: T.Tensor((batch + 1,), "int32"),
        off: T.Tensor((batch + 1, 2), "int32"),
        scale: T.Tensor((batch, kh), "float32") if fp8 else None,
        cos: T.Tensor((P, rot // 2), dtype) if rot else None,
        sin: T.Tensor((P, rot // 2), dtype) if rot else None,
        packed: T.Tensor((kh, capacity, dp), dtype),
    ):
        with T.Kernel(64) as core:
            num = T.alloc_var("int32")
            num = off[batch, which] // step
            for task in T.serial(core, kh * num, 64):
                h = task // num
                first = task % num * step
                b = T.alloc_var("int32", init=0)
                hi = T.alloc_var("int32", init=batch)
                while b < hi:
                    mid = (b + hi) // 2
                    if off[mid + 1, which] <= first:
                        b = mid + 1
                    else:
                        hi = mid
                base = cq[b] if is_q else ck[b]
                length = (cq[b + 1] - cq[b]) if is_q else (ck[b + 1] - ck[b])
                packed_start = off[b, which]
                shift = (ck[b + 1] - ck[b] - length) if is_q else 0
                temp = T.alloc_shared((step, dp), dtype)
                with T.SimtVF(threads=256):
                    for row, d in T.Parallel(step, dp):
                        local = first + row - packed_start
                        pos = local // group if is_q else local
                        ih = h * group + local % group if is_q else h
                        if pos < length and d < dim:
                            v = T.alloc_local((1,), "float32")
                            v[0] = decode(x[base + pos, ih, d])
                            if rot:  # noqa: SIM102 - specialize away absent RoPE tables
                                if d < rot:
                                    pair = d // 2 if interleaved else d % (rot // 2)
                                    part = (d % 2 == 0) if interleaved else (d < rot // 2)
                                    partner = (d ^ 1) if interleaved else (d + rot // 2) % rot
                                    u = decode(x[base + pos, ih, partner])
                                    v[0] = v[0] * T.cast(
                                        cos[pos + shift, pair], "float32"
                                    ) + T.if_then_else(part, -u, u) * T.cast(
                                        sin[pos + shift, pair], "float32"
                                    )
                            if fp8:
                                v[0] *= scale[b, h]
                            temp[row, d] = T.cast(v[0], dtype)
                        else:
                            temp[row, d] = 0
                T.copy(temp, packed[h, first : first + step, :])

    grouped = dim == dp and row_tile % (group if is_q else 1) == 0
    direct = grouped and not rot and not fp8
    native_fp8 = grouped and fp8 and not rot and dim % 64 == 0
    native_rope = grouped and rot and not fp8 and rot % 16 == 0
    if direct:
        program = _reorder_plain(batch, nh, kh, dim, row_tile, dtype, which, True)
    elif native_fp8:
        program = _pack_e4m3(batch, nh, kh, dim, row_tile, dtype, which)
    elif native_rope:
        program = _pack_rotary(batch, nh, kh, dim, row_tile, dtype, which, rot, interleaved)
    else:
        program = tilelang.compile(main, target="ascend", out_idx=[])
    output_type = getattr(torch, dtype)

    def pack(x, *metadata):
        # Rounding each request separately needs at most one extra tile per request.
        rows = x.shape[0] * (group if is_q else 1)
        capacity = ((rows + row_tile - 1) // row_tile + batch) * row_tile
        packed = torch.empty((kh, capacity, dp), device=x.device, dtype=output_type)
        if direct:
            program(
                x.reshape(x.shape[0], nh * dim), packed.view(kh, -1), metadata[which], metadata[2]
            )
        elif native_fp8:
            values = x.view(torch.float8_e4m3fn).reshape(x.shape[0], nh * dim)
            program(values, metadata[which], metadata[2], metadata[3], packed.view(kh, -1))
        elif native_rope:
            program(x.reshape(x.shape[0], nh * dim), *metadata, packed.view(kh, -1))
        else:
            program(x.contiguous(), *metadata, packed)
        return packed

    return pack


@functools.lru_cache(maxsize=64)
def _unpack_attention(batch, heads, kh, dv, dtype, bm, with_lse, which=0):
    tq = T.dynamic("queries")
    tiles = T.dynamic("tiles")
    group = heads // kh
    rows = bm // 2
    vp = (dv + 15) // 16 * 16

    @T.prim_func
    def main(
        src: T.Tensor((kh, tiles, 2, rows, vp), dtype),
        stats: T.Tensor((kh, tiles, 2, rows), "float32") if with_lse else None,
        cq: T.Tensor((batch + 1,), "int32"),
        off: T.Tensor((batch + 1, 2), "int32"),
        out: T.Tensor((tq, heads, dv), dtype),
        lse: T.Tensor((heads, tq), "float32") if with_lse else None,
    ):
        with T.Kernel(64) as core:
            for job in T.serial(core, tq * kh, 64):
                token = job // kh
                kv_head = job % kh
                lo = T.alloc_var("int32", init=0)
                hi = T.alloc_var("int32", init=batch)
                while lo < hi:
                    mid = (lo + hi) // 2
                    if cq[mid + 1] <= token:
                        lo = mid + 1
                    else:
                        hi = mid
                start = off[lo, which] + (token - cq[lo]) * group
                with T.SimtVF(threads=128):
                    for h, d in T.Parallel(group, dv):
                        r = start + h
                        out[token, kv_head * group + h, d] = src[
                            kv_head, r // bm, r % bm // rows, r % rows, d
                        ]
                    if with_lse:
                        for h in T.Parallel(group):
                            r = start + h
                            lse[kv_head * group + h, token] = stats[
                                kv_head, r // bm, r % bm // rows, r % rows
                            ]

    if not with_lse and vp == dv and bm % group == 0:
        program = _reorder_plain(batch, heads, kh, dv, bm, dtype, which, False)

        def unpack(padded, cu, off, out):
            program(out.view(out.shape[0], heads * dv), padded.view(kh, -1), cu, off)

        return unpack

    return tilelang.compile(main, target="ascend", out_idx=[])


@functools.lru_cache(maxsize=128)
def _attention(
    batch,
    heads,
    kh,
    dim,
    dtype,
    causal,
    wl,
    wr,
    scale,
    cap,
    sinks,
    bm,
    bn,
    pool=None,
    value_dim=None,
    with_lse=False,
    raw_kh=0,
    omit_aligned=False,
    packed_kh=0,
):
    tiles = T.dynamic("output_tiles")
    qcap = T.dynamic("packed_queries")
    kcap = T.dynamic("packed_keys")
    group = heads // kh
    dp = (dim + 15) // 16 * 16
    dv = dim if value_dim is None else value_dim
    vp = (dv + 15) // 16 * 16
    rows = bm // 2

    ps, pool_rows, pages, fp8kv, group_scales, rot, interleaved = pool or (
        0,
        0,
        0,
        False,
        False,
        0,
        False,
    )
    direct = ps > 0 and ps % bn == 0 and not fp8kv and not rot and dp == dim
    narrow_value_tile = bool(packed_kh) and dim > 128
    depth_step = math.gcd(dp, 64)
    value_tile = math.gcd(vp, 128) if narrow_value_tile else vp
    storage_heads = packed_kh or kh
    kshape = (pool_rows, kh, dim) if ps else (storage_heads, kcap, dp)
    vshape = kshape if ps else (storage_heads, kcap, vp)
    if raw_kh:
        assert not ps and not with_lse and dp == dim and dv == dim and heads == kh
        kshape = (kcap, raw_kh, dim)
        vshape = kshape
    qshape = (qcap, heads, dim) if raw_kh else (kh, qcap, dp)
    kdtype = "uint8" if fp8kv else dtype
    scale_shape = (batch, kh) if group_scales else (1,)
    P = T.dynamic("positions")

    def read_cache(x, row, head, col):
        if not fp8kv:
            return T.cast(x[row, head, col], "float32")
        u = T.cast(x[row, head, col], "uint32")
        e = (u >> 3) & 15
        m = u & 7
        mag = T.if_then_else(
            e == 0,
            T.cast(m, "float32") * 0.001953125,
            T.reinterpret(((e + 120) << 23) | (m << 20), "float32"),
        )
        return T.if_then_else(
            (u & 127) == 127,
            T.reinterpret(T.uint32(0x7FC00000), "float32"),
            T.if_then_else((u & 128) != 0, -mag, mag),
        )

    @T.macro
    def unpack_k(buf, dest, table, sc, cos, sin, batch_id, group_id, start, length, sub):
        with T.SimtVF(threads=256):
            for row, col in T.Parallel(bn // 2, dp):
                logical = start + sub * (bn // 2) + row
                if logical < length and col < dim:
                    physical = table[batch_id, logical // ps] * ps + logical % ps
                    z = T.alloc_local((1,), "float32")
                    z[0] = read_cache(buf, physical, group_id, col)
                    if rot:  # noqa: SIM102 - specialize away absent RoPE tables
                        if col < rot:
                            pair = col // 2 if interleaved else col % (rot // 2)
                            first = col % 2 == 0 if interleaved else col < rot // 2
                            other = (col ^ 1) if interleaved else (col + rot // 2) % rot
                            v = read_cache(buf, physical, group_id, other)
                            z[0] = z[0] * T.cast(cos[logical, pair], "float32") + T.if_then_else(
                                first, -v, v
                            ) * T.cast(sin[logical, pair], "float32")
                    if fp8kv:
                        z[0] *= sc[batch_id, group_id] if group_scales else sc[0]
                    dest[row, col] = T.cast(z[0], dtype)
                else:
                    dest[row, col] = 0

    @T.macro
    def unpack_v(buf, dest, table, sc, batch_id, group_id, begin, length, sub):
        # The right operand of PV is staged in transposed order.
        with T.SimtVF(threads=256):
            for rr, col in T.Parallel(dp // 2, bn):
                d = sub * (dp // 2) + rr
                logical = begin + col
                if logical < length and d < dim:
                    physical = table[batch_id, logical // ps] * ps + logical % ps
                    v = read_cache(buf, physical, group_id, d)
                    if fp8kv:
                        v *= sc[batch_id, group_id] if group_scales else sc[0]
                    dest[rr, col] = T.cast(v, dtype)
                else:
                    dest[rr, col] = 0

    @T.prim_func
    def main(
        q: T.Tensor(qshape, dtype),
        k: T.Tensor(kshape, kdtype),
        v: T.Tensor(vshape, kdtype),
        cq: T.Tensor((batch + 1,), "int32"),
        ck: T.Tensor((batch + 1,), "int32"),
        off: T.Tensor((batch + 1, 2), "int32"),
        sink: T.Tensor((heads,), "float32") if sinks else None,
        table: T.Tensor((batch, pages), "int32") if ps else None,
        ks: T.Tensor(scale_shape, "float32") if fp8kv else None,
        vs: T.Tensor(scale_shape, "float32") if fp8kv else None,
        cos: T.Tensor((P, rot // 2), dtype) if rot else None,
        sin: T.Tensor((P, rot // 2), dtype) if rot else None,
        out: T.Tensor((kh, tiles, 2, rows, vp), dtype),
        lse: T.Tensor((kh, tiles, 2, rows), "float32") if with_lse else None,
    ):
        with T.MixedKernel(32) as (bx, sub):
            if raw_kh:
                qu = T.alloc_shared((rows, dp), dtype)
            if ps or raw_kh:
                ku = T.alloc_shared((bn // 2, dp), dtype)
                vu = T.alloc_shared((dp // 2, bn), dtype)
            if narrow_value_tile:
                qa = T.alloc_l0a((bm, depth_step), dtype)
                kb_l0 = T.alloc_l0b((bn, depth_step), dtype)
                pa_l0 = T.alloc_l0a((bm, bn), dtype)
                vb_l0 = T.alloc_l0b((value_tile, bn), dtype)
            ql = T.alloc_l1((bm, dp), dtype)
            kl = T.alloc_l1((bn, dp), dtype)
            vl = T.alloc_l1((value_tile, bn), dtype)
            pl = T.alloc_l1((bm, bn), dtype)
            score = T.alloc_l0c((bm, bn), "float32")
            pv = T.alloc_l0c((bm, value_tile), "float32")
            su = T.alloc_shared((rows, bn), "float32")
            masked = T.alloc_shared((rows, max(64, bn)), "float32")
            pu = T.alloc_shared((rows, max(64, bn)), dtype)
            partial = T.alloc_shared((rows, value_tile), "float32")
            acc = T.alloc_shared((rows, vp), "float32")
            maxv = T.alloc_shared((rows,), "float32")
            denom = T.alloc_shared((rows,), "float32")
            alpha = T.alloc_shared((rows,), "float32")
            result = T.alloc_shared((rows, value_tile), dtype)
            task = T.alloc_var("int32", init=bx)
            for b in T.serial(batch):
                lq = cq[b + 1] - cq[b]
                lk = ck[b + 1] - ck[b]
                offset = lk - lq
                jobs = T.ceildiv(lq * group, bm) * kh
                if omit_aligned:
                    jobs = T.if_then_else(
                        lq > 0 and lk >= lq and lq % 128 == 0 and lk % 128 == 0, 0, jobs
                    )
                while task < jobs:
                    g = task % kh
                    start = task // kh * bm
                    qp = off[b, 0] + start
                    kbase = off[b, 1]
                    if raw_kh:
                        qo = cq[b] + start
                        if start + bm <= lq:
                            T.copy(q[qo : qo + bm, g, :], ql)
                        else:
                            with T.SimtVF(threads=128):
                                for rr, dd in T.Parallel(rows, dp):
                                    row = start + sub * rows + rr
                                    if row < lq:
                                        qu[rr, dd] = q[cq[b] + row, g, dd]
                                    else:
                                        qu[rr, dd] = 0
                            T.dual_copy(qu, ql)
                    else:
                        T.copy(q[g, qp : qp + bm, :], ql)
                    with T.SimtVF(threads=128):
                        for rr, d in T.Parallel(rows, vp):
                            acc[rr, d] = 0
                        for rr in T.Parallel(rows):
                            m = T.alloc_local((1,), "float32")
                            m[0] = -3.402823466e38
                            denom[rr] = 0
                            if sinks:
                                s = sink[g * group + (start + sub * rows + rr) % group]
                                if s > -T.infinity("float32"):
                                    m[0] = T.min(s, T.float32(3.402823466e38))
                                    denom[rr] = 1
                            maxv[rr] = m[0]
                    qlo = start // group
                    qhi = T.min((start + bm - 1) // group, lq - 1)
                    first = T.max(0, qlo + offset - wl) // bn if wl >= 0 else 0
                    end = T.alloc_var("int32", init=lk)
                    if causal:
                        end = T.min(end, qhi + offset + 1)
                    if wr >= 0:
                        end = T.min(end, qhi + offset + wr + 1)
                    for j in T.serial(first, T.max(first, T.ceildiv(T.max(end, 0), bn))):
                        kb = kbase + j * bn
                        if raw_kh:
                            ph = g // (heads // raw_kh)
                            ko = ck[b] + j * bn
                            if (j + 1) * bn <= lk:
                                T.copy(k[ko : ko + bn, ph, :], kl)
                            else:
                                with T.SimtVF(threads=128):
                                    for rr, dd in T.Parallel(bn // 2, dp):
                                        row = j * bn + sub * (bn // 2) + rr
                                        if row < lk:
                                            ku[rr, dd] = k[ck[b] + row, ph, dd]
                                        else:
                                            ku[rr, dd] = 0
                                T.dual_copy(ku, kl)
                        elif ps:
                            if direct:
                                physical = table[b, j * bn // ps] * ps + j * bn % ps
                                T.copy(k[physical : physical + bn, g, :], kl)
                            else:
                                unpack_k(k, ku, table, ks, cos, sin, b, g, j * bn, lk, sub)
                                T.dual_copy(ku, kl)
                        else:
                            T.copy(k[g // (kh // storage_heads), kb : kb + bn, :], kl)
                        if narrow_value_tile:
                            for depth_tile in T.serial(dp // depth_step):
                                depth_start = depth_tile * depth_step
                                T.copy(ql[:, depth_start : depth_start + depth_step], qa)
                                T.copy(kl[:, depth_start : depth_start + depth_step], kb_l0)
                                T.gemm(
                                    qa, kb_l0, score, transpose_B=True, clear_accum=depth_tile == 0
                                )
                        else:
                            T.gemm(ql, kl, score, transpose_B=True, clear_accum=True)
                        T.dual_copy(score, su)
                        if cap or bn < 64:
                            with T.SimtVF(threads=128):
                                for rr, kk in T.Parallel(rows, max(64, bn)):
                                    qi = (start + sub * rows + rr) // group
                                    ki = j * bn + kk
                                    visible = kk < bn and ki < lk and qi < lq
                                    if causal:
                                        visible = visible and ki <= qi + offset
                                    if wl >= 0:
                                        visible = visible and ki >= qi + offset - wl
                                    if wr >= 0:
                                        visible = visible and ki <= qi + offset + wr
                                    if visible:
                                        score_value = su[rr, kk] * scale
                                        if cap:
                                            score_value = cap * T.tanh(score_value / cap)
                                        masked[rr, kk] = score_value
                                    else:
                                        masked[rr, kk] = -T.infinity("float32")
                        with T.SimdVF():
                            full = T.simd.pset(32)
                            one = T.simd.pset(32, "PAT_VL1")
                            for rr in T.serial(rows):
                                if cap or bn < 64:
                                    left = T.simd.vld(masked[rr, 0])
                                    right = (
                                        T.simd.vld(masked[rr, 64])
                                        if bn == 128
                                        else T.simd.vdup(-T.infinity("float32"), "float32", full)
                                    )
                                else:
                                    qi = (start + sub * rows + rr) // group
                                    lower = T.max(0, qi + offset - wl - j * bn) if wl >= 0 else 0
                                    upper = T.min(bn, lk - j * bn)
                                    if causal:
                                        upper = T.min(upper, qi + offset + 1 - j * bn)
                                    if wr >= 0:
                                        upper = T.min(upper, qi + offset + wr + 1 - j * bn)
                                    upper = T.if_then_else(qi < lq, upper, 0)
                                    neg = T.simd.vdup(-T.infinity("float32"), "float32", full)
                                    left = T.simd.vmuls(
                                        T.simd.vld(su[rr, 0]), T.float32(scale), full
                                    )
                                    left = T.simd.vsel(
                                        left,
                                        neg,
                                        T.simd.update_mask(T.min(64, T.max(upper, 0)), 32),
                                    )
                                    if wl >= 0:
                                        left = T.simd.vsel(
                                            neg, left, T.simd.update_mask(T.min(64, lower), 32)
                                        )
                                    if bn == 128:
                                        right = T.simd.vmuls(
                                            T.simd.vld(su[rr, 64]), T.float32(scale), full
                                        )
                                        right = T.simd.vsel(
                                            right,
                                            neg,
                                            T.simd.update_mask(T.min(64, T.max(upper - 64, 0)), 32),
                                        )
                                        if wl >= 0:
                                            right = T.simd.vsel(
                                                neg,
                                                right,
                                                T.simd.update_mask(
                                                    T.min(64, T.max(lower - 64, 0)), 32
                                                ),
                                            )
                                    else:
                                        right = neg
                                old = T.simd.vld(maxv[rr], dist="BRC_B32")
                                peak = T.simd.vmax(
                                    old,
                                    T.simd.vdupv(
                                        T.simd.vcmax(T.simd.vmax(left, right, full), full),
                                        full,
                                    ),
                                    full,
                                )
                                adjust = T.simd.vexpdif(old, peak, full)
                                p0 = T.simd.vexpdif(left, peak, full)
                                if bn == 128:
                                    p1 = T.simd.vexpdif(right, peak, full)
                                    row_sum = T.simd.vcadd(T.simd.vadd(p0, p1, full), full)
                                    T.simd.vsts(
                                        pu[rr, 64],
                                        T.simd.vcvt(
                                            p1,
                                            dtype,
                                            round="ROUND_R",
                                            sat=False,
                                            part=0,
                                        ),
                                        full,
                                        dist="PK_B32",
                                    )
                                else:
                                    row_sum = T.simd.vcadd(p0, full)
                                total = T.simd.vadd(
                                    T.simd.vmul(
                                        T.simd.vld(denom[rr], dist="BRC_B32"),
                                        adjust,
                                        full,
                                    ),
                                    row_sum,
                                    one,
                                )
                                T.simd.vsts(denom[rr], total, one, dist="ONEPT_B32")
                                T.simd.vsts(maxv[rr], peak, one, dist="ONEPT_B32")
                                T.simd.vsts(alpha[rr], adjust, one, dist="ONEPT_B32")
                                T.simd.vsts(
                                    pu[rr, 0],
                                    T.simd.vcvt(p0, dtype, round="ROUND_R", sat=False, part=0),
                                    full,
                                    dist="PK_B32",
                                )
                        T.dual_copy(pu[:, :bn], pl)
                        for value_part in T.serial(vp // value_tile):
                            col = value_part * value_tile
                            if raw_kh:
                                ph = g // (heads // raw_kh)
                                ko = ck[b] + j * bn
                                if (j + 1) * bn <= lk:
                                    T.copy(v[ko : ko + bn, ph, :], vl, transpose=True)
                                else:
                                    with T.SimtVF(threads=128):
                                        for rr, cc in T.Parallel(dp // 2, bn):
                                            dd = sub * (dp // 2) + rr
                                            row = j * bn + cc
                                            if row < lk:
                                                vu[rr, cc] = v[ck[b] + row, ph, dd]
                                            else:
                                                vu[rr, cc] = 0
                                    T.dual_copy(vu, vl)
                            elif ps:
                                if direct and (j + 1) * bn <= lk:
                                    physical = table[b, j * bn // ps] * ps + j * bn % ps
                                    T.copy(
                                        v[physical : physical + bn, g, :],
                                        vl,
                                        transpose=True,
                                    )
                                else:
                                    unpack_v(v, vu, table, vs, b, g, j * bn, lk, sub)
                                    T.dual_copy(vu, vl)
                            else:
                                T.copy(
                                    v[
                                        g // (kh // storage_heads),
                                        kb : kb + bn,
                                        col : col + value_tile,
                                    ],
                                    vl,
                                    transpose=True,
                                )
                            if narrow_value_tile:
                                T.copy(pl, pa_l0)
                                T.copy(vl, vb_l0)
                                T.gemm(pa_l0, vb_l0, pv, transpose_B=True, clear_accum=True)
                            else:
                                T.gemm(pl, vl, pv, transpose_B=True, clear_accum=True)
                            T.dual_copy(pv, partial)
                            if value_tile % 64 == 0:
                                with T.SimdVF():
                                    mask = T.simd.pset(32)
                                    for rr in T.serial(rows):
                                        weight = T.simd.vld(alpha[rr], dist="BRC_B32")
                                        for dd in T.serial(value_tile // 64):
                                            cur = T.simd.vld(partial[rr, dd * 64])
                                            old = T.simd.vld(acc[rr, col + dd * 64])
                                            val = T.simd.vadd(
                                                T.simd.vmul(old, weight, mask), cur, mask
                                            )
                                            T.simd.vsts(
                                                acc[rr, col + dd * 64], val, mask, dist="NORM_B32"
                                            )
                            else:
                                with T.SimtVF(threads=128):
                                    for rr, dd in T.Parallel(rows, value_tile):
                                        acc[rr, col + dd] = (
                                            acc[rr, col + dd] * alpha[rr] + partial[rr, dd]
                                        )
                    for value_part in T.serial(vp // value_tile):
                        col = value_part * value_tile
                        if value_tile % 64 == 0:
                            with T.SimdVF():
                                mask = T.simd.pset(32)
                                one_vec = T.simd.vdup(T.float32(1), "float32", mask)
                                for rr in T.serial(rows):
                                    divisor = T.simd.vmax(
                                        one_vec, T.simd.vld(denom[rr], dist="BRC_B32"), mask
                                    )
                                    for dd in T.serial(value_tile // 64):
                                        val = T.simd.vdiv(
                                            T.simd.vld(acc[rr, col + dd * 64]), divisor, mask
                                        )
                                        result_vec = T.simd.vcvt(
                                            val, dtype, round="ROUND_R", sat=False, part=0
                                        )
                                        T.simd.vsts(
                                            result[rr, dd * 64], result_vec, mask, dist="PK_B32"
                                        )
                        else:
                            with T.SimtVF(threads=128):
                                for rr, dd in T.Parallel(rows, value_tile):
                                    result[rr, dd] = T.cast(
                                        T.if_then_else(
                                            denom[rr] > 0, acc[rr, col + dd] / denom[rr], 0
                                        ),
                                        dtype,
                                    )
                        T.copy(result, out[g, qp // bm, sub, :, col : col + value_tile])
                    if with_lse:
                        with T.SimtVF(threads=128):
                            for rr in T.Parallel(rows):
                                maxv[rr] = T.if_then_else(
                                    denom[rr] > 0,
                                    maxv[rr] + T.log(denom[rr]),
                                    -T.infinity("float32"),
                                )
                    if with_lse:
                        T.copy(maxv, lse[g, qp // bm, sub, :])
                    task += 32
                task -= jobs

    program = tilelang.compile(main, target="ascend", out_idx=[])
    unpack = _unpack_attention(batch, heads, kh, dv, dtype, bm, with_lse)

    def run(*args):
        q = args[0]
        output_count = 2 if with_lse else 1
        tiles_count = (q.shape[0] + bm - 1) // bm + batch if raw_kh else q.shape[1] // bm
        shape = (kh, tiles_count, 2, rows)
        temp = [q.new_empty((*shape, vp))]
        if with_lse:
            temp.append(q.new_empty(shape, dtype=torch.float32))
        program(*args[:-output_count], *temp)
        unpack(*temp, args[3], args[5], *args[-output_count:])

    return run


class AscendGQAVarlenKernel(Kernel, GQAVarlenFwdInterface):
    devices = frozenset({"npu"})

    @classmethod
    def entry_for(cls, call):
        return call, lambda: cls(call)

    def __init__(self, call):
        super().__init__(device_index=call.device.index)
        self.call = call
        self.dtype = call.dtype
        self.aligned = None
        if (
            not call.is_fp8
            and not call.fuse_rope
            and call.is_causal
            and not call.softcap
            and call.window_size_right == -1
            and call.dim in (64, 128)
        ):
            self.aligned = tilelang.compile(
                prefill_query_major(call, 32, packed=True),
                target="ascend",
                out_idx=[],
                pass_configs=pass_configs(),
            )

    def forward(
        self,
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_kv,
        q_scale=None,
        k_scale=None,
        v_scale=None,
        rope_cos=None,
        rope_sin=None,
        sinks=None,
    ):
        c = self.call
        tq, h, dim = q.shape
        tk, g, _ = k.shape
        if not tq or not tk:
            return q.new_zeros((tq, h, dim), dtype=c.dtype)
        group = h // g
        bm, bn = 32, 128
        if dim > 128:
            bm = 64 if dim <= 512 and dim % 128 == 0 else bm
            # Separate packing avoids reserving three wide tail tiles in each Vector core.
            # Query tiles share one physical KV head across all heads of its group.
            off = _layout(c.batch, 1, bm, bn)(cu_seqlens_q, cu_seqlens_kv)
            packed = []
            scales_q = q_scale.repeat_interleave(group, 1) if c.is_fp8 and group > 1 else q_scale
            for kind, x, nheads, rows_per_tile, scale in (
                ("q", q, h, bm, scales_q),
                ("k", k, g, bn, k_scale),
                ("v", v, g, bn, v_scale),
            ):
                rot = c.rotary_dim if c.fuse_rope and kind != "v" else 0
                pack = _pack(
                    c.batch,
                    nheads,
                    nheads,
                    dim,
                    rows_per_tile,
                    self.dtype_str,
                    c.is_fp8,
                    kind,
                    rot,
                    c.rope_layout == "interleaved",
                )
                operands = [
                    x.view(torch.uint8) if c.is_fp8 else x,
                    cu_seqlens_q,
                    cu_seqlens_kv,
                    off,
                ]
                if c.is_fp8:
                    operands.append(scale)
                if rot:
                    operands.extend((rope_cos, rope_sin))
                packed.append(pack(*operands))
            core = _attention(
                c.batch,
                h,
                h,
                dim,
                self.dtype_str,
                c.is_causal,
                c.window_size_left,
                c.window_size_right,
                dim**-0.5 if c.sm_scale is None else c.sm_scale,
                c.softcap or 0.0,
                sinks is not None,
                bm,
                bn,
                packed_kh=g,
            )
            args = [*packed, cu_seqlens_q, cu_seqlens_kv, off]
            if sinks is not None:
                args.append(sinks)
            out = q.new_empty((tq, h, dim), dtype=c.dtype)
            core(*args, out)
            return out
        if not c.is_fp8 and not c.fuse_rope and dim % 16 == 0:
            bm = 64
            off = _layout(c.batch, 1, bm, bn, self.aligned is not None)(cu_seqlens_q, cu_seqlens_kv)
            core = _attention(
                c.batch,
                h,
                h,
                dim,
                self.dtype_str,
                c.is_causal,
                c.window_size_left,
                c.window_size_right,
                dim**-0.5 if c.sm_scale is None else c.sm_scale,
                c.softcap or 0.0,
                sinks is not None,
                bm,
                bn,
                raw_kh=g,
                omit_aligned=self.aligned is not None,
            )
            args = (q, k, v, cu_seqlens_q, cu_seqlens_kv, off)
            if sinks is not None:
                args = (*args, sinks)
            out = q.new_empty((tq, h, dim), dtype=c.dtype)
            if self.aligned is not None:
                den = sinks if sinks is not None else q.new_empty((h,), dtype=torch.float32)
                self.aligned(
                    q.unsqueeze(0),
                    k.unsqueeze(0),
                    v.unsqueeze(0),
                    den,
                    out.unsqueeze(0),
                    cu_seqlens_q,
                    cu_seqlens_kv,
                )
            core(*args, out)
            return out
        off = _layout(c.batch, group, bm, bn)(cu_seqlens_q, cu_seqlens_kv)
        packed = []
        for kind, x, sc in (("q", q, q_scale), ("k", k, k_scale), ("v", v, v_scale)):
            row_tile = bm if kind == "q" else bn
            rot = c.rotary_dim if c.fuse_rope and kind != "v" else 0
            fn = _pack(
                c.batch,
                h,
                g,
                dim,
                row_tile,
                self.dtype_str,
                c.is_fp8,
                kind,
                rot,
                c.rope_layout == "interleaved",
            )
            args = [
                x.view(torch.uint8) if c.is_fp8 else x,
                cu_seqlens_q,
                cu_seqlens_kv,
                off,
            ]
            if c.is_fp8:
                args.append(sc)
            if rot:
                args.extend((rope_cos, rope_sin))
            packed.append(fn(*args))
        fn = _attention(
            c.batch,
            h,
            g,
            dim,
            self.dtype_str,
            c.is_causal,
            c.window_size_left,
            c.window_size_right,
            c.sm_scale if c.sm_scale is not None else dim**-0.5,
            c.softcap or 0.0,
            sinks is not None,
            bm,
            bn,
        )
        args = (*packed, cu_seqlens_q, cu_seqlens_kv, off)
        if sinks is not None:
            args = (*args, sinks)
        out = q.new_empty((tq, h, dim), dtype=c.dtype)
        fn(*args, out)
        return out


register_kernel_type("GQAVarlenFwdOp", "ascend", AscendGQAVarlenKernel)
