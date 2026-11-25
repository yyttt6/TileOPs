"""Paged grouped-query attention with live block tables and packed request boundaries."""

import functools
import math
from dataclasses import replace

import tilelang
import tilelang.ascend.language as T
import torch

from tileops._kernels.ascend.attention.gqa_dense_prefill import (
    pass_configs,
    prefill_query_major,
)
from tileops._kernels.ascend.attention.gqa_varlen import (
    AscendGQAVarlenKernel,
    _attention,
    _layout,
    _pack,
)
from tileops._kernels.attention.call_spec import GQAPagedFwdInterface
from tileops._kernels.kernel_base import Kernel
from tileops.backend._registry import register_kernel_type


@functools.lru_cache(maxsize=32)
def _prefix(batch):
    @T.prim_func
    def main(lengths: T.Tensor((batch,), "int32"), ends: T.Tensor((batch + 1,), "int32")):
        with T.Kernel(1), T.SimtVF(threads=32):
            for _ in T.Parallel(1):
                total = T.alloc_local((1,), "int32")
                total[0] = 0
                ends[0] = 0
                for b in T.serial(batch):
                    total[0] += lengths[b]
                    ends[b + 1] = total[0]

    return tilelang.compile(main, target="ascend", out_idx=[1])


@functools.lru_cache(maxsize=48)
def _gather_prefill(batch, page, heads_kv, dim, dtype, aligned=False):
    """Pack live physical pages into request order for the dense prefill pipeline."""
    pool = T.dynamic("pool_rows")
    width = T.dynamic("max_pages")
    capacity = T.dynamic("packed_capacity")
    cols = heads_kv * dim
    per = math.gcd(page, 16)
    col_tile = math.gcd(cols, max(1, 8192 // per))
    segments = cols // col_tile

    @T.prim_func
    def main(
        k: T.Tensor((pool, cols), dtype),
        v: T.Tensor((pool, cols), dtype),
        lengths: T.Tensor((batch,), "int32"),
        table: T.Tensor((batch, width), "int32"),
        ends: T.Tensor((batch + 1,), "int32"),
        kout: T.Tensor((capacity, cols), dtype),
        vout: T.Tensor((capacity, cols), dtype),
    ):
        with T.Kernel(64) as core:
            kb = T.alloc_shared((per, col_tile), dtype)
            vb = T.alloc_shared((per, col_tile), dtype)
            tiles = (
                (T.ceildiv(width * page, 128) * (128 // per)) if aligned else width * (page // per)
            )
            per_request = tiles * segments
            for job in T.serial(core, batch * per_request, 64):
                b = job // per_request
                segment = job % segments
                row = (job % per_request) // segments * per
                if aligned:
                    if row < ends[b + 1] - ends[b]:
                        live = T.max(0, T.min(per, lengths[b] - row))
                        start = segment * col_tile
                        if live == per:
                            phys = table[b, row // page] * page + row % page
                            T.copy(k[phys : phys + per, start : start + col_tile], kb)
                            T.copy(v[phys : phys + per, start : start + col_tile], vb)
                        else:
                            T.fill(kb, 0)
                            T.fill(vb, 0)
                            if live > 0:
                                phys = table[b, row // page] * page + row % page
                                T.copy(
                                    k[phys : phys + live, start : start + col_tile],
                                    kb[:live, :],
                                )
                                T.copy(
                                    v[phys : phys + live, start : start + col_tile],
                                    vb[:live, :],
                                )
                        dst = ends[b] + row
                        T.copy(kb, kout[dst : dst + per, start : start + col_tile])
                        T.copy(vb, vout[dst : dst + per, start : start + col_tile])
                else:
                    live = T.min(per, lengths[b] - row)
                    if live > 0:
                        phys = table[b, row // page] * page + row % page
                        dst = ends[b] + row
                        start = segment * col_tile
                        T.copy(
                            k[phys : phys + live, start : start + col_tile],
                            kb[:live, :],
                        )
                        T.copy(
                            v[phys : phys + live, start : start + col_tile],
                            vb[:live, :],
                        )
                        T.copy(
                            kb[:live, :],
                            kout[dst : dst + live, start : start + col_tile],
                        )
                        T.copy(
                            vb[:live, :],
                            vout[dst : dst + live, start : start + col_tile],
                        )

    return tilelang.compile(main, target="ascend", out_idx=[])


@functools.lru_cache(maxsize=64)
def _short_partials(
    batch,
    heads,
    hk,
    dim,
    page,
    dtype,
    q8,
    k8,
    per_group,
    causal,
    left,
    right,
    cap,
    scale,
    splits,
):
    group = heads // hk
    bm = 32
    head_tiles = (group + bm - 1) // bm
    bn = 64
    half = bm // 2
    unit = math.gcd(page, bn // 2)
    tq = T.dynamic("queries")
    slots = T.dynamic("pool_slots")
    width = T.dynamic("pages")
    stype = (batch, hk) if per_group else (1,)
    qt = "uint8" if q8 else dtype
    kt = "uint8" if k8 else dtype

    @T.prim_func
    def main(
        q: T.Tensor((tq, heads, dim), qt),
        k: T.Tensor((slots, hk, dim), kt),
        v: T.Tensor((slots, hk, dim), kt),
        cu: T.Tensor((batch + 1,), "int32"),
        lengths: T.Tensor((batch,), "int32"),
        pages: T.Tensor((batch, width), "int32"),
        qs: T.Tensor((batch, hk), "float32") if q8 else None,
        ks: T.Tensor(stype, "float32") if k8 else None,
        vs: T.Tensor(stype, "float32") if k8 else None,
        origin: T.int32,
        numer: T.Tensor((tq, hk, head_tiles, splits, 2, half, dim), "float32"),
        stats: T.Tensor((tq, hk, head_tiles, splits, 2, 2, half), "float32"),
    ):
        with T.MixedKernel(tq * hk * head_tiles * splits) as (bx, sub):
            token = bx // (hk * head_tiles * splits)
            h = bx // (head_tiles * splits) % hk
            head_tile = bx // splits % head_tiles
            split = bx % splits
            logical_token = token + origin
            request = T.alloc_var("int32", init=0)
            high = T.alloc_var("int32", init=batch)
            while request < high:
                mid = (request + high) // 2
                if cu[mid + 1] <= logical_token:
                    request = mid + 1
                else:
                    high = mid
            length = lengths[request]
            position = logical_token - cu[request + 1] + length
            first = T.max(0, position - left) if left >= 0 else 0
            stop = T.min(length, position + 1) if causal else length
            if right >= 0:
                stop = T.min(stop, position + right + 1)
            t0 = first // bn
            ntiles = T.max(0, T.ceildiv(T.max(0, stop), bn) - t0)
            begin = t0 + ntiles * split // splits
            end = t0 + ntiles * (split + 1) // splits
            qu = T.alloc_shared((half, dim), dtype)
            if q8:
                qbytes = T.alloc_shared((half, dim), "uint8")
                qwords = T.view(qbytes, (half * dim // 4,), dtype="uint32")
                qcodes = T.view(qbytes, (half * dim,), dtype="float8_e4m3fn")
                qflat = T.reshape(qu, (half * dim,))
            ku = T.alloc_shared((bn // 2, dim), dtype)
            vu = T.alloc_shared((bn // 2, dim), dtype)
            if k8:
                kbytes = T.alloc_shared((bn // 2, dim), "uint8")
                vbytes = T.alloc_shared((bn // 2, dim), "uint8")
                kwords = T.view(kbytes, (bn * dim // 8,), dtype="uint32")
                vwords = T.view(vbytes, (bn * dim // 8,), dtype="uint32")
                kcodes = T.view(kbytes, (bn * dim // 2,), dtype="float8_e4m3fn")
                vcodes = T.view(vbytes, (bn * dim // 2,), dtype="float8_e4m3fn")
                kflat = T.reshape(ku, (bn * dim // 2,))
                vflat = T.reshape(vu, (bn * dim // 2,))
            ql = T.alloc_l1((bm, dim), dtype)
            kl = T.alloc_l1((bn, dim), dtype)
            vl = T.alloc_l1((bn, dim), dtype)
            pl = T.alloc_l1((bm, bn), dtype)
            qa_l0 = T.alloc_l0a((bm, 64), dtype)
            kb_l0 = T.alloc_l0b((bn, 64), dtype)
            a = T.alloc_l0a((bm, bn), dtype)
            b = T.alloc_l0b((dim, bn), dtype)
            score = T.alloc_l0c((bm, bn), "float32")
            pv = T.alloc_l0c((bm, dim), "float32")
            su = T.alloc_shared((half, bn), "float32")
            pu = T.alloc_shared((half, bn), dtype)
            part = T.alloc_shared((half, dim), "float32")
            acc = T.alloc_shared((half, dim), "float32")
            peak = T.alloc_shared((half,), "float32")
            den = T.alloc_shared((half,), "float32")
            alpha = T.alloc_shared((half,), "float32")
            factors = T.alloc_shared((8,), "float32")
            first_head = head_tile * bm + sub * half
            live_heads = T.max(0, T.min(half, group - first_head))
            source_head = h * group + first_head
            if q8:
                T.fill(qwords, T.uint32(0))
                if live_heads > 0:
                    T.copy(
                        q[token, source_head : source_head + live_heads, :],
                        qbytes[:live_heads, :],
                    )
                with T.SimdVF():
                    mask = T.simd.pset(32)
                    for j in T.serial(half * dim // 64):
                        values = T.simd.vcvt(T.simd.vld(qcodes[j * 64], dist="UNPK4_B8"), "float32")
                        T.simd.vsts(
                            qflat[j * 64],
                            T.simd.vcvt(values, dtype, round="ROUND_R", sat=False, part=0),
                            mask,
                            dist="PK_B32",
                        )
            else:
                T.fill(qu, 0)
                if live_heads > 0:
                    T.copy(
                        q[token, source_head : source_head + live_heads, :],
                        qu[:live_heads, :],
                    )
            with T.SimtVF(threads=128):
                for rr, dd in T.Parallel(half, dim):
                    acc[rr, dd] = 0
                for rr in T.Parallel(half):
                    peak[rr] = -3.402823466e38
                    den[rr] = 0
                for _ in T.Parallel(1):
                    qf = qs[request, h] if q8 else 1
                    kf = (ks[request, h] if per_group else ks[0]) if k8 else 1
                    vf = (vs[request, h] if per_group else vs[0]) if k8 else 1
                    factors[0] = scale * qf * kf
                    factors[1] = vf
            T.dual_copy(qu, ql)
            for it in T.serial(begin, end):
                start = it * bn
                if not k8 and page % bn == 0 and start >= first and start + bn <= stop:
                    addr = pages[request, start // page] * page + start % page
                    T.copy(k[addr : addr + bn, h, :], kl)
                    T.copy(v[addr : addr + bn, h, :], vl)
                else:
                    if k8:
                        T.fill(kwords, T.uint32(0))
                        T.fill(vwords, T.uint32(0))
                    else:
                        T.fill(ku, 0)
                        T.fill(vu, 0)
                    for pg in T.serial(bn // (2 * unit)):
                        row = start + sub * (bn // 2) + pg * unit
                        lo = T.max(first, row)
                        hi = T.min(stop, row + unit)
                        if lo < hi:
                            addr = pages[request, lo // page] * page + lo % page
                            dst = pg * unit + lo - row
                            count = hi - lo
                            if k8:
                                T.copy(
                                    k[addr : addr + count, h, :],
                                    kbytes[dst : dst + count, :],
                                )
                                T.copy(
                                    v[addr : addr + count, h, :],
                                    vbytes[dst : dst + count, :],
                                )
                            else:
                                T.copy(
                                    k[addr : addr + count, h, :],
                                    ku[dst : dst + count, :],
                                )
                                T.copy(
                                    v[addr : addr + count, h, :],
                                    vu[dst : dst + count, :],
                                )
                    if k8:
                        with T.SimdVF():
                            mask = T.simd.pset(32)
                            for j in T.serial(bn * dim // 128):
                                key = T.simd.vcvt(
                                    T.simd.vld(kcodes[j * 64], dist="UNPK4_B8"),
                                    "float32",
                                )
                                value = T.simd.vcvt(
                                    T.simd.vld(vcodes[j * 64], dist="UNPK4_B8"),
                                    "float32",
                                )
                                T.simd.vsts(
                                    kflat[j * 64],
                                    T.simd.vcvt(key, dtype, round="ROUND_R", sat=False, part=0),
                                    mask,
                                    dist="PK_B32",
                                )
                                T.simd.vsts(
                                    vflat[j * 64],
                                    T.simd.vcvt(value, dtype, round="ROUND_R", sat=False, part=0),
                                    mask,
                                    dist="PK_B32",
                                )
                    T.dual_copy(ku, kl)
                    T.dual_copy(vu, vl)
                for depth in T.serial(dim // 64):
                    start_d = depth * 64
                    T.copy(ql[:, start_d : start_d + 64], qa_l0)
                    T.copy(kl[:, start_d : start_d + 64], kb_l0)
                    T.gemm(qa_l0, kb_l0, score, transpose_B=True, clear_accum=depth == 0)
                T.dual_copy(score, su)
                if cap:
                    with T.SimtVF(threads=128):
                        for rr, col in T.Parallel(half, bn):
                            su[rr, col] = cap * T.tanh(su[rr, col] * factors[0] / cap)
                with T.SimdVF():
                    mask = T.simd.pset(32)
                    one = T.simd.pset(32, "PAT_VL1")
                    neg = T.simd.vdup(-T.infinity("float32"), "float32", mask)
                    s = T.simd.vld(factors[0], dist="BRC_B32")
                    for rr in T.serial(half):
                        z = T.simd.vld(su[rr, 0])
                        if not cap:
                            z = T.simd.vmul(z, s, mask)
                        z = T.simd.vsel(
                            z,
                            neg,
                            T.simd.update_mask(T.min(bn, T.max(0, stop - start)), 32),
                        )
                        z = T.simd.vsel(
                            neg,
                            z,
                            T.simd.update_mask(T.min(bn, T.max(0, first - start)), 32),
                        )
                        old = T.simd.vld(peak[rr], dist="BRC_B32")
                        next_peak = T.simd.vmax(
                            old, T.simd.vdupv(T.simd.vcmax(z, mask), mask), mask
                        )
                        factor = T.simd.vexpdif(old, next_peak, mask)
                        p = T.simd.vexpdif(z, next_peak, mask)
                        dsum = T.simd.vadd(
                            T.simd.vmul(T.simd.vld(den[rr], dist="BRC_B32"), factor, mask),
                            T.simd.vcadd(p, mask),
                            one,
                        )
                        T.simd.vsts(peak[rr], next_peak, one, dist="ONEPT_B32")
                        T.simd.vsts(den[rr], dsum, one, dist="ONEPT_B32")
                        T.simd.vsts(alpha[rr], factor, one, dist="ONEPT_B32")
                        T.simd.vsts(
                            pu[rr, 0],
                            T.simd.vcvt(p, dtype, round="ROUND_R", sat=False, part=0),
                            mask,
                            dist="PK_B32",
                        )
                T.dual_copy(pu, pl)
                T.copy(pl, a)
                T.copy(vl, b, transpose=True)
                T.gemm(a, b, pv, transpose_B=True, clear_accum=True)
                T.dual_copy(pv, part)
                with T.SimdVF():
                    mask = T.simd.pset(32)
                    for rr in T.serial(half):
                        factor = T.simd.vld(alpha[rr], dist="BRC_B32")
                        for dd in T.serial(dim // 64):
                            old = T.simd.vld(acc[rr, dd * 64])
                            cur = T.simd.vld(part[rr, dd * 64])
                            T.simd.vsts(
                                acc[rr, dd * 64],
                                T.simd.vadd(T.simd.vmul(old, factor, mask), cur, mask),
                                mask,
                                dist="NORM_B32",
                            )
            if k8:
                with T.SimdVF():
                    mask = T.simd.pset(32)
                    s = T.simd.vld(factors[1], dist="BRC_B32")
                    for rr in T.serial(half):
                        for dd in T.serial(dim // 64):
                            T.simd.vsts(
                                acc[rr, dd * 64],
                                T.simd.vmul(T.simd.vld(acc[rr, dd * 64]), s, mask),
                                mask,
                                dist="NORM_B32",
                            )
            T.copy(acc, numer[token, h, head_tile, split, sub, :, :])
            T.copy(peak, stats[token, h, head_tile, split, sub, 0, :])
            T.copy(den, stats[token, h, head_tile, split, sub, 1, :])

    return tilelang.compile(main, target="ascend", out_idx=[-2, -1])


@functools.lru_cache(maxsize=64)
def _short_merge(heads, hk, dim, dtype, splits, sinks):
    group = heads // hk
    bm = 32
    head_tiles = (group + bm - 1) // bm
    half = bm // 2
    tq = T.dynamic("queries")

    @T.prim_func
    def main(
        num: T.Tensor((tq, hk, head_tiles, splits, 2, half, dim), "float32"),
        stats: T.Tensor((tq, hk, head_tiles, splits, 2, 2, half), "float32"),
        sink: T.Tensor((heads,), "float32") if sinks else None,
        out: T.Tensor((tq, heads, dim), dtype),
    ):
        with T.Kernel(64) as block:
            buf = T.alloc_shared((splits, dim), "float32")
            factor = T.alloc_shared((splits,), "float32")
            result = T.alloc_shared((dim,), dtype)
            for job in T.serial(block, tq * heads, 64):
                token = job // heads
                head = job % heads
                kh = head // group
                head_tile = head % group // bm
                rr = head % group % bm
                sub = rr // half
                row = rr % half
                for j in T.serial(splits):
                    T.copy(num[token, kh, head_tile, j, sub, row, :], buf[j, :])
                with T.SimtVF(threads=32):
                    for _ in T.Parallel(1):
                        peak = T.alloc_local((1,), "float32")
                        den = T.alloc_local((1,), "float32")
                        peak[0] = -3.402823466e38
                        if sinks:
                            peak[0] = T.max(peak[0], T.min(sink[head], T.float32(3.402823466e38)))
                        for j in T.serial(splits):
                            peak[0] = T.max(peak[0], stats[token, kh, head_tile, j, sub, 0, row])
                        den[0] = 0
                        if sinks:
                            den[0] = T.exp(T.min(sink[head], T.float32(3.402823466e38)) - peak[0])
                        for j in T.serial(splits):
                            weight = T.exp(stats[token, kh, head_tile, j, sub, 0, row] - peak[0])
                            den[0] += weight * stats[token, kh, head_tile, j, sub, 1, row]
                        for j in T.serial(splits):
                            w = T.exp(stats[token, kh, head_tile, j, sub, 0, row] - peak[0])
                            factor[j] = T.if_then_else(den[0] > 0, w / den[0], 0)
                with T.SimdVF():
                    mask = T.simd.pset(32)
                    for col in T.serial(dim // 64):
                        accum = T.simd.alloc_local((1,), "float32")
                        accum[0] = T.simd.vdup(T.float32(0), "float32", mask)
                        for j in T.serial(splits):
                            v = T.simd.vld(buf[j, col * 64])
                            w = T.simd.vld(factor[j], dist="BRC_B32")
                            accum[0] = T.simd.vadd(accum[0], T.simd.vmul(v, w, mask), mask)
                        T.simd.vsts(
                            result[col * 64],
                            T.simd.vcvt(accum[0], dtype, round="ROUND_R", sat=False, part=0),
                            mask,
                            dist="PK_B32",
                        )
                T.copy(result, out[token, head, :])

    return tilelang.compile(main, target="ascend", out_idx=[-1])


@functools.lru_cache(maxsize=32)
def _unpage_fp8(batch, hk, dim, page, tile, dtype):
    """Gather raw E4M3 page rows into exactly representable Cube operands."""
    unit = math.gcd(16, page)
    total = T.dynamic("physical_rows")
    pages = T.dynamic("page_width")
    cap = T.dynamic("capacity")

    @T.prim_func
    def main(
        k: T.Tensor((total, hk, dim), "uint8"),
        v: T.Tensor((total, hk, dim), "uint8"),
        lengths: T.Tensor((batch,), "int32"),
        table: T.Tensor((batch, pages), "int32"),
        off: T.Tensor((batch + 1, 2), "int32"),
        keys: T.Tensor((hk, cap, dim), dtype),
        values: T.Tensor((hk, cap, dim), dtype),
    ):
        with T.Kernel(64) as core:
            k8 = T.alloc_shared((unit, dim), "uint8")
            v8 = T.alloc_shared((unit, dim), "uint8")
            kw = T.view(k8, (unit * dim // 4,), dtype="uint32")
            vw = T.view(v8, (unit * dim // 4,), dtype="uint32")
            kc = T.view(k8, (unit * dim,), dtype="float8_e4m3fn")
            vc = T.view(v8, (unit * dim,), dtype="float8_e4m3fn")
            ko = T.alloc_shared((unit, dim), dtype)
            vo = T.alloc_shared((unit, dim), dtype)
            kf = T.reshape(ko, (unit * dim,))
            vf = T.reshape(vo, (unit * dim,))
            limit = T.ceildiv(pages * page, tile) * (tile // unit)
            for job in T.serial(core, batch * limit * hk, 64):
                b = job // (hk * limit)
                head = job % hk
                row = job // hk % limit * unit
                length = lengths[b]
                if row < T.ceildiv(length, tile) * tile:
                    n = T.min(unit, T.max(0, length - row))
                    if n < unit:
                        T.fill(kw, T.uint32(0))
                        T.fill(vw, T.uint32(0))
                    if n > 0:
                        src = table[b, row // page] * page + row % page
                        T.copy(k[src : src + n, head, :], k8[:n, :])
                        T.copy(v[src : src + n, head, :], v8[:n, :])
                    with T.SimdVF():
                        pred = T.simd.pset(32)
                        for part in T.serial(unit * dim // 64):
                            kr = T.simd.vcvt(T.simd.vld(kc[part * 64], dist="UNPK4_B8"), "float32")
                            vr = T.simd.vcvt(T.simd.vld(vc[part * 64], dist="UNPK4_B8"), "float32")
                            T.simd.vsts(
                                kf[part * 64],
                                T.simd.vcvt(kr, dtype, round="ROUND_R", sat=False, part=0),
                                pred,
                                dist="PK_B32",
                            )
                            T.simd.vsts(
                                vf[part * 64],
                                T.simd.vcvt(vr, dtype, round="ROUND_R", sat=False, part=0),
                                pred,
                                dist="PK_B32",
                            )
                    dest = off[b, 1] + row
                    T.copy(ko, keys[head, dest : dest + unit, :])
                    T.copy(vo, values[head, dest : dest + unit, :])

    return tilelang.compile(main, target="ascend", out_idx=[])


@functools.lru_cache(maxsize=48)
def _decode_cache(batch, page, hk, dim, dtype, bn, tail_only):
    """Copy complete logical caches or just the final partial key tile into padded storage."""
    pool = T.dynamic("pool_rows")
    width = T.dynamic("table_width")
    capacity = T.dynamic("destination_rows")
    cols = hk * dim
    unit = math.gcd(page, 16)
    col_tile = math.gcd(cols, max(1, 8192 // unit))
    segments = cols // col_tile

    @T.prim_func
    def main(
        k: T.Tensor((pool, cols), dtype),
        v: T.Tensor((pool, cols), dtype),
        lengths: T.Tensor((batch,), "int32"),
        table: T.Tensor((batch, width), "int32"),
        keys: T.Tensor((batch, capacity, cols), dtype),
        values: T.Tensor((batch, capacity, cols), dtype),
    ):
        with T.Kernel(64) as core:
            kb = T.alloc_shared((unit, col_tile), dtype)
            vb = T.alloc_shared((unit, col_tile), dtype)
            blocks = (bn // unit) if tail_only else T.ceildiv(width * page, bn) * (bn // unit)
            for job in T.serial(core, batch * blocks * segments, 64):
                req = job // (blocks * segments)
                segment = job % segments
                dst = job // segments % blocks * unit
                length = lengths[req]
                active = (length % bn != 0) if tail_only else (dst < T.ceildiv(length, bn) * bn)
                if active:
                    row = (length // bn * bn + dst) if tail_only else dst
                    n = T.min(unit, T.max(0, length - row))
                    col = segment * col_tile
                    if n < unit:
                        T.fill(kb, 0)
                        T.fill(vb, 0)
                    if n > 0:
                        src = table[req, row // page] * page + row % page
                        T.copy(k[src : src + n, col : col + col_tile], kb[:n, :])
                        T.copy(v[src : src + n, col : col + col_tile], vb[:n, :])
                    T.copy(kb, keys[req, dst : dst + unit, col : col + col_tile])
                    T.copy(vb, values[req, dst : dst + unit, col : col + col_tile])

    return tilelang.compile(main, target="ascend", out_idx=[])


@functools.lru_cache(maxsize=64)
def _aligned_ranges(batch):
    @T.prim_func
    def main(
        cu: T.Tensor((batch + 1,), "int32"),
        lengths: T.Tensor((batch,), "int32"),
        qout: T.Tensor((batch + 1,), "int32"),
        kout: T.Tensor((batch + 1,), "int32"),
    ):
        with T.Kernel(1), T.SimtVF(threads=32):
            for _ in T.Parallel(1):
                qa = T.alloc_local((1,), "int32")
                ka = T.alloc_local((1,), "int32")
                qa[0] = 0
                ka[0] = 0
                for req in T.serial(batch):
                    qout[req] = qa[0]
                    kout[req] = ka[0]
                    lk = lengths[req]
                    live = T.min(cu[req + 1] - cu[req], lk)
                    if live > 0:
                        nk = T.ceildiv(lk, 128) * 128
                        qa[0] += T.ceildiv(live + nk - lk, 128) * 128
                        ka[0] += nk
                qout[batch] = qa[0]
                kout[batch] = ka[0]

    return tilelang.compile(main, target="ascend", out_idx=[2, 3])


@functools.lru_cache(maxsize=64)
def _align_queries(batch, heads, dim, dtype, pack):
    width = heads * dim
    elements = 8192
    tq = T.dynamic("tokens")
    cap = T.dynamic("buffer_rows")

    @T.prim_func
    def main(
        raw: T.Tensor((tq, width), dtype),
        padded: T.Tensor((cap, width), dtype),
        cu: T.Tensor((batch + 1,), "int32"),
        lengths: T.Tensor((batch,), "int32"),
        qoffset: T.Tensor((batch + 1,), "int32"),
        koffset: T.Tensor((batch + 1,), "int32"),
    ):
        with T.Kernel(64) as core:
            buf = T.alloc_shared((elements,), dtype)
            raw_flat = T.reshape(raw, (tq * width,))
            padded_flat = T.reshape(padded, (cap * width,))
            for req in T.serial(batch):
                qs = cu[req]
                lq = cu[req + 1] - qs
                lk = lengths[req]
                live = T.min(lq, lk)
                start_raw = qs + lq - live
                pq = qoffset[req]
                nq = qoffset[req + 1] - pq
                nk = koffset[req + 1] - koffset[req]
                start_padded = pq + nq - live - (nk - lk)
                for job in T.serial(core, T.ceildiv(live * width, elements), 64):
                    offset = job * elements
                    n = T.min(elements, live * width - offset)
                    r0 = start_raw * width + offset
                    p0 = start_padded * width + offset
                    if n == elements:
                        if pack:
                            T.copy(raw_flat[r0 : r0 + elements], buf)
                            T.copy(buf, padded_flat[p0 : p0 + elements])
                        else:
                            T.copy(padded_flat[p0 : p0 + elements], buf)
                            T.copy(buf, raw_flat[r0 : r0 + elements])
                    else:
                        if pack:
                            T.copy(raw_flat[r0 : r0 + n], buf[:n])
                            T.copy(buf[:n], padded_flat[p0 : p0 + n])
                        else:
                            T.copy(padded_flat[p0 : p0 + n], buf[:n])
                            T.copy(buf[:n], raw_flat[r0 : r0 + n])

    return tilelang.compile(main, target="ascend", out_idx=[])


@functools.lru_cache(maxsize=32)
def _aligned_prefill(call):
    return tilelang.compile(
        prefill_query_major(call, 32, packed=True),
        target="ascend",
        out_idx=[],
        pass_configs=pass_configs(),
    )


class AscendGQAPagedKernel(Kernel, GQAPagedFwdInterface):
    devices = frozenset({"npu"})

    @classmethod
    def entry_for(cls, call):
        return call, lambda: cls(call)

    def __init__(self, call):
        super().__init__(device_index=call.device.index)
        self.call = call
        self.dtype = call.dtype
        self.prefill = None

    def forward(
        self,
        q,
        k_pool,
        v_pool,
        cache_seqlens,
        page_table,
        cu_seqlens_q=None,
        q_scale=None,
        k_scale=None,
        v_scale=None,
        rope_cos=None,
        rope_sin=None,
        sinks=None,
    ):
        c = self.call
        tq, h, dim = q.shape
        total, g, _ = k_pool.shape
        b, pages = page_table.shape
        if not tq or not total or not pages:
            return q.new_zeros((tq, h, dim), dtype=c.dtype)
        group = h // g
        ps = c.page_size
        bm = 32
        # One score vector spans 64 keys. Sub-page transfers may cross physical pages.
        bn = 128 if dim <= 128 and ps % 128 == 0 else 64
        cq = cu_seqlens_q
        if cq is None:
            if not c.is_uniform:
                raise ValueError("packed query boundaries are required")
            cq = torch.arange(b + 1, device=q.device, dtype=torch.int32) * (tq // b)
        rot = c.rotary_dim if c.fuse_rope else 0
        interleaved = c.rope_layout == "interleaved"
        q8 = q.dtype == torch.float8_e4m3fn
        k8 = k_pool.dtype == torch.float8_e4m3fn
        compute_dtype = torch.float16 if q8 else c.dtype
        compute_name = str(compute_dtype).removeprefix("torch.")
        if rot and rope_cos.dtype != compute_dtype:
            rope_cos, rope_sin = rope_cos.to(compute_dtype), rope_sin.to(compute_dtype)
        if (
            not c.is_fp8
            and not rot
            and c.is_causal
            and not c.softcap
            and c.window_size_right == -1
            and dim in (64, 128)
            and ps % 64
            and tq >= 64 * b
        ):
            qcap = ((tq + 254 * b + 127) // 128) * 128
            kcap = b * ((pages * ps + 127) // 128) * 128
            if 4 * (qcap * h + kcap * g) * dim <= 2 * 1024**3:
                pq, pk = _aligned_ranges(b)(cq, cache_seqlens)
                query = q.new_zeros((qcap, h, dim))
                keys = q.new_empty((kcap, g, dim))
                values = torch.empty_like(keys)
                _align_queries(b, h, dim, compute_name, True)(
                    q.view(tq, h * dim),
                    query.view(qcap, h * dim),
                    cq,
                    cache_seqlens,
                    pq,
                    pk,
                )
                _gather_prefill(b, ps, g, dim, compute_name, True)(
                    k_pool.view(total, g * dim),
                    v_pool.view(total, g * dim),
                    cache_seqlens,
                    page_table,
                    pk,
                    keys.view(kcap, g * dim),
                    values.view(kcap, g * dim),
                )
                result = torch.empty_like(query)
                sink = sinks if sinks is not None else q.new_empty((h,), dtype=torch.float32)
                _aligned_prefill(c)(
                    query.unsqueeze(0),
                    keys.unsqueeze(0),
                    values.unsqueeze(0),
                    sink,
                    result.unsqueeze(0),
                    pq,
                    pk,
                )
                out = torch.zeros_like(q)
                _align_queries(b, h, dim, compute_name, False)(
                    out.view(tq, h * dim),
                    result.view(qcap, h * dim),
                    cq,
                    cache_seqlens,
                    pq,
                    pk,
                )
                return out
        capacity = b * pages * ps
        if (
            k8
            and not rot
            and tq > 4 * b
            and dim in (64, 128, 256)
            and 4 * capacity * g * dim <= 3 * 1024**3
        ):
            bm, bn = (64 if dim > 128 else 32), 128
            qgroups = h if dim > 128 else g
            ck = _prefix(b)(cache_seqlens)
            off = _layout(b, h // qgroups, bm, bn)(cq, ck)
            cap = ((capacity + bn - 1) // bn + b) * bn
            keys = torch.empty((g, cap, dim), device=q.device, dtype=compute_dtype)
            values = torch.empty_like(keys)
            _unpage_fp8(b, g, dim, ps, bn, compute_name)(
                k_pool.view(torch.uint8),
                v_pool.view(torch.uint8),
                cache_seqlens,
                page_table,
                off,
                keys,
                values,
            )
            prep = _pack(b, h, qgroups, dim, bm, compute_name, q8, "q", 0, False)
            if q8:
                unit_scales = torch.ones((b, qgroups), device=q.device, dtype=torch.float32)
                qp = prep(q.view(torch.uint8), cq, ck, off, unit_scales)
            else:
                qp = prep(q, cq, ck, off)
            kf = k_scale.expand(b, g)
            vf = v_scale.expand(b, g)
            factors = torch.stack(((q_scale * kf) if q8 else kf, vf), dim=-1)
            core = _attention(
                b,
                h,
                qgroups,
                dim,
                compute_name,
                c.is_causal,
                c.window_size_left,
                c.window_size_right,
                c.sm_scale,
                c.softcap or 0.0,
                sinks is not None,
                bm,
                bn,
                packed_kh=(g if dim > 128 else 0),
                operand_scales=True,
                return_dtype=self.dtype_str,
            )
            args = [qp, keys, values, cq, ck, off]
            if sinks is not None:
                args.append(sinks)
            out = q.new_empty((tq, h, dim), dtype=c.dtype)
            core(*args, factors, out)
            return out
        if (k8 or tq <= 4 * b) and not rot and dim in (64, 128, 256):
            if not k8 and ps % 64:
                # Coalesce logical pages into DMA tiles, including zeroes beyond the last live key.
                capacity = ((pages * ps + 63) // 64) * 64
                if 4 * b * capacity * g * dim <= 2 * 1024**3:
                    keys = q.new_empty((b, capacity, g, dim))
                    values = torch.empty_like(keys)
                    prepare = _decode_cache(b, ps, g, dim, compute_name, 64, False)
                    prepare(
                        k_pool.reshape(total, g * dim),
                        v_pool.reshape(total, g * dim),
                        cache_seqlens,
                        page_table,
                        keys.view(b, capacity, g * dim),
                        values.view(b, capacity, g * dim),
                    )
                    k_pool, v_pool = (
                        keys.reshape(-1, g, dim),
                        values.reshape(-1, g, dim),
                    )
                    ps, pages = 64, capacity // 64
                    page_table = torch.arange(
                        b * pages, device=q.device, dtype=torch.int32
                    ).reshape(b, pages)
            # Keep E4M3 operands exact for the products; the three scales enter in FP32.
            splits = min(8, max(1, (128 + tq * g - 1) // (tq * g)), (pages * ps + 63) // 64)
            fn = _short_partials(
                b,
                h,
                g,
                dim,
                ps,
                compute_name,
                q8,
                k8,
                c.per_request_kv_scale,
                c.is_causal,
                c.window_size_left,
                c.window_size_right,
                c.softcap or 0.0,
                c.sm_scale,
                splits,
            )
            bm = 32
            padded_heads = (group + bm - 1) // bm * bm
            bytes_per_token = g * splits * padded_heads * (dim + 2) * 4
            chunk = max(1, (128 * 1024**2) // bytes_per_token)
            merge = _short_merge(h, g, dim, self.dtype_str, splits, sinks is not None)
            outs = []
            for origin in range(0, tq, chunk):
                part = q[origin : origin + chunk]
                args = [
                    part.view(torch.uint8) if q8 else part,
                    k_pool.view(torch.uint8) if k8 else k_pool,
                    v_pool.view(torch.uint8) if k8 else v_pool,
                    cq,
                    cache_seqlens,
                    page_table,
                ]
                if q8:
                    args.append(q_scale)
                if k8:
                    args.extend((k_scale, v_scale))
                args.append(origin)
                num, stats = fn(*args)
                outs.append(merge(num, stats, sinks) if sinks is not None else merge(num, stats))
            return outs[0] if len(outs) == 1 else torch.cat(outs, dim=0)
        ck = _prefix(b)(cache_seqlens)
        # Capacity includes independent requests that reference the same physical page.
        capacity = b * pages * ps
        if (tq >= 4 * b or rot) and 2 * capacity * g * dim * q.element_size() <= (
            3 * 1024**3 if not k8 and dim > 128 and capacity <= 2 * total else 512 * 1024**2
        ):
            raw_dtype = torch.uint8 if k8 else k_pool.dtype
            keys = torch.empty((capacity, g, dim), device=q.device, dtype=raw_dtype)
            values = torch.empty_like(keys)
            gather = _gather_prefill(b, ps, g, dim, str(raw_dtype).removeprefix("torch."))
            gather(
                k_pool.view(raw_dtype).view(total, g * dim),
                v_pool.view(raw_dtype).view(total, g * dim),
                cache_seqlens,
                page_table,
                ck,
                keys.view(capacity, g * dim),
                values.view(capacity, g * dim),
            )
            if not k8:
                if self.prefill is None:
                    self.prefill = AscendGQAVarlenKernel(replace(c, page_size=0, is_uniform=False))
                return self.prefill(
                    q,
                    keys,
                    values,
                    cq,
                    ck,
                    rope_cos=rope_cos,
                    rope_sin=rope_sin,
                    sinks=sinks,
                )
            # Convert each visited key and value once before all of its query tiles.
            off = _layout(b, group, bm, 128)(cq, ck)
            packed = []
            for kind, tensor, scale, lowbit in (
                ("q", q, q_scale, q8),
                ("k", keys, k_scale, True),
                ("v", values, v_scale, True),
            ):
                rotate = rot if kind != "v" else 0
                fn = _pack(
                    b,
                    h,
                    g,
                    dim,
                    bm if kind == "q" else 128,
                    compute_name,
                    lowbit,
                    kind,
                    rotate,
                    interleaved,
                )
                args = [tensor.view(torch.uint8) if lowbit else tensor, cq, ck, off]
                if lowbit:
                    args.append(scale.expand(b, g).contiguous() if scale.ndim == 1 else scale)
                if rotate:
                    args.extend((rope_cos, rope_sin))
                packed.append(fn(*args))
            core = _attention(
                b,
                h,
                g,
                dim,
                compute_name,
                c.is_causal,
                c.window_size_left,
                c.window_size_right,
                c.sm_scale,
                c.softcap or 0.0,
                sinks is not None,
                bm,
                128,
            )
            args = [*packed, cq, ck, off]
            if sinks is not None:
                args.append(sinks)
            out = q.new_empty((tq, h, dim), dtype=compute_dtype)
            core(*args, out)
            return out.to(c.dtype)
        off = _layout(b, group, bm, bn)(cq, ck)
        pack_q = _pack(b, h, g, dim, bm, compute_name, q8, "q", rot, interleaved)
        inputs = [q.view(torch.uint8) if q8 else q, cq, ck, off]
        if q8:
            inputs.append(q_scale)
        if rot:
            inputs.extend((rope_cos, rope_sin))
        qp = pack_q(*inputs)
        pool = (ps, total, pages, k8, c.per_request_kv_scale, rot, interleaved)
        core = _attention(
            b,
            h,
            g,
            dim,
            compute_name,
            c.is_causal,
            c.window_size_left,
            c.window_size_right,
            c.sm_scale if c.sm_scale is not None else dim**-0.5,
            c.softcap or 0.0,
            sinks is not None,
            bm,
            bn,
            pool,
        )
        args = [
            qp,
            k_pool.view(torch.uint8) if k8 else k_pool,
            v_pool.view(torch.uint8) if k8 else v_pool,
            cq,
            ck,
            off,
        ]
        if sinks is not None:
            args.append(sinks)
        args.append(page_table)
        if k8:
            args.extend((k_scale, v_scale))
        if rot:
            args.extend((rope_cos, rope_sin))
        out = q.new_empty((tq, h, dim), dtype=compute_dtype)
        core(*args, out)
        return out.to(c.dtype)


register_kernel_type("GQAPagedFwdOp", "ascend", AscendGQAPagedKernel)
