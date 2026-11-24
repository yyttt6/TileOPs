"""Split-KV latent attention for dense and paged decode on Ascend."""

import functools

import tilelang
import tilelang.ascend.language as T
import torch
import torch.nn.functional as F

from tileops._kernels.attention.call_spec import (
    MLADecodeFwdInterface,
    MLAPagedFwdInterface,
)
from tileops._kernels.kernel_base import Kernel
from tileops.backend._registry import register_kernel_type


@functools.lru_cache(maxsize=64)
def _partials(
    sq,
    h,
    g,
    dim,
    rope,
    dtype,
    scale,
    ns,
    ps,
    fp8,
    causal,
    fuse_output=False,
    wide_keys=False,
):
    dense = not ps
    b = T.dynamic("batch")
    nk = T.dynamic("keys_or_pool")
    width = T.dynamic("page_table_width")
    hg = h // g
    bm = 32 if hg <= 32 else 64
    bn = 128 if wide_keys else 64
    assert not wide_keys or (dense and not fp8)
    ht = (hg + bm - 1) // bm
    qp = (dim + 63) // 64 * 64
    rp = (rope + 63) // 64 * 64
    dq = qp + rp
    native_page = fp8 and ps % bn == 0 and dim == qp and rope == rp
    kt = "uint8" if fp8 else dtype
    shape = (b, nk, g, qp) if dense else (nk, dim + rope)
    partial_shape = (b, sq, h, dim) if fuse_output else (b, sq, g, ht, ns, bm, qp)
    assert not fuse_output or (dense and ns == 1 and hg % bm == 0 and dim == qp)

    def decode(raw):
        if not fp8:
            return raw
        u = T.cast(raw, "uint32")
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

    @T.prim_func
    def main(
        q: T.Tensor((b, sq, g, ht * bm, dq), dtype),
        keys: T.Tensor(shape, kt),
        kpe: T.Tensor((b, nk, g, rp), dtype) if dense and rope else None,
        table: T.Tensor((b, width), "int32") if ps else None,
        lengths: T.Tensor((b,), "int32") if ps else None,
        kv_scale: T.Tensor((1,), "float32") if fp8 else None,
        live_dense: T.int32,
        numer: T.Tensor(partial_shape, dtype if fuse_output else "float32"),
        stats: T.Tensor((b, sq, g, ht, ns, bm, 2), "float32") if not fuse_output else None,
    ):
        jobs = b * sq * g * ht * ns
        with T.MixedKernel(jobs) as (bx, sub):
            split = bx % ns
            tile = bx // ns % ht
            group = bx // (ns * ht) % g
            query = bx // (ns * ht * g) % sq
            batch = bx // (ns * ht * g * sq)
            length = live_dense if dense else lengths[batch]
            live = length - sq + query + 1 if ps and causal else length
            tiles = T.ceildiv(T.max(live, 0), bn)
            per = T.ceildiv(tiles, ns)
            ql = T.alloc_l1((bm, dq), dtype)
            kl = T.alloc_l1((bn, qp), dtype)
            if rope:
                rl = T.alloc_l1((bn, rp), dtype)
            prob = T.alloc_l1((bm, bn), dtype)
            la = T.alloc_l0a((bm, 64), dtype)
            lb = T.alloc_l0b((bn, 64), dtype)
            l0c = T.alloc_l0c((bm, bn), "float32")
            pa = T.alloc_l0a((bm, bn), dtype)
            vb = T.alloc_l0b((64, bn), dtype)
            pv = T.alloc_l0c((bm, 64), "float32")
            su = T.alloc_shared((bm // 2, bn), "float32")
            pu = T.alloc_shared((bm // 2, bn), dtype)
            acc = T.alloc_shared((bm // 2, qp), "float32")
            part = T.alloc_shared((bm // 2, 64), "float32")
            mx = T.alloc_shared((bm // 2,), "float32")
            den = T.alloc_shared((bm // 2,), "float32")
            alpha = T.alloc_shared((bm // 2,), "float32")
            T.annotate_buffer_versions({la: 2, lb: 2, vb: 2, pv: 2, part: 2})
            if ps:
                ku = T.alloc_shared((bn // 2, qp), dtype)
                if rope:
                    ru = T.alloc_shared((bn // 2, rp), dtype)
            if native_page:
                raw_rows = T.alloc_shared((bn // 2, dq), "uint8")
                codes = T.view(raw_rows, (bn // 2, dq), dtype="float8_e4m3fn")
                words = T.view(raw_rows, (bn // 2, dq // 4), dtype="uint32")
            if fp8:
                cache_factor = T.alloc_shared((8,), "float32")
                T.copy(kv_scale[:1], cache_factor[:1])
            T.copy(q[batch, query, group, tile * bm : (tile + 1) * bm, :], ql)
            with T.SimtVF(threads=128):
                for rr, d in T.Parallel(bm // 2, qp):
                    acc[rr, d] = 0
                for rr in T.Parallel(bm // 2):
                    mx[rr] = -3.402823466e38
                    den[rr] = 0
            for it in T.serial(split * per, T.min((split + 1) * per, tiles)):
                start = it * bn
                if dense:
                    T.copy(keys[batch, start : start + bn, group, :], kl)
                    if rope:
                        T.copy(kpe[batch, start : start + bn, group, :], rl)
                elif native_page:
                    # One contiguous page segment per vector lane group. Clear only
                    # a partial segment so padded NaNs cannot enter the value product.
                    first = start + sub * (bn // 2)
                    used = T.min(bn // 2, T.max(0, live - first))
                    if used < bn // 2:
                        T.fill(words, T.uint32(0))
                    if used > 0:
                        base = table[batch, start // ps] * ps + first % ps
                        T.copy(keys[base : base + used, :], raw_rows[:used, :])
                    with T.SimdVF():
                        mask = T.simd.pset(32)
                        for rr in T.serial(bn // 2):
                            for col in T.serial(qp // 64):
                                payload = T.simd.vld(codes[rr, col * 64], dist="UNPK4_B8")
                                value = T.simd.vcvt(
                                    T.simd.vcvt(payload, "float32"),
                                    dtype,
                                    round="ROUND_R",
                                    sat=False,
                                    part=0,
                                )
                                T.simd.vsts(ku[rr, col * 64], value, mask, dist="PK_B32")
                            if rope:
                                for col in T.serial(rp // 64):
                                    payload = T.simd.vld(codes[rr, qp + col * 64], dist="UNPK4_B8")
                                    value = T.simd.vcvt(
                                        T.simd.vcvt(payload, "float32"),
                                        dtype,
                                        round="ROUND_R",
                                        sat=False,
                                        part=0,
                                    )
                                    T.simd.vsts(ru[rr, col * 64], value, mask, dist="PK_B32")
                    T.dual_copy(ku, kl)
                    if rope:
                        T.dual_copy(ru, rl)
                else:
                    if not fp8 and ps % bn == 0 and dim == qp and rope == rp and start + bn <= live:
                        base = table[batch, start // ps] * ps + start % ps
                        T.copy(keys[base : base + bn, :dim], kl)
                        if rope:
                            T.copy(keys[base : base + bn, dim : dim + rope], rl)
                    else:
                        with T.SimtVF(threads=256):
                            for rr, d in T.Parallel(bn // 2, qp):
                                idx = start + sub * (bn // 2) + rr
                                if idx < live and d < dim:
                                    base = table[batch, idx // ps] * ps + idx % ps
                                    ku[rr, d] = T.cast(decode(keys[base, d]), dtype)
                                else:
                                    ku[rr, d] = 0
                            if rope:
                                for rr, d in T.Parallel(bn // 2, rp):
                                    idx = start + sub * (bn // 2) + rr
                                    if idx < live and d < rope:
                                        base = table[batch, idx // ps] * ps + idx % ps
                                        ru[rr, d] = T.cast(decode(keys[base, dim + d]), dtype)
                                    else:
                                        ru[rr, d] = 0
                        T.dual_copy(ku, kl)
                        if rope:
                            T.dual_copy(ru, rl)
                for j in T.Pipelined(qp // 64, num_stages=2):
                    c = j * 64
                    T.copy(ql[:, c : c + 64], la)
                    T.copy(kl[:, c : c + 64], lb)
                    T.gemm(la, lb, l0c, transpose_B=True, clear_accum=j == 0)
                if rope:
                    for j in T.serial(rp // 64):
                        c = j * 64
                        T.copy(ql[:, qp + c : qp + c + 64], la)
                        T.copy(rl[:, c : c + 64], lb)
                        T.gemm(la, lb, l0c, transpose_B=True, clear_accum=False)
                T.dual_copy(l0c, su)
                with T.SimdVF():
                    mask = T.simd.pset(32)
                    one = T.simd.pset(32, "PAT_VL1")
                    if not wide_keys:
                        visible = T.simd.update_mask(T.min(bn, T.max(0, live - start)), 32)
                        negative = T.simd.vdup(-T.infinity("float32"), "float32", mask)
                    for rr in T.serial(bm // 2):
                        z = T.simd.vmuls(T.simd.vld(su[rr, 0]), T.float32(scale), mask)
                        if fp8:
                            z = T.simd.vmul(z, T.simd.vld(cache_factor[0], dist="BRC_B32"), mask)
                        if wide_keys:
                            upper = T.simd.vmuls(T.simd.vld(su[rr, 64]), T.float32(scale), mask)
                            largest = T.simd.vmax(z, upper, mask)
                        else:
                            z = T.simd.vsel(z, negative, visible)
                            largest = z
                        old = T.simd.vld(mx[rr], dist="BRC_B32")
                        peak = T.simd.vmax(
                            old, T.simd.vdupv(T.simd.vcmax(largest, mask), mask), mask
                        )
                        adj = T.simd.vexpdif(old, peak, mask)
                        p = T.simd.vexpdif(z, peak, mask)
                        if wide_keys:
                            p1 = T.simd.vexpdif(upper, peak, mask)
                            summands = T.simd.vadd(p, p1, mask)
                            T.simd.vsts(
                                pu[rr, 64],
                                T.simd.vcvt(p1, dtype, round="ROUND_R", sat=False, part=0),
                                mask,
                                dist="PK_B32",
                            )
                        else:
                            summands = p
                        dsum = T.simd.vadd(
                            T.simd.vmul(T.simd.vld(den[rr], dist="BRC_B32"), adj, mask),
                            T.simd.vcadd(summands, mask),
                            one,
                        )
                        T.simd.vsts(mx[rr], peak, one, dist="ONEPT_B32")
                        T.simd.vsts(den[rr], dsum, one, dist="ONEPT_B32")
                        T.simd.vsts(alpha[rr], adj, one, dist="ONEPT_B32")
                        T.simd.vsts(
                            pu[rr, 0],
                            T.simd.vcvt(p, dtype, round="ROUND_R", sat=False, part=0),
                            mask,
                            dist="PK_B32",
                        )
                T.dual_copy(pu, prob)
                T.copy(prob, pa)
                for j in T.Pipelined(qp // 64, num_stages=2):
                    c = j * 64
                    T.copy(kl[:, c : c + 64], vb, transpose=True)
                    T.gemm(pa, vb, pv, transpose_B=True, clear_accum=True)
                    T.dual_copy(pv, part)
                    with T.SimdVF():
                        mask = T.simd.pset(32)
                        for rr in T.serial(bm // 2):
                            prior = T.simd.vmul(
                                T.simd.vld(acc[rr, c]),
                                T.simd.vld(alpha[rr], dist="BRC_B32"),
                                mask,
                            )
                            T.simd.vsts(
                                acc[rr, c],
                                T.simd.vadd(prior, T.simd.vld(part[rr, 0]), mask),
                                mask,
                            )
            if fuse_output:
                result = T.alloc_shared((bm // 2, qp), dtype)
                with T.SimdVF():
                    mask = T.simd.pset(32)
                    for rr in T.serial(bm // 2):
                        divisor = T.simd.vld(den[rr], dist="BRC_B32")
                        for col in T.serial(qp // 64):
                            v = T.simd.vdiv(T.simd.vld(acc[rr, col * 64]), divisor, mask)
                            T.simd.vsts(
                                result[rr, col * 64],
                                T.simd.vcvt(v, dtype, round="ROUND_R", sat=False, part=0),
                                mask,
                                dist="PK_B32",
                            )
                first_head = group * hg + tile * bm + sub * (bm // 2)
                T.copy(result, numer[batch, query, first_head : first_head + bm // 2, :])
            else:
                T.copy(
                    acc,
                    numer[
                        batch,
                        query,
                        group,
                        tile,
                        split,
                        sub * (bm // 2) : (sub + 1) * (bm // 2),
                        :,
                    ],
                )
                with T.SimtVF(threads=64):
                    for rr in T.Parallel(bm // 2):
                        stats[batch, query, group, tile, split, sub * (bm // 2) + rr, 0] = mx[rr]
                        stats[batch, query, group, tile, split, sub * (bm // 2) + rr, 1] = den[rr]

    return tilelang.compile(main, target="ascend", out_idx=-1 if fuse_output else [-2, -1])


@functools.lru_cache(maxsize=48)
def _finish(sq, h, g, dim, dtype, ns, fp8, with_lse):
    b = T.dynamic("batch")
    hg = h // g
    bm = 32 if hg <= 32 else 64
    ht = (hg + bm - 1) // bm
    qp = (dim + 63) // 64 * 64

    @T.prim_func
    def main(
        numer: T.Tensor((b, sq, g, ht, ns, bm, qp), "float32"),
        stats: T.Tensor((b, sq, g, ht, ns, bm, 2), "float32"),
        kv_scale: T.Tensor((1,), "float32") if fp8 else None,
        out: T.Tensor((b, sq, h, dim), dtype),
        lse: T.Tensor((b, h, sq), "float32") if with_lse else None,
    ):
        with T.Kernel(64) as bx:
            tilebuf = T.alloc_shared((ns, qp), "float32")
            weights = T.alloc_shared((ns,), "float32")
            result = T.alloc_shared((qp,), dtype)
            for task in T.serial(bx, b * sq * h, 64):
                batch = task // (sq * h)
                query = task // h % sq
                head = task % h
                group = head // hg
                tile = head % hg // bm
                rr = head % hg % bm
                for j in T.serial(ns):
                    T.copy(numer[batch, query, group, tile, j, rr, :], tilebuf[j, :])
                with T.SimtVF(threads=32):
                    for _ in T.Parallel(1):
                        peak = T.alloc_local((1,), "float32")
                        den = T.alloc_local((1,), "float32")
                        peak[0] = -3.402823466e38
                        den[0] = 0
                        for j in T.serial(ns):
                            peak[0] = T.max(peak[0], stats[batch, query, group, tile, j, rr, 0])
                        for j in T.serial(ns):
                            w = T.exp(stats[batch, query, group, tile, j, rr, 0] - peak[0])
                            den[0] += w * stats[batch, query, group, tile, j, rr, 1]
                        for j in T.serial(ns):
                            w = T.exp(stats[batch, query, group, tile, j, rr, 0] - peak[0])
                            weight = T.if_then_else(den[0] > 0, w / den[0], 0)
                            weights[j] = weight * kv_scale[0] if fp8 else weight
                        if with_lse:
                            lse[batch, head, query] = T.if_then_else(
                                den[0] > 0,
                                peak[0] + T.log(den[0]),
                                -T.infinity("float32"),
                            )
                with T.SimdVF():
                    mask = T.simd.pset(32)
                    for col in T.serial(qp // 64):
                        acc = T.simd.alloc_local((1,), "float32")
                        acc[0] = T.simd.vdup(T.float32(0), "float32", mask)
                        for j in T.serial(ns):
                            v = T.simd.vld(tilebuf[j, col * 64])
                            w = T.simd.vld(weights[j], dist="BRC_B32")
                            acc[0] = T.simd.vadd(acc[0], T.simd.vmul(v, w, mask), mask)
                        T.simd.vsts(
                            result[col * 64],
                            T.simd.vcvt(acc[0], dtype, round="ROUND_R", sat=False, part=0),
                            mask,
                            dist="PK_B32",
                        )
                T.copy(result[:dim], out[batch, query, head, :])

    return tilelang.compile(main, target="ascend", out_idx=[-2, -1] if with_lse else -1)


def _qpack(q, rope, g):
    b, s, h, d = q.shape
    hg = h // g
    bm = 32 if hg <= 32 else 64
    if d % 64:
        q = F.pad(q, (0, (-d) % 64))
    if rope.shape[-1] % 64:
        rope = F.pad(rope, (0, (-rope.shape[-1]) % 64))
    merged = torch.cat((q, rope), -1).reshape(b, s, g, hg, -1)
    return F.pad(merged, (0, 0, 0, (-hg) % bm)) if hg % bm else merged


class AscendMLADenseDecodeKernel(Kernel, MLADecodeFwdInterface):
    devices = frozenset({"npu"})

    @classmethod
    def entry_for(cls, call):
        return call, lambda: cls(call)

    def __init__(self, call):
        super().__init__(device_index=call.device.index)
        self.dtype = call.dtype

    def forward(self, q, q_pe, k, k_pe):
        b, h, d = q.shape
        _, sk, g, _ = k.shape
        pe = q_pe.shape[-1]
        if not (b and h and sk and d):
            return torch.zeros_like(q)
        bm = 32 if h // g <= 32 else 64
        jobs = b * g * ((h // g + bm - 1) // bm)
        # Two score vectors share one row maximum. Restrict this dispatch to
        # complete 128-key tiles so the dense kernel needs no per-lane tail mask.
        wide_keys = sk % 128 == 0 and d >= 256
        key_tile = 128 if wide_keys else 64
        ns = min(32, max(1, (96 + jobs - 1) // jobs), (sk + key_tile - 1) // key_tile)
        # If there is at least one full Cube wave, each owner can finish its
        # heads locally. This avoids spilling a wide FP32 output for a merge.
        fuse_output = jobs >= 32 and (h // g) % bm == 0 and d % 64 == 0
        if fuse_output:
            ns = 1
        packed = _qpack(q[:, None], q_pe[:, None], g)
        if d % 64 or sk % 64:
            k = F.pad(k, (0, (-d) % 64, 0, 0, 0, (-sk) % 64))
        if pe and (pe % 64 or sk % 64):
            k_pe = F.pad(k_pe, (0, (-pe) % 64, 0, 0, 0, (-sk) % 64))
        run = _partials(
            1,
            h,
            g,
            d,
            pe,
            self.dtype_str,
            (d + pe) ** -0.5,
            ns,
            0,
            False,
            False,
            fuse_output,
            wide_keys,
        )
        args = (
            (packed, k.contiguous(), k_pe.contiguous(), sk) if pe else (packed, k.contiguous(), sk)
        )
        if fuse_output:
            return run(*args)[:, 0]
        numer, stats = run(*args)
        return _finish(1, h, g, d, self.dtype_str, ns, False, False)(numer, stats)[:, 0]


class AscendMLAPagedKernel(Kernel, MLAPagedFwdInterface):
    devices = frozenset({"npu"})

    @classmethod
    def entry_for(cls, call):
        return call, lambda: cls(call)

    def __init__(self, call):
        super().__init__(device_index=call.device.index)
        self.dtype = call.dtype
        self.call = call

    def forward(self, q, kv_cache, block_table, cache_seqlens, kv_scale=None):
        c = self.call
        b, s, h, dk = q.shape
        d = c.dim_v
        pe = dk - d
        if not (b and s and h and kv_cache.numel() and block_table.shape[1]):
            return q.new_zeros((b, s, h, d)), q.new_full(
                (b, h, s), -float("inf"), dtype=torch.float32
            )
        ns = 8
        packed = _qpack(q[..., :d], q[..., d:], 1)
        cache = kv_cache.reshape(-1, dk).contiguous()
        args = [
            packed,
            cache.view(torch.uint8) if c.quantized else cache,
            block_table.contiguous(),
            cache_seqlens.contiguous(),
        ]
        if c.quantized:
            args.append(kv_scale.contiguous())
        args.append(0)
        numer, stats = _partials(
            s,
            h,
            1,
            d,
            pe,
            self.dtype_str,
            dk**-0.5 if c.sm_scale is None else c.sm_scale,
            ns,
            c.page_size,
            c.quantized,
            c.is_causal,
        )(*args)
        finish = _finish(s, h, 1, d, self.dtype_str, ns, c.quantized, True)
        return tuple(finish(numer, stats, kv_scale) if c.quantized else finish(numer, stats))


register_kernel_type("MLADenseDecodeFwdOp", "ascend", AscendMLADenseDecodeKernel)
register_kernel_type("MLAPagedFwdOp", "ascend", AscendMLAPagedKernel)
