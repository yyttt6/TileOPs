"""Dense grouped-query attention gradients with FP32, privately owned accumulators."""

import functools

import tilelang
import tilelang.ascend.language as T
import torch
import torch.nn.functional as F

from tileops._kernels.attention.call_spec import (
    GQABwdInterface,
    GQAPreprocessBwdInterface,
)
from tileops._kernels.kernel_base import Kernel
from tileops.backend._registry import register_kernel_type


@functools.lru_cache(maxsize=32)
def _row_stats(batch, length, heads, dim, scale, dtype):
    tile = 16
    chunks = (dim + 63) // 64
    tiles = (length + tile - 1) // tile

    @T.prim_func
    def main(
        o: T.Tensor((batch, length, heads, dim), dtype),
        do: T.Tensor((batch, length, heads, dim), dtype),
        lse: T.Tensor((batch, heads, length), "float32"),
        delta: T.Tensor((batch, heads, length), "float32"),
        log2: T.Tensor((batch, heads, length), "float32"),
    ):
        with T.Kernel(batch * heads * tiles) as bx:
            b = bx // (heads * tiles)
            h = bx // tiles % heads
            first = bx % tiles * tile
            used = T.min(tile, length - first)
            x = T.alloc_shared((tile, chunks * 64), dtype)
            dy = T.alloc_shared((tile, chunks * 64), dtype)
            d = T.alloc_shared((tile,), "float32")
            T.copy(o[b, first : first + used, h, :], x[:used, :dim])
            T.copy(do[b, first : first + used, h, :], dy[:used, :dim])
            with T.SimdVF():
                full = T.simd.pset(32)
                one = T.simd.pset(32, "PAT_VL1")
                for r in T.serial(used):
                    acc = T.simd.alloc_local((1,), "float32")
                    acc[0] = T.simd.vdup(T.float32(0), "float32", full)
                    for v in T.serial(chunks):
                        mask = T.simd.update_mask(T.min(64, dim - 64 * v), 32)
                        a = T.simd.vcvt(
                            T.simd.vld(x[r, v * 64], dist="UNPK_B16"), "float32", part=0
                        )
                        g = T.simd.vcvt(
                            T.simd.vld(dy[r, v * 64], dist="UNPK_B16"),
                            "float32",
                            part=0,
                        )
                        acc[0] = T.simd.vadd(acc[0], T.simd.vmul(a, g, mask), full)
                    total = T.simd.vmuls(T.simd.vcadd(acc[0], full), T.float32(scale), one)
                    T.simd.vsts(d[r], total, one, dist="ONEPT_B32")
            T.copy(d[:used], delta[b, h, first : first + used])
            with T.SimtVF(threads=32):
                for r in T.Parallel(tile):
                    if r < used:
                        v = lse[b, h, first + r]
                        log2[b, h, first + r] = T.if_then_else(
                            v == -T.infinity("float32"),
                            T.infinity("float32"),
                            v * 1.4426950408889634,
                        )

    return tilelang.compile(main, target="ascend", out_idx=[3, 4])


@functools.lru_cache(maxsize=64)
def _backward(batch, length, heads, kv_heads, dim, dtype, causal, wl, wr, scale, cap, kv_owner):
    tile = 64
    tiles = (length + tile - 1) // tile
    sp = tiles * tile
    dp = (dim + 15) // 16 * 16
    group = heads // kv_heads
    owners = kv_heads if kv_owner else heads
    rows = tile // 2
    correct_dk = kv_owner and dtype == "bfloat16" and group > 1
    vector_softmax = cap == 0 and wl < 0 and wr < 0

    @T.prim_func
    def main(
        q: T.Tensor((batch, sp, heads, dp), dtype),
        k: T.Tensor((batch, sp, kv_heads, dp), dtype),
        v: T.Tensor((batch, sp, kv_heads, dp), dtype),
        do: T.Tensor((batch, sp, heads, dp), dtype),
        log2: T.Tensor((batch, heads, sp), "float32"),
        delta: T.Tensor((batch, heads, sp), "float32"),
        grad: T.Tensor((batch, sp, owners, dp), dtype),
        grad_v: T.Tensor((batch, sp, kv_heads, dp), dtype) if kv_owner else None,
    ):
        with T.MixedKernel(batch * owners * tiles) as (bx, sub):
            b = bx // (owners * tiles)
            owner = bx // tiles % owners
            own_tile = bx % tiles
            own = own_tile * tile
            left_qk = T.alloc_l1((tile, dp), dtype)
            right_qk = T.alloc_l1((tile, dp), dtype)
            left_dp = T.alloc_l1((tile, dp), dtype)
            right_dp = T.alloc_l1((tile, dp), dtype)
            ds_l1 = T.alloc_l1((tile, tile), dtype)
            p_l1 = T.alloc_l1((tile, tile), dtype)
            right_t = T.alloc_l1((dp, tile), dtype)
            score = T.alloc_l0c((tile, tile), "float32")
            dprob = T.alloc_l0c((tile, tile), "float32")
            accum = T.alloc_l0c((tile, dp), "float32")
            accum_v = T.alloc_l0c((tile, dp), "float32")
            su = T.alloc_shared((rows, tile), "float32")
            du = T.alloc_shared((rows, tile), "float32")
            prob = T.alloc_shared((rows, tile), dtype)
            deriv = T.alloc_shared((rows, tile), dtype)
            if correct_dk:
                ds_lo = T.alloc_shared((rows, tile), dtype)
            if vector_softmax:
                lu = T.alloc_shared((tile,), "float32")
                dl = T.alloc_shared((tile,), "float32")
                if not kv_owner:
                    T.copy(log2[b, owner, own : own + tile], lu)
                    T.copy(delta[b, owner, own : own + tile], dl)
            if kv_owner:
                T.copy(k[b, own : own + tile, owner, :], left_qk)
                T.copy(v[b, own : own + tile, owner, :], left_dp)
            else:
                T.copy(q[b, own : own + tile, owner, :], left_qk)
                T.copy(do[b, own : own + tile, owner, :], left_dp)
            begin = own_tile if kv_owner and causal else 0
            limit = own_tile + 1 if not kv_owner and causal else tiles
            for g in T.serial(group if kv_owner else 1):
                h = owner * group + g if kv_owner else owner
                kh = owner if kv_owner else owner // group
                for j in T.serial(begin, limit):
                    other = j * tile
                    if kv_owner:
                        T.copy(q[b, other : other + tile, h, :], right_qk)
                        T.copy(do[b, other : other + tile, h, :], right_dp)
                    else:
                        T.copy(k[b, other : other + tile, kh, :], right_qk)
                        T.copy(v[b, other : other + tile, kh, :], right_dp)
                    T.gemm(left_qk, right_qk, score, transpose_B=True, clear_accum=True)
                    T.dual_copy(score, su)
                    T.gemm(left_dp, right_dp, dprob, transpose_B=True, clear_accum=True)
                    T.dual_copy(dprob, du)
                    if vector_softmax:
                        if kv_owner:
                            T.copy(log2[b, h, other : other + tile], lu)
                            T.copy(delta[b, h, other : other + tile], dl)
                        with T.SimdVF():
                            full = T.simd.pset(32)
                            zero = T.simd.vdup(T.float32(0), "float32", full)
                            for rr in T.serial(rows):
                                pos = own + sub * rows + rr
                                if kv_owner:
                                    ls = T.simd.vld(lu[0])
                                    de = T.simd.vld(dl[0])
                                else:
                                    ls = T.simd.vld(lu[sub * rows + rr], dist="BRC_B32")
                                    de = T.simd.vld(dl[sub * rows + rr], dist="BRC_B32")
                                z = T.simd.vmuls(T.simd.vld(su[rr, 0]), T.float32(scale), full)
                                p = T.simd.vexpdif(
                                    z,
                                    T.simd.vmuls(ls, T.float32(0.6931471805599453), full),
                                    full,
                                )
                                valid_count = T.min(tile, length - other)
                                if causal and not kv_owner:
                                    valid_count = T.min(valid_count, pos - other + 1)
                                mask = T.simd.update_mask(
                                    T.if_then_else(pos < length, T.max(0, valid_count), 0), 32
                                )
                                p = T.simd.vsel(p, zero, mask)
                                if causal and kv_owner:
                                    excluded = T.simd.update_mask(
                                        T.max(0, T.min(tile, pos - other)), 32
                                    )
                                    p = T.simd.vsel(zero, p, excluded)
                                dscore = T.simd.vmul(
                                    p,
                                    T.simd.vsub(
                                        T.simd.vmuls(
                                            T.simd.vld(du[rr, 0]),
                                            T.float32(scale),
                                            full,
                                        ),
                                        de,
                                        full,
                                    ),
                                    full,
                                )
                                rounded = T.simd.vcvt(
                                    dscore, dtype, round="ROUND_R", sat=False, part=0
                                )
                                T.simd.vsts(deriv[rr, 0], rounded, full, dist="PK_B32")
                                if correct_dk:
                                    err = T.simd.vsub(
                                        dscore, T.simd.vcvt(rounded, "float32", part=0), full
                                    )
                                    T.simd.vsts(
                                        ds_lo[rr, 0],
                                        T.simd.vcvt(err, dtype, round="ROUND_R", sat=False, part=0),
                                        full,
                                        dist="PK_B32",
                                    )
                                if kv_owner:
                                    T.simd.vsts(
                                        prob[rr, 0],
                                        T.simd.vcvt(p, dtype, round="ROUND_R", sat=False, part=0),
                                        full,
                                        dist="PK_B32",
                                    )
                    else:
                        with T.SimtVF(threads=128):
                            for rr, c in T.Parallel(rows, tile):
                                qi = other + c if kv_owner else own + sub * rows + rr
                                ki = own + sub * rows + rr if kv_owner else other + c
                                visible = qi < length and ki < length
                                if causal:
                                    visible = visible and ki <= qi
                                if wl >= 0:
                                    visible = visible and ki >= qi - wl
                                if wr >= 0:
                                    visible = visible and ki <= qi + wr
                                prob[rr, c] = 0
                                deriv[rr, c] = 0
                                if correct_dk:
                                    ds_lo[rr, c] = 0
                                if visible:
                                    z = su[rr, c] * scale
                                    dz = T.alloc_local((1,), "float32")
                                    dz[0] = du[rr, c] * scale - delta[b, h, qi]
                                    if cap > 0:
                                        clipped = T.tanh(z / cap)
                                        z = cap * clipped
                                        dz[0] *= 1 - clipped * clipped
                                    p = T.exp2(z * 1.4426950408889634 - log2[b, h, qi])
                                    if kv_owner:
                                        prob[rr, c] = p
                                    raw = p * dz[0]
                                    rounded = T.cast(raw, dtype)
                                    deriv[rr, c] = rounded
                                    if correct_dk:
                                        ds_lo[rr, c] = T.cast(
                                            raw - T.cast(rounded, "float32"), dtype
                                        )
                    T.dual_copy(deriv, ds_l1)
                    if kv_owner:
                        T.copy(q[b, other : other + tile, h, :], right_t, transpose=True)
                    else:
                        T.copy(k[b, other : other + tile, kh, :], right_t, transpose=True)
                    T.gemm(
                        ds_l1,
                        right_t,
                        accum,
                        transpose_B=True,
                        clear_accum=(g == 0 and j == begin),
                    )
                    if correct_dk:
                        # BF16 rounds dS before the Q contraction. Recover the low
                        # term in FP32 dK when multiple query heads contribute.
                        T.dual_copy(ds_lo, ds_l1)
                        T.gemm(ds_l1, right_t, accum, transpose_B=True, clear_accum=False)
                    if kv_owner:
                        T.dual_copy(prob, p_l1)
                        T.copy(do[b, other : other + tile, h, :], right_t, transpose=True)
                        T.gemm(
                            p_l1,
                            right_t,
                            accum_v,
                            transpose_B=True,
                            clear_accum=(g == 0 and j == begin),
                        )
            T.copy(accum, grad[b, own : own + tile, owner, :])
            if kv_owner:
                T.copy(accum_v, grad_v[b, own : own + tile, owner, :])

    return tilelang.compile(main, target="ascend", out_idx=[6, 7] if kv_owner else [6])


class AscendGQABwdPreprocess(Kernel, GQAPreprocessBwdInterface):
    devices = frozenset({"npu"})

    @classmethod
    def entry_for(cls, call):
        return call, lambda: cls(call)

    def __init__(self, call):
        super().__init__(device_index=call.device.index)
        self.dtype = call.dtype
        self.stats = _row_stats(
            call.batch,
            call.max_seqlen_q,
            call.heads,
            call.dim,
            call.sm_scale if call.sm_scale is not None else call.dim**-0.5,
            self.dtype_str,
        )

    def forward(self, o, do, lse, with_dq_accum=True, launch_dependents=False):
        delta, log2 = self.stats(o.contiguous(), do.contiguous(), lse.contiguous())
        return (
            delta,
            log2,
            torch.zeros_like(o, dtype=torch.float32) if with_dq_accum else None,
        )


class AscendGQADenseBwdKernel(Kernel, GQABwdInterface):
    devices = frozenset({"npu"})
    owns_dq_accum = True

    @classmethod
    def entry_for(cls, call):
        return call, lambda: cls(call)

    def __init__(self, call):
        super().__init__(device_index=call.device.index)
        self.call = call
        self.dtype = call.dtype
        args = (
            call.batch,
            call.max_seqlen_q,
            call.heads,
            call.heads_kv,
            call.dim,
            self.dtype_str,
            call.is_causal,
            call.window_size_left,
            call.window_size_right,
            call.sm_scale if call.sm_scale is not None else call.dim**-0.5,
            call.softcap or 0.0,
        )
        self.q_grad = _backward(*args, False)
        self.kv_grad = _backward(*args, True)

    def forward(self, q, k, v, do, lse, delta, dq_accum):
        # Padding is only for whole Cube fragments; masks and row statistics use true bounds.
        s, d = q.shape[1], q.shape[-1]

        def aligned(x):
            if s % 64 or d % 16:
                return F.pad(x, (0, (-d) % 16, 0, 0, 0, (-s) % 64)).contiguous()
            return x.contiguous()

        if s % 64:
            lse = F.pad(lse, (0, (-s) % 64), value=float("inf"))
            delta = F.pad(delta, (0, (-s) % 64))
        args = (*(aligned(x) for x in (q, k, v, do)), lse, delta)
        dq = self.q_grad(*args)
        dk, dv = self.kv_grad(*args)
        return tuple(x[:, :s, :, :d].contiguous() for x in (dq, dk, dv))


register_kernel_type("GQADenseBwdOp", "ascend_preprocess", AscendGQABwdPreprocess)
register_kernel_type("GQADenseBwdOp", "ascend", AscendGQADenseBwdKernel)
