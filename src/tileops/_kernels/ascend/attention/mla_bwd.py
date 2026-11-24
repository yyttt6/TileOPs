"""Streaming MLA backward for rectangular, fixed or ragged requests."""

import functools
import math

import tilelang
import tilelang.ascend.language as T
import torch

from tileops._kernels.ascend.attention.gqa_dense_bwd import _row_stats
from tileops._kernels.ascend.attention.gqa_varlen import (
    _layout,
    _pack,
    _unpack_attention,
)
from tileops._kernels.attention.call_spec import (
    MLADenseBwdInterface,
    MLAVarlenBwdInterface,
)
from tileops._kernels.kernel_base import Kernel
from tileops.backend._registry import register_kernel_type


@functools.lru_cache(maxsize=32)
def _gradient(batch, heads, dq, dv, dtype, scale, causal, key_owner):
    tile = 64
    rows = tile // 2
    qp = (dq + 15) // 16 * 16
    vp = (dv + 15) // 16 * 16
    depth_step = math.gcd(tile, math.gcd(qp, vp))
    tq = T.dynamic("query_tokens")
    tiles = T.dynamic("owner_tiles")
    qcap = T.dynamic("query_buffer")
    kcap = T.dynamic("kv_buffer")

    @T.prim_func
    def main(
        q: T.Tensor((heads, qcap, qp), dtype),
        k: T.Tensor((heads, kcap, qp), dtype),
        v: T.Tensor((heads, kcap, vp), dtype),
        dout: T.Tensor((heads, qcap, vp), dtype),
        logs: T.Tensor((heads, tq), "float32"),
        delta: T.Tensor((heads, tq), "float32"),
        cq: T.Tensor((batch + 1,), "int32"),
        ck: T.Tensor((batch + 1,), "int32"),
        off: T.Tensor((batch + 1, 2), "int32"),
        grad: T.Tensor((heads, tiles, 2, rows, qp), dtype),
        grad_v: T.Tensor((heads, tiles, 2, rows, vp), dtype) if key_owner else None,
    ):
        with T.MixedKernel(32) as (bx, sub):
            a = T.alloc_l1((tile, qp), dtype)
            b = T.alloc_l1((tile, qp), dtype)
            dv_a = T.alloc_l1((tile, vp), dtype)
            dv_b = T.alloc_l1((tile, vp), dtype)
            deriv = T.alloc_l1((tile, tile), dtype)
            prob = T.alloc_l1((tile, tile), dtype)
            qa = T.alloc_l0a((tile, depth_step), dtype)
            kb = T.alloc_l0b((tile, depth_step), dtype)
            la = T.alloc_l0a((tile, tile), dtype)
            lb = T.alloc_l0b((qp, tile), dtype)
            lb_v = T.alloc_l0b((vp, tile), dtype)
            score = T.alloc_l0c((tile, tile), "float32")
            dp = T.alloc_l0c((tile, tile), "float32")
            accum = T.alloc_l0c((tile, qp), "float32")
            accum_v = T.alloc_l0c((tile, vp), "float32")
            su = T.alloc_shared((rows, tile), "float32")
            du = T.alloc_shared((rows, tile), "float32")
            ds = T.alloc_shared((rows, tile), dtype)
            pu = T.alloc_shared((rows, tile), dtype)
            lu = T.alloc_shared((tile,), "float32")
            dl = T.alloc_shared((tile,), "float32")
            gout = T.alloc_shared((rows, qp), "float32")
            vout = T.alloc_shared((rows, vp), "float32")
            result = T.alloc_shared((rows, qp), dtype)
            result_v = T.alloc_shared((rows, vp), dtype)
            work = T.alloc_var("int32", init=bx)
            for request in T.serial(batch):
                lq = cq[request + 1] - cq[request]
                lk = ck[request + 1] - ck[request]
                nq = T.ceildiv(lq, tile)
                nk = T.ceildiv(lk, tile)
                shift = lk - lq
                owned = nk if key_owner else nq
                count = owned * heads
                while work < count:
                    head = work // owned
                    own = work % owned * tile
                    start = T.alloc_var("int32", init=0)
                    end = T.alloc_var("int32")
                    if key_owner:
                        end = nq
                        if causal:
                            start = T.min(nq, T.max(0, own - shift) // tile)
                        base = off[request, 1] + own
                        T.copy(k[head, base : base + tile, :], a)
                        T.copy(v[head, base : base + tile, :], dv_a)
                    else:
                        end = nk
                        if causal:
                            end = T.min(nk, T.ceildiv(T.max(0, own + tile + shift), tile))
                        base = off[request, 0] + own
                        T.copy(q[head, base : base + tile, :], a)
                        T.copy(dout[head, base : base + tile, :], dv_a)
                    for j in T.serial(start, end):
                        other = j * tile
                        if key_owner:
                            base_q = off[request, 0] + other
                            T.copy(q[head, base_q : base_q + tile, :], b)
                            T.copy(dout[head, base_q : base_q + tile, :], dv_b)
                        else:
                            base_k = off[request, 1] + other
                            T.copy(k[head, base_k : base_k + tile, :], b)
                            T.copy(v[head, base_k : base_k + tile, :], dv_b)
                        for depth in T.serial(qp // depth_step):
                            first_col = depth * depth_step
                            T.copy(a[:, first_col : first_col + depth_step], qa)
                            T.copy(b[:, first_col : first_col + depth_step], kb)
                            T.gemm(qa, kb, score, transpose_B=True, clear_accum=depth == 0)
                        T.dual_copy(score, su)
                        for depth in T.serial(vp // depth_step):
                            first_col = depth * depth_step
                            T.copy(dv_a[:, first_col : first_col + depth_step], qa)
                            T.copy(dv_b[:, first_col : first_col + depth_step], kb)
                            T.gemm(qa, kb, dp, transpose_B=True, clear_accum=depth == 0)
                        T.dual_copy(dp, du)
                        query_first = other if key_owner else own
                        with T.SimtVF(threads=64):
                            for col in T.Parallel(tile):
                                qi = query_first + col
                                if qi < lq:
                                    lu[col] = logs[head, cq[request] + qi]
                                    dl[col] = delta[head, cq[request] + qi]
                                else:
                                    lu[col] = T.infinity("float32")
                                    dl[col] = 0
                        with T.SimdVF():
                            full = T.simd.pset(32)
                            zero = T.simd.vdup(T.float32(0), "float32", full)
                            for rr in T.serial(rows):
                                pos = own + sub * rows + rr
                                if key_owner:
                                    ls = T.simd.vld(lu[0])
                                    de = T.simd.vld(dl[0])
                                    visible = T.if_then_else(pos < lk, T.min(tile, lq - other), 0)
                                else:
                                    ls = T.simd.vld(lu[sub * rows + rr], dist="BRC_B32")
                                    de = T.simd.vld(dl[sub * rows + rr], dist="BRC_B32")
                                    visible = T.if_then_else(pos < lq, T.min(tile, lk - other), 0)
                                    if causal:
                                        visible = T.min(visible, pos + shift - other + 1)
                                z = T.simd.vmuls(T.simd.vld(su[rr, 0]), T.float32(scale), full)
                                p = T.simd.vexpdif(
                                    z,
                                    T.simd.vmuls(ls, T.float32(0.6931471805599453), full),
                                    full,
                                )
                                p = T.simd.vsel(p, zero, T.simd.update_mask(T.max(0, visible), 32))
                                if key_owner and causal:
                                    excluded = T.simd.update_mask(
                                        T.min(tile, T.max(0, pos - shift - other)), 32
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
                                T.simd.vsts(
                                    ds[rr, 0],
                                    T.simd.vcvt(
                                        dscore,
                                        dtype,
                                        round="ROUND_R",
                                        sat=False,
                                        part=0,
                                    ),
                                    full,
                                    dist="PK_B32",
                                )
                                if key_owner:
                                    T.simd.vsts(
                                        pu[rr, 0],
                                        T.simd.vcvt(p, dtype, round="ROUND_R", sat=False, part=0),
                                        full,
                                        dist="PK_B32",
                                    )
                        T.dual_copy(ds, deriv)
                        T.copy(deriv, la)
                        T.copy(b, lb, transpose=True)
                        T.gemm(la, lb, accum, transpose_B=True, clear_accum=j == start)
                        if key_owner:
                            T.dual_copy(pu, prob)
                            T.copy(prob, la)
                            T.copy(dv_b, lb_v, transpose=True)
                            T.gemm(
                                la,
                                lb_v,
                                accum_v,
                                transpose_B=True,
                                clear_accum=j == start,
                            )
                    if start < end:
                        T.dual_copy(accum, gout)
                        if key_owner:
                            T.dual_copy(accum_v, vout)
                    with T.SimtVF(threads=128):
                        for rr, d in T.Parallel(rows, qp):
                            result[rr, d] = T.cast(
                                T.if_then_else(start < end, gout[rr, d], 0), dtype
                            )
                        if key_owner:
                            for rr, d in T.Parallel(rows, vp):
                                result_v[rr, d] = T.cast(
                                    T.if_then_else(start < end, vout[rr, d], 0), dtype
                                )
                    owned_tile = (off[request, 1 if key_owner else 0] + own) // tile
                    T.copy(result, grad[head, owned_tile, sub, :, :])
                    if key_owner:
                        T.copy(result_v, grad_v[head, owned_tile, sub, :, :])
                    work += 32
                work -= count

    program = tilelang.compile(main, target="ascend", out_idx=[])
    which = int(key_owner)
    unpack = _unpack_attention(batch, heads, heads, dq, dtype, tile, False, which)
    if key_owner:
        unpack_v = _unpack_attention(batch, heads, heads, dv, dtype, tile, False, which)

    def run(*args):
        owner = args[which]
        shape = (heads, owner.shape[1] // tile, 2, rows)
        packed_grad = owner.new_empty((*shape, qp))
        inputs, outputs = args[:9], args[9:]
        buffers = [packed_grad]
        if key_owner:
            buffers.append(owner.new_empty((*shape, vp)))
        program(*inputs, *buffers)
        cu, off = inputs[6 + which], inputs[8]
        unpack(packed_grad, cu, off, outputs[0])
        if key_owner:
            unpack_v(buffers[1], cu, off, outputs[1])

    return run


def _packed_row_stats(o, dout, lse, scale, dtype):
    tokens, heads, dim = o.shape
    # The shared row-statistics program launches one AIV block per 16 rows
    # per head. Bound that grid for long packed batches, keeping head-major
    # LSE and both results in their original token order.
    token_chunk = 16 * max(1, 32768 // heads)
    deltas = []
    logs = []
    for first in range(0, tokens, token_chunk):
        count = min(token_chunk, tokens - first)
        stop = first + count
        stats = _row_stats(1, count, heads, dim, scale, dtype)
        delta, log = stats(
            o[first:stop].unsqueeze(0).contiguous(),
            dout[first:stop].unsqueeze(0).contiguous(),
            lse[:, first:stop].unsqueeze(0).contiguous(),
        )
        if count == tokens:
            return delta[0], log[0]
        deltas.append(delta[0])
        logs.append(log[0])
    return torch.cat(deltas, dim=1), torch.cat(logs, dim=1)


def _packed_grad(q, k, v, o, do, lse, cq, ck, causal, scale):
    tq, h, d = q.shape
    tk = k.shape[0]
    dv = v.shape[-1]
    batch = cq.numel() - 1
    if not tq or not tk or not dv:
        return torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v)
    dtype = str(q.dtype).removeprefix("torch.")
    off = _layout(batch, 1, 64, 64)(cq, ck)
    packed = []
    for mode, x in (("q", q), ("k", k), ("v", v), ("q", do)):
        fn = _pack(batch, h, h, x.shape[-1], 64, dtype, False, mode, 0, False)
        packed.append(fn(x.contiguous(), cq, ck, off))
    delta, logs = _packed_row_stats(o, do, lse, scale, dtype)
    args = (*packed, logs, delta, cq, ck, off)
    dq = torch.empty_like(q, memory_format=torch.contiguous_format)
    dk = torch.empty_like(k, memory_format=torch.contiguous_format)
    grad_v = torch.empty_like(v, memory_format=torch.contiguous_format)
    _gradient(batch, h, d, dv, dtype, scale, causal, False)(*args, dq)
    _gradient(batch, h, d, dv, dtype, scale, causal, True)(*args, dk, grad_v)
    return dq, dk, grad_v


class AscendMLADenseBwdKernel(Kernel, MLADenseBwdInterface):
    devices = frozenset({"npu"})

    @classmethod
    def entry_for(cls, call):
        return call, lambda: cls(call)

    def __init__(self, call):
        super().__init__(device_index=call.device.index)
        self.call = call

    def forward(self, q, k, v, o, do, lse):
        b, s, h, d = q.shape
        sk = k.shape[1]
        offsets = torch.arange(b + 1, device=q.device, dtype=torch.int32)
        args = [x.flatten(0, 1) for x in (q, k, v, o, do)]
        grads = _packed_grad(
            *args,
            lse.transpose(0, 1).reshape(h, b * s),
            offsets * s,
            offsets * sk,
            self.call.is_causal,
            d**-0.5 if self.call.sm_scale is None else self.call.sm_scale,
        )
        return tuple(x.view_as(y) for x, y in zip(grads, (q, k, v), strict=True))


class AscendMLAVarlenBwdKernel(Kernel, MLAVarlenBwdInterface):
    devices = frozenset({"npu"})

    @classmethod
    def entry_for(cls, call):
        return call, lambda: cls(call)

    def __init__(self, call):
        super().__init__(device_index=call.device.index)
        self.call = call

    def forward(self, q, k, v, o, do, lse, cu_seqlens_q, cu_seqlens_kv):
        c = self.call
        return _packed_grad(
            q,
            k,
            v,
            o,
            do,
            lse,
            cu_seqlens_q,
            cu_seqlens_kv,
            c.is_causal,
            c.dim_qk**-0.5 if c.sm_scale is None else c.sm_scale,
        )


register_kernel_type("MLADenseBwdOp", "ascend", AscendMLADenseBwdKernel)
register_kernel_type("MLAVarlenBwdOp", "ascend", AscendMLAVarlenBwdKernel)
