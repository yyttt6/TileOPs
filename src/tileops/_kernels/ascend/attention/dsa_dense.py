"""Sparse latent attention over the selected dense-cache rows on Ascend."""

import functools

import tilelang
import tilelang.ascend.language as T
import torch
import torch.nn.functional as F

from tileops._kernels.attention.call_spec import SparseMLADecodeFwdInterface
from tileops._kernels.kernel_base import Kernel
from tileops.backend._registry import register_kernel_type


@functools.lru_cache(maxsize=32)
def _select(batch, seq, groups, k, sk, stride, offset):
    """Select each valid key once and pad the last Cube tile with -1."""
    slots = (k + int(stride > 1) + 63) // 64 * 64

    @T.prim_func
    def main(
        sorted_ids: T.Tensor((batch, seq, groups, k), "int32"),
        selected: T.Tensor((batch, seq, groups, slots), "int32"),
    ):
        with T.Kernel(batch * seq * groups) as bx:
            b = bx // (seq * groups)
            row = bx // groups % seq
            g = bx % groups
            with T.SimtVF(threads=128):
                for j in T.Parallel(slots):
                    key = T.alloc_local((1,), "int32")
                    key[0] = -1
                    if j < k:
                        idx = sorted_ids[b, row, g, j]
                        if (
                            idx >= 0
                            and idx < sk
                            and (idx != 0 or row >= stride - 1)
                            and (idx + 1) * stride - 1 <= offset + row
                        ):
                            if j == 0:  # noqa: SIM114 - never load slot -1 on the first index
                                key[0] = idx
                            elif sorted_ids[b, row, g, j - 1] != idx:
                                key[0] = idx
                    if stride > 1 and j == k and row < stride - 1:
                        # The reference includes key zero for these early queries.
                        key[0] = 0
                    selected[b, row, g, j] = key[0]

    return tilelang.compile(main, target="ascend", out_idx=[1])


@functools.lru_cache(maxsize=32)
def _attend(
    batch,
    seq,
    groups,
    head_count,
    sk,
    slots,
    dim,
    latent,
    dtype,
    scale,
    with_lse=False,
    contiguous_slots=False,
    dense_membership=False,
):
    bm = 32 if head_count <= 32 else 64
    bn, vd = 64, 64
    hm = (head_count + bm - 1) // bm
    width = (latent + vd - 1) // vd * vd
    depth = (dim + 63) // 64 * 64
    vr = bm // 2

    @T.prim_func
    def main(
        q: T.Tensor((batch, seq, groups, hm * bm, depth), dtype),
        kv: T.Tensor(
            (
                batch,
                sk + int(not contiguous_slots and not dense_membership),
                groups,
                depth,
            ),
            dtype,
        ),
        selected: T.Tensor((batch, seq, groups, slots), "int32"),
        counts: T.Tensor((batch, seq, groups, slots // bn), "int32") if dense_membership else None,
        output: T.Tensor((batch, seq, groups, hm * bm, width), dtype),
        lse: T.Tensor((batch, seq, groups, hm * bm), "float32") if with_lse else None,
    ):
        with T.MixedKernel(batch * seq * groups * hm) as (bx, sub):
            b = bx // (seq * groups * hm)
            row = bx // (groups * hm) % seq
            g = bx // hm % groups
            head = (bx % hm) * bm
            ql = T.alloc_l1((bm, depth), dtype)
            kl = T.alloc_l1((bn, depth), dtype)
            pl = T.alloc_l1((bm, bn), dtype)
            a = T.alloc_l0a((bm, 64), dtype)
            pa = T.alloc_l0a((bm, bn), dtype)
            bk = T.alloc_l0b((bn, 64), dtype)
            bv = T.alloc_l0b((vd, bn), dtype)
            score = T.alloc_l0c((bm, bn), "float32")
            pv = T.alloc_l0c((bm, vd), "float32")
            if not contiguous_slots:
                ku = T.alloc_shared((bn // 2, depth), dtype)
            ids = T.alloc_shared((bn,), "int32")
            su = T.alloc_shared((vr, bn), "float32")
            pu = T.alloc_shared((vr, bn), dtype)
            partial = T.alloc_shared((vr, vd), "float32")
            acc = T.alloc_shared((vr, width), "float32")
            mx = T.alloc_shared((vr,), "float32")
            den = T.alloc_shared((vr,), "float32")
            alpha = T.alloc_shared((vr,), "float32")
            result = T.alloc_shared((vr, vd), dtype)
            if with_lse:
                row_lse = T.alloc_shared((vr,), "float32")
            T.annotate_buffer_versions({a: 2, bk: 2, bv: 2, pv: 2, partial: 2})
            T.copy(q[b, row, g, head : head + bm, :], ql)
            with T.SimtVF(threads=128):
                for rr, j in T.Parallel(vr, width):
                    acc[rr, j] = 0
                for rr in T.Parallel(vr):
                    mx[rr] = -3.402823466e38
                    den[rr] = 0
            for block in T.serial(slots // bn):
                if not dense_membership or counts[b, row, g, block] > 0:
                    T.copy(selected[b, row, g, block * bn : (block + 1) * bn], ids)
                    if dense_membership:
                        if counts[b, row, g, block] == bn:
                            T.copy(kv[b, block * bn : (block + 1) * bn, g, :], kl)
                        else:
                            # Contiguous loads cover this physical key tile. Mask invalid rows before PV.
                            token = block * bn + sub * (bn // 2)
                            T.copy(kv[b, token : token + bn // 2, g, :], ku)
                            with T.SimtVF(threads=256):
                                for rr, dd in T.Parallel(bn // 2, depth):
                                    if ids[sub * (bn // 2) + rr] < 0:
                                        ku[rr, dd] = 0
                            T.dual_copy(ku, kl)
                    elif contiguous_slots:
                        # Paged rows have already been gathered and invalid slots zeroed.
                        T.copy(kv[b, block * bn : (block + 1) * bn, g, :], kl)
                    else:
                        for fragment in T.serial(bn // 8):
                            local_row = fragment * 4
                            j0 = sub * (bn // 2) + local_row
                            first = ids[j0]
                            if (
                                first >= 0
                                and ids[j0 + 1] == first + 1
                                and ids[j0 + 2] == first + 2
                                and ids[j0 + 3] == first + 3
                            ):
                                T.copy(
                                    kv[b, first : first + 4, g, :],
                                    ku[local_row : local_row + 4, :],
                                )
                            else:
                                for one in T.unroll(4):
                                    index = ids[j0 + one]
                                    address = T.if_then_else(index >= 0, index, sk)
                                    T.copy(kv[b, address, g, :], ku[local_row + one, :])
                        T.dual_copy(ku, kl)
                    for step in T.Pipelined(depth // 64, num_stages=2):
                        start = step * 64
                        T.copy(ql[:, start : start + 64], a)
                        T.copy(kl[:, start : start + 64], bk)
                        T.gemm(a, bk, score, transpose_B=True, clear_accum=step == 0)
                    T.dual_copy(score, su)
                    with T.SimdVF():
                        mask = T.simd.pset(32)
                        one = T.simd.pset(32, "PAT_VL1")
                        valid = T.simd.vcmps(T.simd.vld(ids[0]), 0, mask, "ge")
                        neg = T.simd.vdup(-T.infinity("float32"), "float32", mask)
                        for rr in T.serial(vr):
                            z = T.simd.vsel(
                                T.simd.vmuls(T.simd.vld(su[rr, 0]), T.float32(scale), mask),
                                neg,
                                valid,
                            )
                            old = T.simd.vld(mx[rr], dist="BRC_B32")
                            next_peak = T.simd.vmax(
                                old, T.simd.vdupv(T.simd.vcmax(z, mask), mask), mask
                            )
                            factor = T.simd.vexpdif(old, next_peak, mask)
                            p = T.simd.vexpdif(z, next_peak, mask)
                            d = T.simd.vadd(
                                T.simd.vmul(T.simd.vld(den[rr], dist="BRC_B32"), factor, mask),
                                T.simd.vdupv(T.simd.vcadd(p, mask), mask),
                                mask,
                            )
                            T.simd.vsts(mx[rr], next_peak, one, dist="ONEPT_B32")
                            T.simd.vsts(den[rr], d, one, dist="ONEPT_B32")
                            T.simd.vsts(alpha[rr], factor, one, dist="ONEPT_B32")
                            T.simd.vsts(
                                pu[rr, 0],
                                T.simd.vcvt(p, dtype, round="ROUND_R", sat=False, part=0),
                                mask,
                                dist="PK_B32",
                            )
                    T.dual_copy(pu, pl)
                    T.copy(pl, pa)
                    for part in T.Pipelined(width // vd, num_stages=2):
                        col = part * vd
                        T.copy(kl[:, col : col + vd], bv, transpose=True)
                        T.gemm(pa, bv, pv, transpose_B=True, clear_accum=True)
                        T.dual_copy(pv, partial)
                        with T.SimdVF():
                            mask = T.simd.pset(32)
                            for rr in T.serial(vr):
                                prev = T.simd.vmul(
                                    T.simd.vld(acc[rr, col]),
                                    T.simd.vld(alpha[rr], dist="BRC_B32"),
                                    mask,
                                )
                                T.simd.vsts(
                                    acc[rr, col],
                                    T.simd.vadd(prev, T.simd.vld(partial[rr, 0]), mask),
                                    mask,
                                )
            for part in T.serial(width // vd):
                col = part * vd
                with T.SimdVF():
                    mask = T.simd.pset(32)
                    one = T.simd.vdup(T.float32(1), "float32", mask)
                    for rr in T.serial(vr):
                        div = T.simd.vmax(one, T.simd.vld(den[rr], dist="BRC_B32"), mask)
                        v = T.simd.vdiv(T.simd.vld(acc[rr, col]), div, mask)
                        T.simd.vsts(
                            result[rr, 0],
                            T.simd.vcvt(v, dtype, round="ROUND_R", sat=False, part=0),
                            mask,
                            dist="PK_B32",
                        )
                T.copy(
                    result,
                    output[
                        b,
                        row,
                        g,
                        head + sub * vr : head + (sub + 1) * vr,
                        col : col + vd,
                    ],
                )
            if with_lse:
                with T.SimtVF(threads=64):
                    for rr in T.Parallel(vr):
                        row_lse[rr] = T.if_then_else(
                            den[rr] > 0, mx[rr] + T.log(den[rr]), -T.infinity("float32")
                        )
                T.copy(row_lse, lse[b, row, g, head + sub * vr : head + (sub + 1) * vr])

    return tilelang.compile(main, target="ascend", out_idx=[-2, -1] if with_lse else [-1])


def device_core_counts(device):
    """Use the physical counts capped by the current process's device limit."""
    device_index = device.index
    if device_index is None:
        device_index = torch.npu.current_device()
    properties = torch.npu.get_device_properties(device_index)
    get_limit = getattr(torch.npu, "get_device_limit", None)
    if get_limit is not None and not callable(get_limit):
        raise RuntimeError("NPU get_device_limit is not callable")
    limits = get_limit(device_index) if get_limit is not None else {}
    if not isinstance(limits, dict):
        raise RuntimeError("invalid Ascend device limits")
    counts = []
    for name in ("cube_core_num", "vector_core_num"):
        physical = int(getattr(properties, name))
        limit = limits.get(name, physical)
        if type(limit) is not int or min(physical, limit) <= 0:
            raise RuntimeError(f"invalid Ascend {name}")
        counts.append(min(physical, limit))
    return counts[0], counts[1]


def uses_d512_schedule(call, q, kv, indices):
    """Select the tiled BF16 path when the call meets its buffer contract."""
    topk = indices.shape[-1]
    return (
        q.dtype == torch.bfloat16
        and kv.dtype == q.dtype
        and q.shape[2] in (64, 128)
        and call.kv_group == 1
        and call.kv_stride == 1
        and call.is_causal
        and call.dim == 512
        and call.tail_dim == 0
        and isinstance(call.q_start_index_s, int)
        and 0 < topk <= 4096
        and topk % 128 == 0
    )


class AscendDSADenseKernel(Kernel, SparseMLADecodeFwdInterface):
    devices = frozenset({"npu"})

    @classmethod
    def refusal(cls, call):
        return None if call.is_causal else "sparse MLA decode requires causal attention"

    @classmethod
    def entry_for(cls, call):
        return call, lambda: cls(call)

    def __init__(self, call):
        super().__init__(device_index=call.device.index)
        self.call = call
        self.dtype = call.dtype

    def forward(self, q, kv, indices):
        c = self.call
        b, seq, h, dim = q.shape
        g = c.kv_group
        hg = h // g
        if not b or not seq or not h or not c.dim:
            return q.new_empty((b, seq, h, c.dim))
        if not c.seq_len_kv or not c.topk:
            out = q.new_zeros((b, seq, h, c.dim))
            if c.seq_len_kv and c.kv_stride > 1:
                # Compression still selects key zero for the first stride-1 queries.
                out[:, : c.kv_stride - 1].unflatten(2, (g, hg)).copy_(
                    kv[:, None, 0, :, None, : c.dim]
                )
            return out
        if uses_d512_schedule(c, q, kv, indices):
            from tileops._kernels.ascend.attention.dsa_dense_d512 import forward_d512

            cube_cores, vector_cores = device_core_counts(q.device)
            scale = 512**-0.5 if c.sm_scale is None else c.sm_scale
            return forward_d512(q, kv, indices, c.q_start_index_s, scale, cube_cores, vector_cores)
        cap = min(c.seq_len_kv, max(1, (c.q_start_index_s + seq) // c.kv_stride))
        slots = (cap + 63) // 64 * 64
        dense = (
            slots <= c.seq_len_kv
            and slots <= 2 * c.topk
            and b * seq * g * (slots + 1) * 4 <= 64 * 1024**2
        )
        if dense:
            # One flag per reachable key implements set membership even for repeated indices.
            row = torch.arange(seq, dtype=torch.int32, device=q.device)
            limit = ((row + c.q_start_index_s + 1) // c.kv_stride).view(1, seq, 1, 1)
            live = (indices >= 0) & (indices < limit) & (indices < slots)
            scatter_index = torch.where(live, indices, slots).long()
            flags = torch.full((b, seq, g, slots + 1), -1, device=q.device, dtype=torch.int32)
            flags.scatter_(-1, scatter_index, 0)
            if c.kv_stride > 1:
                flags[:, : c.kv_stride - 1, :, 0] = 0
            ids = flags[..., :slots].contiguous()
        else:
            ordered = indices.sort(dim=-1).values
            ids = _select(b, seq, g, c.topk, c.seq_len_kv, c.kv_stride, c.q_start_index_s)(ordered)
        bm = 32 if hg <= 32 else 64
        q = q.reshape(b, seq, g, hg, dim)
        if hg % bm or dim % 64:
            q = F.pad(q, (0, (-dim) % 64, 0, (-hg) % bm))
        if not dense or dim % 64:
            # A single sentinel row absorbs padded and duplicate sorted slots.
            kv = F.pad(kv, (0, (-dim) % 64, 0, 0, 0, int(not dense)))
        program = _attend(
            b,
            seq,
            g,
            hg,
            c.seq_len_kv,
            ids.shape[-1],
            dim,
            c.dim,
            self.dtype_str,
            dim**-0.5 if c.sm_scale is None else c.sm_scale,
            dense_membership=dense,
        )
        args = [q.contiguous(), kv.contiguous(), ids]
        if dense:
            counts = (ids >= 0).view(b, seq, g, slots // 64, 64).sum(-1, dtype=torch.int32)
            args.append(counts)
        result = program(*args)
        return result[:, :, :, :hg, : c.dim].reshape(b, seq, h, c.dim).contiguous()


register_kernel_type("DSADenseFwdOp", "ascend", AscendDSADenseKernel)
