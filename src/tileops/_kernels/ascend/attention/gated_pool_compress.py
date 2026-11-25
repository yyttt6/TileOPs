"""Pool closing windows against the pre-call rings, then commit only surviving ring rows."""

from functools import lru_cache

import tilelang
import tilelang.ascend.language as T
import torch

from tileops._kernels.attention.gated_pool_compress import GatedPoolCompressFwdInterface
from tileops._kernels.kernel_base import Kernel
from tileops.backend._registry import register_kernel_type


@lru_cache(maxsize=64)
def _programs(tokens, batch, slots, ratio, overlap, dim, dtype):
    span = ratio * (2 if overlap else 1)
    width = dim * (2 if overlap else 1)

    compact = tokens >= 512
    pool_tasks = (tokens + ratio - 1) // ratio + batch if compact else tokens
    commit_rows = min(tokens, span)

    @T.macro
    def load_value(kv, score, ape, kv_state, score_state, tok, pos, first, slot, j, channel, v, s):
        p = pos - span + 1 + j
        col = channel + T.if_then_else(overlap and j >= ratio, dim, 0)
        if p < 0:
            v[0] = 0
            s[0] = -T.infinity("float32")
        elif p < first:
            v[0] = kv_state[slot, p % span, col]
            s[0] = score_state[slot, p % span, col]
        else:
            v[0] = T.cast(kv[tok + p - pos, col], "float32")
            s[0] = T.cast(score[tok + p - pos, col], "float32") + T.cast(
                ape[p % ratio, col], "float32"
            )

    @T.prim_func
    def pool(
        kv: T.Tensor((tokens, width), dtype),
        score: T.Tensor((tokens, width), dtype),
        ape: T.Tensor((ratio, width), dtype),
        kv_state: T.Tensor((slots, span, width), "float32"),
        score_state: T.Tensor((slots, span, width), "float32"),
        seq: T.Tensor((pool_tasks,), "int32"),
        cu: T.Tensor((batch + 1,), "int32"),
        starts: T.Tensor((batch,), "int32"),
        slot_ids: T.Tensor((batch,), "int32"),
        result: T.Tensor((tokens, dim), "bfloat16"),
        valid: T.Tensor((tokens,), "bool"),
    ):
        # Each request owns consecutive work items. Its extra slot covers a partial
        # global ratio-sized chunk; the end bound excludes that slot when empty.
        with T.Kernel(pool_tasks) as task:
            request = seq[task]
            first = starts[request]
            begin = cu[request]
            end = cu[request + 1]
            local = task - (begin // ratio + request) if compact else 0
            if compact:
                chunk = begin + local * ratio
                tok = chunk + (ratio - 1 - first % ratio)
                with T.SimtVF(threads=min(128, ratio)):
                    for offset in T.Parallel(ratio):
                        if chunk + offset < end:
                            valid[chunk + offset] = (first + offset + 1) % ratio == 0
            else:
                tok = task
                valid[tok] = (first + tok - begin + 1) % ratio == 0
            pos = first + tok - begin
            slot = slot_ids[request]
            if tok < end and (compact or (pos + 1) % ratio == 0):
                with T.SimtVF(threads=min(128, dim)):
                    for col in T.Parallel(dim):
                        v = T.alloc_local((1,), "float32")
                        s = T.alloc_local((1,), "float32")
                        peak = T.alloc_local((1,), "float32")
                        total = T.alloc_local((1,), "float32")
                        value = T.alloc_local((1,), "float32")
                        peak[0] = -T.infinity("float32")
                        total[0] = 0
                        value[0] = 0
                        for j in T.serial(span):
                            load_value(
                                kv,
                                score,
                                ape,
                                kv_state,
                                score_state,
                                tok,
                                pos,
                                first,
                                slot,
                                j,
                                col,
                                v,
                                s,
                            )
                            peak[0] = T.max(peak[0], s[0])
                        for j in T.serial(span):
                            load_value(
                                kv,
                                score,
                                ape,
                                kv_state,
                                score_state,
                                tok,
                                pos,
                                first,
                                slot,
                                j,
                                col,
                                v,
                                s,
                            )
                            total[0] += T.exp(s[0] - peak[0])
                        for j in T.serial(span):
                            load_value(
                                kv,
                                score,
                                ape,
                                kv_state,
                                score_state,
                                tok,
                                pos,
                                first,
                                slot,
                                j,
                                col,
                                v,
                                s,
                            )
                            weight = T.cast(T.exp(s[0] - peak[0]) / total[0], dtype)
                            value[0] += T.cast(
                                T.cast(weight * T.cast(v[0], dtype), dtype), "float32"
                            )
                        result[tok, col] = T.cast(value[0], "bfloat16")

    channel_tile = min(dim, 512 if span <= 8 else 64)
    groups = dim // channel_tile
    staged_blocks = min(64, pool_tasks * max(1, groups))

    @T.macro
    def stage_half(
        kv,
        score,
        ape,
        kv_state,
        score_state,
        tok,
        pos,
        first,
        slot,
        channel,
        start_j,
        src_col,
        vals,
        scores,
        biases,
        ring_vals,
        ring_scores,
    ):
        # Each half starts at a ratio boundary, including in the old ring.
        start = pos - span + 1 + start_j
        prefix = T.min(ratio, T.max(0, first - start))
        absent = T.min(ratio, T.max(0, -start))
        col = channel + src_col
        if prefix > absent:
            old = (start + absent) % span
            T.copy(
                kv_state[slot, old : old + prefix - absent, col : col + channel_tile],
                ring_vals[start_j + absent : start_j + prefix, :],
            )
            T.copy(
                score_state[slot, old : old + prefix - absent, col : col + channel_tile],
                ring_scores[start_j + absent : start_j + prefix, :],
            )
        if prefix < ratio:
            current = tok + start + prefix - pos
            T.copy(
                kv[current : current + ratio - prefix, col : col + channel_tile],
                vals[start_j + prefix : start_j + ratio, :],
            )
            T.copy(
                score[current : current + ratio - prefix, col : col + channel_tile],
                scores[start_j + prefix : start_j + ratio, :],
            )
            T.copy(
                ape[prefix:ratio, col : col + channel_tile],
                biases[start_j + prefix : start_j + ratio, :],
            )

    @T.prim_func
    def staged_pool(
        kv: T.Tensor((tokens, width), dtype),
        score: T.Tensor((tokens, width), dtype),
        ape: T.Tensor((ratio, width), dtype),
        kv_state: T.Tensor((slots, span, width), "float32"),
        score_state: T.Tensor((slots, span, width), "float32"),
        seq: T.Tensor((pool_tasks,), "int32"),
        cu: T.Tensor((batch + 1,), "int32"),
        starts: T.Tensor((batch,), "int32"),
        slot_ids: T.Tensor((batch,), "int32"),
        result: T.Tensor((tokens, dim), "bfloat16"),
        valid: T.Tensor((tokens,), "bool"),
    ):
        with T.Kernel(staged_blocks) as bx:
            vals = T.alloc_shared((span, channel_tile), dtype)
            scores = T.alloc_shared((span, channel_tile), dtype)
            biases = T.alloc_shared((span, channel_tile), dtype)
            ring_vals = T.alloc_shared((span, channel_tile), "float32")
            ring_scores = T.alloc_shared((span, channel_tile), "float32")
            out = T.alloc_shared((channel_tile,), "bfloat16")
            for step in T.serial(T.ceildiv(pool_tasks * groups, staged_blocks)):
                job = step * staged_blocks + bx
                if job < pool_tasks * groups:
                    task = job // groups
                    section = job % groups
                    channel = section * channel_tile
                    request = seq[task]
                    first = starts[request]
                    begin = cu[request]
                    end = cu[request + 1]
                    slot = slot_ids[request]
                    if compact:
                        chunk = begin + (task - (begin // ratio + request)) * ratio
                        tok = chunk + ratio - 1 - first % ratio
                        if section == 0:
                            with T.SimtVF(threads=min(128, ratio)):
                                for i in T.Parallel(ratio):
                                    if chunk + i < end:
                                        valid[chunk + i] = (first + i + 1) % ratio == 0
                    else:
                        tok = task
                        if section == 0:
                            valid[tok] = (first + tok - begin + 1) % ratio == 0
                    pos = first + tok - begin
                    if tok < end and (compact or (pos + 1) % ratio == 0):
                        stage_half(
                            kv,
                            score,
                            ape,
                            kv_state,
                            score_state,
                            tok,
                            pos,
                            first,
                            slot,
                            channel,
                            0,
                            0,
                            vals,
                            scores,
                            biases,
                            ring_vals,
                            ring_scores,
                        )
                        if overlap:
                            stage_half(
                                kv,
                                score,
                                ape,
                                kv_state,
                                score_state,
                                tok,
                                pos,
                                first,
                                slot,
                                channel,
                                ratio,
                                dim,
                                vals,
                                scores,
                                biases,
                                ring_vals,
                                ring_scores,
                            )
                        with T.SimdVF():
                            mask = T.simd.pset(32)
                            for v in range(channel_tile // 64):
                                c = v * 64
                                top = T.simd.alloc_local(1, "float32")
                                denom = T.simd.alloc_local(1, "float32")
                                acc = T.simd.alloc_local(1, "float32")
                                top[0] = T.simd.vdup(-T.infinity("float32"), "float32")
                                for j in T.serial(span):
                                    at = pos - span + 1 + j
                                    score_vec = T.simd.alloc_local(1, "float32")
                                    val_vec = T.simd.alloc_local(1, "float32")
                                    if at < 0:
                                        score_vec[0] = T.simd.vdup(
                                            -T.infinity("float32"), "float32"
                                        )
                                        val_vec[0] = T.simd.vdup(T.float32(0), "float32")
                                    elif at < first:
                                        score_vec[0] = T.simd.vld(ring_scores[j, c])
                                        val_vec[0] = T.simd.vld(ring_vals[j, c])
                                    else:
                                        if dtype == "float32":
                                            val_vec[0] = T.simd.vld(vals[j, c])
                                            sc = T.simd.vld(scores[j, c])
                                            bias = T.simd.vld(biases[j, c])
                                        else:
                                            val_vec[0] = T.simd.vcvt(
                                                T.simd.vld(vals[j, c], dist="UNPK_B16"),
                                                "float32",
                                                part=0,
                                            )
                                            sc = T.simd.vcvt(
                                                T.simd.vld(scores[j, c], dist="UNPK_B16"),
                                                "float32",
                                                part=0,
                                            )
                                            bias = T.simd.vcvt(
                                                T.simd.vld(biases[j, c], dist="UNPK_B16"),
                                                "float32",
                                                part=0,
                                            )
                                        score_vec[0] = T.simd.vadd(sc, bias, mask)
                                    top[0] = T.simd.vmax(top[0], score_vec[0], mask)
                                    T.simd.vsts(ring_scores[j, c], score_vec[0], mask)
                                    T.simd.vsts(ring_vals[j, c], val_vec[0], mask)
                                T.simd.mem_bar("VST_VLD")
                                denom[0] = T.simd.vdup(T.float32(0), "float32")
                                for j in T.serial(span):
                                    exp = T.simd.vexp(
                                        T.simd.vsub(T.simd.vld(ring_scores[j, c]), top[0], mask),
                                        mask,
                                    )
                                    T.simd.vsts(ring_scores[j, c], exp, mask)
                                    denom[0] = T.simd.vadd(denom[0], exp, mask)
                                T.simd.mem_bar("VST_VLD")
                                acc[0] = T.simd.vdup(T.float32(0), "float32")
                                for j in T.serial(span):
                                    w = T.simd.vdiv(T.simd.vld(ring_scores[j, c]), denom[0], mask)
                                    value = T.simd.vld(ring_vals[j, c])
                                    if dtype == "bfloat16":
                                        w = T.simd.vcvt(
                                            T.simd.vcvt(
                                                w, "bfloat16", part=0, round="ROUND_R", sat=False
                                            ),
                                            "float32",
                                            part=0,
                                        )
                                        value = T.simd.vcvt(
                                            T.simd.vcvt(
                                                value,
                                                "bfloat16",
                                                part=0,
                                                round="ROUND_R",
                                                sat=False,
                                            ),
                                            "float32",
                                            part=0,
                                        )
                                    product = T.simd.vmul(w, value, mask)
                                    if dtype == "bfloat16":
                                        product = T.simd.vcvt(
                                            T.simd.vcvt(
                                                product,
                                                "bfloat16",
                                                part=0,
                                                round="ROUND_R",
                                                sat=False,
                                            ),
                                            "float32",
                                            part=0,
                                        )
                                    acc[0] = T.simd.vadd(acc[0], product, mask)
                                packed = T.simd.vcvt(
                                    acc[0], "bfloat16", part=0, round="ROUND_R", sat=False
                                )
                                T.simd.vsts(out[c], packed, mask, dist="PK_B32")
                        T.copy(out, result[tok, channel : channel + channel_tile], l2_cache_ctrl=0)

    @T.prim_func
    def commit(
        kv: T.Tensor((tokens, width), dtype),
        score: T.Tensor((tokens, width), dtype),
        ape: T.Tensor((ratio, width), dtype),
        kv_state: T.Tensor((slots, span, width), "float32"),
        score_state: T.Tensor((slots, span, width), "float32"),
        seq: T.Tensor((pool_tasks,), "int32"),
        cu: T.Tensor((batch + 1,), "int32"),
        starts: T.Tensor((batch,), "int32"),
        slot_ids: T.Tensor((batch,), "int32"),
    ):
        with T.Kernel(batch * commit_rows if compact else tokens) as task:
            if compact:
                req = task // commit_rows
                tok = cu[req + 1] - 1 - task % commit_rows
            else:
                tok = task
                req = seq[task]
            if tok >= cu[req] and tok >= cu[req + 1] - span:
                pos = starts[req] + tok - cu[req]
                slot = slot_ids[req]
                with T.SimtVF(threads=min(128, width)):
                    for col in T.Parallel(width):
                        kv_state[slot, pos % span, col] = T.cast(kv[tok, col], "float32")
                        score_state[slot, pos % span, col] = T.cast(
                            score[tok, col], "float32"
                        ) + T.cast(ape[pos % ratio, col], "float32")

    # The staged path is specialized for complete FP32 vectors and a bounded UB window.
    selected = (
        staged_pool
        if (compact or ratio >= 64)
        and dim % 64 == 0
        and dim % channel_tile == 0
        and 0 < span <= 128
        and dtype in ("float32", "bfloat16")
        else pool
    )
    return tilelang.compile(selected, target="ascend", out_idx=[9, 10]), tilelang.compile(
        commit, target="ascend", out_idx=[]
    )


class AscendGatedPoolCompressKernel(Kernel, GatedPoolCompressFwdInterface):
    devices = frozenset({"npu"})
    supported_archs = None

    def __init__(self, call):
        super().__init__(device_index=call.device.index)
        self.call = call

    def forward(self, kv, score, ape, kv_state, score_state, cu_seqlens, start_pos, state_slots):
        tokens = kv.shape[0]
        if not tokens:
            return kv.new_empty((0, self.call.dim), dtype=torch.bfloat16), kv.new_empty(
                (0,), dtype=torch.bool
            )
        compact = tokens >= 512
        if compact:
            ratio = self.call.ratio
            batch = start_pos.numel()
            boundaries = cu_seqlens[1:-1] // ratio + torch.arange(
                1, batch, device=kv.device, dtype=torch.int32
            )
            tasks = (tokens + ratio - 1) // ratio + batch
        else:
            boundaries = cu_seqlens[1:]
            tasks = tokens
        seq = torch.searchsorted(
            boundaries,
            torch.arange(tasks, device=kv.device, dtype=torch.int32),
            right=True,
            out_int32=True,
        )
        args = (
            kv,
            score,
            ape,
            kv_state,
            score_state,
            seq,
            cu_seqlens,
            start_pos,
            state_slots,
        )
        pool, commit = _programs(
            tokens,
            start_pos.numel(),
            kv_state.shape[0],
            self.call.ratio,
            self.call.overlap,
            self.call.dim,
            self.dtype_to_str(kv.dtype),
        )
        entries, valid = pool(*args)
        commit(*args)
        return entries, valid


register_kernel_type("GatedPoolCompressFwdOp", "ascend", AscendGatedPoolCompressKernel)
