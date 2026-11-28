"""Opt-in D512 split-head and shared-L0B QK256 candidates; hardware validation pending.

DMA and SIMD packing derive from PR #3322, commit
4c0cb5fcdaf466c2a44e63ef1404553b94909da1 (MIT), adapted by the validated
kernel_dense_dma.py. This module is additive and is not a measured winner.
"""

import functools

import torch
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S
from tilelang.layout import (
    make_ascend_compact_nz_layout,
    make_ascend_major_k_layout,
    make_ascend_major_mn_layout,
)


from tvm import tirx


def dense_attention_next(
    batch, seq, heads, kv_len, topk, scale, num_blocks, profile, contiguous=False, *, wide_qk=False
):
    if profile == "paired_pv" and heads != 128:
        raise ValueError("paired_pv serves H128 only")
    if profile == "kv_double_buffer" and heads != 128:
        raise ValueError("KV double buffer candidate serves H128 only")
    plan = tuning_plan(
        "fused128" if profile in ("paired_pv", "reciprocal", "fma", "registers", "kv_coalesced_dma", "kv_l2_persistent", "kv_double_buffer") and heads == 128 else "wide64",
        heads,
    )
    if min(batch, seq, kv_len, topk) <= 0 or topk % 128:
        raise ValueError("positive dimensions and K multiple of 128 required")
    M = plan["head_tile"]
    HEADS_PER_AIV = M // 2
    BLOCK_TOP_K = 128
    VECTOR_LANES = 64
    SOFTMAX_SCALE = scale
    dtype = "bfloat16"
    accum_dtype = "float32"
    WIDE = plan["wide_pv"]
    SHARE = plan["share_scratch"]
    if profile not in ("split64", "qk256", "paired_pv", "reciprocal", "fma", "registers", "kv_coalesced_dma", "kv_l2_persistent", "kv_double_buffer"):
        raise ValueError("unknown next-generation profile")
    FMA = profile == "fma"
    RECIPROCAL = profile in ("reciprocal", "fma", "registers", "kv_coalesced_dma", "kv_l2_persistent", "kv_double_buffer")
    # Reuse one 128-column PV scratch tile to make room for both KV versions.
    # H128 and the 16-row DMA/format-conversion packet stay intact.
    PAIRED = profile == "paired_pv" or (
        RECIPROCAL and heads == 128 and profile != "kv_double_buffer"
    )
    nd_versions, nz_versions, prob_versions = (1, 1, 1) if PAIRED else (2, 2, 1)
    QK_WIDTH = 256 if profile == "qk256" or wide_qk else 128
    OUTPUT_CACHE = 4 if (batch * seq * (heads // M) + num_blocks - 1) // num_blocks > 128 else 0
    COALESCE_KV = profile == "kv_coalesced_dma"
    KV_CACHE_HINT = "NORMAL_PERS" if profile == "kv_l2_persistent" else "NORMAL_FV"
    head_tiles = heads // M
    jobs = batch * seq * head_tiles
    cores = min(jobs, num_blocks)
    blocks = topk // 128

    @T.macro
    def gather(batch_id, offset, count, sid, KV, idx, nd, nz, l1):
        for chunk in T.Serial(4, annotations={"multi_buffer_eligible": [nd]}):
            row_start = sid * 64 + chunk * 16
            rows = T.max(T.min(count - row_start, 16), 0)
            # Full chunks overwrite every byte; only a tail/empty chunk needs
            # clearing. The adapter guarantees a sorted, unique valid prefix.
            if rows < 16:
                with T.SimdVF():
                    T.fill(nd, T.float32(0))
            if contiguous:
                if rows > 0:
                    T.copy(
                        KV[batch_id, offset + row_start : offset + row_start + rows, 0, :],
                        nd[:rows, :],
                    )
            elif COALESCE_KV:
                for group in T.Serial(rows // 4):
                    first = 4 * group
                    i0 = T.int32(idx[row_start + first])
                    i1 = T.int32(idx[row_start + first + 1])
                    i2 = T.int32(idx[row_start + first + 2])
                    i3 = T.int32(idx[row_start + first + 3])
                    stride = i1 - i0
                    T.assume(i0 >= 0)
                    T.assume(i0 < i1)
                    T.assume(i1 < i2)
                    T.assume(i2 < i3)
                    T.assume(i3 < kv_len)
                    if (i2 - i1 == stride) and (i3 - i2 == stride):
                        _copy_kv_rows(
                            T.access_ptr(nd[first, 0], "w"),
                            T.access_ptr(KV[batch_id, i0, 0, 0], "r"),
                            4, stride,
                        )
                    else:
                        _copy_pair(
                            T.access_ptr(nd[first, 0], "w"),
                            T.access_ptr(KV[batch_id, i0, 0, 0], "r"), stride,
                        )
                        _copy_pair(
                            T.access_ptr(nd[first + 2, 0], "w"),
                            T.access_ptr(KV[batch_id, i2, 0, 0], "r"), i3 - i2,
                        )
                rest = (rows // 4) * 4
                if rows % 4 >= 2:
                    i0 = T.int32(idx[row_start + rest])
                    i1 = T.int32(idx[row_start + rest + 1])
                    T.assume(i0 >= 0)
                    T.assume(i0 < i1)
                    T.assume(i1 < kv_len)
                    _copy_pair(
                        T.access_ptr(nd[rest, 0], "w"),
                        T.access_ptr(KV[batch_id, i0, 0, 0], "r"), i1 - i0,
                    )
                if rows % 2 != 0:
                    tail = rows - 1
                    index = T.int32(idx[row_start + tail])
                    T.assume(index >= 0)
                    T.assume(index < kv_len)
                    T.copy(KV[batch_id, index, 0, :], nd[tail, :])
            else:
                for pair in T.Serial(rows // 2):
                    i0 = T.int32(idx[row_start + 2 * pair])
                    i1 = T.int32(idx[row_start + 2 * pair + 1])
                    T.assume(i0 >= 0)
                    T.assume(i0 < i1)
                    T.assume(i1 < kv_len)
                    if profile == "kv_l2_persistent":
                        _copy_kv_rows(
                            T.access_ptr(nd[2 * pair, 0], "w"),
                            T.access_ptr(KV[batch_id, i0, 0, 0], "r"),
                            2, i1 - i0, 2,
                        )
                    else:
                        _copy_pair(
                            T.access_ptr(nd[2 * pair, 0], "w"),
                            T.access_ptr(KV[batch_id, i0, 0, 0], "r"),
                            i1 - i0,
                        )
                if rows % 2 != 0:
                    tail = rows - 1
                    index = T.int32(idx[row_start + tail])
                    T.assume(index >= 0)
                    T.assume(index < kv_len)
                    T.copy(KV[batch_id, index, 0, :], nd[tail, :], l2_cache_ctrl=KV_CACHE_HINT)
            T.copy(nd, nz[:16, :])
            T.copy(nz[:16, :], l1[row_start : row_start + 16, :])

    @T.macro
    def softmax_pack(S_ub, P_nz_ub, m_ub, l_ub, alpha_ub, old_m_ub, valid_count):
        # Linear work-size estimate from the validated 32-row hint, not a
        # measured H128 cycle count. Device profiling remains authoritative.
        with T.SimdVF(latency=735 * HEADS_PER_AIV // 32):
            fp32_mask = S.pset(32, "PAT_ALL")
            scalar_mask = S.pset(32, "PAT_VL1")
            bf16_mask = S.pset(16, "PAT_VL128")
            # A finite negative sentinel makes masked exp(score - max) zero
            # for normal logits; it is not IEEE -inf.
            mask_value = S.vdup(T.float32(-3.402823e38), accum_dtype, fp32_mask)
            p_nz_ptr = S.make_ubuf_ptr(
                T.access_ptr(P_nz_ub[0, 0], "w", HEADS_PER_AIV + 1, BLOCK_TOP_K),
                dtype,
            )
            nz_stride = T.int32(((HEADS_PER_AIV + 1) << 16) | 1)
            valid_lo = S.update_mask(T.uint32(T.min(valid_count, VECTOR_LANES)), width=32)
            valid_hi = S.update_mask(T.uint32(T.max(valid_count - VECTOR_LANES, 0)), width=32)

            for head in range(HEADS_PER_AIV):
                scores_lo = S.vmuls(
                    S.vld(S_ub[head, 0], "NORM"), T.float32(SOFTMAX_SCALE), fp32_mask
                )
                scores_hi = S.vmuls(
                    S.vld(S_ub[head, VECTOR_LANES], "NORM"),
                    T.float32(SOFTMAX_SCALE),
                    fp32_mask,
                )
                masked_lo = S.vsel(scores_lo, mask_value, valid_lo)
                masked_hi = S.vsel(scores_hi, mask_value, valid_hi)
                S.vsts(S_ub[head, 0], masked_lo, fp32_mask, "NORM_B32")
                S.vsts(S_ub[head, VECTOR_LANES], masked_hi, fp32_mask, "NORM_B32")
                max_acc = T.alloc_var("float32x64")
                max_acc = S.vmax(masked_lo, masked_hi, fp32_mask)
                # Online softmax: m_new = max(m_old, block_max).
                old_max = S.vld(m_ub[head], "BRC_B32")
                S.vsts(old_m_ub[head], old_max, scalar_mask, "ONEPT_B32")
                max_acc = S.vmax(max_acc, old_max, fp32_mask)
                row_max = S.vcmax(max_acc, fp32_mask)
                S.vsts(m_ub[head], row_max, scalar_mask, "ONEPT_B32")

            S.mem_bar("VST_VLD")

            # Pack unnormalized P = exp(score - m_new) directly in BF16 NZ.
            zeros = S.vdup(T.float32(0), accum_dtype, fp32_mask)
            even_valid = S.update_mask(T.uint32((valid_count + 1) // 2), width=32)
            odd_valid = S.update_mask(T.uint32(valid_count // 2), width=32)
            for head in range(HEADS_PER_AIV):
                row_max = S.vld(m_ub[head], "BRC_B32")
                even_scores, odd_scores = S.vld2(S_ub[head, 0], "DINTLV_B32")
                even_prob = S.vexpdif(even_scores, row_max, fp32_mask)
                odd_prob = S.vexpdif(odd_scores, row_max, fp32_mask)
                # Explicit zeroing is essential for all-empty rows: finite
                # sentinel minus itself otherwise produces exp(0) == 1.
                even_prob = S.vsel(even_prob, zeros, even_valid)
                odd_prob = S.vsel(odd_prob, zeros, odd_valid)
                sum_acc = S.vadd(even_prob, odd_prob, fp32_mask)
                even_bf16 = S.vcvt(even_prob, dtype, fp32_mask, sat=False, part=0)
                odd_bf16 = S.vcvt(odd_prob, dtype, fp32_mask, sat=False, part=1)
                packed = S.vor(
                    T.reinterpret(even_bf16, "uint16x128"),
                    T.reinterpret(odd_bf16, "uint16x128"),
                    bf16_mask,
                )
                p_nz_ptr = S.vsstb(
                    T.reinterpret(packed, "bfloat16x128"),
                    p_nz_ptr,
                    nz_stride,
                    bf16_mask,
                    update=True,
                )
                prob_sum = S.vcadd(sum_acc, fp32_mask)
                # alpha = exp(m_old - m_new); l_new = alpha * l_old + sum(P).
                alpha = S.vexpdif(S.vld(old_m_ub[head], "BRC_B32"), row_max, fp32_mask)
                S.vsts(alpha_ub[head], alpha, scalar_mask, "ONEPT_B32")
                if FMA:
                    denominator = S.alloc_var("float32")
                    denominator = S.vld(l_ub[head], "BRC_B32")
                    S.vmadd(denominator, alpha, prob_sum, scalar_mask)
                else:
                    denominator = S.vadd(
                        S.vmul(S.vld(l_ub[head], "BRC_B32"), alpha, fp32_mask),
                        prob_sum,
                        scalar_mask,
                    )
                S.vsts(l_ub[head], denominator, scalar_mask, "ONEPT_B32")

    @T.macro
    def softmax_registers(S_ub, P_nz_ub, m_ub, l_ub, alpha_ub, valid_count):
        # Linear work-size estimate from the validated 32-row hint, not a
        # measured H128 cycle count. Device profiling remains authoritative.
        with T.SimdVF(latency=735 * HEADS_PER_AIV // 32):
            fp32_mask = S.pset(32, "PAT_ALL")
            scalar_mask = S.pset(32, "PAT_VL1")
            bf16_mask = S.pset(16, "PAT_VL128")
            # A finite negative sentinel makes masked exp(score - max) zero
            # for normal logits; it is not IEEE -inf.
            sentinel = S.vdup(T.float32(-3.402823e38), accum_dtype, fp32_mask)
            p_nz_ptr = S.make_ubuf_ptr(
                T.access_ptr(P_nz_ub[0, 0], "w", HEADS_PER_AIV + 1, BLOCK_TOP_K),
                dtype,
            )
            nz_stride = T.int32(((HEADS_PER_AIV + 1) << 16) | 1)
            zeros = S.vdup(T.float32(0), accum_dtype, fp32_mask)
            even_valid = S.update_mask(T.uint32((valid_count + 1) // 2), width=32)
            odd_valid = S.update_mask(T.uint32(valid_count // 2), width=32)
            for head in range(HEADS_PER_AIV):
                even_scores, odd_scores = S.vld2(S_ub[head, 0], "DINTLV_B32")
                even_scores = S.vmuls(even_scores, T.float32(SOFTMAX_SCALE), fp32_mask)
                odd_scores = S.vmuls(odd_scores, T.float32(SOFTMAX_SCALE), fp32_mask)
                old_max = S.vld(m_ub[head], "BRC_B32")
                even_scores = S.vsel(even_scores, sentinel, even_valid)
                odd_scores = S.vsel(odd_scores, sentinel, odd_valid)
                masked_max = S.vmax(even_scores, odd_scores, fp32_mask)
                reduced = S.vcmax(S.vmax(masked_max, old_max, fp32_mask), fp32_mask)
                row_max = S.vdupv(reduced, fp32_mask, pos="POS_LOWEST")
                S.vsts(m_ub[head], row_max, scalar_mask, "ONEPT_B32")
                even_prob = S.vexpdif(even_scores, row_max, fp32_mask)
                odd_prob = S.vexpdif(odd_scores, row_max, fp32_mask)
                # Explicit zeroing is essential for all-empty rows: finite
                # sentinel minus itself otherwise produces exp(0) == 1.
                even_prob = S.vsel(even_prob, zeros, even_valid)
                odd_prob = S.vsel(odd_prob, zeros, odd_valid)
                sum_acc = S.vadd(even_prob, odd_prob, fp32_mask)
                even_bf16 = S.vcvt(even_prob, dtype, fp32_mask, sat=False, part=0)
                odd_bf16 = S.vcvt(odd_prob, dtype, fp32_mask, sat=False, part=1)
                packed = S.vor(
                    T.reinterpret(even_bf16, "uint16x128"),
                    T.reinterpret(odd_bf16, "uint16x128"),
                    bf16_mask,
                )
                p_nz_ptr = S.vsstb(
                    T.reinterpret(packed, "bfloat16x128"),
                    p_nz_ptr,
                    nz_stride,
                    bf16_mask,
                    update=True,
                )
                prob_sum = S.vcadd(sum_acc, fp32_mask)
                # alpha = exp(m_old - m_new); l_new = alpha * l_old + sum(P).
                alpha = S.vexpdif(old_max, row_max, fp32_mask)
                S.vsts(alpha_ub[head], alpha, scalar_mask, "ONEPT_B32")
                denominator = S.vadd(
                    S.vmul(S.vld(l_ub[head], "BRC_B32"), alpha, fp32_mask),
                    prob_sum, scalar_mask)
                S.vsts(l_ub[head], denominator, scalar_mask, "ONEPT_B32")

    @T.macro
    def pv_chunk(vd, kv_l1, pa, vb, cc, partial):
        T.copy(kv_l1[:, vd * 128 : (vd + 1) * 128], vb[:128, :])
        T.gemm(pa, vb[:128, :], cc, transpose_B=False, clear_accum=True)
        T.dual_copy(cc, partial[:, vd * 128 : (vd + 1) * 128])

    @T.prim_func
    def main(
        Q: T.Tensor((batch, seq, heads, 512), dtype),
        KV: T.Tensor((batch, kv_len, 1, 512), dtype),
        Indices: T.Tensor((batch, seq, 1, topk), "int32"),
        Lengths: T.Tensor((batch, seq, 1), "int32"),
        Output: T.Tensor((batch, seq, heads, 512), dtype),
    ):
        with T.MixedKernel(cores, sids=2) as (bx, sid):
            q_l1 = T.alloc_l1((M, 512), dtype)
            kv_l1 = T.alloc_l1((128, 512), dtype)
            p_l1 = T.alloc_l1((M, 128), dtype)
            qa = T.alloc_l0a((M, QK_WIDTH), dtype)
            kb = T.alloc_l0b((128, QK_WIDTH), dtype)
            if wide_qk:
                pa = T.view(qa, (QK_WIDTH, M), dtype=dtype)
            else:
                pa = T.alloc_l0a((M, 128), dtype)
            if QK_WIDTH == 256:
                # Separate layout views share exactly 64 KiB of L0B storage.
                vb = T.view(kb, (QK_WIDTH, 128))
            else:
                vb = T.alloc_l0b((128, 128), dtype)
            score_c = T.alloc_l0c((M, 128), accum_dtype)
            # Separate ordinary 2D tiles retain the validated MAD interface;
            # no nonzero-origin trailing C slice or higher-rank C is required.
            values_c = T.alloc_l0c((M, 128), accum_dtype)
            if WIDE or PAIRED:
                values_c1 = T.alloc_l0c((M, 128), accum_dtype)
            if WIDE:
                values_c2 = T.alloc_l0c((M, 128), accum_dtype)
                values_c3 = T.alloc_l0c((M, 128), accum_dtype)
            kv_nd = T.alloc_shared((16, 512), dtype)
            kv_nz = T.alloc_shared((17, 512), dtype)
            partial = T.alloc_shared(
                (HEADS_PER_AIV, 512 if WIDE else (256 if PAIRED else 128)), accum_dtype
            )
            if SHARE:
                # Views preserve the entire allocation; only the first half
                # is a score tile when paired PV uses a 256-column scratch.
                scores = T.view(partial, (HEADS_PER_AIV * (2 if PAIRED else 1), 128))
            else:
                scores = T.alloc_shared((HEADS_PER_AIV, 128), accum_dtype)
            prob = T.alloc_shared((HEADS_PER_AIV + 1, 128), dtype)
            accum = T.alloc_shared((HEADS_PER_AIV, 512), accum_dtype)
            output = T.view(accum, (HEADS_PER_AIV * 2, 512), dtype=dtype)
            maxima = T.alloc_shared((64,), accum_dtype)
            denom = T.alloc_shared((64,), accum_dtype)
            alpha = T.alloc_shared((64,), accum_dtype)
            old_m = T.alloc_shared((64,), accum_dtype)
            idx = T.alloc_shared((128,), "int32")
            T.annotate_layout(
                {
                    kv_nz: make_ascend_compact_nz_layout(kv_nz),
                    prob: make_ascend_compact_nz_layout(prob),
                    kb: make_ascend_major_k_layout(kb),
                    vb: make_ascend_major_mn_layout(vb),
                }
            )
            T.annotate_buffer_versions(
                {
                    q_l1: 1,
                    kv_l1: 2,
                    p_l1: 2,
                    qa: 1,
                    kb: 1,
                    pa: 1,
                    vb: 1,
                    score_c: 1,
                    values_c: 1,
                    kv_nd: nd_versions,
                    kv_nz: nz_versions,
                    prob: prob_versions,
                    scores: 1,
                    partial: 1,
                    accum: 1,
                    output: 1,
                    maxima: 1,
                    denom: 1,
                    alpha: 1,
                    old_m: 1,
                    idx: 2,
                }
            )
            if WIDE or PAIRED:
                T.annotate_buffer_versions({values_c1: 1})
            if WIDE:
                T.annotate_buffer_versions({values_c2: 1, values_c3: 1})
            tasks = T.ceildiv(jobs - bx, cores)
            # Flatten query/head tasks and KV blocks into one pipeline, as in
            # PR #3322; the M/L/numerator carry remains a single physical copy.
            for work in T.Pipelined(
                tasks * blocks, num_stages=2, annotations={"enable_offset": True}
            ):
                job = work // blocks * cores + bx
                tile = work % blocks
                ht = job % head_tiles
                t = job // head_tiles % seq
                b = job // (head_tiles * seq)
                if tile == 0:
                    T.copy(Q[b, t, ht * M : (ht + 1) * M, :], q_l1)
                    with T.SimdVF():
                        T.fill(accum, T.float32(0))
                        T.fill(maxima, T.float32(-3.402823e38))
                        T.fill(denom, T.float32(0))
                count = T.max(T.min(Lengths[b, t, 0] - tile * 128, 128), 0)
                T.copy(Indices[b, t, 0, tile * 128 : (tile + 1) * 128], idx)
                gather(b, tile * 128, count, sid, KV, idx, kv_nd, kv_nz, kv_l1)
                for kd in T.serial(512 // QK_WIDTH):
                    T.copy(q_l1[:, kd * QK_WIDTH : (kd + 1) * QK_WIDTH], qa)
                    T.copy(kv_l1[:, kd * QK_WIDTH : (kd + 1) * QK_WIDTH], kb)
                    T.gemm(qa, kb, score_c, transpose_B=True, clear_accum=kd == 0)
                T.dual_copy(score_c, scores[:HEADS_PER_AIV, :])
                if profile == "registers":
                    softmax_registers(scores, prob, maxima, denom, alpha, count)
                else:
                    softmax_pack(scores, prob, maxima, denom, alpha, old_m, count)
                T.dual_copy(prob[:HEADS_PER_AIV, :], p_l1)
                if wide_qk:
                    T.copy(p_l1, pa[:128, :])
                else:
                    T.copy(p_l1, pa)
                if WIDE:
                    pv_chunk(0, kv_l1, pa, vb, values_c, partial)
                    pv_chunk(1, kv_l1, pa, vb, values_c1, partial)
                    pv_chunk(2, kv_l1, pa, vb, values_c2, partial)
                    pv_chunk(3, kv_l1, pa, vb, values_c3, partial)
                elif PAIRED:
                    # Two PV tiles share one alpha-rescale VF; mathematical order
                    # per output element is unchanged, with a 256-wide UB slice.
                    for pair in T.unroll(2):
                        T.copy(kv_l1[:, pair * 256 : pair * 256 + 128], vb[:128, :])
                        if wide_qk:
                            T.gemm(pa[:128, :], vb[:128, :], values_c, transpose_B=False, clear_accum=True)
                        else:
                            T.gemm(pa, vb, values_c, transpose_B=False, clear_accum=True)
                        T.dual_copy(values_c, partial[:, :128])
                        T.copy(kv_l1[:, pair * 256 + 128 : (pair + 1) * 256], vb[:128, :])
                        if wide_qk:
                            T.gemm(pa[:128, :], vb[:128, :], values_c1, transpose_B=False, clear_accum=True)
                        else:
                            T.gemm(pa, vb, values_c1, transpose_B=False, clear_accum=True)
                        T.dual_copy(values_c1, partial[:, 128:256])
                        with T.SimdVF():
                            mask = S.pset(32, "PAT_ALL")
                            for r in range(HEADS_PER_AIV):
                                factor = S.vld(alpha[r], "BRC_B32")
                                for col in range(0, 256, 64):
                                    previous = S.vld(accum[r, pair * 256 + col], "NORM")
                                    contribution = S.vld(partial[r, col], "NORM")
                                    if FMA:
                                        updated = S.alloc_var("float32")
                                        updated = previous
                                        S.vmadd(updated, factor, contribution, mask)
                                    else:
                                        updated = S.vadd(
                                            S.vmul(previous, factor, mask), contribution, mask
                                        )
                                    S.vsts(accum[r, pair * 256 + col], updated, mask, "NORM_B32")
                else:
                    for vd in T.unroll(4):
                        T.copy(kv_l1[:, vd * 128 : (vd + 1) * 128], vb[:128, :])
                        T.gemm(pa, vb[:128, :], values_c, transpose_B=False, clear_accum=True)
                        T.dual_copy(values_c, partial)
                        with T.SimdVF():
                            mask = S.pset(32, "PAT_ALL")
                            for r in range(HEADS_PER_AIV):
                                factor = S.vld(alpha[r], "BRC_B32")
                                for col in range(0, 128, 64):
                                    previous = S.vld(accum[r, vd * 128 + col], "NORM")
                                    contribution = S.vld(partial[r, col], "NORM")
                                    if FMA:
                                        updated = S.alloc_var("float32")
                                        updated = previous
                                        S.vmadd(updated, factor, contribution, mask)
                                    else:
                                        updated = S.vadd(
                                            S.vmul(previous, factor, mask),
                                            contribution,
                                            mask,
                                        )
                                    S.vsts(
                                        accum[r, vd * 128 + col],
                                        updated,
                                        mask,
                                        "NORM_B32",
                                    )
                if WIDE:
                    # All PV chunks are ready: one VF updates the full numerator.
                    with T.SimdVF():
                        mask = S.pset(32, "PAT_ALL")
                        for r in range(HEADS_PER_AIV):
                            factor = S.vld(alpha[r], "BRC_B32")
                            for col in range(0, 512, 64):
                                previous = S.vld(accum[r, col], "NORM")
                                contribution = S.vld(partial[r, col], "NORM")
                                if FMA:
                                    updated = S.alloc_var("float32")
                                    updated = previous
                                    S.vmadd(updated, factor, contribution, mask)
                                else:
                                    updated = S.vadd(
                                        S.vmul(previous, factor, mask), contribution, mask
                                    )
                                S.vsts(accum[r, col], updated, mask, "NORM_B32")
                if tile == blocks - 1:
                    with T.SimdVF():
                        mask = S.pset(32, "PAT_ALL")
                        for r in range(HEADS_PER_AIV):
                            divisor = S.vmaxs(S.vld(denom[r], "BRC_B32"), T.float32(1e-30), mask)
                            inverse = T.alloc_var("float32x64")
                            if RECIPROCAL:
                                inverse = S.vdiv(
                                    S.vdup(T.float32(1), accum_dtype, mask), divisor, mask
                                )
                            for col in range(0, 512, 64):
                                if RECIPROCAL:
                                    normalized = S.vmul(S.vld(accum[r, col], "NORM"), inverse, mask)
                                else:
                                    normalized = S.vdiv(S.vld(accum[r, col], "NORM"), divisor, mask)
                                packed = S.vcvt(normalized, dtype, mask, sat=False)
                                S.vsts(output[r, col], packed, mask, "PK_B32")
                    T.dual_copy(
                        output[:HEADS_PER_AIV, :],
                        Output[b, t, ht * M : (ht + 1) * M, :],
                        l2_cache_ctrl=OUTPUT_CACHE,
                    )

    return main


CANDIDATES = ("lean64", "fused128", "wide64")


def tuning_plan(variant, heads):
    if variant not in CANDIDATES or heads not in (64, 128):
        raise ValueError("expected a named tuning candidate and H64/H128")
    fused = variant == "fused128"
    wide = variant == "wide64"
    m = heads if fused else 64
    rows = m // 2
    versions = 2 if variant == "lean64" else 1
    share = fused
    # Buffers/views sharing an allocation are counted once. Logical bytes;
    # alignment and compiler-introduced storage need the generated code.
    ub = (
        versions * (16 * 512 * 2 + 17 * 512 * 2 + (rows + 1) * 128 * 2)
        + rows * 512 * 4
        + rows * (512 if wide else 128) * 4
        + (0 if share else rows * 128 * 4)
        + 4 * 64 * 4
        + 2 * 128 * 4
    )
    return dict(
        variant=variant,
        head_tile=m,
        head_tiles=heads // m,
        wide_pv=wide,
        share_scratch=share,
        staging_versions=versions,
        logical_bytes=dict(
            UB=ub,
            L1=m * 512 * 2 + 2 * 128 * 512 * 2 + 2 * m * 128 * 2,
            L0A=2 * m * 128 * 2,
            L0B=2 * 128 * 128 * 2,
            L0C=(1 + (4 if wide else 1)) * m * 128 * 4,
        ),
    )


def _copy_pair(dst, src, stride):
    return _copy_kv_rows(dst, src, 2, stride)


def _copy_kv_rows(dst, src, rows, stride, cache_hint=0):
    """Issue one burst descriptor for equally spaced 1024-byte KV rows."""
    return tirx.call_intrin(
        "void",
        tirx.op.Op.get("tl.ascend_copy_gm_to_ubuf"),
        dst,
        src,
        0,
        rows,
        1024,
        0,
        0,
        0,
        cache_hint,
        stride * 1024,
        1024,
    )


def build(rows, seq, width, kv_len, start, cores):
    import tilelang.ascend.language as T
    from tilelang.ascend.language import simd as S

    row_tile = 8
    jobs = rows // row_tile
    blocks = min(jobs, cores)

    @T.prim_func
    def main(
        Ordered: T.Tensor((rows, width), "int32"),
        Selected: T.Tensor((rows, width), "int32"),
        Lengths: T.Tensor((rows,), "int32"),
    ):
        with T.Kernel(blocks) as bx:
            ids = T.alloc_shared((row_tile, width), "int32")
            packed = T.alloc_shared((row_tile, width), "int32")
            counts = T.alloc_shared((row_tile,), "int32")
            for task in T.Serial(T.ceildiv(jobs - bx, blocks)):
                base = (task * blocks + bx) * row_tile
                T.copy(Ordered[base : base + row_tile, :], ids)
                with T.SimdVF():
                    T.fill(packed, T.int32(kv_len))
                for r in T.Serial(row_tile):
                    # Compute in int64 before clamping to the index dtype.
                    bound = T.cast(T.max(-1, T.min(kv_len - 1, T.int64(start) + (base + r) % seq)), "int32")
                    with T.SimdVF():
                        full = S.pset(32)
                        first_lane = S.pset(32, "PAT_VL1")
                        zero = S.vdup(T.int32(0), "int32", full)
                        one = S.vdup(T.int32(1), "int32", full)
                        previous = S.alloc_var("int32")
                        previous = S.vdup(T.int32(-1), "int32", full)
                        count = S.alloc_var("int32")
                        count = S.vdup(T.int32(0), "int32", full)
                        lanes = S.vci(T.int32(0), "int32")
                        preceding = S.vmaxs(S.vadds(lanes, T.int32(-1), full), T.int32(0), full)
                        for group in range(width // 64):
                            values = S.vld(ids[r, group * 64], "NORM")
                            neighbors = S.vselr(values, T.reinterpret(preceding, "uint32x64"))
                            neighbors = S.vsel(previous, neighbors, first_lane)
                            distinct = S.vcmp(values, neighbors, full, "gt")
                            valid = S.pand(S.vcmps(values, T.int32(0), full, "ge"),
                                           S.vcmps(values, bound, full, "le"), full)
                            keep = S.pand(distinct, valid, full)
                            ranks = S.vadd(count, S.vusqz(keep, "int32"), full)
                            S.vscatter(values, T.access_ptr(packed[r, 0], "w", extent=width),
                                       T.reinterpret(ranks, "uint32x64"), keep)
                            subtotal = S.vcadd(S.vsel(one, zero, keep), full)
                            count = S.vadd(count, S.vdupv(subtotal, full, pos="POS_LOWEST"), full)
                            previous = S.vdupv(values, full, pos="POS_HIGHEST")
                        S.vsts(counts[r], count, first_lane, "ONEPT_B32")
                T.copy(packed, Selected[base : base + row_tile, :])
                T.copy(counts, Lengths[base : base + row_tile])

    return main


@functools.lru_cache(maxsize=32)
def compile_program(rows, seq, width, kv_len, start, cores, device):
    import tilelang
    target = tilelang.tvm.target.Target("ascend")
    with target:
        return tilelang.compile(build(rows, seq, width, kv_len, start, cores),
                                target=target, out_idx=[])


def compact(indices, kv_len, start, cores):
    batch, seq, groups, width = indices.shape
    rows = batch * seq
    if groups != 1 or rows % 8 or width != 1024:
        raise ValueError("vector compaction requires one KV head, aligned rows, and TopK 1024")
    ordered = indices.reshape(rows, width).sort(dim=-1).values
    selected = torch.empty_like(ordered)
    lengths = indices.new_empty((rows,))
    fn = compile_program(rows, seq, width, kv_len, start, cores, indices.device.index)
    fn(ordered, selected, lengths)
    return selected.reshape_as(indices), lengths.reshape(batch, seq, 1)

def compact_builder(rows, seq, topk, kv_len, start, num_blocks):
    import tilelang.ascend.language as T

    extent = max(128, 1 << (topk - 1).bit_length())
    output_width = ((topk + 127) // 128) * 128
    levels = (extent - 1).bit_length()
    row_tile = 4 if topk <= 1024 and rows % 4 == 0 else 1
    jobs = rows // row_tile
    cores = min(jobs, num_blocks)

    @T.prim_func
    def main(
        Sorted: T.Tensor((rows, topk), "int32"),
        Selected: T.Tensor((rows, output_width), "int32"),
        Lengths: T.Tensor((rows, 8), "int32"),
    ):
        with T.Kernel(cores) as bx:
            ids = T.alloc_shared((row_tile, extent), "int32")
            flags = T.alloc_shared((row_tile, extent), "int32")
            scan = T.alloc_shared((row_tile, extent), "int32")
            scratch = T.alloc_shared((row_tile, extent), "int32")
            packed = T.alloc_shared((row_tile, extent), "int32")
            length = T.alloc_shared((row_tile, 8), "int32")
            for task in T.Serial(T.ceildiv(jobs - bx, cores)):
                row = (task * cores + bx) * row_tile
                with T.SimtVF(threads=min(256, extent)):
                    for r, i in T.Parallel(row_tile, extent):
                        ids[r, i] = kv_len
                        packed[r, i] = kv_len
                T.copy(Sorted[row : row + row_tile, :], ids[:, :topk])
                with T.SimtVF(threads=min(256, extent)):
                    for r, i in T.Parallel(row_tile, extent):
                        valid = (i < topk) & (ids[r, i] >= 0) & (ids[r, i] < kv_len)
                        causal = ids[r, i] <= start + (row + r) % seq
                        first = (i == 0) | (ids[r, i] != ids[r, T.max(i - 1, 0)])
                        flags[r, i] = T.if_then_else(valid & causal & first, 1, 0)
                        # Store the predicate directly: do not reread a just
                        # written UB flag through the scalar memory pipeline.
                        scan[r, i] = T.if_then_else(valid & causal & first, 1, 0)
                # Every scan step reads an immutable previous step. Separate
                # VF/copy boundaries avoid read/write races between lanes.
                for level in T.unroll(levels):
                    distance = 1 << level
                    with T.SimtVF(threads=min(256, extent)):
                        for r, i in T.Parallel(row_tile, extent):
                            scratch[r, i] = scan[r, i] + T.if_then_else(
                                i >= distance, scan[r, T.max(i - distance, 0)], 0
                            )
                    # T.copy(UB,UB) lowered to scalar UB load/store in the
                    # returned device code, outside the VF's vector pipeline.
                    # Keep all prefix steps in ordered VF calls instead.
                    with T.SimtVF(threads=min(256, extent)):
                        for r, i in T.Parallel(row_tile, extent):
                            scan[r, i] = scratch[r, i]
                with T.SimtVF(threads=min(256, extent)):
                    for r, i in T.Parallel(row_tile, extent):
                        if flags[r, i] != 0:
                            packed[r, T.max(scan[r, i] - 1, 0)] = ids[r, i]
                # Dedicated VF avoids mixing the scatter's lane domain with
                # a second, smaller parallel loop. Each output row is 32-byte
                # aligned and carries eight identical count values.
                with T.SimtVF(threads=32):
                    for r, column in T.Parallel(row_tile, 8):
                        length[r, column] = scan[r, extent - 1]
                T.copy(packed[:, :output_width], Selected[row : row + row_tile, :])
                T.copy(length, Lengths[row : row + row_tile, :])

    return main


@functools.lru_cache(maxsize=64)
def _compile(rows, seq, topk, kv_len, start, blocks, device_index):
    import tilelang

    target = tilelang.tvm.target.Target("ascend")
    with target:
        return tilelang.compile(
            compact_builder(rows, seq, topk, kv_len, start, blocks), target=target, out_idx=[]
        )


def compact_indices(indices, kv_len, start, vector_cores):
    b, s, g, k = indices.shape
    if g != 1 or not 1 <= k <= 4096:
        raise ValueError("compact path requires Hkv1 and TopK in [1,4096]")
    ordered = indices.contiguous().reshape(b * s, k).sort(dim=-1).values
    width = ((k + 127) // 128) * 128
    selected = torch.empty((b * s, width), device=indices.device, dtype=torch.int32)
    lengths = torch.empty((b * s, 8), device=indices.device, dtype=torch.int32)
    native = _compile(b * s, s, k, kv_len, start, vector_cores, indices.device.index)
    native(ordered, selected, lengths)
    return selected.reshape(b, s, 1, width), lengths[:, 0].contiguous().reshape(b, s, 1)


@functools.lru_cache(maxsize=32)
def compile_attention(batch, seq, heads, kv_len, topk, scale, cores, device, wide_qk=False):
    import tilelang

    target = tilelang.tvm.target.Target("ascend")
    with target:
        program = dense_attention_next(batch, seq, heads, kv_len, topk, scale, cores, "reciprocal", wide_qk=wide_qk)
        return tilelang.compile(program, target=target, out_idx=[])


def forward_d512(q, kv, indices, start, scale, cube_cores, vector_cores):
    batch, seq, heads, _ = q.shape
    kv_len = kv.shape[1]
    if heads == 128 and indices.shape[-1] == 1024 and (batch * seq) % 8 == 0:
        selected, lengths = compact(indices.contiguous(), kv_len, start, vector_cores)
    else:
        selected, lengths = compact_indices(indices, kv_len, start, vector_cores)
    out = q.new_empty((batch, seq, heads, 512))
    wide_qk = heads == 128
    attention = compile_attention(batch, seq, heads, kv_len, selected.shape[-1], scale,
                                  cube_cores, q.device.index, wide_qk)
    attention(q.contiguous(), kv.contiguous(), selected, lengths, out)
    return out
