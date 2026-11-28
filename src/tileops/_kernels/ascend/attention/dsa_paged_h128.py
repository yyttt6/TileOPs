"""H128 paged attention with inline FP8 decoding and split-KV reduction.

See THIRD_PARTY_NOTICES for the TileLang source attribution.
"""

import functools

import tilelang.ascend.language as T
import torch
from tilelang.ascend.language import simd as S
from tilelang.layout import make_ascend_compact_nz_layout, make_ascend_major_mn_layout

from tileops._kernels.ascend.attention.dsa_paged_indices import metadata, metadata_row_tile


@T.macro
def decode_shared_kv(raw_codes, raw_scales, raw_rope, nz):
    with T.SimdVF():
        mask = S.pset(32, "PAT_ALL")
        block_mask = S.pset(16, "PAT_VL128")
        for r in range(16):
            for group in range(4):
                sf = S.vld(raw_scales[r, 128 + group], "BRC_B32")
                first = S.vld(raw_codes[r, group * 128], "UNPK4_B8")
                second = S.vld(raw_codes[r, group * 128 + 64], "UNPK4_B8")
                lo = S.vmul(S.vcvt(first, "float32"), sf, mask)
                hi = S.vmul(S.vcvt(second, "float32"), sf, mask)
                even, odd = S.vdintlv(lo, hi)
                a = S.vcvt(even, "bfloat16", mask, sat=False, part=0)
                b = S.vcvt(odd, "bfloat16", mask, sat=False, part=1)
                packed = S.vor(
                    T.reinterpret(a, "uint16x128"), T.reinterpret(b, "uint16x128"), block_mask
                )
                # Eight 16-BF16 column blocks. Neighboring NZ blocks are 17 rows apart.
                S.vsstb(
                    T.reinterpret(packed, "bfloat16x128"),
                    nz[r, group * 128],
                    T.int32(17 << 16),
                    block_mask,
                )
            # Byte 512 is aligned; RoPE begins eight uint16 elements later.
            rope_mask = S.pset(16, "PAT_VL64")
            offsets = T.reinterpret(S.vci(T.int16(8), "int16"), "uint16x128")
            bits = S.vgather2(raw_rope[r, 256], offsets, rope_mask)
            S.vsstb(T.reinterpret(bits, "bfloat16x128"), nz[r, 512], T.int32(17 << 16), rope_mask)


@T.macro
def decode_shared_kv_fast(raw_codes, raw_scales, raw_rope, nz):
    with T.SimdVF():
        mask = S.pset(32, "PAT_ALL")
        fp8_mask = S.pset(8, "PAT_ALL")
        block_mask = S.pset(16, "PAT_VL128")
        for r in range(16):
            for group in range(4):
                sf = S.vld(raw_scales[r, 128 + group], "BRC_B32")
                codes = S.vld(raw_codes[r, group * 128], "UNPK_B8")
                even = S.vmul(S.vcvt(codes, "float32", fp8_mask, part=0), sf, mask)
                odd = S.vmul(S.vcvt(codes, "float32", fp8_mask, part=2), sf, mask)
                a = S.vcvt(even, "bfloat16", mask, sat=False, part=0)
                b = S.vcvt(odd, "bfloat16", mask, sat=False, part=1)
                packed = S.vor(
                    T.reinterpret(a, "uint16x128"), T.reinterpret(b, "uint16x128"), block_mask
                )
                # Eight 16-BF16 column blocks. Neighboring NZ blocks are 17 rows apart.
                S.vsstb(
                    T.reinterpret(packed, "bfloat16x128"),
                    nz[r, group * 128],
                    T.int32(17 << 16),
                    block_mask,
                )
            # Byte 512 is aligned; RoPE begins eight uint16 elements later.
            rope_mask = S.pset(16, "PAT_VL64")
            offsets = T.reinterpret(S.vci(T.int16(8), "int16"), "uint16x128")
            bits = S.vgather2(raw_rope[r, 256], offsets, rope_mask)
            S.vsstb(T.reinterpret(bits, "bfloat16x128"), nz[r, 512], T.int32(17 << 16), rope_mask)


def paged_attention_inline_split(
    batch, seq, heads, kv_len, topk, scale, variant, num_blocks, query_splits=1, schedule="base"
):
    if heads != 128 or variant != "fused128":
        raise ValueError("expected H128 fused128")
    plan = dict(head_tile=128, wide_pv=False, share_scratch=True, staging_versions=1)
    qk_dim = 576
    # Four full latent chunks followed by an exact 64-column RoPE tail.
    qk_chunk = 192
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
    STAGING = plan["staging_versions"]
    head_tiles = heads // M
    jobs = batch * seq * head_tiles
    cores = min(jobs, num_blocks)
    blocks = topk // 128
    overlap = schedule in ("overlap", "balanced_split")
    balanced_split = schedule == "balanced_split"
    metadata_batch = batch // query_splits if balanced_split else batch
    short_cores = 2 * metadata_batch - 4 * cores if balanced_split else 0
    if balanced_split and (
        seq != 1
        or topk != 2048
        or query_splits != 2
        or short_cores * 4 != metadata_batch
        or short_cores <= 0
    ):
        raise ValueError("shape and core-count combination cannot use the balanced 7+9 schedule")

    @T.macro
    def gather(
        batch_id, count, sid, KV, idx, nz, l1, raw, raw_words, raw_codes, raw_scales, raw_rope
    ):
        for chunk in T.Serial(4):
            row_start = sid * 64 + chunk * 16
            rows = T.max(T.min(count - row_start, 16), 0)
            if rows < 16:
                with T.SimdVF():
                    T.fill(raw_words, T.uint32(0))
            for r in T.Serial(rows):
                physical = T.int32(idx[row_start + r])
                T.copy(KV[physical, :], raw[r, :656])
            if query_splits == 2:
                decode_shared_kv_fast(raw_codes, raw_scales, raw_rope, nz)
            else:
                decode_shared_kv(raw_codes, raw_scales, raw_rope, nz)
            T.copy(nz[:16, :], l1[row_start : row_start + 16, :])

    @T.macro
    def gather_overlap(
        count, sid, KV, idx, nz, l1, raw, raw_words, raw_codes, raw_scales, raw_rope
    ):
        # Process four 16-row chunks per AIV. The copy for the next
        # full chunk may run while the current chunk is decoded.
        for chunk in T.Pipelined(4, num_stages=2, annotations={"enable_offset": True}):
            row_start = sid * 64 + chunk * 16
            rows = T.max(T.min(count - row_start, 16), 0)
            with T.Stage(0):
                if rows < 16:
                    with T.SimdVF():
                        T.fill(raw_words, T.uint32(0))
                for r in T.Serial(rows):
                    physical = T.int32(idx[row_start + r])
                    T.copy(KV[physical, :], raw[r, :656])
            with T.Stage(1):
                if query_splits == 2:
                    decode_shared_kv_fast(raw_codes, raw_scales, raw_rope, nz)
                else:
                    decode_shared_kv(raw_codes, raw_scales, raw_rope, nz)
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
                denominator = S.vadd(
                    S.vmul(S.vld(l_ub[head], "BRC_B32"), alpha, fp32_mask),
                    prob_sum,
                    scalar_mask,
                )
                S.vsts(l_ub[head], denominator, scalar_mask, "ONEPT_B32")

    @T.macro
    def pv_chunk(vd, kv_l1, pa, vb, cc, partial):
        T.copy(kv_l1[:, vd * 128 : (vd + 1) * 128], vb[:128, :])
        T.gemm(pa[:128, :], vb[:128, :], cc, transpose_B=False, clear_accum=True)
        T.dual_copy(cc, partial[:, vd * 128 : (vd + 1) * 128])

    out_shape = (batch, 2, HEADS_PER_AIV, 512) if balanced_split else (batch, seq, heads, 512)
    lse_shape = (batch, 2, HEADS_PER_AIV) if balanced_split else (batch, seq, heads)

    @T.prim_func
    def main(
        Q: T.Tensor((batch // query_splits, seq, heads, qk_dim), dtype),
        KV: T.Tensor((kv_len, 656), "uint8"),
        Indices: T.Tensor((metadata_batch, seq, 1, topk), "int32"),
        Lengths: T.Tensor((metadata_batch, seq, 8), "int32"),
        Output: T.Tensor(out_shape, dtype),
        LSE: T.Tensor(lse_shape, accum_dtype),
    ):
        LSE_flat = T.reshape(LSE, (batch * seq * heads,))
        with T.MixedKernel(cores, sids=2) as (bx, sid):
            q_l1 = T.alloc_l1((M, qk_dim), dtype)
            kv_l1 = T.alloc_l1((128, qk_dim), dtype)
            p_l1 = T.alloc_l1((M, 128), dtype)
            qa = T.alloc_l0a((M, qk_chunk), dtype)
            kb = T.alloc_l0b((128, qk_chunk), dtype)
            pa = T.view(qa, (192, M), dtype=dtype)
            vb = T.view(kb, (192, 128), dtype=dtype)
            score_c = T.alloc_l0c((M, 128), accum_dtype)
            # Separate ordinary 2D tiles retain the validated MAD interface;
            # no nonzero-origin trailing C slice or higher-rank C is required.
            values_c = T.alloc_l0c((M, 128), accum_dtype)
            if WIDE:
                values_c1 = T.alloc_l0c((M, 128), accum_dtype)
                values_c2 = T.alloc_l0c((M, 128), accum_dtype)
                values_c3 = T.alloc_l0c((M, 128), accum_dtype)
            raw = T.alloc_shared((16, 672), "uint8")
            raw_words = T.view(raw, (16, 168), dtype="uint32")
            raw_codes = T.view(raw, dtype="float8_e4m3fn")
            raw_scales = T.view(raw, (16, 168), dtype="float32")
            raw_rope = T.view(raw, (16, 336), dtype="uint16")
            kv_nz = T.alloc_shared((17, qk_dim), dtype)
            partial = T.alloc_shared((HEADS_PER_AIV, 512 if WIDE else 128), accum_dtype)
            if SHARE:
                scores = T.view(partial, (HEADS_PER_AIV, 128))
            else:
                scores = T.alloc_shared((HEADS_PER_AIV, 128), accum_dtype)
            prob = T.alloc_shared((HEADS_PER_AIV + 1, 128), dtype)
            accum = T.alloc_shared((HEADS_PER_AIV, 512), accum_dtype)
            output = T.view(accum, (HEADS_PER_AIV * 2, 512), dtype=dtype)
            maxima = T.alloc_shared((64,), accum_dtype)
            denom = T.alloc_shared((64,), accum_dtype)
            lse = T.alloc_shared((64,), accum_dtype)
            alpha = T.alloc_shared((64,), accum_dtype)
            old_m = T.alloc_shared((64,), accum_dtype)
            idx = T.alloc_shared((128,), "int32")
            T.annotate_layout(
                {
                    kv_nz: make_ascend_compact_nz_layout(kv_nz),
                    prob: make_ascend_compact_nz_layout(prob),
                    vb: make_ascend_major_mn_layout(vb),
                }
            )
            T.annotate_buffer_versions(
                {
                    raw: 2 if overlap else 1,
                    q_l1: 1,
                    kv_l1: 2,
                    p_l1: 2,
                    qa: 1,
                    kb: 1,
                    score_c: 1,
                    values_c: 1,
                    kv_nz: STAGING,
                    prob: STAGING,
                    scores: 1,
                    partial: 1,
                    accum: 1,
                    output: 1,
                    maxima: 1,
                    denom: 1,
                    lse: 1,
                    alpha: 1,
                    old_m: 1,
                    idx: 2,
                }
            )
            if WIDE:
                T.annotate_buffer_versions({values_c1: 1, values_c2: 1, values_c3: 1})
            if balanced_split:
                # Cores with five jobs do four 7-tile parts and one 9-tile part.
                # Cores with four jobs do four 9-tile parts.
                tile_work = T.if_then_else(bx < short_cores, 37, 36)
            else:
                tile_work = T.ceildiv(jobs - bx, cores) * blocks
            # The M/L/numerator carry remains a single physical copy.
            for work in T.Pipelined(tile_work, num_stages=2, annotations={"enable_offset": True}):
                if balanced_split:
                    has_short_jobs = bx < short_cores
                    is_short = has_short_jobs & (work < 28)
                    row = T.if_then_else(
                        has_short_jobs,
                        T.if_then_else(is_short, bx * 4 + work // 7, bx),
                        short_cores + (bx - short_cores) * 4 + work // 9,
                    )
                    tile = T.if_then_else(
                        is_short, work % 7, T.if_then_else(has_short_jobs, work - 28, work % 9)
                    )
                    shard = T.if_then_else(is_short, 0, 1)
                    job = 2 * row + shard
                    offset = shard * 7 * 128
                    shard_tiles = T.if_then_else(is_short, 7, 9)
                    index_row = row
                else:
                    job = work // blocks * cores + bx
                    tile = work % blocks
                    offset = 0
                    shard_tiles = blocks
                    index_row = job // (head_tiles * seq)
                ht = job % head_tiles
                t = job // head_tiles % seq
                b = job // (head_tiles * seq)
                if tile == 0:
                    T.copy(Q[b // query_splits, t, ht * M : (ht + 1) * M, :], q_l1)
                    with T.SimdVF():
                        T.fill(accum, T.float32(0))
                        T.fill(maxima, T.float32(-3.402823e38))
                        T.fill(denom, T.float32(0))
                valid_slots = Lengths[index_row, t, 0]
                if balanced_split:
                    valid_slots = T.max(T.min(valid_slots - offset, shard_tiles * 128), 0)
                count = T.max(T.min(valid_slots - tile * 128, 128), 0)
                first = offset + tile * 128
                T.copy(Indices[index_row, t, 0, first : first + 128], idx)
                if overlap:
                    gather_overlap(
                        count,
                        sid,
                        KV,
                        idx,
                        kv_nz,
                        kv_l1,
                        raw,
                        raw_words,
                        raw_codes,
                        raw_scales,
                        raw_rope,
                    )
                else:
                    gather(
                        b,
                        count,
                        sid,
                        KV,
                        idx,
                        kv_nz,
                        kv_l1,
                        raw,
                        raw_words,
                        raw_codes,
                        raw_scales,
                        raw_rope,
                    )
                for kd in T.serial(qk_dim // qk_chunk):
                    T.copy(q_l1[:, kd * qk_chunk : (kd + 1) * qk_chunk], qa)
                    T.copy(kv_l1[:, kd * qk_chunk : (kd + 1) * qk_chunk], kb)
                    T.gemm(qa, kb, score_c, transpose_B=True, clear_accum=kd == 0)
                T.dual_copy(score_c, scores)
                softmax_pack(scores, prob, maxima, denom, alpha, old_m, count)
                T.dual_copy(prob[:HEADS_PER_AIV, :], p_l1)
                T.copy(p_l1, pa[:128, :])
                if WIDE:
                    pv_chunk(0, kv_l1, pa, vb, values_c, partial)
                    pv_chunk(1, kv_l1, pa, vb, values_c1, partial)
                    pv_chunk(2, kv_l1, pa, vb, values_c2, partial)
                    pv_chunk(3, kv_l1, pa, vb, values_c3, partial)
                else:
                    for vd in T.unroll(4):
                        T.copy(kv_l1[:, vd * 128 : (vd + 1) * 128], vb[:128, :])
                        T.gemm(
                            pa[:128, :], vb[:128, :], values_c, transpose_B=False, clear_accum=True
                        )
                        T.dual_copy(values_c, partial)
                        with T.SimdVF():
                            mask = S.pset(32, "PAT_ALL")
                            for r in range(HEADS_PER_AIV):
                                factor = S.vld(alpha[r], "BRC_B32")
                                for col in range(0, 128, 64):
                                    previous = S.vld(accum[r, vd * 128 + col], "NORM")
                                    contribution = S.vld(partial[r, col], "NORM")
                                    if overlap:
                                        updated = S.alloc_var("float32")
                                        updated = previous
                                        S.vmadd(updated, factor, contribution, mask)
                                    else:
                                        updated = S.vadd(
                                            S.vmul(previous, factor, mask), contribution, mask
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
                                updated = S.vadd(S.vmul(previous, factor, mask), contribution, mask)
                                S.vsts(accum[r, col], updated, mask, "NORM_B32")
                if tile == shard_tiles - 1:
                    with T.SimdVF():
                        mask = S.pset(32, "PAT_ALL")
                        for r in range(HEADS_PER_AIV):
                            divisor = S.vmaxs(S.vld(denom[r], "BRC_B32"), T.float32(1e-30), mask)
                            inverse = S.vdiv(S.vdup(T.float32(1), "float32", mask), divisor, mask)
                            for col in range(0, 512, 64):
                                normalized = S.vmul(S.vld(accum[r, col], "NORM"), inverse, mask)
                                packed_out = S.vcvt(normalized, dtype, mask, sat=False)
                                S.vsts(output[r, col], packed_out, mask, "PK_B32")
                    if balanced_split:
                        T.copy(output[:HEADS_PER_AIV, :], Output[b, sid, :, :])
                    else:
                        T.dual_copy(
                            output[:HEADS_PER_AIV, :],
                            Output[b, t, ht * M : (ht + 1) * M, :],
                        )
                    if valid_slots == 0:
                        with T.SimdVF():
                            T.fill(lse, -T.infinity(accum_dtype))
                    else:
                        with T.SimdVF():
                            lse_mask = S.pset(32, "PAT_ALL")
                            value = S.vadd(
                                S.vln(S.vld(denom[0], "NORM"), lse_mask),
                                S.vld(maxima[0], "NORM"),
                                lse_mask,
                            )
                            S.vsts(lse[0], value, lse_mask, "NORM_B32")
                    if balanced_split:
                        T.copy(lse[:HEADS_PER_AIV], LSE[b, sid, :])
                    else:
                        T.dual_copy(
                            lse[:HEADS_PER_AIV],
                            LSE_flat[
                                (b * seq + t) * heads + ht * M : (b * seq + t) * heads
                                + (ht + 1) * M
                            ],
                        )

    return main


def merge_builder(rows, splits, heads, cores, first_split_init=False):
    import tilelang.ascend.language as T
    from tilelang.ascend.language import simd as S

    partial_dtype = "bfloat16" if heads == 128 else "float32"
    first_split_init = first_split_init and heads == 128 and splits > 1
    tile = 16
    jobs = rows * (heads // tile)
    blocks = min(jobs, cores)

    @T.prim_func
    def main(
        Part: T.Tensor((rows, splits, heads, 512), partial_dtype),
        Log: T.Tensor((rows, splits, heads), "float32"),
        Out: T.Tensor((rows, heads, 512), "bfloat16"),
        Lse: T.Tensor((rows, heads), "float32"),
    ):
        with T.Kernel(blocks) as bx:
            logs = T.alloc_shared((splits, 64), "float32")
            weights = T.alloc_shared((splits, 64), "float32")
            inverse_ub = T.alloc_shared((64,), "float32")
            merged_lse = T.alloc_shared((64,), "float32")
            values = T.alloc_shared((tile, 512), partial_dtype)
            accum = T.alloc_shared((tile, 512), "float32")
            packed = T.alloc_shared((tile, 512), "bfloat16")
            for task in T.Serial(T.ceildiv(jobs - bx, blocks)):
                job = task * blocks + bx
                row = job // (heads // tile)
                start = job % (heads // tile) * tile
                with T.SimdVF():
                    T.fill(logs, -T.infinity("float32"))
                    if not first_split_init:
                        T.fill(accum, T.float32(0))
                for split in T.Serial(splits):
                    T.copy(Log[row, split, start : start + tile], logs[split, :tile])
                with T.SimdVF():
                    mask = S.pset(32, "PAT_ALL")
                    # Finite max avoids -inf - -inf for an entirely empty head.
                    mx = T.alloc_var("float32x64")
                    mx = S.vdup(T.float32(-3.402823e38), "float32", mask)
                    for split in range(splits):
                        mx = S.vmax(mx, S.vld(logs[split, 0], "NORM"), mask)
                    total = T.alloc_var("float32x64")
                    total = S.vdup(T.float32(0), "float32", mask)
                    for split in range(splits):
                        weight = S.vexpdif(S.vld(logs[split, 0], "NORM"), mx, mask)
                        S.vsts(weights[split, 0], weight, mask, "NORM_B32")
                        total = S.vadd(total, weight, mask)
                    divisor = S.vmaxs(total, T.float32(1e-30), mask)
                    inv = S.vdiv(S.vdup(T.float32(1), "float32", mask), divisor, mask)
                    S.vsts(inverse_ub[0], inv, mask, "NORM_B32")
                    log_total = S.vadd(S.vln(total, mask), mx, mask)
                    positive = S.vcmps(total, T.float32(0), mask, op="gt")
                    log_total = S.vsel(
                        log_total,
                        S.vdup(-T.infinity("float32"), "float32", mask),
                        positive,
                    )
                    S.vsts(merged_lse[0], log_total, mask, "NORM_B32")
                if first_split_init:
                    T.copy(Part[row, 0, start : start + tile, :], values)
                    with T.SimdVF():
                        mask = S.pset(32, "PAT_ALL")
                        for r in range(tile):
                            weight = S.vld(weights[0, r], "BRC_B32")
                            for col in range(0, 512, 64):
                                v = S.vcvt(S.vld(values[r, col], "UNPK_B16"), "float32")
                                S.vsts(accum[r, col], S.vmul(v, weight, mask), mask, "NORM_B32")
                for split in T.Serial(1 if first_split_init else 0, splits - 1):
                    T.copy(Part[row, split, start : start + tile, :], values)
                    with T.SimdVF():
                        mask = S.pset(32, "PAT_ALL")
                        for r in range(tile):
                            weight = S.vld(weights[split, r], "BRC_B32")
                            for col in range(0, 512, 64):
                                if heads == 128:
                                    loaded = S.vcvt(S.vld(values[r, col], "UNPK_B16"), "float32")
                                    value = S.vmul(loaded, weight, mask)
                                else:
                                    value = S.vmul(S.vld(values[r, col], "NORM"), weight, mask)
                                value = S.vadd(S.vld(accum[r, col], "NORM"), value, mask)
                                S.vsts(accum[r, col], value, mask, "NORM_B32")
                T.copy(Part[row, splits - 1, start : start + tile, :], values)
                with T.SimdVF():
                    mask = S.pset(32, "PAT_ALL")
                    for r in range(tile):
                        weight = S.vld(weights[splits - 1, r], "BRC_B32")
                        inverse = S.vld(inverse_ub[r], "BRC_B32")
                        for col in range(0, 512, 64):
                            if heads == 128:
                                loaded = S.vcvt(S.vld(values[r, col], "UNPK_B16"), "float32")
                            else:
                                loaded = S.vld(values[r, col], "NORM")
                            value = S.vmul(loaded, weight, mask)
                            value = S.vadd(S.vld(accum[r, col], "NORM"), value, mask)
                            normalized = S.vmul(value, inverse, mask)
                            bf16 = S.vcvt(normalized, "bfloat16", mask, sat=False)
                            S.vsts(packed[r, col], bf16, mask, "PK_B32")
                T.copy(packed[:tile, :], Out[row, start : start + tile, :])
                T.copy(merged_lse[:tile], Lse[row, start : start + tile])

    return main


@functools.lru_cache(maxsize=32)
def compile_merge(rows, splits, heads, cores, device, first_split_init=False):
    import tilelang

    target = tilelang.tvm.target.Target("ascend")
    with target:
        return tilelang.compile(
            merge_builder(rows, splits, heads, cores, first_split_init), target=target, out_idx=[]
        )


@functools.lru_cache(maxsize=32)
def compile_attention(rows, capacity, part, scale, cores, device, splits, schedule="base"):
    import tilelang

    target = tilelang.tvm.target.Target("ascend")
    with target:
        program = paged_attention_inline_split(
            rows * splits, 1, 128, capacity, part, scale, "fused128", cores, splits, schedule
        )
        return tilelang.compile(program, target=target, out_idx=[])


def forward_h128(
    q,
    cache,
    table,
    limits,
    indices,
    scale,
    splits,
    cube_cores,
    vector_cores,
    *,
    first_split_init=False,
    schedule="base",
):
    b, s, heads, depth = q.shape
    rows = b * s
    metadata_splits = 1 if schedule == "balanced_split" else splits
    page_size = cache.shape[1]
    topk = indices.shape[-1]
    part = topk // metadata_splits
    full_row_fast = (
        splits == 2
        and schedule in ("balanced_split", "overlap")
        and page_size == 64
        and part >= 128
        and part % 128 == 0
        and part & (part - 1) == 0
        and topk == part * metadata_splits
        and metadata_row_tile(rows * metadata_splits, part, table.shape[1]) == 1
    )
    physical, lengths = metadata(
        indices,
        limits,
        table,
        page_size,
        metadata_splits,
        vector_cores,
        full_row_fast=full_row_fast,
    )
    part = physical.shape[-1]
    query = q.reshape(rows, 1, heads, depth).contiguous()
    packed = cache.reshape(-1, 656).contiguous()
    partial = q.new_empty((rows * splits, 1, heads, 512))
    logs = torch.empty((rows * splits, 1, heads), dtype=torch.float32, device=q.device)
    out = q.new_empty((rows, heads, 512))
    lse = torch.empty((rows, heads), dtype=torch.float32, device=q.device)
    device = q.device.index
    attention = compile_attention(
        rows, packed.shape[0], part, scale, cube_cores, device, splits, schedule
    )
    merge = compile_merge(
        rows, splits, heads, vector_cores, device, first_split_init or splits == 4
    )
    if schedule == "balanced_split":
        attention(
            query,
            packed,
            physical,
            lengths,
            partial.view(rows * splits, 2, heads // 2, 512),
            logs.view(rows * splits, 2, heads // 2),
        )
    else:
        attention(query, packed, physical, lengths, partial, logs)
    merge(partial.reshape(rows, splits, heads, 512), logs.reshape(rows, splits, heads), out, lse)
    return out.reshape(b, s, heads, 512), lse.reshape(b, s, heads).transpose(1, 2).contiguous()
