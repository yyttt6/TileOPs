"""Paged prototype core: validated tail DMA arithmetic plus online-state LSE.

Experimental D576/BF16 DMA path, derived from the measured D512 program.

DMA and SIMD packing derive from PR #3322, commit
4c0cb5fcdaf466c2a44e63ef1404553b94909da1 (MIT), adapted by the validated
kernel_dense_dma.py. This module is additive and is not a measured winner.
"""

import functools

import torch
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S
from tilelang.layout import make_ascend_compact_nz_layout, make_ascend_major_mn_layout

from .dsa_paged_indices import metadata


def paged_attention_core(batch, seq, heads, kv_len, topk, scale, variant, num_blocks):
    if heads != 64:
        raise ValueError("expected H64")
    plan = dict(head_tile=64, wide_pv=True, share_scratch=False, staging_versions=1)
    qk_dim = 576
    # Three equal K reductions cover the full 576-dimensional query.
    qk_chunk = 192
    if variant != "wide64":
        raise ValueError("PV256 requires H64 wide64")
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

    @T.macro
    def gather(batch_id, tile, count, sid, KV, nd, nz, l1):
        for chunk in T.Serial(4):
            row_start = sid * 64 + chunk * 16
            rows = T.max(T.min(count - row_start, 16), 0)
            begin = tile * 128 + row_start
            if rows < 16:
                with T.SimdVF():
                    T.fill(nd, T.float32(0))
            if rows > 0:
                T.copy(KV[batch_id, begin : begin + rows, 0, :], nd[:rows, :])
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
                denominator = S.vadd(
                    S.vmul(S.vld(l_ub[head], "BRC_B32"), alpha, fp32_mask),
                    prob_sum,
                    scalar_mask,
                )
                S.vsts(l_ub[head], denominator, scalar_mask, "ONEPT_B32")

    @T.macro
    def pv_chunk(vd, kv_l1, pa, vb, cc, partial):
        T.copy(kv_l1[:, vd * 256 : (vd + 1) * 256], vb)
        T.gemm(pa, vb, cc, transpose_B=False, clear_accum=True)
        T.dual_copy(cc, partial[:, vd * 256 : (vd + 1) * 256])

    @T.macro
    def qk_step(offset, clear, q_l1, kv_l1, qa, kb, score_c):
        T.copy(q_l1[:, offset : offset + qk_chunk], qa)
        T.copy(kv_l1[:, offset : offset + qk_chunk], kb[:, :192])
        T.gemm(qa, kb[:, :192], score_c, transpose_B=True, clear_accum=clear)

    @T.prim_func
    def main(
        Q: T.Tensor((batch, seq, heads, qk_dim), dtype),
        KV: T.Tensor((batch, kv_len, 1, qk_dim), dtype),
        Lengths: T.Tensor((batch, seq, 8), "int32"),
        Output: T.Tensor((batch, seq, heads, 512), dtype),
        LSE: T.Tensor((batch, seq, heads), accum_dtype),
    ):
        LSE_flat = T.reshape(LSE, (batch * seq * heads,))
        with T.MixedKernel(cores, sids=2) as (bx, sid):
            q_l1 = T.alloc_l1((M, qk_dim), dtype)
            kv_l1 = T.alloc_l1((128, qk_dim), dtype)
            p_l1 = T.alloc_l1((M, 128), dtype)
            qa = T.alloc_l0a((M, qk_chunk), dtype)
            kb = T.alloc_l0b((128, 256), dtype)
            pa = T.alloc_l0a((M, 128), dtype)
            vb = T.view(kb, (128, 256), dtype=dtype)
            score_c = T.alloc_l0c((M, 128), accum_dtype)
            # Separate ordinary 2D tiles retain the validated MAD interface;
            # no nonzero-origin trailing C slice or higher-rank C is required.
            values_c = T.alloc_l0c((M, 256), accum_dtype)
            if WIDE:
                values_c1 = T.alloc_l0c((M, 256), accum_dtype)
            kv_nd = T.alloc_shared((16, qk_dim), dtype)
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
            T.annotate_layout(
                {
                    kv_nz: make_ascend_compact_nz_layout(kv_nz),
                    prob: make_ascend_compact_nz_layout(prob),
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

                    score_c: 1,
                    values_c: 1,
                    kv_nd: STAGING,
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
                }
            )
            if WIDE:
                T.annotate_buffer_versions({values_c1: 1})
            tasks = T.ceildiv(jobs - bx, cores)
            for task in T.Serial(tasks):
                job = task * cores + bx
                ht = job % head_tiles
                t = job // head_tiles % seq
                b = job // (head_tiles * seq)
                live = T.min(T.max(Lengths[b, t, 0], 0), topk)
                active_tiles = T.max(T.ceildiv(live, 128), 1)
                T.copy(Q[b, t, ht * M : (ht + 1) * M, :], q_l1)
                with T.SimdVF():
                    T.fill(accum, T.float32(0))
                    T.fill(maxima, T.float32(-3.402823e38))
                    T.fill(denom, T.float32(0))
                for tile in T.Pipelined(
                    active_tiles, num_stages=2, annotations={"enable_offset": True}
                ):
                    count = T.max(T.min(live - tile * 128, 128), 0)
                    if count == 128:
                        T.copy(KV[b, tile * 128 : (tile + 1) * 128, 0, :], kv_l1)
                    else:
                        gather(b, tile, count, sid, KV, kv_nd, kv_nz, kv_l1)
                    qk_step(0, True, q_l1, kv_l1, qa, kb, score_c)
                    qk_step(192, False, q_l1, kv_l1, qa, kb, score_c)
                    qk_step(384, False, q_l1, kv_l1, qa, kb, score_c)
                    T.dual_copy(score_c, scores)
                    softmax_pack(scores, prob, maxima, denom, alpha, old_m, count)
                    T.dual_copy(prob[:HEADS_PER_AIV, :], p_l1)
                    T.copy(p_l1, pa)
                    pv_chunk(0, kv_l1, pa, vb, values_c, partial)
                    pv_chunk(1, kv_l1, pa, vb, values_c1, partial)
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
                with T.SimdVF():
                    mask = S.pset(32, "PAT_ALL")
                    for r in range(HEADS_PER_AIV):
                        divisor = S.vmaxs(S.vld(denom[r], "BRC_B32"), T.float32(1e-30), mask)
                        inverse = S.vdiv(S.vdup(T.float32(1), "float32", mask), divisor, mask)
                        for col in range(0, 512, 64):
                            normalized = S.vmul(S.vld(accum[r, col], "NORM"), inverse, mask)
                            packed = S.vcvt(normalized, dtype, mask, sat=False)
                            S.vsts(output[r, col], packed, mask, "PK_B32")
                T.dual_copy(
                    output[:HEADS_PER_AIV, :],
                    Output[b, t, ht * M : (ht + 1) * M, :],
                )
                if Lengths[b, t, 0] == 0:
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
                T.dual_copy(
                    lse[:HEADS_PER_AIV],
                    LSE_flat[
                        (b * seq + t) * heads + ht * M : (b * seq + t) * heads + (ht + 1) * M
                    ],
                )

    return main


def gather_schedule(queries, tiles, vector_cores):
    """Partition the compact rows using all available AIVs when useful."""
    helpers = min(max(vector_cores - queries, 0), queries)
    if not helpers:
        return min(vector_cores, queries), tiles, 0

    # A two-thirds occupancy estimate sets the shared window. The primary AIV
    # still owns all remaining tiles if more slots are live at runtime.
    expected_live_tiles = (2 * tiles + 2) // 3
    rows_per_helper = (queries + helpers - 1) // helpers
    shared_tiles = expected_live_tiles // (rows_per_helper + 1)
    if not shared_tiles:
        return queries, tiles, 0
    return queries + helpers, expected_live_tiles - shared_tiles, shared_tiles


def builder(b, s, k, capacity, cores):
    import tilelang.ascend.language as T
    from tilelang.ascend.language import simd as S

    # UNPK4_B8 and packed output accesses must start at 32-byte boundaries.
    # Broadcast FP32 scales use naturally aligned 4-byte scalar addresses.
    assert all((r * 672 + chunk * 64) % 32 == 0 for r in range(32) for chunk in range(8))
    assert all((r * 576 * 2 + chunk * 64 * 2) % 32 == 0 for r in range(32) for chunk in range(8))
    assert all((r * 672 + 512 + group * 4) % 4 == 0 for r in range(32) for group in range(4))
    tile = 32
    if k % tile:
        raise ValueError("requires K multiple of 32")
    queries = b * s
    blocks, base_tiles, shared_tiles = gather_schedule(queries, k // tile, cores)
    balanced = shared_tiles > 0
    helpers = max(1, blocks - queries)

    @T.prim_func
    def main(
        Cache: T.Tensor((capacity, 656), "uint8"),
        Index: T.Tensor((b * s, k), "int32"),
        Lens: T.Tensor((b * s, 8), "int32"),
        Output: T.Tensor((b * s * k, 576), "bfloat16"),
    ):
        with T.Kernel(blocks) as bx:
            index_ub = T.alloc_shared((k,), "int32")
            raw = T.alloc_shared((tile, 672), "uint8")
            raw_words = T.view(raw, shape=(tile, 168), dtype="uint32")
            codes = T.view(raw, dtype="float8_e4m3fn")
            scales = T.view(raw, shape=(tile, 168), dtype="float32")
            rope = T.view(raw, shape=(tile, 336), dtype="bfloat16")
            decoded = T.alloc_shared((tile, 576), "bfloat16")
            # All raw views share raw.data; ownership/versioning follows that
            # allocation. Metadata remains single-version and immutable per job.
            T.annotate_buffer_versions({raw: 2, decoded: 2, index_ub: 1})
            limit = T.alloc_var("int32")
            if balanced:
                primary = bx < queries
                helper = T.max(bx - queries, 0)
                first = helper * queries // helpers
                last = (helper + 1) * queries // helpers
                tasks = T.if_then_else(primary, 1, last - first)
            else:
                tasks = T.ceildiv(queries - bx, blocks)
            for task in T.Serial(tasks):
                if balanced:
                    query_id = T.if_then_else(primary, bx, first + task)
                else:
                    query_id = task * blocks + bx
                if balanced:
                    if primary:
                        T.copy(Index[query_id, :], index_ub)
                    else:
                        begin = base_tiles * tile
                        end = (base_tiles + shared_tiles) * tile
                        T.copy(Index[query_id, begin:end], index_ub[begin:end])
                else:
                    T.copy(Index[query_id, :], index_ub)
                limit = Lens[query_id, 0]
                active_tiles = T.ceildiv(limit, tile)
                if balanced:
                    core_tiles = T.min(active_tiles, base_tiles)
                    overflow_tiles = T.max(active_tiles - base_tiles - shared_tiles, 0)
                    shared_count = T.min(T.max(active_tiles - base_tiles, 0), shared_tiles)
                    steps = T.if_then_else(primary, core_tiles + overflow_tiles, shared_count)
                else:
                    steps = active_tiles
                for step in T.Pipelined(
                    steps,
                    num_stages=2,
                    annotations={"enable_offset": True},
                ):
                    if balanced:
                        tile_index = T.if_then_else(
                            primary, step + T.if_then_else(step < base_tiles, 0, shared_tiles), base_tiles + step
                        )
                        slot_begin = tile_index * tile
                    else:
                        slot_begin = step * tile
                    valid_rows = T.min(limit - slot_begin, tile)
                    if valid_rows < tile:
                        with T.SimdVF():
                            T.fill(raw_words, T.uint32(0))
                    for r in T.Serial(valid_rows):
                        physical = index_ub[slot_begin + r]
                        T.copy(Cache[physical, :], raw[r, :656])
                    with T.SimdVF():
                        mask = S.pset(32, "PAT_ALL")
                        for r in range(tile):
                            for group in range(4):
                                sf = S.vld(scales[r, 128 + group], "BRC_B32")
                                for half in range(2):
                                    col = group * 128 + half * 64
                                    packed = S.vld(codes[r, col], "UNPK4_B8")
                                    value = S.vcvt(packed, "float32")
                                    scaled = S.vmul(value, sf, mask)
                                    bf = S.vcvt(scaled, "bfloat16", mask, sat=False)
                                    S.vsts(decoded[r, col], bf, mask, "PK_B32")
                    # RoPE starts at byte 528 (16 modulo 32), so a vector
                    # UNPK_B16 load there is illegal. Retain the validated
                    # scalar-element SIMT copy for just these 64 BF16 values.
                    with T.SimtVF(threads=128):
                        for r, d in T.Parallel(tile, 64):
                            decoded[r, 512 + d] = rope[r, 264 + d]
                    begin = query_id * k + slot_begin
                    T.copy(decoded, Output[begin : begin + tile, :])

    return main


@functools.lru_cache(maxsize=64)
def compile_gather(b, s, k, capacity, cores, device):
    import tilelang

    target = tilelang.tvm.target.Target("ascend")
    with target:
        return tilelang.compile(builder(b, s, k, capacity, cores), target=target, out_idx=[])


def gather(cache, index, lens, batch, seq, vector_cores):
    k = index.shape[-1]
    packed = cache.reshape(-1, 656).contiguous()
    output = torch.empty((batch * seq * k, 576), dtype=torch.bfloat16, device=index.device)
    native = compile_gather(batch, seq, k, packed.shape[0], vector_cores, index.device.index)
    native(packed, index.reshape(batch * seq, k), lens.reshape(batch * seq, 8), output)
    return output.reshape(batch * seq, k, 1, 576)

@functools.lru_cache(maxsize=32)
def compile_attention(rows, topk, scale, cores, device):
    import tilelang

    target = tilelang.tvm.target.Target("ascend")
    with target:
        program = paged_attention_core(rows, 1, 64, topk, topk, scale, "wide64", cores)
        return tilelang.compile(program, target=target, out_idx=[])


def forward_h64(q, cache, table, limits, indices, scale, cube_cores, vector_cores):
    b, s, heads, depth = q.shape
    rows = b * s
    physical, lengths = metadata(indices, limits, table, cache.shape[1], 1, vector_cores)
    kv = gather(cache, physical, lengths, b, s, vector_cores)
    query = q.reshape(rows, 1, heads, depth).contiguous()
    out = q.new_empty((rows, 1, heads, 512))
    lse = torch.empty((rows, 1, heads), dtype=torch.float32, device=q.device)
    attention = compile_attention(rows, kv.shape[1], scale, cube_cores, q.device.index)
    attention(query, kv, lengths, out, lse)
    return out.reshape(b, s, heads, 512), lse.reshape(b, s, heads).transpose(1, 2).contiguous()
