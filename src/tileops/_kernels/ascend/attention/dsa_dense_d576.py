"""D576 BF16 sparse attention with a fused H128 tile and DMA KV gather.

See THIRD_PARTY_NOTICES for the TileLang source attribution.
"""

import functools

import tilelang
import tilelang.ascend.language as T
import torch
from tilelang.ascend.language import simd as S
from tilelang.layout import make_ascend_compact_nz_layout, make_ascend_major_mn_layout
from tvm import tirx

from tileops._kernels.ascend.attention.dsa_dense_d512 import compact


def dense_attention_tail(batch, seq, heads, kv_len, topk, scale, num_blocks):
    qk_dim = 576
    # A 64-wide QK tile handles the 64-column rope tail without padding or
    # stale L0 lanes. It also keeps the L0 footprint below the D512 path.
    qk_chunk = 64
    row_bytes = qk_dim * 2

    def copy_pair(dst, src, stride):
        return tirx.call_intrin(
            "void",
            tirx.op.Op.get("tl.ascend_copy_gm_to_ubuf"),
            dst,
            src,
            0,
            2,
            row_bytes,
            0,
            0,
            0,
            0,
            stride * row_bytes,
            row_bytes,
        )

    if min(batch, seq, kv_len, topk) <= 0 or topk % 128:
        raise ValueError("positive dimensions and K multiple of 128 required")
    M = heads
    HEADS_PER_AIV = M // 2
    BLOCK_TOP_K = 128
    VECTOR_LANES = 64
    SOFTMAX_SCALE = scale
    dtype = "bfloat16"
    accum_dtype = "float32"
    head_tiles = heads // M
    jobs = batch * seq * head_tiles
    cores = min(jobs, num_blocks)
    blocks = topk // 128

    @T.macro
    def gather(batch_id, count, sid, KV, idx, nd, nz, l1):
        for chunk in T.Serial(4, annotations={"multi_buffer_eligible": [nd]}):
            row_start = sid * 64 + chunk * 16
            rows = T.max(T.min(count - row_start, 16), 0)
            # Full chunks overwrite every byte; only a tail/empty chunk needs
            # clearing. The adapter guarantees a sorted, unique valid prefix.
            if rows < 16:
                with T.SimdVF():
                    T.fill(nd, T.float32(0))
            for pair in T.Serial(rows // 2):
                i0 = T.int32(idx[row_start + 2 * pair])
                i1 = T.int32(idx[row_start + 2 * pair + 1])
                T.assume(i0 >= 0)
                T.assume(i0 < i1)
                T.assume(i1 < kv_len)
                copy_pair(
                    T.access_ptr(nd[2 * pair, 0], "w"),
                    T.access_ptr(KV[batch_id, i0, 0, 0], "r"),
                    i1 - i0,
                )
            if rows % 2 != 0:
                tail = rows - 1
                index = T.int32(idx[row_start + tail])
                T.assume(index >= 0)
                T.assume(index < kv_len)
                T.copy(KV[batch_id, index, 0, :], nd[tail, :])
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

    @T.prim_func
    def main(
        Q: T.Tensor((batch, seq, heads, qk_dim), dtype),
        KV: T.Tensor((batch, kv_len, 1, qk_dim), dtype),
        Indices: T.Tensor((batch, seq, 1, topk), "int32"),
        Lengths: T.Tensor((batch, seq, 1), "int32"),
        Output: T.Tensor((batch, seq, heads, 512), dtype),
    ):
        with T.MixedKernel(cores, sids=2) as (bx, sid):
            q_l1 = T.alloc_l1((M, qk_dim), dtype)
            kv_l1 = T.alloc_l1((128, qk_dim), dtype)
            p_l1 = T.alloc_l1((M, 128), dtype)
            qa = T.alloc_l0a((M, qk_chunk), dtype)
            kb = T.alloc_l0b((128, qk_chunk), dtype)
            pa = T.alloc_l0a((M, 128), dtype)
            vb = T.alloc_l0b((128, 128), dtype)
            score_c = T.alloc_l0c((M, 128), accum_dtype)
            # Separate ordinary 2D tiles retain the validated MAD interface;
            # no nonzero-origin trailing C slice or higher-rank C is required.
            values_c = T.alloc_l0c((M, 128), accum_dtype)
            kv_nd = T.alloc_shared((16, qk_dim), dtype)
            kv_nz = T.alloc_shared((17, qk_dim), dtype)
            partial = T.alloc_shared((HEADS_PER_AIV, 128), accum_dtype)
            scores = T.view(partial, (HEADS_PER_AIV, 128))
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
                    kv_nd: 1,
                    kv_nz: 1,
                    prob: 1,
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
            tasks = T.ceildiv(jobs - bx, cores)
            # Flatten query/head tasks and KV blocks into one pipeline, as in
            # the D512 path. The softmax state remains a single physical copy.
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
                gather(b, count, sid, KV, idx, kv_nd, kv_nz, kv_l1)
                for kd in T.serial(qk_dim // qk_chunk):
                    T.copy(q_l1[:, kd * qk_chunk : (kd + 1) * qk_chunk], qa)
                    T.copy(kv_l1[:, kd * qk_chunk : (kd + 1) * qk_chunk], kb)
                    T.gemm(qa, kb, score_c, transpose_B=True, clear_accum=kd == 0)
                T.dual_copy(score_c, scores)
                softmax_pack(scores, prob, maxima, denom, alpha, old_m, count)
                T.dual_copy(prob[:HEADS_PER_AIV, :], p_l1)
                T.copy(p_l1, pa)
                for vd in T.unroll(4):
                    T.copy(kv_l1[:, vd * 128 : (vd + 1) * 128], vb)
                    T.gemm(pa, vb, values_c, transpose_B=False, clear_accum=True)
                    T.dual_copy(values_c, partial)
                    with T.SimdVF():
                        mask = S.pset(32, "PAT_ALL")
                        for r in range(HEADS_PER_AIV):
                            factor = S.vld(alpha[r], "BRC_B32")
                            for col in range(0, 128, 64):
                                previous = S.vld(accum[r, vd * 128 + col], "NORM")
                                contribution = S.vld(partial[r, col], "NORM")
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
                if tile == blocks - 1:
                    with T.SimdVF():
                        mask = S.pset(32, "PAT_ALL")
                        for r in range(HEADS_PER_AIV):
                            divisor = S.vmaxs(S.vld(denom[r], "BRC_B32"), T.float32(1e-30), mask)
                            inverse = S.vdiv(S.vdup(T.float32(1), accum_dtype, mask), divisor, mask)
                            for col in range(0, 512, 64):
                                normalized = S.vmul(S.vld(accum[r, col], "NORM"), inverse, mask)
                                packed = S.vcvt(normalized, dtype, mask, sat=False)
                                S.vsts(output[r, col], packed, mask, "PK_B32")
                    T.dual_copy(
                        output[:HEADS_PER_AIV, :],
                        Output[b, t, ht * M : (ht + 1) * M, :],
                    )

    return main


def prepare_indices(indices, kv_len, start):
    """Return a sorted, unique prefix and its length for each query row."""
    positions = torch.arange(indices.shape[1], device=indices.device, dtype=torch.int64)
    cutoff = (positions + start).view(1, -1, 1, 1)
    valid = (indices >= 0) & (indices < kv_len) & (indices <= cutoff)
    ordered = torch.where(valid, indices, kv_len).sort(dim=-1).values
    repeated = ordered[..., 1:] == ordered[..., :-1]
    selected = (
        torch.cat((ordered[..., :1], torch.where(repeated, kv_len, ordered[..., 1:])), dim=-1)
        .sort(dim=-1)
        .values
    )
    return selected, (selected < kv_len).sum(-1, dtype=torch.int32).contiguous()


@functools.lru_cache(maxsize=32)
def compile_tail(batch, seq, heads, kv_len, topk, scale, cube_cores, device):
    target = tilelang.tvm.target.Target("ascend")
    with target:
        program = dense_attention_tail(batch, seq, heads, kv_len, topk, scale, cube_cores)
        return tilelang.compile(program, target=target, out_idx=[])


def forward_d576(q, kv, indices, start, scale, cube_cores, vector_cores):
    batch, seq, heads, _ = q.shape
    if indices.shape[-1] in (1024, 2048) and (batch * seq) % 8 == 0:
        selected, lengths = compact(indices.contiguous(), kv.shape[1], start, vector_cores)
    else:
        selected, lengths = prepare_indices(indices, kv.shape[1], start)
    result = q.new_empty((batch, seq, heads, 512))
    kernel = compile_tail(
        batch, seq, heads, kv.shape[1], selected.shape[-1], scale, cube_cores, q.device.index
    )
    kernel(q.contiguous(), kv.contiguous(), selected, lengths, result)
    return result
