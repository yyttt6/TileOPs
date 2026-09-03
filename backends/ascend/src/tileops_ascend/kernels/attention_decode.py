"""Decode attention kernels for continuous and paged KV caches.

The decode path is deliberately separate from the prefill attention kernel.  A
decode task owns one (batch, head) pair and scans the cache in block_N chunks;
the query sequence is one token, so there is no useful Q-sequence tiling axis.
"""

import functools

import tilelang
from tilelang import language as T


PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}
NUM_AI_CORES = 25


@functools.lru_cache(maxsize=32)
def _decode_kernel(
    batch,
    heads,
    heads_kv,
    max_seqlen_kv,
    dim,
    pe_dim,
    dtype_name,
    q_has_seq,
    is_mla,
    paged=False,
    page_size=0,
):
    block_m = 16
    block_n = 64
    block_num = batch * heads
    launch_blocks = min(block_num, NUM_AI_CORES)
    task_rounds = (block_num + launch_blocks - 1) // launch_blocks
    max_iters = (max_seqlen_kv + block_n - 1) // block_n
    blocks_per_page = page_size // block_n if paged else 1
    v_rows = block_m // 2
    sm_scale = (1.0 / (dim + pe_dim if is_mla else dim)) ** 0.5

    @tilelang.jit(
        out_idx=[7],
        workspace_idx=[8, 9, 10],
        pass_configs=PASS_CONFIGS,
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _build():
        q_shape = [batch, 1, heads, dim] if q_has_seq else [batch, heads, dim]
        kv_shape = (
            [max_seqlen_kv, heads_kv, dim]
            if paged
            else [batch, max_seqlen_kv, heads_kv, dim]
        )
        q_pe_shape = [batch, heads, pe_dim] if is_mla else q_shape
        k_pe_shape = [batch, max_seqlen_kv, heads_kv, pe_dim] if is_mla else kv_shape
        cache_shape = [batch]
        block_table_shape = [
            batch,
            (max_seqlen_kv + page_size - 1) // page_size if paged else 1,
        ]
        output_shape = q_shape
        score_ws_shape = [launch_blocks, block_m, block_n]
        prob_ws_shape = [launch_blocks, block_m, block_n]
        value_ws_shape = [launch_blocks, block_m, dim]
        @T.prim_func
        def main(
            Q: T.Tensor(q_shape, dtype_name),
            QPe: T.Tensor(q_pe_shape, dtype_name),
            K: T.Tensor(kv_shape, dtype_name),
            KPe: T.Tensor(k_pe_shape, dtype_name),
            V: T.Tensor(kv_shape, dtype_name),
            CacheSeqLens: T.Tensor(cache_shape, "int32"),
            BlockTable: T.Tensor(block_table_shape, "int32"),
            Output: T.Tensor(output_shape, dtype_name),
            ScoreWS: T.Tensor(score_ws_shape, "float"),
            ProbWS: T.Tensor(prob_ws_shape, dtype_name),
            ValueWS: T.Tensor(value_ws_shape, "float"),
        ):
            with T.Kernel(launch_blocks, is_npu=True) as (cid, vid):
                q_l1 = T.alloc_L1([block_m, dim], dtype_name)
                q_pe_l1 = T.alloc_L1([block_m, pe_dim], dtype_name) if is_mla else q_l1
                k_l1 = T.alloc_L1([block_n, dim], dtype_name)
                k_pe_l1 = T.alloc_L1([block_n, pe_dim], dtype_name) if is_mla else k_l1
                v_l1 = T.alloc_L1([block_n, dim], dtype_name)
                score_l1 = T.alloc_L1([block_m, block_n], dtype_name)
                score_l0c = T.alloc_L0C([block_m, block_n], "float")
                value_l0c = T.alloc_L0C([block_m, dim], "float")

                acc_o = T.alloc_ub([v_rows, dim], "float")
                scores = T.alloc_ub([v_rows, block_n], "float")
                scores_other = T.alloc_ub([v_rows, block_n], "float")
                probs = T.alloc_ub([v_rows, block_n], dtype_name)
                value = T.alloc_ub([v_rows, dim], "float")
                max_prev = T.alloc_ub([v_rows], "float")
                max_cur = T.alloc_ub([v_rows], "float")
                sum_prev = T.alloc_ub([v_rows], "float")
                sum_cur = T.alloc_ub([v_rows], "float")
                tmp = T.alloc_ub([v_rows, dim], dtype_name)

                score_ready, score_free = 0, 1
                prob_ready, prob_free = 2, 3
                value_ready, value_free = 4, 5

                with T.Scope("C"):
                    T.set_cross_flag("MTE2", prob_free)
                    for task_round in T.serial(task_rounds):
                        task_id = cid + task_round * launch_blocks
                        if task_id < block_num:
                            bid = task_id // heads
                            hid = task_id % heads
                            kv_hid = hid // (heads // heads_kv)
                            cache_len = CacheSeqLens[bid]
                            if q_has_seq:
                                T.copy(Q[bid, 0, hid, :], q_l1[0, :])
                            else:
                                T.copy(Q[bid, hid, :], q_l1[0, :])
                            if is_mla:
                                T.copy(QPe[bid, hid, :], q_pe_l1[0, :])
                            for k_block in T.serial(max_iters):
                                if k_block * block_n < cache_len:
                                    T.wait_cross_flag(score_free)
                                    if paged:
                                        T.copy(
                                            K[
                                                BlockTable[
                                                    bid, k_block // blocks_per_page
                                                ]
                                                * page_size
                                                + (k_block % blocks_per_page)
                                                * block_n : BlockTable[
                                                    bid, k_block // blocks_per_page
                                                ]
                                                * page_size
                                                + ((k_block % blocks_per_page) + 1)
                                                * block_n,
                                                kv_hid,
                                                :,
                                            ],
                                            k_l1,
                                        )
                                    else:
                                        T.copy(
                                            K[
                                                bid,
                                                k_block * block_n : (k_block + 1)
                                                * block_n,
                                                kv_hid,
                                                :,
                                            ],
                                            k_l1,
                                        )
                                    T.gemm_v0(
                                        q_l1,
                                        k_l1,
                                        score_l0c,
                                        transpose_B=True,
                                        init=True,
                                    )
                                    if is_mla:
                                        if paged:
                                            T.copy(
                                                KPe[
                                                    BlockTable[
                                                        bid, k_block // blocks_per_page
                                                    ]
                                                    * page_size
                                                    + (k_block % blocks_per_page)
                                                    * block_n : BlockTable[
                                                        bid, k_block // blocks_per_page
                                                    ]
                                                    * page_size
                                                    + ((k_block % blocks_per_page) + 1)
                                                    * block_n,
                                                    kv_hid,
                                                    :,
                                                ],
                                                k_pe_l1,
                                            )
                                        else:
                                            T.copy(
                                                KPe[
                                                    bid,
                                                    k_block * block_n : (k_block + 1)
                                                    * block_n,
                                                    kv_hid,
                                                    :,
                                                ],
                                                k_pe_l1,
                                            )
                                        T.gemm_v0(
                                            q_pe_l1,
                                            k_pe_l1,
                                            score_l0c,
                                            transpose_B=True,
                                            init=False,
                                        )
                                    T.copy(score_l0c, ScoreWS[cid, :, :])
                                    T.set_cross_flag("FIX", score_ready)

                                    T.wait_cross_flag(prob_ready)
                                    T.copy(ProbWS[cid, :, :], score_l1)
                                    T.set_cross_flag("MTE2", prob_free)
                                    T.wait_cross_flag(value_free)
                                    if paged:
                                        T.copy(
                                            V[
                                                BlockTable[
                                                    bid, k_block // blocks_per_page
                                                ]
                                                * page_size
                                                + (k_block % blocks_per_page)
                                                * block_n : BlockTable[
                                                    bid, k_block // blocks_per_page
                                                ]
                                                * page_size
                                                + ((k_block % blocks_per_page) + 1)
                                                * block_n,
                                                kv_hid,
                                                :,
                                            ],
                                            v_l1,
                                        )
                                    else:
                                        T.copy(
                                            V[
                                                bid,
                                                k_block * block_n : (k_block + 1)
                                                * block_n,
                                                kv_hid,
                                                :,
                                            ],
                                            v_l1,
                                        )
                                    T.gemm_v0(score_l1, v_l1, value_l0c, init=True)
                                    T.copy(value_l0c, ValueWS[cid, :, :])
                                    T.set_cross_flag("FIX", value_ready)

                with T.Scope("V"):
                    T.set_cross_flag("MTE2", score_free)
                    T.set_cross_flag("MTE2", value_free)
                    for task_round in T.serial(task_rounds):
                        task_id = cid + task_round * launch_blocks
                        if task_id < block_num:
                            bid = task_id // heads
                            hid = task_id % heads
                            cache_len = CacheSeqLens[bid]
                            T.tile.fill(acc_o, 0)
                            T.tile.fill(sum_prev, 0)
                            T.tile.fill(max_cur, -(2.0**30))
                            for k_block in T.serial(max_iters):
                                if k_block * block_n < cache_len:
                                    T.wait_cross_flag(score_ready)
                                    T.tile.fill(scores, 0)
                                    T.copy(max_cur, max_prev)
                                    T.copy(
                                        ScoreWS[
                                            cid, vid * v_rows : (vid + 1) * v_rows, :
                                        ],
                                        scores_other,
                                    )
                                    T.set_cross_flag("MTE2", score_free)
                                    T.tile.add(scores, scores, scores_other)
                                    T.tile.mul(scores, scores, sm_scale)
                                    for i, j in T.Parallel(v_rows, block_n):
                                        scores[i, j] = T.if_then_else(
                                            k_block * block_n + j < cache_len,
                                            scores[i, j],
                                            -T.infinity("float"),
                                        )
                                    T.reduce_max(scores, max_cur, dim=-1)
                                    T.tile.max(max_cur, max_cur, max_prev)
                                    T.tile.sub(max_prev, max_prev, max_cur)
                                    T.tile.exp(max_prev, max_prev)
                                    for i in range(v_rows):
                                        T.tile.sub(
                                            scores[i, :], scores[i, :], max_cur[i]
                                        )
                                    T.tile.exp(scores, scores)
                                    T.reduce_sum(scores, sum_cur, dim=-1)
                                    T.tile.mul(sum_prev, sum_prev, max_prev)
                                    T.tile.add(sum_prev, sum_prev, sum_cur)
                                    T.copy(scores, probs)
                                    T.wait_cross_flag(prob_free)
                                    T.copy(
                                        probs,
                                        ProbWS[
                                            cid, vid * v_rows : (vid + 1) * v_rows, :
                                        ],
                                    )
                                    T.set_cross_flag("MTE3", prob_ready)

                                    T.wait_cross_flag(value_ready)
                                    T.copy(
                                        ValueWS[
                                            cid, vid * v_rows : (vid + 1) * v_rows, :
                                        ],
                                        value,
                                    )
                                    T.set_cross_flag("MTE2", value_free)
                                    for i in range(v_rows):
                                        T.tile.mul(
                                            acc_o[i, :], acc_o[i, :], max_prev[i]
                                        )
                                    T.tile.add(acc_o, acc_o, value)

                            for i in range(v_rows):
                                T.tile.div(acc_o[i, :], acc_o[i, :], sum_prev[i])
                            T.copy(acc_o, tmp)
                            if vid == 0:
                                if q_has_seq:
                                    T.copy(tmp[0, :], Output[bid, 0, hid, :])
                                else:
                                    T.copy(tmp[0, :], Output[bid, hid, :])

        return main

    return _build()


def build_mha_decode_kernel(batch, heads, max_seqlen_kv, dim, dtype):
    """Return a callable with ``(Q, K, V, cache_seqlens)`` ABI."""
    import torch

    if dim != 128:
        raise ValueError(f"decode attention pilot requires dim=128, got {dim}")
    if dtype == torch.float16:
        dtype_name = "float16"
    elif dtype == torch.bfloat16:
        dtype_name = "bfloat16"
    else:
        raise TypeError(f"decode attention supports float16/bfloat16, got {dtype}")
    if batch <= 0 or heads <= 0 or max_seqlen_kv <= 0:
        raise ValueError("decode attention dimensions must be positive")
    raw = _decode_kernel(
        batch, heads, heads, max_seqlen_kv, dim, 0, dtype_name, True, False
    )

    def invoke(q, k, v, cache_seqlens):
        block_table = torch.zeros((batch, 1), dtype=torch.int32, device=q.device)
        return raw(q, q, k, k, v, cache_seqlens, block_table)

    return invoke


def build_gqa_decode_kernel(batch, heads, heads_kv, max_seqlen_kv, dim, dtype):
    """Return a continuous-cache GQA decode callable."""
    import torch

    if dim != 128:
        raise ValueError(f"GQA decode pilot requires dim=128, got {dim}")
    if heads_kv <= 0 or heads % heads_kv:
        raise ValueError(f"heads={heads} must be divisible by heads_kv={heads_kv}")
    from .attention import _require_fp16_gqa

    _require_fp16_gqa(dtype, "GroupedQueryAttentionDecodeWithKVCacheFwdOp")
    dtype_name = "float16"
    raw = _decode_kernel(
        batch, heads, heads_kv, max_seqlen_kv, dim, 0, dtype_name, False, False
    )

    def invoke(q, k, v, cache_seqlens):
        block_table = torch.zeros((batch, 1), dtype=torch.int32, device=q.device)
        return raw(q, q, k, k, v, cache_seqlens, block_table)

    return invoke


def build_mla_decode_kernel(batch, heads, heads_kv, max_seqlen_kv, dim, pe_dim, dtype):
    """Return a compressed-latent MLA decode callable."""
    import torch

    if dim != 128 or pe_dim != 64:
        raise ValueError(
            f"MLA decode pilot requires dim=128, pe_dim=64; got {dim}, {pe_dim}"
        )
    if heads_kv <= 0 or heads % heads_kv:
        raise ValueError(f"heads={heads} must be divisible by heads_kv={heads_kv}")
    if dtype == torch.float16:
        dtype_name = "float16"
    elif dtype == torch.bfloat16:
        dtype_name = "bfloat16"
    else:
        raise TypeError(f"MLA decode supports float16/bfloat16, got {dtype}")
    raw = _decode_kernel(
        batch, heads, heads_kv, max_seqlen_kv, dim, pe_dim, dtype_name, False, True
    )

    def invoke(q, q_pe, k, k_pe, cache_seqlens):
        block_table = torch.zeros((batch, 1), dtype=torch.int32, device=q.device)
        return raw(q, q_pe, k, k_pe, k, cache_seqlens, block_table)

    return invoke


def build_mha_decode_paged_kernel(
    batch, heads, seqlen_q, max_seqlen_kv, dim, page_size, is_causal, dtype
):
    """Return a paged MHA decode callable with runtime block-table lookup."""
    import torch

    if seqlen_q != 1:
        raise ValueError(f"paged decode requires seqlen_q=1, got {seqlen_q}")
    if dim <= 0 or page_size <= 0 or max_seqlen_kv <= 0:
        raise ValueError("paged decode dimensions must be positive")
    if max_seqlen_kv % page_size:
        raise ValueError("max_seqlen_kv must be divisible by page_size")
    if page_size % 64:
        raise ValueError("page_size must be divisible by kernel block_n=64")
    if dtype == torch.float16:
        dtype_name = "float16"
    elif dtype == torch.bfloat16:
        dtype_name = "bfloat16"
    else:
        raise TypeError(f"paged decode supports float16/bfloat16, got {dtype}")
    raw = _decode_kernel(
        batch,
        heads,
        heads,
        max_seqlen_kv,
        dim,
        0,
        dtype_name,
        True,
        False,
        True,
        page_size,
    )

    def invoke(q, k, v, cache_seqlens, block_table):
        return raw(q, q, k, k, v, cache_seqlens, block_table)

    return invoke


def build_gqa_decode_paged_kernel(
    batch, heads, heads_kv, max_seqlen_kv, dim, page_size, dtype
):
    """Return paged GQA decode with runtime block-table lookup."""
    from .attention import _require_fp16_gqa

    if dim != 128:
        raise ValueError(f"paged GQA decode requires dim=128, got {dim}")
    if heads_kv <= 0 or heads % heads_kv:
        raise ValueError(f"heads={heads} must be divisible by heads_kv={heads_kv}")
    if max_seqlen_kv <= 0 or page_size <= 0:
        raise ValueError("paged GQA decode dimensions must be positive")
    if max_seqlen_kv % page_size:
        raise ValueError("max_seqlen_kv must be divisible by page_size")
    if page_size % 64:
        raise ValueError("page_size must be divisible by kernel block_n=64")
    _require_fp16_gqa(dtype, "GroupedQueryAttentionDecodePagedWithKVCacheFwdOp")
    raw = _decode_kernel(
        batch,
        heads,
        heads_kv,
        max_seqlen_kv,
        dim,
        0,
        "float16",
        False,
        False,
        True,
        page_size,
    )

    def invoke(q, k, v, cache_seqlens, block_table):
        return raw(q, q, k, k, v, cache_seqlens, block_table)

    return invoke


@functools.lru_cache(maxsize=16)
def _gqa_prefill_paged_append_kernel(
    batch,
    heads_kv,
    total_q,
    max_seqlen_q,
    physical_tokens,
    max_pages_per_req,
    page_size,
    dim,
    dtype_name,
):
    logical_tasks = batch * max_seqlen_q * heads_kv
    launch_blocks = min((logical_tasks + 1) // 2, NUM_AI_CORES)
    tasks_per_round = launch_blocks * 2
    task_rounds = (logical_tasks + tasks_per_round - 1) // tasks_per_round

    @tilelang.jit(out_idx=[], pass_configs=PASS_CONFIGS, compile_flags=["-O3"])
    def _build():
        @T.prim_func
        def main(
            KNew: T.Tensor([total_q, heads_kv, dim], dtype_name),
            VNew: T.Tensor([total_q, heads_kv, dim], dtype_name),
            CuSeqLensQ: T.Tensor([batch + 1], "int32"),
            CacheSeqLens: T.Tensor([batch], "int32"),
            BlockTable: T.Tensor([batch, max_pages_per_req], "int32"),
            KPages: T.Tensor([physical_tokens, heads_kv, dim], dtype_name),
            VPages: T.Tensor([physical_tokens, heads_kv, dim], dtype_name),
        ):
            with T.Kernel(launch_blocks, is_npu=True) as (cid, vid):
                k_ub = T.alloc_ub([1, dim], dtype_name)
                v_ub = T.alloc_ub([1, dim], dtype_name)
                with T.Scope("V"):
                    for task_round in T.serial(task_rounds):
                        task_id = cid * 2 + vid + task_round * tasks_per_round
                        if task_id < logical_tasks:
                            kv_hid = task_id % heads_kv
                            batch_pos = task_id // heads_kv
                            q_pos = batch_pos % max_seqlen_q
                            bid = batch_pos // max_seqlen_q
                            q_start = CuSeqLensQ[bid]
                            q_len = CuSeqLensQ[bid + 1] - q_start
                            if q_pos < q_len:
                                logical_pos = CacheSeqLens[bid] + q_pos
                                physical_pos = (
                                    BlockTable[bid, logical_pos // page_size] * page_size
                                    + logical_pos % page_size
                                )
                                T.copy(KNew[q_start + q_pos, kv_hid, :], k_ub[0, :])
                                T.copy(VNew[q_start + q_pos, kv_hid, :], v_ub[0, :])
                                T.copy(k_ub[0, :], KPages[physical_pos, kv_hid, :])
                                T.copy(v_ub[0, :], VPages[physical_pos, kv_hid, :])

        return main

    return _build()


@functools.lru_cache(maxsize=16)
def _gqa_prefill_paged_attention_kernel(
    batch,
    heads,
    heads_kv,
    total_q,
    max_seqlen_q,
    physical_tokens,
    max_pages_per_req,
    page_size,
    dim,
    dtype_name,
    is_causal,
    sm_scale,
):
    logical_tasks = batch * max_seqlen_q * heads
    launch_blocks = min((logical_tasks + 1) // 2, NUM_AI_CORES)
    tasks_per_round = launch_blocks * 2
    task_rounds = (logical_tasks + tasks_per_round - 1) // tasks_per_round
    max_cache_len = max_pages_per_req * page_size

    @tilelang.jit(
        out_idx=[6],
        pass_configs=PASS_CONFIGS,
        compile_flags=["-O3"],
    )
    def _build():
        @T.prim_func
        def main(
            Q: T.Tensor([total_q, heads, dim], dtype_name),
            KPages: T.Tensor([physical_tokens, heads_kv, dim], dtype_name),
            VPages: T.Tensor([physical_tokens, heads_kv, dim], dtype_name),
            CuSeqLensQ: T.Tensor([batch + 1], "int32"),
            CacheSeqLens: T.Tensor([batch], "int32"),
            BlockTable: T.Tensor([batch, max_pages_per_req], "int32"),
            Output: T.Tensor([total_q, heads, dim], dtype_name),
        ):
            with T.Kernel(launch_blocks, is_npu=True) as (cid, vid):
                q_local = T.alloc_ub([1, dim], dtype_name)
                k_local = T.alloc_ub([1, dim], dtype_name)
                v_local = T.alloc_ub([1, dim], dtype_name)
                q_float = T.alloc_ub([1, dim], "float")
                k_float = T.alloc_ub([1, dim], "float")
                v_float = T.alloc_ub([1, dim], "float")
                weighted_v = T.alloc_ub([1, dim], "float")
                product = T.alloc_ub([8, dim], "float")
                acc = T.alloc_ub([1, dim], "float")
                out_local = T.alloc_ub([1, dim], dtype_name)
                score = T.alloc_ub([8], "float")
                max_prev = T.alloc_ub([8], "float")
                max_cur = T.alloc_ub([8], "float")
                sum_prev = T.alloc_ub([8], "float")

                with T.Scope("V"):
                    for task_round in T.serial(task_rounds):
                        task_id = cid * 2 + vid + task_round * tasks_per_round
                        if task_id < logical_tasks:
                            hid = task_id % heads
                            batch_pos = task_id // heads
                            q_pos = batch_pos % max_seqlen_q
                            bid = batch_pos // max_seqlen_q
                            q_start = CuSeqLensQ[bid]
                            q_len = CuSeqLensQ[bid + 1] - q_start
                            if q_pos < q_len:
                                q_idx = q_start + q_pos
                                kv_hid = hid // (heads // heads_kv)
                                attended_len = CacheSeqLens[bid] + (
                                    q_pos + 1 if is_causal else q_len
                                )
                                T.copy(Q[q_idx, hid, :], q_local[0, :])
                                T.copy(q_local, q_float)
                                T.tile.fill(acc, 0)
                                T.tile.fill(sum_prev, 0)
                                T.tile.fill(max_cur, -T.infinity("float"))

                                for k_pos in T.serial(max_cache_len):
                                    if k_pos < attended_len:
                                        physical_pos = (
                                            BlockTable[bid, k_pos // page_size] * page_size
                                            + k_pos % page_size
                                        )
                                        T.copy(
                                            KPages[physical_pos, kv_hid, :],
                                            k_local[0, :],
                                        )
                                        T.copy(k_local, k_float)
                                        T.tile.fill(product, 0)
                                        T.tile.mul(
                                            product[0, :], q_float[0, :], k_float[0, :]
                                        )
                                        T.reduce_sum(product, score, dim=-1)
                                        T.tile.mul(score, score, sm_scale)
                                        T.copy(max_cur, max_prev)
                                        T.copy(score, max_cur)
                                        T.tile.max(max_cur, max_cur, max_prev)
                                        T.tile.sub(max_prev, max_prev, max_cur)
                                        T.tile.exp(max_prev, max_prev)
                                        T.tile.sub(score, score, max_cur)
                                        T.tile.exp(score, score)
                                        T.tile.mul(sum_prev, sum_prev, max_prev)
                                        T.tile.add(sum_prev, sum_prev, score)
                                        physical_pos = (
                                            BlockTable[bid, k_pos // page_size] * page_size
                                            + k_pos % page_size
                                        )
                                        T.copy(
                                            VPages[physical_pos, kv_hid, :],
                                            v_local[0, :],
                                        )
                                        T.copy(v_local, v_float)
                                        T.tile.mul(acc, acc, max_prev[0])
                                        T.tile.mul(weighted_v, v_float, score[0])
                                        T.tile.add(acc, acc, weighted_v)

                                T.tile.div(acc, acc, sum_prev[0])
                                T.copy(acc, out_local)
                                T.copy(out_local[0, :], Output[q_idx, hid, :])

        return main

    return _build()


def build_gqa_prefill_paged_kernel(
    batch,
    heads,
    heads_kv,
    total_q,
    max_seqlen_q,
    physical_tokens,
    max_pages_per_req,
    page_size,
    dim,
    is_causal,
    dtype,
    sm_scale=None,
):
    """Append packed KV then run paged GQA prefill without host-side gathering."""
    from .attention import _require_fp16_gqa

    if heads_kv <= 0 or heads % heads_kv:
        raise ValueError(f"heads={heads} must be divisible by heads_kv={heads_kv}")
    if any(value <= 0 for value in (batch, total_q, max_seqlen_q, page_size, dim)):
        raise ValueError("paged GQA prefill dimensions must be positive")
    if physical_tokens % page_size:
        raise ValueError("physical_tokens must be divisible by page_size")
    _require_fp16_gqa(dtype, "GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp")
    scale = dim**-0.5 if sm_scale is None else float(sm_scale)
    append = _gqa_prefill_paged_append_kernel(
        batch,
        heads_kv,
        total_q,
        max_seqlen_q,
        physical_tokens,
        max_pages_per_req,
        page_size,
        dim,
        "float16",
    )
    attention = _gqa_prefill_paged_attention_kernel(
        batch,
        heads,
        heads_kv,
        total_q,
        max_seqlen_q,
        physical_tokens,
        max_pages_per_req,
        page_size,
        dim,
        "float16",
        bool(is_causal),
        scale,
    )

    def invoke(q, k_new, v_new, k_pages, v_pages, cu_seqlens_q, cache_seqlens, block_table):
        append(k_new, v_new, cu_seqlens_q, cache_seqlens, block_table, k_pages, v_pages)
        return attention(q, k_pages, v_pages, cu_seqlens_q, cache_seqlens, block_table)

    return invoke


__all__ = [
    "build_gqa_decode_kernel",
    "build_gqa_decode_paged_kernel",
    "build_gqa_prefill_paged_kernel",
    "build_mha_decode_kernel",
    "build_mha_decode_paged_kernel",
    "build_mla_decode_kernel",
]
