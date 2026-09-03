"""Pilot dense MHA backward kernel for Ascend.

The implementation deliberately keeps the backward dataflow visible: each
Q-tile recomputes ``S = Q K^T`` and the softmax probabilities, then emits dQ,
dK and dV.  dK/dV are shared by Q tiles, so their fp32 GM destinations are
updated with the Ascend ``tile.atomic_add`` primitive.

This file has no postponed-annotation import.  TileLang needs concrete dtype
objects while parsing the ``T.Tensor`` annotations.
"""

from functools import lru_cache
import math

import tilelang
from tilelang import language as T
import torch

from .common import grid_repeat_count, launch_block_count


_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


def _dtype_name(dtype):
    if dtype == torch.float16:
        return "float16"
    if dtype == torch.bfloat16:
        raise TypeError(
            "MultiHeadAttentionBwdOp requires dtype=torch.float16; received torch.bfloat16. "
            "bf16 is explicitly unsupported by the current attention Cube workspace ABI."
        )
    raise TypeError(f"MultiHeadAttentionBwdOp requires dtype=torch.float16; received {dtype}")


@lru_cache(maxsize=32)
def _compile_single_fused_probe(batch, seq_len, heads, dim, is_causal, dtype_name):
    # 32x64 keeps the three fp32 L0C accumulators within the 910B1 budget.
    block_m = 32
    block_n = 64
    q_tiles = math.ceil(seq_len / block_m)
    k_tiles = math.ceil(seq_len / block_n)
    logical_blocks = batch * heads * q_tiles
    launch_blocks = launch_block_count(logical_blocks)
    repeats = grid_repeat_count(logical_blocks, launch_blocks)
    accum = "float"
    scale = (1.0 / dim) ** 0.5

    @tilelang.jit(out_idx=[], pass_configs=_PASS_CONFIGS, compile_flags=["-O3", "-DENABLE_BF16"])
    def factory():
        q_shape = (batch * seq_len, heads, dim)
        lse_shape = (batch, heads, seq_len)

        @T.prim_func
        def main(
            Q: T.Tensor(q_shape, dtype_name),
            K: T.Tensor(q_shape, dtype_name),
            V: T.Tensor(q_shape, dtype_name),
            Out: T.Tensor(q_shape, dtype_name),
            dO: T.Tensor(q_shape, dtype_name),
            LSE: T.Tensor(lse_shape, accum),
            dQ: T.Tensor(q_shape, accum),
            dK: T.Tensor(q_shape, accum),
            dV: T.Tensor(q_shape, accum),
        ):
            # A one-dimensional legal blockDim plus a grid-stride loop covers
            # workloads larger than CANN's 65535 block limit.
            with T.Kernel(launch_blocks, is_npu=True) as (cid, vid):
                q_l1 = T.alloc_shared((block_m, dim), dtype_name)
                k_l1 = T.alloc_shared((block_n, dim), dtype_name)
                v_l1 = T.alloc_shared((block_n, dim), dtype_name)
                do_l1 = T.alloc_shared((block_m, dim), dtype_name)
                o_l1 = T.alloc_shared((block_m, dim), dtype_name)
                p_l1 = T.alloc_shared((block_m, block_n), dtype_name)
                ds_l1 = T.alloc_shared((block_m, block_n), dtype_name)

                score_l0 = T.alloc_fragment((block_m, block_n), accum)
                dp_l0 = T.alloc_fragment((block_m, block_n), accum)
                dq_l0 = T.alloc_fragment((block_m, dim), accum)
                dk_l0 = T.alloc_fragment((block_n, dim), accum)
                dv_l0 = T.alloc_fragment((block_n, dim), accum)

                # ``alloc_shared`` is intentionally used for the CV-fused
                # scratch.  Auto CV combine assigns these shared buffers to
                # UB while avoiding duplicate declarations across C/V views.
                # The three fp32 matrix buffers have disjoint phase lifetimes:
                # score becomes P in-place, dP becomes dS in-place, and the
                # initial O*dO product becomes the row-broadcast buffer.
                score_prob = T.alloc_shared((block_m, block_n), accum)
                dp_ds = T.alloc_shared((block_m, block_n), accum)
                delta = T.alloc_shared((block_m,), accum)
                lse_ub = T.alloc_shared((block_m,), accum)
                product_bcast = T.alloc_shared((block_m, block_n), accum)
                o_ub = T.alloc_shared((block_m, dim), accum)
                grad_ub = T.alloc_shared((block_m, dim), accum)
                p_half = T.alloc_shared((block_m, block_n), dtype_name)
                ds_half = T.alloc_shared((block_m, block_n), dtype_name)

                for rep in T.serial(repeats):
                    task = cid + rep * launch_blocks
                    if task < logical_blocks:
                        qb = task % q_tiles
                        h = (task // q_tiles) % heads
                        b = task // (q_tiles * heads)
                        q0 = qb * block_m
                        q_rows = T.min(block_m, seq_len - q0)
                        qbase = b * seq_len

                        # Tail-safe loads.  Full tiles use bulk copies; only
                        # the boundary tile falls back to guarded scalar UB
                        # initialization, preserving the generic shape domain.
                        if q_rows == block_m:
                            T.copy(Q[qbase + q0 : qbase + q0 + block_m, h, :], q_l1)
                            T.copy(Out[qbase + q0 : qbase + q0 + block_m, h, :], o_l1)
                            T.copy(dO[qbase + q0 : qbase + q0 + block_m, h, :], do_l1)
                        else:
                            T.tile.fill(q_l1, 0)
                            T.tile.fill(o_l1, 0)
                            T.tile.fill(do_l1, 0)
                            for i in T.serial(block_m):
                                if i < q_rows:
                                    for j in T.serial(dim):
                                        q_l1[i, j] = Q[qbase + q0 + i, h, j]
                                        o_l1[i, j] = Out[qbase + q0 + i, h, j]
                                        do_l1[i, j] = dO[qbase + q0 + i, h, j]

                        T.copy(o_l1, o_ub)
                        T.copy(do_l1, grad_ub)
                        T.tile.mul(product_bcast[:, :dim], o_ub, grad_ub)
                        T.reduce_sum(product_bcast[:, :dim], delta, dim=-1)
                        if q_rows != block_m:
                            for i in T.serial(block_m):
                                if i >= q_rows:
                                    delta[i] = 0.0

                        if q_rows == block_m:
                            T.copy(LSE[b, h, q0 : q0 + block_m], lse_ub)
                        else:
                            T.tile.fill(lse_ub, 0)
                            for i in T.serial(block_m):
                                if i < q_rows:
                                    lse_ub[i] = LSE[b, h, q0 + i]

                        for kt in T.serial(k_tiles):
                            k0 = kt * block_n
                            k_rows = T.min(block_n, seq_len - k0)
                            if k_rows == block_n:
                                T.copy(K[qbase + k0 : qbase + k0 + block_n, h, :], k_l1)
                                T.copy(V[qbase + k0 : qbase + k0 + block_n, h, :], v_l1)
                            else:
                                T.tile.fill(k_l1, 0)
                                T.tile.fill(v_l1, 0)
                                for i in T.serial(block_n):
                                    if i < k_rows:
                                        for j in T.serial(dim):
                                            k_l1[i, j] = K[qbase + k0 + i, h, j]
                                            v_l1[i, j] = V[qbase + k0 + i, h, j]

                            # Recompute S and P from Q/K and the saved LSE.
                            T.gemm_v0(q_l1, k_l1, score_l0, transpose_B=True, init=True)
                            T.copy(score_l0, score_prob)
                            for i, j in T.Parallel(block_m, block_n):
                                valid = (i < q_rows) & (j < k_rows)
                                visible = (not is_causal) | (k0 + j <= q0 + i)
                                score_prob[i, j] = T.if_then_else(
                                    valid & visible,
                                    score_prob[i, j] * scale - lse_ub[i],
                                    -1.0e30,
                                )
                            T.tile.exp(score_prob, score_prob)
                            if q_rows != block_m or k_rows != block_n:
                                for i, j in T.Parallel(block_m, block_n):
                                    if (i >= q_rows) | (j >= k_rows):
                                        score_prob[i, j] = 0.0

                            # dP = dO @ V^T, followed by dS = P*(dP-delta)*scale.
                            T.gemm_v0(do_l1, v_l1, dp_l0, transpose_B=True, init=True)
                            T.copy(dp_l0, dp_ds)
                            T.tile.broadcast(product_bcast, delta, axis=1)
                            T.tile.sub(dp_ds, dp_ds, product_bcast)
                            T.tile.mul(dp_ds, score_prob, dp_ds)
                            T.tile.mul(dp_ds, dp_ds, scale)
                            T.copy(score_prob, p_half)
                            T.copy(dp_ds, ds_half)
                            T.copy(p_half, p_l1)
                            T.copy(ds_half, ds_l1)

                            # Each Q tile owns dQ, while dK/dV overlap across
                            # Q tiles.  All three destinations are fp32 and
                            # initialized by the builder before launch.
                            T.gemm_v0(p_l1, do_l1, dv_l0, transpose_A=True, init=True)
                            T.tile.atomic_add(dV[qbase + k0 : qbase + k0 + block_n, h, :], dv_l0)
                            T.gemm_v0(ds_l1, q_l1, dk_l0, transpose_A=True, init=True)
                            T.tile.atomic_add(dK[qbase + k0 : qbase + k0 + block_n, h, :], dk_l0)
                            T.gemm_v0(ds_l1, k_l1, dq_l0, init=True)
                            T.tile.atomic_add(dQ[qbase + q0 : qbase + q0 + block_m, h, :], dq_l0)

        return main

    return factory()


@lru_cache(maxsize=32)
def _compile_dv(batch, seq_len, heads, heads_kv, dim, is_causal, dtype_name):
    """Compile the first split launch: recompute P and accumulate dV."""
    block_m = 32
    block_n = 64
    q_tiles = math.ceil(seq_len / block_m)
    k_tiles = math.ceil(seq_len / block_n)
    logical_tasks = batch * heads * q_tiles
    cores = min(24, logical_tasks)
    tasks_per_core = logical_tasks // cores
    extra_tasks = logical_tasks % cores
    k_padded = k_tiles * block_n
    accum = "float"
    scale = (1.0 / dim) ** 0.5

    # These are block-local AIC/AIV semaphores, not cross-block barriers.
    score_ready = 0  # Cube -> Vector
    score_free = 1  # Vector -> Cube
    prob_ready = 2  # Vector -> Cube
    prob_free = 3  # Cube -> Vector

    @tilelang.jit(out_idx=[], pass_configs=_PASS_CONFIGS, compile_flags=["-O3", "-DENABLE_BF16"])
    def factory():
        q_shape = (batch * seq_len, heads, dim)
        kv_shape = (batch * seq_len, heads_kv, dim)
        lse_shape = (batch, heads, seq_len)
        dv_shape = (batch, heads_kv, k_padded, dim)
        score_shape = (cores, block_m, block_n)

        @T.prim_func
        def main(
            Q: T.Tensor(q_shape, dtype_name),
            K: T.Tensor(kv_shape, dtype_name),
            dO: T.Tensor(q_shape, dtype_name),
            LSE: T.Tensor(lse_shape, accum),
            dV: T.Tensor(dv_shape, accum),
            score_ws: T.Tensor(score_shape, accum),
            prob_ws: T.Tensor(score_shape, dtype_name),
        ):
            with T.Kernel(cores, is_npu=True) as (cid, vid):
                q_l1 = T.alloc_L1((block_m, dim), dtype_name)
                k_l1 = T.alloc_L1((block_n, dim), dtype_name)
                do_l1 = T.alloc_L1((block_m, dim), dtype_name)
                p_l1 = T.alloc_L1((block_m, block_n), dtype_name)
                score_l0 = T.alloc_L0C((block_m, block_n), accum)
                dv_l0 = T.alloc_L0C((block_n, dim), accum)

                # Allocation extents must remain Python constants.  ``half_m``
                # below is a TIR value used only in slice/index expressions.
                score_ub = T.alloc_ub((block_m // 2, block_n), accum)
                prob_ub = T.alloc_ub((block_m // 2, block_n), accum)
                lse_ub = T.alloc_ub((block_m // 2,), accum)
                prob_half = T.alloc_ub((block_m // 2, block_n), dtype_name)
                half_m = block_m // 2

                my_start = cid * tasks_per_core + T.if_then_else(cid < extra_tasks, cid, extra_tasks)
                my_count = tasks_per_core + T.if_then_else(cid < extra_tasks, 1, 0)

                with T.Scope("C"):
                    T.set_cross_flag("MTE2", prob_free)
                    for task_offset in T.serial(my_count):
                        task = my_start + task_offset
                        qb = task % q_tiles
                        h = (task // q_tiles) % heads
                        kv_head = h // (heads // heads_kv)
                        b = task // (q_tiles * heads)
                        q0 = qb * block_m
                        base = b * seq_len
                        T.copy(Q[base + q0 : base + q0 + block_m, h, :], q_l1)
                        T.copy(dO[base + q0 : base + q0 + block_m, h, :], do_l1)

                        for kt in T.serial(k_tiles):
                            k0 = kt * block_n
                            T.wait_cross_flag(score_free)
                            T.copy(K[base + k0 : base + k0 + block_n, kv_head, :], k_l1)
                            T.gemm_v0(q_l1, k_l1, score_l0, transpose_B=True, init=True)
                            T.copy(score_l0, score_ws[cid, :, :])
                            T.set_cross_flag("FIX", score_ready)

                            T.wait_cross_flag(prob_ready)
                            T.copy(prob_ws[cid, :, :], p_l1)
                            T.gemm_v0(p_l1, do_l1, dv_l0, transpose_A=True, init=True)
                            T.tile.atomic_add(dV[b, kv_head, k0 : k0 + block_n, :], dv_l0)
                            T.set_cross_flag("MTE2", prob_free)

                    T.wait_cross_flag(score_free)

                with T.Scope("V"):
                    T.set_cross_flag("MTE2", score_free)
                    for task_offset in T.serial(my_count):
                        task = my_start + task_offset
                        qb = task % q_tiles
                        h = (task // q_tiles) % heads
                        b = task // (q_tiles * heads)
                        q0 = qb * block_m
                        q_rows = T.min(block_m, seq_len - q0)
                        T.copy(
                            LSE[b, h, q0 + vid * half_m : q0 + vid * half_m + half_m],
                            lse_ub,
                        )

                        for kt in T.serial(k_tiles):
                            k0 = kt * block_n
                            k_rows = T.min(block_n, seq_len - k0)
                            T.wait_cross_flag(score_ready)
                            T.copy(
                                score_ws[cid, vid * half_m : vid * half_m + half_m, :],
                                score_ub,
                            )
                            T.set_cross_flag("MTE2", score_free)

                            for i, j in T.Parallel(half_m, block_n):
                                row = vid * half_m + i
                                valid = (row < q_rows) & (j < k_rows)
                                visible = (not is_causal) | (k0 + j <= q0 + row)
                                score_ub[i, j] = T.if_then_else(
                                    valid & visible,
                                    score_ub[i, j] * scale - lse_ub[i],
                                    -1.0e30,
                                )
                            T.tile.exp(prob_ub, score_ub)
                            for i, j in T.Parallel(half_m, block_n):
                                if (vid * half_m + i >= q_rows) | (j >= k_rows):
                                    prob_ub[i, j] = 0.0
                            T.tile.cast(prob_half, prob_ub, "CAST_RINT", half_m * block_n)

                            T.wait_cross_flag(prob_free)
                            T.copy(
                                prob_half,
                                prob_ws[cid, vid * half_m : vid * half_m + half_m, :],
                            )
                            T.set_cross_flag("MTE3", prob_ready)

                    T.wait_cross_flag(prob_free)

        return main

    return factory()


@lru_cache(maxsize=32)
def _compile_dq_dk(batch, seq_len, heads, heads_kv, dim, is_causal, dtype_name):
    """Compile the second split launch: recompute dS, then accumulate dQ/dK."""
    block_m = 32
    block_n = 64
    q_tiles = math.ceil(seq_len / block_m)
    k_tiles = math.ceil(seq_len / block_n)
    logical_tasks = batch * heads * q_tiles
    cores = min(24, logical_tasks)
    tasks_per_core = logical_tasks // cores
    extra_tasks = logical_tasks % cores
    q_padded = q_tiles * block_m
    k_padded = k_tiles * block_n
    accum = "float"
    scale = (1.0 / dim) ** 0.5

    raw_ready = 0  # Cube -> Vector (score and dP are both ready)
    raw_free = 1  # Vector -> Cube
    ds_ready = 2  # Vector -> Cube
    ds_free = 3  # Cube -> Vector

    @tilelang.jit(out_idx=[], pass_configs=_PASS_CONFIGS, compile_flags=["-O3", "-DENABLE_BF16"])
    def factory():
        q_shape = (batch * seq_len, heads, dim)
        kv_shape = (batch * seq_len, heads_kv, dim)
        lse_shape = (batch, heads, seq_len)
        dq_shape = (batch, heads, q_padded, dim)
        dk_shape = (batch, heads_kv, k_padded, dim)
        matrix_shape = (cores, block_m, block_n)

        @T.prim_func
        def main(
            Q: T.Tensor(q_shape, dtype_name),
            K: T.Tensor(kv_shape, dtype_name),
            V: T.Tensor(kv_shape, dtype_name),
            Out: T.Tensor(q_shape, dtype_name),
            dO: T.Tensor(q_shape, dtype_name),
            LSE: T.Tensor(lse_shape, accum),
            dQ: T.Tensor(dq_shape, accum),
            dK: T.Tensor(dk_shape, accum),
            score_ws: T.Tensor(matrix_shape, accum),
            dp_ws: T.Tensor(matrix_shape, accum),
            ds_ws: T.Tensor(matrix_shape, dtype_name),
        ):
            with T.Kernel(cores, is_npu=True) as (cid, vid):
                q_l1 = T.alloc_L1((block_m, dim), dtype_name)
                k_l1 = T.alloc_L1((block_n, dim), dtype_name)
                v_l1 = T.alloc_L1((block_n, dim), dtype_name)
                do_l1 = T.alloc_L1((block_m, dim), dtype_name)
                ds_l1 = T.alloc_L1((block_m, block_n), dtype_name)
                score_l0 = T.alloc_L0C((block_m, block_n), accum)
                dp_l0 = T.alloc_L0C((block_m, block_n), accum)
                dq_l0 = T.alloc_L0C((block_m, dim), accum)
                dk_l0 = T.alloc_L0C((block_n, dim), accum)

                score_ub = T.alloc_ub((block_m // 2, block_n), accum)
                dp_ub = T.alloc_ub((block_m // 2, block_n), accum)
                prob_ub = T.alloc_ub((block_m // 2, block_n), accum)
                delta_bcast = T.alloc_ub((block_m // 2, block_n), accum)
                ds_ub = T.alloc_ub((block_m // 2, block_n), accum)
                o_half = T.alloc_ub((block_m // 2, dim), dtype_name)
                do_half = T.alloc_ub((block_m // 2, dim), dtype_name)
                o_ub = T.alloc_ub((block_m // 2, dim), accum)
                grad_ub = T.alloc_ub((block_m // 2, dim), accum)
                product_ub = T.alloc_ub((block_m // 2, dim), accum)
                delta = T.alloc_ub((block_m // 2,), accum)
                lse_ub = T.alloc_ub((block_m // 2,), accum)
                ds_half = T.alloc_ub((block_m // 2, block_n), dtype_name)
                half_m = block_m // 2

                my_start = cid * tasks_per_core + T.if_then_else(cid < extra_tasks, cid, extra_tasks)
                my_count = tasks_per_core + T.if_then_else(cid < extra_tasks, 1, 0)

                with T.Scope("C"):
                    T.set_cross_flag("MTE2", ds_free)
                    for task_offset in T.serial(my_count):
                        task = my_start + task_offset
                        qb = task % q_tiles
                        h = (task // q_tiles) % heads
                        kv_head = h // (heads // heads_kv)
                        b = task // (q_tiles * heads)
                        q0 = qb * block_m
                        base = b * seq_len
                        T.copy(Q[base + q0 : base + q0 + block_m, h, :], q_l1)
                        T.copy(dO[base + q0 : base + q0 + block_m, h, :], do_l1)

                        for kt in T.serial(k_tiles):
                            k0 = kt * block_n
                            T.wait_cross_flag(raw_free)
                            T.copy(K[base + k0 : base + k0 + block_n, kv_head, :], k_l1)
                            T.copy(V[base + k0 : base + k0 + block_n, kv_head, :], v_l1)
                            T.gemm_v0(q_l1, k_l1, score_l0, transpose_B=True, init=True)
                            T.copy(score_l0, score_ws[cid, :, :])
                            T.gemm_v0(do_l1, v_l1, dp_l0, transpose_B=True, init=True)
                            T.copy(dp_l0, dp_ws[cid, :, :])
                            T.set_cross_flag("FIX", raw_ready)

                            T.wait_cross_flag(ds_ready)
                            T.copy(ds_ws[cid, :, :], ds_l1)
                            T.gemm_v0(ds_l1, q_l1, dk_l0, transpose_A=True, init=True)
                            T.tile.atomic_add(dK[b, kv_head, k0 : k0 + block_n, :], dk_l0)
                            T.gemm_v0(ds_l1, k_l1, dq_l0, init=True)
                            T.tile.atomic_add(dQ[b, h, q0 : q0 + block_m, :], dq_l0)
                            T.set_cross_flag("MTE2", ds_free)

                    T.wait_cross_flag(raw_free)

                with T.Scope("V"):
                    T.set_cross_flag("MTE2", raw_free)
                    for task_offset in T.serial(my_count):
                        task = my_start + task_offset
                        qb = task % q_tiles
                        h = (task // q_tiles) % heads
                        b = task // (q_tiles * heads)
                        q0 = qb * block_m
                        q_rows = T.min(block_m, seq_len - q0)
                        base = b * seq_len

                        T.copy(
                            Out[
                                base + q0 + vid * half_m : base + q0 + vid * half_m + half_m,
                                h,
                                :,
                            ],
                            o_half,
                        )
                        T.copy(
                            dO[base + q0 + vid * half_m : base + q0 + vid * half_m + half_m, h, :],
                            do_half,
                        )
                        T.copy(o_half, o_ub)
                        T.copy(do_half, grad_ub)
                        T.tile.mul(product_ub, o_ub, grad_ub)
                        T.reduce_sum(product_ub, delta, dim=-1)
                        T.copy(
                            LSE[b, h, q0 + vid * half_m : q0 + vid * half_m + half_m],
                            lse_ub,
                        )

                        for kt in T.serial(k_tiles):
                            k0 = kt * block_n
                            k_rows = T.min(block_n, seq_len - k0)
                            T.wait_cross_flag(raw_ready)
                            T.copy(
                                score_ws[cid, vid * half_m : vid * half_m + half_m, :],
                                score_ub,
                            )
                            T.copy(
                                dp_ws[cid, vid * half_m : vid * half_m + half_m, :],
                                dp_ub,
                            )
                            T.set_cross_flag("MTE2", raw_free)

                            for i, j in T.Parallel(half_m, block_n):
                                row = vid * half_m + i
                                valid = (row < q_rows) & (j < k_rows)
                                visible = (not is_causal) | (k0 + j <= q0 + row)
                                score_ub[i, j] = T.if_then_else(
                                    valid & visible,
                                    score_ub[i, j] * scale - lse_ub[i],
                                    -1.0e30,
                                )
                            T.tile.exp(prob_ub, score_ub)
                            T.tile.broadcast(delta_bcast, delta, axis=1)
                            T.tile.sub(dp_ub, dp_ub, delta_bcast)
                            T.tile.mul(ds_ub, prob_ub, dp_ub)
                            T.tile.mul(ds_ub, ds_ub, scale)
                            for i, j in T.Parallel(half_m, block_n):
                                if (vid * half_m + i >= q_rows) | (j >= k_rows):
                                    ds_ub[i, j] = 0.0
                            T.tile.cast(ds_half, ds_ub, "CAST_RINT", half_m * block_n)

                            T.wait_cross_flag(ds_free)
                            T.copy(
                                ds_half,
                                ds_ws[cid, vid * half_m : vid * half_m + half_m, :],
                            )
                            T.set_cross_flag("MTE3", ds_ready)

                    T.wait_cross_flag(ds_free)

        return main

    return factory()


@lru_cache(maxsize=32)
def _compile_cast(batch, seq_len, heads, dim, dtype_name):
    total = batch * seq_len * heads * dim
    tile = 1024
    blocks = launch_block_count(math.ceil(total / tile))
    repeats = grid_repeat_count(math.ceil(total / tile), blocks)

    @tilelang.jit(out_idx=[], pass_configs={
        tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
        tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    })
    def factory():
        shape = (total,)

        @T.prim_func
        def main(src: T.Tensor(shape, "float"), dst: T.Tensor(shape, dtype_name)):
            with T.Kernel(blocks, is_npu=True) as (cid, vid):
                ub = T.alloc_shared((tile,), "float")
                out = T.alloc_shared((tile,), dtype_name)
                for rep in T.serial(repeats):
                    start = (cid + rep * blocks) * tile + vid * (tile // 2)
                    if start < total:
                        valid = T.min(tile // 2, total - start)
                        T.copy(src[start : start + tile // 2], ub)
                        T.tile.cast(out, ub, "CAST_RINT", tile // 2)
                        T.copy(out[:valid], dst[start : start + valid])

        return main

    return factory()


@lru_cache(maxsize=32)
def _compile_output_cast(
    batch, seq_len, heads, heads_kv, dim, q_padded, k_padded, dtype_name
):
    """Cast and unpad all three explicit outputs in one vector launch."""
    q_total = batch * seq_len * heads * dim
    kv_total = batch * seq_len * heads_kv * dim
    tile = 1024
    logical_blocks = math.ceil(q_total / tile)
    blocks = launch_block_count(logical_blocks)
    repeats = grid_repeat_count(logical_blocks, blocks)
    src_q_shape = (batch, heads, q_padded, dim)
    src_k_shape = (batch, heads_kv, k_padded, dim)
    q_dst_shape = (q_total,)
    kv_dst_shape = (kv_total,)

    @tilelang.jit(
        out_idx=[],
        pass_configs={
            tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: False,
            tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
            tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
        },
    )
    def factory():
        @T.prim_func
        def main(
            dQ32: T.Tensor(src_q_shape, "float"),
            dK32: T.Tensor(src_k_shape, "float"),
            dV32: T.Tensor(src_k_shape, "float"),
            dQ: T.Tensor(q_dst_shape, dtype_name),
            dK: T.Tensor(kv_dst_shape, dtype_name),
            dV: T.Tensor(kv_dst_shape, dtype_name),
        ):
            with T.Kernel(blocks, is_npu=True) as (cid, vid):
                q_ub = T.alloc_ub((tile // 2,), "float")
                k_ub = T.alloc_ub((tile // 2,), "float")
                v_ub = T.alloc_ub((tile // 2,), "float")
                q_out = T.alloc_ub((tile // 2,), dtype_name)
                k_out = T.alloc_ub((tile // 2,), dtype_name)
                v_out = T.alloc_ub((tile // 2,), dtype_name)

                with T.Scope("V"):
                    for rep in T.serial(repeats):
                        start = (cid + rep * blocks) * tile + vid * (tile // 2)
                        if start < q_total:
                            q_valid = T.min(tile // 2, q_total - start)
                            kv_valid = T.min(tile // 2, kv_total - start)
                            for i in T.serial(tile // 2):
                                if i < q_valid:
                                    flat = start + i
                                    d = flat % dim
                                    h = (flat // dim) % heads
                                    s = (flat // (dim * heads)) % seq_len
                                    b = flat // (dim * heads * seq_len)
                                    q_ub[i] = dQ32[b, h, s, d]
                                else:
                                    q_ub[i] = 0.0
                                if i < kv_valid:
                                    flat_kv = start + i
                                    d_kv = flat_kv % dim
                                    h_kv = (flat_kv // dim) % heads_kv
                                    s_kv = (flat_kv // (dim * heads_kv)) % seq_len
                                    b_kv = flat_kv // (dim * heads_kv * seq_len)
                                    k_ub[i] = dK32[b_kv, h_kv, s_kv, d_kv]
                                    v_ub[i] = dV32[b_kv, h_kv, s_kv, d_kv]
                                else:
                                    k_ub[i] = 0.0
                                    v_ub[i] = 0.0
                            T.tile.cast(q_out, q_ub, "CAST_RINT", tile // 2)
                            T.tile.cast(k_out, k_ub, "CAST_RINT", tile // 2)
                            T.tile.cast(v_out, v_ub, "CAST_RINT", tile // 2)
                            T.copy(q_out[:q_valid], dQ[start : start + q_valid])
                            if start < kv_total:
                                T.copy(k_out[:kv_valid], dK[start : start + kv_valid])
                                T.copy(v_out[:kv_valid], dV[start : start + kv_valid])

        return main

    return factory()


@lru_cache(maxsize=32)
def _compile_output_unpad_fp32(batch, seq_len, heads, heads_kv, dim, q_padded, k_padded):
    """Unpad fp32 accumulators into the explicit buffers owned by TileOPs."""
    q_total = batch * seq_len * heads * dim
    kv_total = batch * seq_len * heads_kv * dim
    tile = 1024
    logical_blocks = math.ceil(q_total / tile)
    blocks = launch_block_count(logical_blocks)
    repeats = grid_repeat_count(logical_blocks, blocks)
    src_q_shape = (batch, heads, q_padded, dim)
    src_k_shape = (batch, heads_kv, k_padded, dim)
    q_dst_shape = (q_total,)
    kv_dst_shape = (kv_total,)

    @tilelang.jit(
        out_idx=[],
        pass_configs={
            tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: False,
            tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
            tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
        },
    )
    def factory():
        @T.prim_func
        def main(
            dQ_padded: T.Tensor(src_q_shape, "float"),
            dK_padded: T.Tensor(src_k_shape, "float"),
            dV_padded: T.Tensor(src_k_shape, "float"),
            dQ: T.Tensor(q_dst_shape, "float"),
            dK: T.Tensor(kv_dst_shape, "float"),
            dV: T.Tensor(kv_dst_shape, "float"),
        ):
            with T.Kernel(blocks, is_npu=True) as (cid, vid):
                q_ub = T.alloc_ub((tile // 2,), "float")
                k_ub = T.alloc_ub((tile // 2,), "float")
                v_ub = T.alloc_ub((tile // 2,), "float")

                with T.Scope("V"):
                    for rep in T.serial(repeats):
                        start = (cid + rep * blocks) * tile + vid * (tile // 2)
                        if start < q_total:
                            q_valid = T.min(tile // 2, q_total - start)
                            kv_valid = T.min(tile // 2, kv_total - start)
                            for i in T.serial(tile // 2):
                                if i < q_valid:
                                    flat = start + i
                                    d = flat % dim
                                    h = (flat // dim) % heads
                                    s = (flat // (dim * heads)) % seq_len
                                    b = flat // (dim * heads * seq_len)
                                    q_ub[i] = dQ_padded[b, h, s, d]
                                else:
                                    q_ub[i] = 0.0
                                if i < kv_valid:
                                    flat_kv = start + i
                                    d_kv = flat_kv % dim
                                    h_kv = (flat_kv // dim) % heads_kv
                                    s_kv = (flat_kv // (dim * heads_kv)) % seq_len
                                    b_kv = flat_kv // (dim * heads_kv * seq_len)
                                    k_ub[i] = dK_padded[b_kv, h_kv, s_kv, d_kv]
                                    v_ub[i] = dV_padded[b_kv, h_kv, s_kv, d_kv]
                                else:
                                    k_ub[i] = 0.0
                                    v_ub[i] = 0.0
                            T.copy(q_ub[:q_valid], dQ[start : start + q_valid])
                            if start < kv_total:
                                T.copy(k_ub[:kv_valid], dK[start : start + kv_valid])
                                T.copy(v_ub[:kv_valid], dV[start : start + kv_valid])

        return main

    return factory()


def build_attention_bwd_role_kernel(
    q_shape, k_shape, v_shape, o_shape, do_shape, lse_shape, dtype, is_causal=True
):
    """Build the GQA op's main-role ABI, including grouped KV heads."""
    if not all(len(shape) == 4 for shape in (q_shape, k_shape, v_shape, o_shape, do_shape)):
        raise ValueError("GroupedQueryAttentionBwdOp expects rank-4 BSHD tensors")
    batch, seq_len, heads, dim = (int(x) for x in q_shape)
    if tuple(o_shape) != tuple(q_shape) or tuple(do_shape) != tuple(q_shape):
        raise ValueError("GQA backward O/dO must match q shape")
    if tuple(v_shape) != tuple(k_shape):
        raise ValueError("GQA backward K/V shapes must match")
    kv_batch, kv_seq_len, heads_kv, kv_dim = (int(x) for x in k_shape)
    if (kv_batch, kv_seq_len, kv_dim) != (batch, seq_len, dim):
        raise ValueError("GQA backward K/V must match Q batch, sequence length, and dim")
    if heads_kv <= 0 or heads % heads_kv:
        raise ValueError(f"GQA backward requires heads % heads_kv == 0, got {heads} and {heads_kv}")
    if tuple(lse_shape) != (batch, heads, seq_len):
        raise ValueError(f"GQA backward expects lse ({batch}, {heads}, {seq_len}), got {lse_shape}")
    dtype_name = _dtype_name(dtype)
    q_tiles = math.ceil(seq_len / 32)
    k_tiles = math.ceil(seq_len / 64)
    q_padded = q_tiles * 32
    k_padded = k_tiles * 64
    cores = min(24, batch * heads * q_tiles)
    dv_kernel = _compile_dv(
        batch, seq_len, heads, heads_kv, dim, bool(is_causal), dtype_name
    )
    dqdk_kernel = _compile_dq_dk(
        batch, seq_len, heads, heads_kv, dim, bool(is_causal), dtype_name
    )
    output_unpad = _compile_output_unpad_fp32(
        batch, seq_len, heads, heads_kv, dim, q_padded, k_padded
    )

    def invoke(q, k, v, do, lse, out, dq, dk, dv):
        q3 = q.reshape(batch * seq_len, heads, dim).contiguous()
        k3 = k.reshape(batch * seq_len, heads_kv, dim).contiguous()
        v3 = v.reshape(batch * seq_len, heads_kv, dim).contiguous()
        out3 = out.reshape(batch * seq_len, heads, dim).contiguous()
        do3 = do.reshape(batch * seq_len, heads, dim).contiguous()
        dq_padded = torch.zeros(
            (batch, heads, q_padded, dim), dtype=torch.float32, device=q.device
        )
        dk_padded = torch.zeros(
            (batch, heads_kv, k_padded, dim), dtype=torch.float32, device=q.device
        )
        dv_padded = torch.zeros_like(dk_padded)
        score_ws = torch.empty((cores, 32, 64), dtype=torch.float32, device=q.device)
        prob_ws = torch.empty((cores, 32, 64), dtype=q.dtype, device=q.device)
        dp_ws = torch.empty_like(score_ws)
        ds_ws = torch.empty_like(prob_ws)
        dv_kernel(q3, k3, do3, lse, dv_padded, score_ws, prob_ws)
        dqdk_kernel(
            q3,
            k3,
            v3,
            out3,
            do3,
            lse,
            dq_padded,
            dk_padded,
            score_ws,
            dp_ws,
            ds_ws,
        )
        output_unpad(
            dq_padded,
            dk_padded,
            dv_padded,
            dq.reshape(-1),
            dk.reshape(-1),
            dv.reshape(-1),
        )

    return invoke


def build_attention_bwd_kernel(q_shape, k_shape, v_shape, o_shape, do_shape, lse_shape, dtype, is_causal=True):
    """Build MHA backward; public tensors remain BSHD and outputs match q/k/v."""
    if not (len(q_shape) == len(k_shape) == len(v_shape) == len(o_shape) == len(do_shape) == 4):
        raise ValueError("MultiHeadAttentionBwdOp expects rank-4 BSHD tensors")
    if tuple(k_shape) != tuple(q_shape) or tuple(v_shape) != tuple(q_shape):
        raise ValueError("dense MHA backward requires equal Q/K/V shapes")
    if tuple(o_shape) != tuple(q_shape) or tuple(do_shape) != tuple(q_shape):
        raise ValueError("MHA backward O/dO must match q shape")
    batch, seq_len, heads, dim = (int(x) for x in q_shape)
    if tuple(lse_shape) != (batch, heads, seq_len):
        raise ValueError(f"MHA backward expects lse ({batch}, {heads}, {seq_len}), got {lse_shape}")
    dtype_name = _dtype_name(dtype)
    q_tiles = math.ceil(seq_len / 32)
    k_tiles = math.ceil(seq_len / 64)
    q_padded = q_tiles * 32
    k_padded = k_tiles * 64
    logical_tasks = batch * heads * q_tiles
    cores = min(24, logical_tasks)
    dv_kernel = _compile_dv(
        batch, seq_len, heads, heads, dim, bool(is_causal), dtype_name
    )
    dqdk_kernel = _compile_dq_dk(
        batch, seq_len, heads, heads, dim, bool(is_causal), dtype_name
    )
    output_cast = _compile_output_cast(
        batch, seq_len, heads, heads, dim, q_padded, k_padded, dtype_name
    )

    def invoke(q, k, v, o, do, lse):
        q3 = q.reshape(batch * seq_len, heads, dim)
        k3 = k.reshape(batch * seq_len, heads, dim)
        v3 = v.reshape(batch * seq_len, heads, dim)
        o3 = o.reshape(batch * seq_len, heads, dim)
        do3 = do.reshape(batch * seq_len, heads, dim)
        # Padded fp32 destinations make full-tile atomic writes tail-safe.  The
        # public BSHD outputs are cropped and cast by ``output_cast`` below.
        dq32 = torch.zeros(
            (batch, heads, q_padded, dim), dtype=torch.float32, device=q.device
        )
        dk32 = torch.zeros(
            (batch, heads, k_padded, dim), dtype=torch.float32, device=q.device
        )
        dv32 = torch.zeros_like(dk32)
        score_ws = torch.empty(
            (cores, 32, 64), dtype=torch.float32, device=q.device
        )
        prob_ws = torch.empty((cores, 32, 64), dtype=q.dtype, device=q.device)
        dp_ws = torch.empty_like(score_ws)
        ds_ws = torch.empty_like(prob_ws)

        dv_kernel(q3, k3, do3, lse, dv32, score_ws, prob_ws)
        dqdk_kernel(
            q3,
            k3,
            v3,
            o3,
            do3,
            lse,
            dq32,
            dk32,
            score_ws,
            dp_ws,
            ds_ws,
        )
        dq = torch.empty_like(q3)
        dk = torch.empty_like(k3)
        dv = torch.empty_like(v3)
        output_cast(
            dq32,
            dk32,
            dv32,
            dq.reshape(-1),
            dk.reshape(-1),
            dv.reshape(-1),
        )
        return dq.reshape(q_shape), dk.reshape(k_shape), dv.reshape(v_shape)

    return invoke


__all__ = ["build_attention_bwd_kernel", "build_attention_bwd_role_kernel"]
