"""Ascend multi-core token-permute kernel.

The routing pass is a counting sort over ``topk_ids``.  Each core first writes a
per-expert histogram, then all cores derive disjoint expert ranges and gather the
hidden rows into the tight output.  Int32 metadata uses vector copies; the int64
prefix has a separately audited scalar writeback because PTO lacks an int64
vector-copy lowering.
"""

from functools import lru_cache
import math

import tilelang
import tilelang.language as T
import torch


_NUM_CORES = 24
_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
}


def _choose_cores(tokens: int) -> int:
    """Use the largest available core count that evenly partitions tokens.

    A partial final core would require scalar GM writes for ``fwd_idx``; dav-2201
    can lower those stores to an impossible AIC/AIV guard, leaving allocator data.
    Keeping the pair partition exact avoids that silent path altogether.
    """
    limit = min(_NUM_CORES, max(1, tokens))
    for candidate in range(limit, 0, -1):
        if tokens % candidate == 0:
            return candidate
    return 1


def _dtype_name(dtype: torch.dtype) -> str:
    if dtype == torch.float16:
        return "float16"
    if dtype == torch.bfloat16:
        return "bfloat16"
    raise TypeError(f"MoePermuteNopadFwdOp supports float16/bfloat16, got {dtype}")


@lru_cache(maxsize=64)
def _compile(
    tokens: int,
    top_k: int,
    experts: int,
    hidden: int,
    dtype_name: str,
    map_present: bool,
):
    cores = _choose_cores(tokens)
    tokens_per_core = math.ceil(tokens / cores)
    chunk = tokens_per_core * top_k
    total_pairs = tokens * top_k
    workspace = cores * experts
    half_hidden = hidden // 2
    if hidden < 2 or hidden % 2:
        raise ValueError(f"hidden size must be a positive even value, got {hidden}")

    @tilelang.jit(
        out_idx=[2, 3, 4, 5, 6],
        workspace_idx=[7],
        pass_configs=_PASS_CONFIGS,
        target="pto",
    )
    def factory():
        @T.prim_func
        def main(
            hidden_gm: T.Tensor((tokens, hidden), dtype_name),
            ids_gm: T.Tensor((tokens, top_k), "int32"),
            perm_gm: T.Tensor((total_pairs, hidden), dtype_name),
            offsets_gm: T.Tensor((experts,), "int32"),
            sizes_gm: T.Tensor((experts,), "int32"),
            first_gm: T.Tensor((experts + 1,), "int64"),
            fwd_gm: T.Tensor((total_pairs,), "int32"),
            workspace_gm: T.Tensor((workspace,), "int32"),
        ):
            with T.Kernel(cores, is_npu=True) as (cid, vid):
                ids_ub = T.alloc_ub((chunk,), "int32")
                hist_ub = T.alloc_ub((experts,), "int32")
                ws_ub = T.alloc_ub((workspace,), "int32")
                offsets_ub = T.alloc_ub((experts,), "int32")
                sizes_ub = T.alloc_ub((experts,), "int32")
                counters_ub = T.alloc_ub((experts,), "int32")
                first_ub = T.alloc_ub((experts + 1,), "int64")
                sio_ub = T.alloc_ub((chunk,), "int32")
                row = T.alloc_ub((half_hidden,), dtype_name)

                my_start = cid * chunk
                with T.Scope("C"):
                    T.sync_all()

                with T.Scope("V"):
                    for e in T.Pipelined(experts):
                        hist_ub[e] = 0
                    # Keep the guarded form even though the compile-time core
                    # choice normally makes every chunk exact.
                    for i in T.serial(chunk):
                        if my_start + i < total_pairs:
                            ids_ub[i] = ids_gm[(my_start + i) // top_k, (my_start + i) % top_k]
                        else:
                            ids_ub[i] = 0

                    for i in T.Pipelined(chunk):
                        if my_start + i < total_pairs:
                            expert = ids_ub[i]
                            if expert >= 0:
                                hist_ub[expert] = hist_ub[expert] + 1

                    T.copy(hist_ub, workspace_gm[cid * experts])
                    T.set_flag("mte3", "mte2", 2)
                    T.wait_flag("mte3", "mte2", 2)
                    T.sync_all()

                    T.copy(workspace_gm[0], ws_ub)
                    T.set_flag("mte2", "v", 3)
                    T.wait_flag("mte2", "v", 3)

                    running = T.alloc_var("int32")
                    running = 0
                    first_ub[0] = 0
                    for e in T.Pipelined(experts):
                        total_e = T.alloc_var("int32")
                        prefix = T.alloc_var("int32")
                        total_e = 0
                        prefix = 0
                        for c in T.Pipelined(cores):
                            total_e = total_e + ws_ub[c * experts + e]
                            if c < cid:
                                prefix = prefix + ws_ub[c * experts + e]
                        offsets_ub[e] = running + prefix
                        sizes_ub[e] = total_e
                        counters_ub[e] = 0
                        first_ub[e] = T.cast(running, "int64")
                        running = running + total_e
                    first_ub[experts] = T.cast(running, "int64")

                    for i in T.Pipelined(chunk):
                        if my_start + i < total_pairs:
                            expert = ids_ub[i]
                            if expert >= 0:
                                slot = offsets_ub[expert] + counters_ub[expert]
                                counters_ub[expert] = counters_ub[expert] + 1
                                sio_ub[i] = slot
                            else:
                                sio_ub[i] = -1
                        else:
                            sio_ub[i] = -1

                    # Every core computes identical metadata.  Restrict the
                    # actual write to cid=0; all other cores still reach the
                    # barrier so no consumer observes a partially written table.
                    if cid == 0:
                        T.copy(offsets_ub, offsets_gm[0])
                        T.copy(sizes_ub, sizes_gm[0])
                        # PTO currently has no int64 vector-copy lowering; try the
                        # scalar form so the compiler can establish whether the
                        # contract is representable at all.
                        for e in T.serial(experts + 1):
                            first_gm[e] = first_ub[e]
                    T.barrier_all()

                    # fwd_idx is a flat-pair mapping.  ``cores`` was chosen to
                    # make every chunk exact, so this
                    # remains a vector copy for every launch.
                    T.copy(sio_ub, fwd_gm[my_start])

                    # Gather one hidden row per routing pair.  vid owns a
                    # disjoint half-row: start = cid*chunk + vid*half_hidden.
                    for pair in T.serial(chunk):
                        if my_start + pair < total_pairs:
                            src_token = (my_start + pair) // top_k
                            slot = sio_ub[pair]
                            if slot >= 0:
                                T.copy(
                                    hidden_gm[src_token, vid * half_hidden],
                                    row,
                                )
                                T.copy(row, perm_gm[slot, vid * half_hidden])

                    T.barrier_all()

        return main

    return factory()


def build_moe_permute_kernel(
    hidden_shape: tuple[int, ...],
    ids_shape: tuple[int, ...],
    dtype: torch.dtype,
    num_experts: int,
    num_experts_local: int,
    map_present: bool,
):
    if len(hidden_shape) != 2 or len(ids_shape) != 2:
        raise ValueError("MoePermuteNopadFwdOp expects hidden_states [T,H] and topk_ids [T,K]")
    if hidden_shape[0] != ids_shape[0]:
        raise ValueError("hidden_states and topk_ids must have the same token count")
    if num_experts_local != num_experts:
        raise NotImplementedError(
            "expert_map EP routing is intentionally outside this first single-rank template"
        )
    if map_present:
        raise NotImplementedError("expert_map path is reserved for the EP follow-up")
    dtype_name = _dtype_name(dtype)
    tokens, hidden = map(int, hidden_shape)
    top_k = int(ids_shape[1])
    compiled = _compile(tokens, top_k, int(num_experts), hidden, dtype_name, False)
    def invoke(hidden_states: torch.Tensor, topk_ids: torch.Tensor):
        if tuple(hidden_states.shape) != tuple(hidden_shape) or tuple(topk_ids.shape) != tuple(ids_shape):
            raise ValueError("MoePermuteNopadFwdOp kernel shape mismatch")
        # The JIT adapter owns output/workspace allocation for out_idx and
        # workspace_idx; passing user-created buffers here would shift the
        # positional ABI and silently return untouched tensors.
        return compiled(hidden_states, topk_ids)

    return invoke


__all__ = ["build_moe_permute_kernel"]


# The two kernels below are deliberately separate from the no-pad counting-sort
# implementation above.  They use explicit output parameters (``out_idx=[]``),
# which keeps the output ABI unambiguous for callers that may supply ``out``.
@lru_cache(maxsize=64)
def _compile_permute_align(total_tokens: int, top_k: int, experts: int, block_size: int):
    pairs = total_tokens * top_k
    padded = pairs + (experts + 1) * (block_size - 1)
    blocks = (padded + block_size - 1) // block_size

    @tilelang.jit(out_idx=[], pass_configs=_PASS_CONFIGS, target="pto")
    def factory():
        @T.prim_func
        def main(
            ids_gm: T.Tensor((total_tokens, top_k), "int32"),
            sorted_gm: T.Tensor((padded,), "int32"),
            expert_gm: T.Tensor((blocks,), "int32"),
            post_gm: T.Tensor((1,), "int32"),
        ):
            with T.Kernel(1, is_npu=True) as (cid, vid):
                counts = T.alloc_ub((experts,), "int32")
                cursor = T.alloc_ub((experts,), "int32")
                if vid == 0:
                    with T.Scope("V"):
                        for e in T.serial(experts):
                            counts[e] = 0
                            cursor[e] = 0
                        for i in T.serial(pairs):
                            expert = ids_gm[i // top_k, i % top_k]
                            if expert >= 0 and expert < experts:
                                counts[expert] = counts[expert] + 1
                        running = T.alloc_var("int32")
                        running = 0
                        for e in T.serial(experts):
                            cursor[e] = running
                            running = running + ((counts[e] + block_size - 1) // block_size) * block_size
                        post_gm[0] = running
                        # TileOPs' ABI uses the flattened assignment count as
                        # the padding sentinel, so it cannot be confused with
                        # a valid assignment index.
                        for i in T.serial(padded):
                            sorted_gm[i] = pairs
                        for b in T.serial(blocks):
                            expert_gm[b] = -1
                        for e in T.serial(experts):
                            start = cursor[e]
                            count = counts[e]
                            first_block = start // block_size
                            nblocks = (count + block_size - 1) // block_size
                            for b in T.serial(nblocks):
                                expert_gm[first_block + b] = e
                        for i in T.serial(pairs):
                            expert = ids_gm[i // top_k, i % top_k]
                            if expert >= 0 and expert < experts:
                                slot = cursor[expert]
                                # Keep the flattened (token, top-k) assignment
                                # index.  The downstream grouped-GEMM contract
                                # indexes the routing pair, not the token row.
                                sorted_gm[slot] = i
                                cursor[expert] = slot + 1
                    T.barrier_all()

        return main

    return factory()


def build_moe_permute_align_kernel(total_tokens: int, top_k: int, experts: int, block_size: int):
    if min(total_tokens, top_k, experts, block_size) <= 0:
        raise ValueError("MoePermuteAlignFwdOp parameters must be positive")
    if block_size > 1024:
        raise ValueError("MoePermuteAlignFwdOp block_size exceeds the supported UB tile")
    compiled = _compile_permute_align(int(total_tokens), int(top_k), int(experts), int(block_size))
    pairs = int(total_tokens) * int(top_k)
    padded = pairs + (int(experts) + 1) * (int(block_size) - 1)
    blocks = (padded + int(block_size) - 1) // int(block_size)

    def invoke(topk_ids: torch.Tensor):
        if tuple(topk_ids.shape) != (int(total_tokens), int(top_k)):
            raise ValueError("MoePermuteAlignFwdOp topk_ids shape mismatch")
        sorted_ids = torch.empty((padded,), dtype=torch.int32, device=topk_ids.device)
        expert_ids = torch.empty((blocks,), dtype=torch.int32, device=topk_ids.device)
        num_post_pad = torch.empty((1,), dtype=torch.int32, device=topk_ids.device)
        compiled(topk_ids, sorted_ids, expert_ids, num_post_pad)
        return sorted_ids, expert_ids, num_post_pad

    return invoke


@lru_cache(maxsize=64)
def _compile_unpermute(
    total_tokens: int,
    top_k: int,
    padded_batch_sum: int,
    hidden: int,
    dtype_name: str,
    scaling: float,
):
    if hidden < 2 or hidden % 2:
        raise ValueError(f"MoeUnpermuteFwdOp hidden size must be positive and even, got {hidden}")
    cores = _choose_cores(total_tokens)
    tokens_per_core = total_tokens // cores
    hidden_per_vector = hidden // 2
    hidden_tile = ((hidden_per_vector + 31) // 32) * 32
    routing_tile = ((top_k + 31) // 32) * 32

    @tilelang.jit(out_idx=[], pass_configs=_PASS_CONFIGS, target="pto")
    def factory():
        @T.prim_func
        def main(
            mm2_gm: T.Tensor((padded_batch_sum, hidden), dtype_name),
            fwd_gm: T.Tensor((total_tokens * top_k,), "int32"),
            weights_gm: T.Tensor((total_tokens, top_k), "float32"),
            out_gm: T.Tensor((total_tokens, hidden), dtype_name),
        ):
            with T.Kernel(cores, is_npu=True) as (cid, vid):
                with T.Scope("V"):
                    source_typed = T.alloc_ub((hidden_tile,), dtype_name)
                    output_typed = T.alloc_ub((hidden_tile,), dtype_name)
                    source_fp32 = T.alloc_ub((hidden_tile,), "float32")
                    weighted_fp32 = T.alloc_ub((hidden_tile,), "float32")
                    accum_fp32 = T.alloc_ub((hidden_tile,), "float32")
                    slots_ub = T.alloc_ub((routing_tile,), "int32")
                    weights_ub = T.alloc_ub((routing_tile,), "float32")

                    for local_token in T.serial(tokens_per_core):
                        token = cid * tokens_per_core + local_token
                        hidden_start = vid * hidden_per_vector
                        T.copy(
                            fwd_gm[token * top_k : (token + 1) * top_k],
                            slots_ub[0:top_k],
                        )
                        T.copy(weights_gm[token, 0:top_k], weights_ub[0:top_k])
                        T.tile.fill(accum_fp32, 0.0)
                        for k in T.serial(top_k):
                            slot = slots_ub[k]
                            if slot >= 0 and slot < padded_batch_sum:
                                T.tile.fill(source_typed, 0.0)
                                T.copy(
                                    mm2_gm[
                                        slot,
                                        hidden_start : hidden_start + hidden_per_vector,
                                    ],
                                    source_typed[0:hidden_per_vector],
                                )
                                T.tile.cast(
                                    source_fp32,
                                    source_typed,
                                    "CAST_NONE",
                                    hidden_tile,
                                )
                                T.tile.mul(
                                    weighted_fp32,
                                    source_fp32,
                                    weights_ub[k],
                                )
                                T.tile.add(accum_fp32, accum_fp32, weighted_fp32)
                        if scaling != 1.0:
                            T.tile.mul(accum_fp32, accum_fp32, scaling)
                        T.tile.cast(
                            output_typed,
                            accum_fp32,
                            "CAST_RINT",
                            hidden_tile,
                        )
                        T.copy(
                            output_typed[0:hidden_per_vector],
                            out_gm[
                                token,
                                hidden_start : hidden_start + hidden_per_vector,
                            ],
                        )
                    T.barrier_all()

        return main

    return factory()


def build_moe_unpermute_kernel(mm2_shape, fwd_shape, weights_shape, dtype, scaling: float = 1.0):
    if len(mm2_shape) != 2 or len(fwd_shape) != 1 or len(weights_shape) != 2:
        raise ValueError("MoeUnpermuteFwdOp expects mm2_pad [T*K,H], fwd_idx [T*K], weights [T,K]")
    total_tokens, top_k = int(weights_shape[0]), int(weights_shape[1])
    if int(fwd_shape[0]) != total_tokens * top_k or int(mm2_shape[0]) < total_tokens * top_k:
        raise ValueError("MoeUnpermuteFwdOp routing shapes are inconsistent")
    dtype_name = _dtype_name(dtype)
    hidden = int(mm2_shape[1])
    compiled = _compile_unpermute(
        total_tokens,
        top_k,
        int(mm2_shape[0]),
        hidden,
        dtype_name,
        float(scaling),
    )

    def invoke(mm2_pad: torch.Tensor, fwd_idx: torch.Tensor, topk_weights: torch.Tensor, out=None):
        output = out if out is not None else torch.empty((total_tokens, hidden), dtype=mm2_pad.dtype, device=mm2_pad.device)
        if tuple(output.shape) != (total_tokens, hidden):
            raise ValueError("MoeUnpermuteFwdOp output shape mismatch")
        compiled(mm2_pad, fwd_idx, topk_weights, output)
        return output

    return invoke


__all__ += ["build_moe_permute_align_kernel", "build_moe_unpermute_kernel"]
