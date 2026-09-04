"""Ascend AIV top-k index selection (``TopkSelectorFwdOp``).

Top-k is the generalisation of the arg-reduction in ``kernels/indexed_reduce.py``:
one vector context owns one flattened row, walks the reduction axis in UB tiles, and
emits indices rather than values. Three things are inherited from that template and
from ``kernels/pool_indices.py`` rather than reinvented:

* the flattened ``[M, N]`` row view built by the builder, with a host ``permute``
  when the reduction axis is not the trailing one (``kv_group > 1``);
* the ``launch_block_count`` / ``grid_repeat_count`` grid-stride cap, so BlockDim
  stays inside CANN 8.5's ``[1, 65535]`` (``docs/PROJECT_STATE.md`` §13.1);
* index arithmetic that is never allowed to touch a 32-bit product: the flat element
  count of the manifest's largest workload is ``32768 * 65536 == 2**31`` exactly, so
  a flattened int32 offset overflows. Every flat offset here is computed in Python
  ``int`` at build time or as an int64 on host; the kernel itself only ever indexes
  one row (``n <= 2**24``), which is why per-row indices are safe as int32 -- and as
  float32, which is what the platform's sort primitives return.

Unlike arg{max,min}, a scalar comparison loop is not viable: top-1024 of 65536 by
insertion is ``O(N*k)`` per row. The platform supplies a vector sort instead --
``T.tile.topk`` -- but it needs its whole source in UB, and the manifest reduction
axis is 65536 fp32 = 256 KiB against a 192 KiB UB
(``tilelang-ascend/tilelang/carver/arch/ascend.py:36``). The platform's own example
(``tilelang-ascend/examples/topk_selector/example_topk_selector.py:127``) records the
same ceiling as "max ~8192 for 192KB UB".

So the row is streamed: each ``chunk``-sized tile is top-k'd on its own, and the
result is merged into a running top-k of the row. Merging ``k`` running winners with
``k`` tile winners and taking the top ``k`` of the ``2k`` is exact, because a global
top-k element is a top-k element of whichever tile contains it.

Index bookkeeping is the only subtle part. ``T.tile.topk`` writes interleaved
``(value, index)`` pairs and generates the index itself, so it is *tile-local*.
``gather_mask`` de-interleaves ("P0101" = values at even slots, "P1010" = indices at
odd slots) and a vector ``adds`` lifts the tile-local index to a row index. The merge
step's indices are positions inside the ``2k`` merge buffer, so they are mapped back
through a vector ``T.tile.gather`` -- byte offsets, uint32, exactly as
``tilelang-ascend/examples/cann-bench/gather/gather.py:80`` does it.

The ``starts``/``ends`` window is applied with the mask/select pair the platform uses
for the same job in ``examples/HISA/block_sparse_mqa_attn_expert_test_for_a5.py:554``:
a float position vector, two scalar compares, ``bitwise_and``, and a
``VSEL_TENSOR_SCALAR_MODE`` select that drops out-of-window lanes to ``-inf``. That is
the same candidate window the in-tree CUDA kernel enforces at
``TileOPs/src/tileops/kernels/topk_selector.py:86-90``.

Every public write is a vector ``T.copy``; there is no scalar GM store, so the F013
AIV-guard class (``tools/audit_aiv_guard.py``) cannot arise here.
"""

from __future__ import annotations

from functools import lru_cache

import tilelang
import tilelang.language as T
import torch

from .common import grid_repeat_count, launch_block_count


#: Largest UB tile handed to ``T.tile.topk``. The platform example's own ceiling is
#: ~8192 fp32 for a kernel that holds nothing else; this kernel also holds the
#: position vector, the merge buffers and the gather scratch, so it stays below that.
_MAX_CHUNK = 4096

#: Smallest UB tile. ``chunk // 8`` is the window-mask byte count, so this keeps the
#: mask itself a whole 32-byte unit.
_MIN_CHUNK = 256

#: ``T.tile.topk`` sorts in 32-element groups (``ascend_tile.py:592``).
_SORT_GRANULE = 32

#: Every UB buffer length is rounded up to this many float32 lanes. A vector
#: instruction addresses UB in 32-byte units; a buffer of, say, 17 float32 makes the
#: *next* allocation start off a 32-byte boundary and the launch aborts with
#: ``507015 ... The UB address accessed by the VEC instruction is not aligned``.
#: Measured on 2026-08-28 with ``topk=17``; see ``R192-data/probe_topk_small.log``.
_UB_LANES = 32

#: A finite stand-in for -inf used to prime the running top-k. Nothing a float32
#: score can legally be compares above it, and unlike ``T.infinity`` it is accepted
#: by ``T.tile.fill``.
_NEG_SENTINEL = -3.0e38


def _round_up_pow2(value: int) -> int:
    result = 1
    while result < value:
        result *= 2
    return result


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _chunk_for(n: int) -> int:
    """The UB tile used to walk one row of length *n*.

    Never smaller than ``_MIN_CHUNK``: the window mask is a ``chunk // 8`` byte
    ``uint8`` buffer, so a small chunk would itself be a sub-32-byte allocation.
    """
    chunk = min(_MAX_CHUNK, _round_up_pow2(n))
    return max(chunk, _MIN_CHUNK)


@lru_cache(maxsize=32)
def _compile(m: int, n: int, k: int, group: int, chunk: int):
    """Compile one static ``(rows, reduction length, k, kv_group)`` signature.

    Args:
        m: Number of flattened rows, ``batch * seq_len * kv_group``.
        n: Reduction length, ``seq_len_kv``.
        k: ``params.topk``. A compile-time constant -- the op stores it on the
            instance and ``Op._manifest_params()`` hands it to the builder by
            keyword, so it is known here and can size UB.
        group: ``kv_group``; row ``r`` reads ``starts``/``ends`` at ``r // group``.
        chunk: UB tile along the reduction axis.
    """
    # Every UB buffer is sized in whole 32-lane units, and the platform sort is
    # asked for ``kp >= k`` winners rather than exactly ``k``. Taking more winners
    # per tile is still exact: the global top-k is a subset of the top-kp of every
    # tile that contains any of it, and the merge keeps the pairs sorted, so the
    # answer is the leading ``k`` of the final buffer.
    kp = _round_up(k, _UB_LANES)
    rows_per_block = 2  # one row per vector context
    m_tiles = (m + rows_per_block - 1) // rows_per_block
    launch_blocks = launch_block_count(m_tiles)
    grid_repeats = grid_repeat_count(m_tiles, launch_blocks)
    n_chunks = (n + chunk - 1) // chunk

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            tilelang.PassConfigKey.TIR_MERGE_STATIC_SMEM: True,
            tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
            tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
            tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
        },
    )
    def factory():
        @T.prim_func
        def main(
            scores: T.Tensor((m, n), "float32"),
            starts: T.Tensor((m // group,), "int32"),
            ends: T.Tensor((m // group,), "int32"),
            indexes: T.Tensor((m, kp), "int32"),
        ):
            with T.Kernel(launch_blocks, is_npu=True) as (cid, vid):
                tile = T.alloc_ub((chunk,), "float32")
                positions = T.alloc_ub((chunk,), "float32")
                keep = T.alloc_ub((chunk // 8,), "uint8")
                bound = T.alloc_ub((chunk // 8,), "uint8")

                pairs = T.alloc_ub((2 * kp,), "float32")
                tile_values = T.alloc_ub((kp,), "float32")
                tile_indices = T.alloc_ub((kp,), "float32")
                merge_values = T.alloc_ub((2 * kp,), "float32")
                merge_indices = T.alloc_ub((2 * kp,), "float32")
                best_values = T.alloc_ub((kp,), "float32")
                best_indices = T.alloc_ub((kp,), "float32")
                slot_i32 = T.alloc_ub((kp,), "int32")
                slot_u32 = T.alloc_ub((kp,), "uint32")

                with T.Scope("V"):
                    # The position vector is the same for every tile; only the two
                    # window bounds move, so it is built once per launch and the
                    # bounds are shifted instead.
                    T.tile.arith_progression(positions, 0.0, 1.0, chunk)

                    for grid_repeat in T.serial(grid_repeats):
                        logical_cid = cid + grid_repeat * launch_blocks
                        if logical_cid < m_tiles:
                            row = logical_cid * rows_per_block + vid
                            if row < m:
                                window_row = row // group
                                start = starts[window_row]
                                end = ends[window_row]

                                T.tile.fill(best_values, _NEG_SENTINEL)
                                T.tile.fill(best_indices, 0.0)

                                for c in T.serial(n_chunks):
                                    base = c * chunk
                                    T.copy(
                                        scores[row, base:base + chunk],
                                        tile,
                                        # Only reachable on the trailing tile of a
                                        # non-divisible row; the `< n - base`
                                        # predicate below discards these lanes, so
                                        # the pad value never reaches the sort.
                                        pad_value=0.0,
                                    )

                                    T.tile.compare(
                                        keep,
                                        positions,
                                        T.Cast("float32", start - base),
                                        "GE",
                                    )
                                    T.tile.compare(
                                        bound,
                                        positions,
                                        T.Cast("float32", end - base),
                                        "LT",
                                    )
                                    T.tile.bitwise_and(keep, keep, bound)
                                    # Trailing-tile predicate. Emitted unconditionally:
                                    # on a divisible row it is always true and costs two
                                    # vector ops against a sort of the same tile, and
                                    # unconditional emission keeps ``n`` a free variable
                                    # of this function -- TVMScript evaluates the
                                    # parameter annotations against the closure, so a
                                    # name that appears only in an annotation is not
                                    # captured and the annotation silently degrades to a
                                    # string under ``from __future__ import annotations``.
                                    T.tile.compare(
                                        bound, positions, T.float32(n - base), "LT"
                                    )
                                    T.tile.bitwise_and(keep, keep, bound)
                                    T.tile.select(
                                        tile,
                                        keep,
                                        tile,
                                        _NEG_SENTINEL,
                                        "VSEL_TENSOR_SCALAR_MODE",
                                    )

                                    T.tile.topk(pairs, tile, kp, chunk)
                                    T.tile.gather_mask(tile_values, pairs, "P0101")
                                    T.tile.gather_mask(tile_indices, pairs, "P1010")
                                    # Tile-local index -> row index.
                                    T.tile.add(tile_indices, tile_indices, T.float32(base))

                                    T.copy(best_values, merge_values[0:kp])
                                    T.copy(tile_values, merge_values[kp:2 * kp])
                                    T.copy(best_indices, merge_indices[0:kp])
                                    T.copy(tile_indices, merge_indices[kp:2 * kp])

                                    T.tile.topk(pairs, merge_values, kp, 2 * kp)
                                    T.tile.gather_mask(best_values, pairs, "P0101")
                                    T.tile.gather_mask(tile_values, pairs, "P1010")
                                    # `tile_values` now holds slots inside
                                    # `merge_indices`, not row indices; map them back
                                    # with a vector gather over byte offsets.
                                    T.tile.cast(slot_i32, tile_values, "CAST_ROUND", kp)
                                    T.tile.mul(slot_i32, slot_i32, 4)
                                    T.reinterpretcast(slot_u32, slot_i32, "uint32_t")
                                    T.tile.gather(best_indices, merge_indices, slot_u32, 0)

                                # `slot_i32` is dead once the last merge has been
                                # gathered, so the int32 writeback reuses it rather
                                # than paying another `kp` lanes of UB.
                                T.tile.cast(slot_i32, best_indices, "CAST_ROUND", kp)
                                T.copy(slot_i32, indexes[row, :])

        return main

    return factory()


def build_topk_selector_kernel(index_score_shape, dtype, starts_shape, ends_shape, topk):
    """Return a callable implementing the TileOPs ``TopkSelectorFwdOp`` contract.

    Args:
        index_score_shape: ``[batch, seq_len, seq_len_kv, kv_group]``.
        dtype: Score dtype. The manifest declares ``float32`` and nothing else.
        starts_shape: ``[batch, seq_len]``.
        ends_shape: ``[batch, seq_len]``.
        topk: ``params.topk``.

    Returns:
        ``launch(index_score, starts, ends) -> indexes`` with
        ``indexes.shape == [batch, seq_len, kv_group, topk]`` and dtype int32, as
        ``TileOPs/src/tileops/manifest/attention_indexing.yaml:62-63`` declares.
    """
    if dtype != torch.float32:
        raise TypeError(
            f"TopkSelectorFwdOp declares index_score dtype float32; received {dtype}"
        )
    if len(index_score_shape) != 4:
        raise ValueError(
            f"TopkSelectorFwdOp expects index_score [batch, seq_len, seq_len_kv, kv_group]; "
            f"received {tuple(index_score_shape)}"
        )
    batch, seq_len, seq_len_kv, kv_group = (int(dim) for dim in index_score_shape)
    topk = int(topk)
    if not 0 < topk <= seq_len_kv:
        raise ValueError(f"topk must satisfy 0 < topk <= seq_len_kv={seq_len_kv}, got {topk}")
    if tuple(int(d) for d in starts_shape) != (batch, seq_len):
        raise ValueError(
            f"starts must be [batch, seq_len] = {(batch, seq_len)}; "
            f"received {tuple(starts_shape)}"
        )
    if tuple(int(d) for d in ends_shape) != (batch, seq_len):
        raise ValueError(
            f"ends must be [batch, seq_len] = {(batch, seq_len)}; received {tuple(ends_shape)}"
        )

    rows = batch * seq_len * kv_group
    chunk = _chunk_for(seq_len_kv)
    padded_topk = _round_up(topk, _UB_LANES)
    if padded_topk > chunk:
        raise ValueError(
            f"this Ascend kernel streams the reduction axis in {chunk}-element UB tiles "
            f"and merges 2*topk candidates per tile, so it needs topk (rounded up to "
            f"{_UB_LANES}) <= {chunk}; got topk={topk}. A larger topk needs a "
            f"multi-level merge, not a wider tile: a wider tile exceeds the 192 KiB UB."
        )
    compiled = _compile(rows, seq_len_kv, topk, kv_group, chunk)

    def launch(index_score, starts, ends):
        if kv_group == 1:
            flat = index_score.reshape(rows, seq_len_kv)
        else:
            # The reduction axis is dim 2, not the trailing one. A [rows, N] view
            # with unit stride does not exist for kv_group > 1, so the axis is moved
            # on host -- the same boundary fallback `kernels/indexed_reduce.py`
            # and `kernels/scan.py` use, and the kernel still owns the whole
            # reduction and every output write.
            flat = index_score.permute(0, 1, 3, 2).reshape(rows, seq_len_kv)
        flat = flat.contiguous()
        # The kernel writes `padded_topk` int32 per row so the writeback is one whole
        # vector copy; the manifest output is the leading `topk` of it, and the pairs
        # come out of the platform sort in descending order, so the leading slice is
        # exactly the top-k.
        out = compiled(flat, starts.reshape(-1), ends.reshape(-1))
        if padded_topk != topk:
            out = out[:, :topk].contiguous()
        return out.reshape(batch, seq_len, kv_group, topk)

    return launch


__all__ = ["build_topk_selector_kernel"]
