"""Ascend AIV argmax row-reduction kernel.

The input is presented as a flattened ``[M, N]`` view by the builder.  Each
vector context owns a disjoint row range, and the output is padded so the
writeback is always a vector copy (scalar GM stores can be removed by the
auto CV combiner on dav-2201).

R198: the per-lane scalar comparison loop that used to carry the reduction
(``_SUB_M * _BLOCK_N`` = 16384 scalar iterations per tile, on the main path)
is replaced by a vector sequence.  The index of the winning lane is recovered
without leaving vector code:

    tile_max            = reduce_max(x)                       # per row
    is_max              = (x == broadcast(tile_max))          # packed mask
    candidate_index     = select(is_max, ramp, +BIG)           # ramp[r, c] = c
    first_index_in_tile = reduce_min(candidate_index) + base

``reduce_min`` over the candidate columns yields the *lowest* matching column,
and the cross-tile merge uses a strict comparison, so the earliest index wins
every tie -- matching ``torch.arg{max,min}``.

Two correctness constraints are load-bearing here (cf. PROJECT_STATE 13.20 /
R090): the neutral element of ``reduce_max`` is -inf, not 0, so the padded
lanes of a trailing tile must be driven to the *signed-infinity* sentinel
before the reduction, and the same lanes must additionally be cleared out of
the index-candidate mask -- a sentinel alone is not enough, because real input
data may legitimately contain +/-inf and would then tie with the padding.
"""

from functools import lru_cache
import math

import tilelang
import tilelang.language as T
import torch

from .common import grid_repeat_count, launch_block_count


_BLOCK_M = 128
_VEC = 2
_SUB_M = _BLOCK_M // _VEC
_PADDED_OUTPUT = 64

# --- Geometry (R198) ---------------------------------------------------------
# ``T.tile.compare`` requires a 256-byte aligned operand (codegen_ascend.cc:1926,
# "Compare alignment error").  The cross-tile merge compares the per-row state
# vectors, so the tile row count is pinned to ``_SUB_M`` = 64 (64 x fp32 = 256 B)
# -- a (32, 512) tile holding the same number of lanes does not compile.
_TILE_ROWS = _SUB_M
# With the rows pinned, UB fits three fp32 tiles of (_TILE_ROWS, _BLOCK_N) plus
# one input-dtype tile.  fp32: 4 * 64 * 128 * 4 = 131072 B; bf16/fp16 needs less.
# The measured sweep (R198-data/probe/probe_argmax_geom.json) shows fp16 and fp32
# take the same time at this geometry, i.e. the kernel is element-rate limited
# rather than byte-rate limited, so widening _BLOCK_N is not the lever -- the
# vector-op count per lane is.  See the trailing-tile guards below.
_BLOCK_N = 128
# fp32 holds integers exactly up to 2**24.  Beyond that the index arithmetic in
# the vector path would start rounding, so those (absent from every manifest
# shape) keep the original scalar comparison loop.
_MAX_VECTOR_INDEX = 1 << 24


def _dtype_name(dtype: torch.dtype) -> str:
    if dtype == torch.float16:
        return "float16"
    if dtype == torch.bfloat16:
        return "bfloat16"
    if dtype == torch.float32:
        return "float32"
    raise TypeError(f"indexed arg-reduce supports floating dtypes, got {dtype}")


@lru_cache(maxsize=64)
def _compile(m: int, n: int, dtype_name: str, op_kind: str):
    """Compile one static row shape.

    Ties resolve to the lowest index: ``reduce_min`` picks the first matching
    column inside a tile and the cross-tile merge only updates on a strict
    comparison.  This matches PyTorch arg{max,min}.
    """

    m_tiles = (m + _BLOCK_M - 1) // _BLOCK_M
    n_tiles = (n + _BLOCK_N - 1) // _BLOCK_N
    has_m_tail = m % _BLOCK_M != 0
    has_n_tail = n % _BLOCK_N != 0
    launch_blocks = launch_block_count(m_tiles)
    grid_repeats = grid_repeat_count(m_tiles, launch_blocks)
    need_cast = dtype_name != "float32"
    # 2**30: exactly representable in fp32 and far above any supported index, so
    # a tile with no matching lane always loses the cross-tile min.
    big_index = float(1 << 30)

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
            tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
            tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
        },
    )
    def factory():
        @T.prim_func
        def main(
            A: T.Tensor((m, n), dtype_name),
            C: T.Tensor((m_tiles * _BLOCK_M,), "int64"),
        ):
            with T.Kernel(launch_blocks, is_npu=True) as (cid, vid):
                values = T.alloc_ub((_TILE_ROWS, _BLOCK_N), dtype_name)
                values_fp32 = T.alloc_ub((_TILE_ROWS, _BLOCK_N), "float32")
                ramp = T.alloc_ub((_TILE_ROWS, _BLOCK_N), "float32")
                work = T.alloc_ub((_TILE_ROWS, _BLOCK_N), "float32")
                row_offset = T.alloc_ub((_TILE_ROWS,), "float32")
                best_values = T.alloc_ub((_TILE_ROWS,), "float32")
                best_index = T.alloc_ub((_TILE_ROWS,), "float32")
                tile_values = T.alloc_ub((_TILE_ROWS,), "float32")
                tile_index = T.alloc_ub((_TILE_ROWS,), "float32")
                best_indices = T.alloc_ub((_SUB_M,), "int64")
                mask = T.alloc_ub((_TILE_ROWS * _BLOCK_N // 8,), "uint8")
                valid = T.alloc_ub((_TILE_ROWS * _BLOCK_N // 8,), "uint8")
                row_mask = T.alloc_ub((_TILE_ROWS // 8,), "uint8")

                with T.Scope("V"):
                    # ``-inf`` for argmax: the neutral element of a max reduction.
                    # Built from T.infinity because a Python float("inf") lowers
                    # to CUDART_INF, which does not exist on dav-2201.
                    sentinel = (
                        -T.infinity("float32")
                        if op_kind == "argmax"
                        else T.infinity("float32")
                    )
                    # ramp[r, c] = c, built once per launch.  arith_progression
                    # fills flattened storage (verified: probe_intrinsics.json
                    # q1, row 1 starts at _BLOCK_N), so the per-row base is
                    # subtracted back off with a broadcast.
                    T.tile.arith_progression(ramp, 0.0, 1.0, _TILE_ROWS * _BLOCK_N)
                    for r in T.serial(_TILE_ROWS):
                        row_offset[r] = T.Cast("float32", r * _BLOCK_N)
                    T.tile.broadcast(work, row_offset)
                    T.tile.sub(ramp, ramp, work)

                    for grid_repeat in T.serial(grid_repeats):
                        logical_cid = cid + grid_repeat * launch_blocks
                        if logical_cid < m_tiles:
                            row_base = logical_cid * _BLOCK_M + vid * _SUB_M
                            T.tile.fill(best_values, sentinel)
                            T.tile.fill(best_index, 0.0)

                            for nt in T.serial(n_tiles):
                                base = nt * _BLOCK_N
                                if not has_m_tail or logical_cid < m_tiles - 1:
                                    # Whole row block is in range: one strided
                                    # 2D transfer instead of _TILE_ROWS of them.
                                    T.copy(A[row_base, base], values, pad_value=0)
                                else:
                                    for r in T.serial(_TILE_ROWS):
                                        if row_base + r < m:
                                            T.copy(
                                                A[row_base + r, base],
                                                values[r, :],
                                                pad_value=0,
                                            )
                                T.barrier_all()   # MTE2 -> V (§13.55 #1/#2)
                                if need_cast:
                                    T.tile.cast(
                                        values_fp32,
                                        values,
                                        "CAST_NONE",
                                        _TILE_ROWS * _BLOCK_N,
                                    )
                                else:
                                    T.copy(values, values_fp32)

                                if has_n_tail and nt == n_tiles - 1:
                                    # Trailing tile only.  Emitting these three
                                    # vector ops unconditionally would tax every
                                    # full tile, and the kernel is element-rate
                                    # limited, so they are guarded.
                                    T.tile.compare(
                                        valid, ramp, T.Cast("float32", n - base), "LT"
                                    )
                                    T.tile.select(
                                        values_fp32,
                                        valid,
                                        values_fp32,
                                        sentinel,
                                        "VSEL_TENSOR_SCALAR_MODE",
                                    )
                                if op_kind == "argmax":
                                    T.reduce_max(
                                        values_fp32, tile_values, dim=-1, clear=True
                                    )
                                else:
                                    T.reduce_min(
                                        values_fp32, tile_values, dim=-1, clear=True
                                    )
                                T.tile.broadcast(work, tile_values)
                                T.tile.compare(mask, values_fp32, work, "EQ")
                                if has_n_tail and nt == n_tiles - 1:
                                    # A sentinel-valued padding lane ties with
                                    # real +/-inf data, so drop the pad lanes
                                    # from the index candidates as well.
                                    T.tile.bitwise_and(mask, mask, valid)
                                T.tile.select(
                                    work, mask, ramp, big_index,
                                    "VSEL_TENSOR_SCALAR_MODE",
                                )
                                T.reduce_min(work, tile_index, dim=-1, clear=True)
                                T.tile.add(
                                    tile_index, tile_index, T.Cast("float32", base)
                                )
                                # Strict comparison: the earliest tile keeps a tie.
                                if op_kind == "argmax":
                                    T.tile.compare(
                                        row_mask, tile_values, best_values, "GT"
                                    )
                                else:
                                    T.tile.compare(
                                        row_mask, tile_values, best_values, "LT"
                                    )
                                T.tile.select(
                                    best_index, row_mask, tile_index, best_index,
                                    "VSEL_TENSOR_TENSOR_MODE",
                                )
                                if op_kind == "argmax":
                                    T.tile.max(best_values, best_values, tile_values)
                                else:
                                    T.tile.min(best_values, best_values, tile_values)

                            # V -> scalar -> MTE3.  ``best_index`` is written
                            # by T.tile.select now; it used to be written by the
                            # scalar comparison loop, which ordered the UB write
                            # against this read implicitly (§13.55 #2).
                            T.barrier_all()
                            for r in T.serial(_SUB_M):
                                best_indices[r] = T.Cast("int64", best_index[r])
                            T.copy(best_indices, C[row_base : row_base + _SUB_M])

        return main

    return factory()


@lru_cache(maxsize=64)
def _compile_scalar_fallback(m: int, n: int, dtype_name: str, op_kind: str):
    """Original per-lane comparison loop.

    Retained for ``n > 2**24``, where fp32 can no longer hold a column index
    exactly and the vector path above would round.  No manifest shape reaches
    this, so it is correctness insurance rather than a performance path.
    """

    _BLOCK_N_SCALAR = 256
    m_tiles = (m + _BLOCK_M - 1) // _BLOCK_M
    n_tiles = (n + _BLOCK_N_SCALAR - 1) // _BLOCK_N_SCALAR
    launch_blocks = launch_block_count(m_tiles)
    grid_repeats = grid_repeat_count(m_tiles, launch_blocks)

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
            tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
            tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
        },
    )
    def factory():
        @T.prim_func
        def main(
            A: T.Tensor((m, n), dtype_name),
            C: T.Tensor((m_tiles * _BLOCK_M,), "int64"),
        ):
            with T.Kernel(launch_blocks, is_npu=True) as (cid, vid):
                values = T.alloc_ub((_SUB_M, _BLOCK_N_SCALAR), dtype_name)
                values_fp32 = T.alloc_ub((_SUB_M, _BLOCK_N_SCALAR), "float32")
                best_values = T.alloc_ub((_SUB_M,), "float32")
                best_indices = T.alloc_ub((_SUB_M,), "int64")

                with T.Scope("V"):
                    for grid_repeat in T.serial(grid_repeats):
                        logical_cid = cid + grid_repeat * launch_blocks
                        if logical_cid < m_tiles:
                            row_base = logical_cid * _BLOCK_M + vid * _SUB_M
                            for r in T.serial(_SUB_M):
                                if row_base + r < m:
                                    best_values[r] = (
                                        -T.infinity("float32")
                                        if op_kind == "argmax"
                                        else T.infinity("float32")
                                    )
                                    best_indices[r] = 0

                            for nt in T.serial(n_tiles):
                                for r in T.serial(_SUB_M):
                                    if row_base + r < m:
                                        T.copy(
                                            A[
                                                row_base + r,
                                                nt * _BLOCK_N_SCALAR : nt
                                                * _BLOCK_N_SCALAR
                                                + _BLOCK_N_SCALAR,
                                            ],
                                            values[r, :],
                                            pad_value=0,
                                        )
                                if dtype_name == "float32":
                                    T.copy(values, values_fp32)
                                else:
                                    T.tile.cast(
                                        values_fp32,
                                        values,
                                        "CAST_NONE",
                                        _SUB_M * _BLOCK_N_SCALAR,
                                    )

                                for r in T.serial(_SUB_M):
                                    if row_base + r < m:
                                        for j in T.serial(_BLOCK_N_SCALAR):
                                            global_index = nt * _BLOCK_N_SCALAR + j
                                            if global_index < n and (
                                                values_fp32[r, j] > best_values[r]
                                                if op_kind == "argmax"
                                                else values_fp32[r, j] < best_values[r]
                                            ):
                                                best_values[r] = values_fp32[r, j]
                                                best_indices[r] = global_index

                            T.copy(best_indices, C[row_base : row_base + _SUB_M])

        return main

    return factory()


def _compile_dispatch(m: int, n: int, dtype_name: str, op_kind: str):
    if n > _MAX_VECTOR_INDEX:
        return _compile_scalar_fallback(m, n, dtype_name, op_kind)
    return _compile(m, n, dtype_name, op_kind)


def _axes(dim, ndim: int) -> tuple[int, ...]:
    values = tuple(range(ndim)) if dim is None else (dim,)
    raw = tuple(int(axis) for axis in values)
    if any(axis < -ndim or axis >= ndim for axis in raw):
        raise ValueError(f"reduction dimension out of range for rank {ndim}: {dim!r}")
    axes = tuple(sorted(axis + ndim if axis < 0 else axis for axis in raw))
    if len(set(axes)) != len(axes):
        raise ValueError(f"reduction dimensions must be unique, got {dim!r}")
    return axes


def _build(input_shape, dtype, dim, keepdim, op_kind: str, op_name: str):
    axes = _axes(dim, len(input_shape))
    reduced = set(axes)
    output_shape = (
        tuple(1 if i in reduced else input_shape[i] for i in range(len(input_shape)))
        if keepdim
        else tuple(input_shape[i] for i in range(len(input_shape)) if i not in reduced)
    )
    m = math.prod(output_shape) if output_shape else 1
    n = math.prod(input_shape[i] for i in axes)
    if n <= 0:
        raise ValueError(f"{op_name} does not support an empty reduction dimension")
    order = tuple(i for i in range(len(input_shape)) if i not in reduced) + axes
    dtype_name = _dtype_name(dtype)
    if dim is not None and len(axes) != 1:
        raise ValueError(f"{op_name} accepts one reduction dimension or None")
    compiled = _compile_dispatch(m, n, dtype_name, op_kind)

    def launch(x):
        view = x if order == tuple(range(len(input_shape))) else x.permute(order)
        # A non-last reduction axis cannot be represented as a contiguous
        # [M, N] view on arbitrary ranks.  This mirrors the existing
        # TileOPs reduction template's fallback; the kernel itself still owns
        # the reduction and all output writes.
        return compiled(view.reshape(m, n).contiguous())[:m].reshape(output_shape)

    return launch


def build_argmax_kernel(input_shape, dtype, dim, keepdim):
    """Return a callable implementing the TileOPs ArgmaxFwdOp contract."""

    return _build(input_shape, dtype, dim, keepdim, "argmax", "ArgmaxFwdOp")


def build_argmin_kernel(input_shape, dtype, dim, keepdim):
    """Return a callable implementing the TileOPs ArgminFwdOp contract."""

    return _build(input_shape, dtype, dim, keepdim, "argmin", "ArgminFwdOp")


__all__ = ["build_argmax_kernel", "build_argmin_kernel"]
