"""Ascend AIV row-reduction kernel for ``torch.count_nonzero``.

The builder presents arbitrary reduction axes as a contiguous ``[M, N]``
view.  Each block owns 128 rows and the two vector contexts own disjoint
64-row halves.  Counts are accumulated in fp32 (the predicate is represented
as 0.0/1.0) and converted to int64 only at the output boundary.

R198: the predicate used to be built by a per-lane scalar loop
(``_SUB_M * _BLOCK_N`` = 16384 scalar iterations per input tile, on the main
path).  It is now ``compare`` into a packed mask plus ``select`` against a tile
of ones -- bit-identical, because ``select`` reproduces
``v != 0.0 ? 1.0 : 0.0`` exactly, including for NaN (IEEE ``NE`` is true for
NaN, so a NaN lane still counts as nonzero).  The tile of ones takes the slot
the separate predicate tile used to hold, but the packed mask is a genuinely
new allocation -- see the _BLOCK_N comment below for what that cost.

The padded lanes of a trailing tile need no special handling here: ``src`` is
zero-filled, 0 is the correct predicate input (it contributes 0 to the count)
*and* 0 is the identity of the sum that consumes it.  This is the easy case of
the R090 hazard -- unlike max (identity -inf) or product (identity 1.0).

R198 round 2: ``dim=None`` (three of the four manifest cases) collapses to
m == 1, and m == 1 gives ``m_tiles == 1`` -- ONE block, i.e. one vector context
out of 96, walking 32768 n-tiles.  Measured on the pre-R198 kernel, card 6
(``R198-data/probe/diag_count_nonzero.json``), execution time is exactly linear
in n and the manifest's own shape costs **22.97 s per call**:

    n=65536 -> 0.180 s   n=262144 -> 0.718 s   n=1048576 -> 2.871 s
    n=8388608 (manifest [2048,4096], dim=None) -> 22.968 s

That, times the harness's warmup+repeats over three ``dim=None`` cases, is the
40-minute ``exit=124`` this op kept hitting -- it was never hanging and it was
never a compile blowup (compile is a flat ~13 s at every size).  The fix is the
same two-stage decomposition ``reduction.py`` uses: view the flat run as
``(rows, n // rows)``, let the row factory count per row across many blocks,
then sum the partials.

The partial counts are carried in fp32 and that is exact, not approximate:
every partial is an integer <= n // rows and every running sum is an integer
<= n, so as long as n < 2**24 each addition lands on an exactly representable
value.  (The single-stage path accumulates in fp32 too, so n >= 2**24 is a
pre-existing limit, not one this decomposition introduces.)
"""

from functools import lru_cache
import math
import os

import tilelang
import tilelang.language as T
import torch

from .common import grid_repeat_count, launch_block_count
from .reduction import _MIN_SPLIT_K, _full_reduction_rows, _group_split


_BLOCK_M = 128
_VEC = 2
_SUB_M = _BLOCK_M // _VEC
# R198 round 2: 128, not 256.  The packed mask the vectorised predicate needs is
# a new UB allocation, and at _BLOCK_N = 256 the fp32 tile set was
#   src 65536 + calc 65536 + ones 65536 + mask 2048 = 198656 bytes
# against an arena of roughly 196352 -- the pre-R198 tile set was already at
# 196608, i.e. right on the edge, so the mask alone tipped it over and fp32
# died with a device-side crash (SIGSEGV, not a compile error).  Halving the
# tile puts fp32 at 99328 bytes.  fp32 is absent from this op's manifest, but
# _dtype_name accepts it, so a crashing path is not acceptable either way.
_BLOCK_N = 128


def _dtype_name(dtype: torch.dtype) -> str:
    names = {
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
        torch.float32: "float32",
    }
    try:
        return names[dtype]
    except KeyError as exc:
        raise TypeError(
            f"CountNonzeroFwdOp supports float16, bfloat16, and float32 inputs, got {dtype}"
        ) from exc


def _axes(dim, ndim: int) -> tuple[int, ...]:
    values = (
        tuple(range(ndim))
        if dim is None
        else ((dim,) if isinstance(dim, int) else tuple(dim))
    )
    axes = tuple(sorted((int(axis) + ndim) % ndim for axis in values))
    if len(set(axes)) != len(axes):
        raise ValueError(f"reduction dimensions must be unique, got {dim!r}")
    return axes


@lru_cache(maxsize=64)
def _compile(m: int, n: int, dtype_name: str, out_dtype: str = "int64",
             predicate: bool = True, groups: int = 1,
             block_n: int = _BLOCK_N):
    """One row-wise pass.

    ``predicate=True``  -> count nonzeros: compare/select to 0.0/1.0, then sum.
    ``predicate=False`` -> plain sum of an already-numeric input.  Stage 2 of
    the full-reduction split needs this: its input is a vector of per-row
    *counts*, and a row that legitimately counted zero must contribute 0 to the
    total, not be re-tested for nonzero-ness.
    """
    m_tiles = (m + _BLOCK_M - 1) // _BLOCK_M
    n_tiles = (n + block_n - 1) // block_n
    has_m_tail = m % _BLOCK_M != 0
    has_n_tail = n % block_n != 0
    launch_blocks = launch_block_count(m_tiles)
    grid_repeats = grid_repeat_count(m_tiles, launch_blocks)

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
            tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
            tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
        },
    )
    def factory():
        @T.prim_func
        def main(
            A: T.Tensor((groups * m, n), dtype_name),
            C: T.Tensor((m_tiles * _BLOCK_M,), out_dtype),
        ):
            with T.Kernel(launch_blocks, is_npu=True) as (cid, vid):
                src = T.alloc_ub((_SUB_M, block_n), dtype_name)
                calc = T.alloc_ub((_SUB_M, block_n), "float32")
                # R198: replaces the old ``nonzero`` predicate tile.  select
                # needs its "true" operand to be a buffer, so this holds 1.0 and
                # is filled once per launch rather than per tile.
                ones = T.alloc_ub((_SUB_M, block_n), "float32")
                mask = T.alloc_ub((_SUB_M * block_n // 8,), "uint8")
                partial = T.alloc_ub((_SUB_M,), "float32")
                accum = T.alloc_ub((_SUB_M,), "float32")
                out = T.alloc_ub((_SUB_M,), out_dtype)

                with T.Scope("V"):
                    T.tile.fill(ones, 1.0)
                    for grid_repeat in T.serial(grid_repeats):
                        logical_cid = cid + grid_repeat * launch_blocks
                        if logical_cid < m_tiles:
                            row_base = logical_cid * _BLOCK_M + vid * _SUB_M
                            T.tile.fill(accum, 0.0)
                            for g in T.serial(groups):
                                for nt in T.serial(n_tiles):
                                    T.tile.fill(src, 0)
                                    if not has_m_tail:
                                        if not has_n_tail or nt < n_tiles - 1:
                                            T.copy(A[g * m + row_base, nt * block_n], src)
                                        else:
                                            for r in T.serial(_SUB_M):
                                                for c in T.serial(block_n):
                                                    if c < n % block_n:
                                                        src[r, c] = A[
                                                            g * m + row_base + r, nt * block_n + c
                                                        ]
                                    elif logical_cid < m_tiles - 1:
                                        if not has_n_tail or nt < n_tiles - 1:
                                            T.copy(A[g * m + row_base, nt * block_n], src)
                                        else:
                                            for r in T.serial(_SUB_M):
                                                for c in T.serial(block_n):
                                                    if c < n % block_n:
                                                        src[r, c] = A[
                                                            g * m + row_base + r, nt * block_n + c
                                                        ]
                                    else:
                                        for r in T.serial(_SUB_M):
                                            if row_base + r < m:
                                                if not has_n_tail or nt < n_tiles - 1:
                                                    T.copy(
                                                        A[g * m + row_base + r, nt * block_n],
                                                        src[r, :],
                                                    )
                                                else:
                                                    for c in T.serial(block_n):
                                                        if c < n % block_n:
                                                            src[r, c] = A[
                                                                g * m + row_base + r,
                                                                nt * block_n + c,
                                                            ]

                                    T.barrier_all()
                                    if dtype_name == "float32":
                                        T.copy(src, calc)
                                    else:
                                        T.tile.cast(
                                            calc, src, "CAST_NONE", _SUB_M * block_n
                                        )
                                    if predicate:
                                        T.tile.compare(mask, calc, 0.0, "NE")
                                        T.tile.select(
                                            calc, mask, ones, 0.0,
                                            "VSEL_TENSOR_SCALAR_MODE",
                                        )
                                    T.reduce_sum(calc, partial, dim=-1, clear=True)
                                    T.barrier_all()
                                    T.tile.add(accum, accum, partial)

                            # V -> scalar / V -> MTE3.  ``accum`` was last
                            # written by T.tile.add, and the predicate that
                            # feeds it is now a vector select instead of the
                            # scalar loop that used to order these implicitly
                            # (PROJECT_STATE 13.55 #2).
                            T.barrier_all()
                            if out_dtype == "int64":
                                for r in T.serial(_SUB_M):
                                    out[r] = T.cast(accum[r], "int64")
                                T.copy(out, C[row_base : row_base + _SUB_M])
                            else:
                                # fp32 partial counts stay exact (see module
                                # docstring), so stage 2 can just add them up.
                                T.copy(accum, C[row_base : row_base + _SUB_M])

        return main

    return factory()


# --- R340: row split for m > 1 -----------------------------------------------
# `[4,128,4096] dim=[0,2]` gives m = 128 -> m_tiles = 1 -> blockDim = 1, i.e. the
# whole 2 M-element count ran on 2 of 96 vector contexts.  Measured alone at
# 115.0 us (R340-data/probe/p2-pristine-decompose.json) against 20.25 us for
# All/Any on the same geometry, which DO get `reduction._axis_split_factor`.
# The two-stage shape is the one already used for m == 1: stage 1 counts per
# row-chunk into fp32 partials, stage 2 SUMS those partials (`predicate=False`,
# because a chunk that legitimately counted zero must contribute 0, not be
# re-tested for nonzero-ness).
_ROW_SPLIT_MIN_BLOCKS = 8
# 16: swept, interior optimum.  k=8 -> 21.00 us, k=16 -> 14.75 us, k=32 -> 16.25 us
# on `[4,128,4096] dim=[0,2]` (R340-data/probe/p3-ksweep1.log).
_ROW_SPLIT_TARGET_BLOCKS = 16
_ROW_SPLIT_MAX_K = 128        # stage 2 tile is (_SUB_M, k) fp32 x3 -> 98304 B


def _row_split_factor(m: int, n: int) -> int | None:
    """Chunks to cut the per-group reduction axis into, or None to leave it."""
    if m <= 1 or n <= 0:
        return None
    if (m + _BLOCK_M - 1) // _BLOCK_M >= _ROW_SPLIT_MIN_BLOCKS:
        return None
    best = None
    k = _MIN_SPLIT_K          # k <= 4 is a WRONG ANSWER; see reduction.py
    k_cap = min(_ROW_SPLIT_MAX_K, n // _BLOCK_N)
    while k <= k_cap:
        if n % k == 0:
            best = k
            if (m * k + _BLOCK_M - 1) // _BLOCK_M >= _ROW_SPLIT_TARGET_BLOCKS:
                break
        k += 1
    return best


def build_count_nonzero_kernel(input_shape, dtype, dim):
    """Return an invocable CountNonzero kernel for a static input signature."""

    if not input_shape or any(int(size) <= 0 for size in input_shape):
        raise ValueError("CountNonzeroFwdOp requires a non-empty input shape")
    axes = _axes(dim, len(input_shape))
    reduced = set(axes)
    output_shape = tuple(
        input_shape[i] for i in range(len(input_shape)) if i not in reduced
    )
    m = math.prod(output_shape) if output_shape else 1
    n = math.prod(input_shape[i] for i in axes) if axes else 1
    order = tuple(i for i in range(len(input_shape)) if i not in reduced) + axes
    # R340: same leading-group split as reduction.py.  n becomes the per-group
    # reduction length; the true length is groups * n.  groups == 1 leaves the
    # old behaviour untouched.
    groups, n = _group_split(input_shape, reduced, order, n)
    dtype_name = _dtype_name(dtype)

    # A full reduction is one block unless it is split; see the module
    # docstring for the 22.97 s that costs on the manifest shape.
    rows = _full_reduction_rows(n) if m == 1 else None
    row_split = _row_split_factor(m, n) if rows is None else None
    if rows is None and row_split is None:
        compiled = _compile(m, n, dtype_name, groups=groups)
        two_stage = None
    elif rows is None:
        # R340: m > 1, starved of row blocks.  Split the per-group reduction
        # axis into `row_split` chunks; stage 1 then has m * k rows.
        k = row_split
        stage1 = _compile(m * k, n // k, dtype_name, "float32", True,
                          groups=groups)
        # block_n = k makes stage 2 exactly one tile with no trailing tile, i.e.
        # no per-lane scalar gather (the R198 trap that cost 6x elsewhere).
        stage2 = _compile(m, k, "float32", "int64", False, block_n=k)
        compiled = stage1
        two_stage = (k, stage1, stage2)
    else:
        stage1 = _compile(rows, n // rows, dtype_name, "float32", True)
        # Stage 2 sums `rows` partials in a single block; `rows` is at most a
        # few thousand, so this is microseconds against stage 1's milliseconds.
        stage2 = _compile(1, rows, "float32", "int64", False)
        compiled = stage1
        two_stage = (rows, stage1, stage2)

    def launch(x):
        if groups > 1:
            # R340: pure view; replaces a whole-tensor permute+contiguous copy.
            flat = x.reshape(groups * m, n)
        else:
            view = x if order == tuple(range(len(input_shape))) else x.permute(order)
            flat = view.reshape(m, n).contiguous()
        if two_stage is None:
            return compiled(flat)[:m].reshape(output_shape)
        split, stage1, stage2 = two_stage
        if row_split is not None:
            # (groups, m, k, n//k) -> (groups*m*k, n//k); a view.
            parts = stage1(flat.reshape(groups * m * split, n // split))
            return stage2(parts[: m * split].reshape(m, split))[:m].reshape(
                output_shape
            )
        partials = stage1(flat.reshape(split, n // split))[:split]
        return stage2(partials.reshape(1, split))[:1].reshape(output_shape)

    return launch


__all__ = ["build_count_nonzero_kernel"]
