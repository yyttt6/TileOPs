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

import tilelang
import tilelang.language as T
import torch

from .common import grid_repeat_count, launch_block_count
from .reduction import _full_reduction_rows


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
             predicate: bool = True):
    """One row-wise pass.

    ``predicate=True``  -> count nonzeros: compare/select to 0.0/1.0, then sum.
    ``predicate=False`` -> plain sum of an already-numeric input.  Stage 2 of
    the full-reduction split needs this: its input is a vector of per-row
    *counts*, and a row that legitimately counted zero must contribute 0 to the
    total, not be re-tested for nonzero-ness.
    """
    m_tiles = (m + _BLOCK_M - 1) // _BLOCK_M
    n_tiles = (n + _BLOCK_N - 1) // _BLOCK_N
    has_m_tail = m % _BLOCK_M != 0
    has_n_tail = n % _BLOCK_N != 0
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
            A: T.Tensor((m, n), dtype_name),
            C: T.Tensor((m_tiles * _BLOCK_M,), out_dtype),
        ):
            with T.Kernel(launch_blocks, is_npu=True) as (cid, vid):
                src = T.alloc_ub((_SUB_M, _BLOCK_N), dtype_name)
                calc = T.alloc_ub((_SUB_M, _BLOCK_N), "float32")
                # R198: replaces the old ``nonzero`` predicate tile.  select
                # needs its "true" operand to be a buffer, so this holds 1.0 and
                # is filled once per launch rather than per tile.
                ones = T.alloc_ub((_SUB_M, _BLOCK_N), "float32")
                mask = T.alloc_ub((_SUB_M * _BLOCK_N // 8,), "uint8")
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
                            for nt in T.serial(n_tiles):
                                T.tile.fill(src, 0)
                                if not has_m_tail:
                                    if not has_n_tail or nt < n_tiles - 1:
                                        T.copy(A[row_base, nt * _BLOCK_N], src)
                                    else:
                                        for r in T.serial(_SUB_M):
                                            for c in T.serial(_BLOCK_N):
                                                if c < n % _BLOCK_N:
                                                    src[r, c] = A[
                                                        row_base + r, nt * _BLOCK_N + c
                                                    ]
                                elif logical_cid < m_tiles - 1:
                                    if not has_n_tail or nt < n_tiles - 1:
                                        T.copy(A[row_base, nt * _BLOCK_N], src)
                                    else:
                                        for r in T.serial(_SUB_M):
                                            for c in T.serial(_BLOCK_N):
                                                if c < n % _BLOCK_N:
                                                    src[r, c] = A[
                                                        row_base + r, nt * _BLOCK_N + c
                                                    ]
                                else:
                                    for r in T.serial(_SUB_M):
                                        if row_base + r < m:
                                            if not has_n_tail or nt < n_tiles - 1:
                                                T.copy(
                                                    A[row_base + r, nt * _BLOCK_N],
                                                    src[r, :],
                                                )
                                            else:
                                                for c in T.serial(_BLOCK_N):
                                                    if c < n % _BLOCK_N:
                                                        src[r, c] = A[
                                                            row_base + r,
                                                            nt * _BLOCK_N + c,
                                                        ]

                                T.barrier_all()
                                if dtype_name == "float32":
                                    T.copy(src, calc)
                                else:
                                    T.tile.cast(
                                        calc, src, "CAST_NONE", _SUB_M * _BLOCK_N
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
    dtype_name = _dtype_name(dtype)

    # A full reduction is one block unless it is split; see the module
    # docstring for the 22.97 s that costs on the manifest shape.
    rows = _full_reduction_rows(n) if m == 1 else None
    if rows is None:
        compiled = _compile(m, n, dtype_name)
        two_stage = None
    else:
        stage1 = _compile(rows, n // rows, dtype_name, "float32", True)
        # Stage 2 sums `rows` partials in a single block; `rows` is at most a
        # few thousand, so this is microseconds against stage 1's milliseconds.
        stage2 = _compile(1, rows, "float32", "int64", False)
        compiled = stage1
        two_stage = (rows, stage1, stage2)

    def launch(x):
        view = x if order == tuple(range(len(input_shape))) else x.permute(order)
        flat = view.reshape(m, n).contiguous()
        if two_stage is None:
            return compiled(flat)[:m].reshape(output_shape)
        split, stage1, stage2 = two_stage
        partials = stage1(flat.reshape(split, n // split))[:split]
        return stage2(partials.reshape(1, split))[:1].reshape(output_shape)

    return launch


__all__ = ["build_count_nonzero_kernel"]
