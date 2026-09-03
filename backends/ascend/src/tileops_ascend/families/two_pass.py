"""Two-pass AIV softmax kernel for the Ascend backend."""

from functools import lru_cache
import math

import tilelang
import tilelang.language as T
import torch

from .._registry import register
from ..kernels.common import (
    grid_repeat_count,
    launch_block_count,
    plan_rowwise_norm,
)


_VEC = 2

#: ⚠️ ``_BLOCK_M``/``_BLOCK_N_MAX`` used to be 128 each, i.e. ``sub_m = 64`` rows
#: by 128 columns per vector lane, for every shape.  R200 replaced both with
#: ``kernels.common.plan_rowwise_norm``; see that function and ``_compile_norm``
#: for the measured reasons.  They are kept only as the fallback grain.
_BLOCK_M = 128
_BLOCK_N_MAX = 128


def _dtype_name(dtype: torch.dtype) -> str:
    if dtype is torch.float16:
        return "float16"
    if dtype is torch.bfloat16:
        return "bfloat16"
    if dtype is torch.float32:
        return "float32"
    raise TypeError(f"SoftmaxFwdOp supports floating dtypes, got {dtype}")


def _implicit_dim(ndim: int) -> int:
    # This mirrors TileOPs' _get_softmax_dim rule for dim=None.
    return 0 if ndim in (0, 1, 3) else 1


@lru_cache(maxsize=128)
def _compile_norm(m: int, n: int, dtype_name: str, kind: str, eps: float):
    """Compile the row-wise normalization subset of archetype 3.

    RMSNorm and LayerNorm both reduce a contiguous trailing row, then perform a
    second pass for the affine epilogue.  The output is padded internally so
    every GM write is a vector copy; the public builder slices it back.
    """
    if dtype_name not in {"float16", "bfloat16", "float32"}:
        raise TypeError(f"normalization supports floating dtypes, got {dtype_name}")
    itemsize = 2 if dtype_name in ("float16", "bfloat16") else 4
    # --- Geometry (R200) -------------------------------------------------
    # UB census read off the allocation list in ``factory`` below.  Getting one
    # entry wrong here is a silent UB overflow, so it is spelled out:
    #   (sub_m, block_n) : src[itemsize] + x32[4] + work[4] + bc[4]
    #   (block_n,)       : wub[itemsize] + bub[itemsize] + w32[4] + b32[4]
    #   (sub_m,)         : 18 fp32 scalars (stat stat2 count m2 row_sum row_sq
    #                      row_m2 tile_mean tile_count delta ratio new_count
    #                      correction mean denom refine inv + 1 spare)
    plan = plan_rowwise_norm(
        m, n, itemsize,
        bytes_per_cell=itemsize + 12,
        bytes_per_col=2 * itemsize + 8,
        bytes_per_row=18 * 4,
    )
    sub_m = plan["sub_m"]
    block_m = plan["block_m"]
    block_n = plan["block_n"]
    m_tiles = plan["m_tiles"]
    n_tiles = plan["n_tiles"]
    m_pad = plan["m_pad"]
    n_pad = plan["n_pad"]
    has_m_tail = plan["has_m_tail"]
    has_n_tail = plan["has_n_tail"]
    launch_blocks = plan["launch_blocks"]
    grid_repeats = plan["grid_repeats"]
    need_cast = dtype_name != "float32"
    # The Welford combine across n-tiles is provably the identity when there is
    # exactly one tile: ``count`` starts at 0, so ``ratio = tile_count /
    # tile_count = 1`` and the second correction term is ``0 * delta**2``.
    # Eliding it is bit-exact, not an approximation.
    single_n_tile = n_tiles == 1

    @tilelang.jit(
        out_idx=[1],
        pass_configs={
            tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
            tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
            tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
        },
    )
    def factory():
        @T.prim_func
        def main(
            A: T.Tensor([m, n], dtype_name),
            B: T.Tensor([m_pad, n_pad], dtype_name),
            W: T.Tensor([n], dtype_name),
            Bias: T.Tensor([n], dtype_name),
        ):
            with T.Kernel(launch_blocks, is_npu=True) as (cid, vid):
                src = T.alloc_ub((sub_m, block_n), dtype_name)
                x32 = T.alloc_ub((sub_m, block_n), "float32")
                work = T.alloc_ub((sub_m, block_n), "float32")
                stat = T.alloc_ub((sub_m,), "float32")
                stat2 = T.alloc_ub((sub_m,), "float32")
                # Welford state: count, mean and M2 are all kept in fp32.
                count = T.alloc_ub((sub_m,), "float32")
                m2 = T.alloc_ub((sub_m,), "float32")
                row_sum = T.alloc_ub((sub_m,), "float32")
                row_sq = T.alloc_ub((sub_m,), "float32")
                row_m2 = T.alloc_ub((sub_m,), "float32")
                tile_mean = T.alloc_ub((sub_m,), "float32")
                tile_count = T.alloc_ub((sub_m,), "float32")
                delta = T.alloc_ub((sub_m,), "float32")
                ratio = T.alloc_ub((sub_m,), "float32")
                new_count = T.alloc_ub((sub_m,), "float32")
                correction = T.alloc_ub((sub_m,), "float32")
                mean = T.alloc_ub((sub_m,), "float32")
                denom = T.alloc_ub((sub_m,), "float32")
                refine = T.alloc_ub((sub_m,), "float32")
                inv = T.alloc_ub((sub_m,), "float32")
                bc = T.alloc_ub((sub_m, block_n), "float32")
                wub = T.alloc_ub((block_n,), dtype_name)
                bub = T.alloc_ub((block_n,), dtype_name)
                w32 = T.alloc_ub((block_n,), "float32")
                b32 = T.alloc_ub((block_n,), "float32")

                with T.Scope("V"):
                    for grid_repeat in T.serial(grid_repeats):
                        logical_cid = cid + grid_repeat * launch_blocks
                        if logical_cid < m_tiles:
                            row_base = logical_cid * block_m + vid * sub_m
                            if kind == "rms":
                                T.tile.fill(stat2, 0.0)
                            else:
                                T.tile.fill(stat, 0.0)
                                T.tile.fill(count, 0.0)
                                T.tile.fill(m2, 0.0)
                            for nt in T.serial(n_tiles):
                                if has_n_tail:
                                    T.tile.fill(src, 0.0)
                                if not has_n_tail:
                                    if not has_m_tail:
                                        T.copy(A[row_base, nt * block_n], src)
                                    else:
                                        for r in T.serial(sub_m):
                                            if row_base + r < m:
                                                T.copy(A[row_base + r, nt * block_n], src[r, :])
                                elif nt < n_tiles - 1:
                                    if not has_m_tail:
                                        T.copy(A[row_base, nt * block_n], src)
                                    else:
                                        for r in T.serial(sub_m):
                                            if row_base + r < m:
                                                T.copy(A[row_base + r, nt * block_n], src[r, :])
                                else:
                                    for r in T.serial(sub_m):
                                        if row_base + r < m:
                                            for c in T.serial(block_n):
                                                if c < n - nt * block_n:
                                                    src[r, c] = A[row_base + r, nt * block_n + c]
                                if need_cast:
                                    T.tile.cast(x32, src, "CAST_NONE", sub_m * block_n)
                                else:
                                    T.copy(src, x32)
                                # --- Dead-code elimination by kind (R200 r2) ------------
                                # Both statistic paths used to run for BOTH kinds, but the
                                # ``denom`` selection below consumes only one of them:
                                #     rms   -> stat2 = sum(x^2)
                                #     layer -> m2 / count  (the Welford state)
                                # so each kind paid for the other's reduction in full, on
                                # every n-tile.  Counted in FULL-TILE vector ops
                                # (sub_m x block_n), per n-tile:
                                #     rms  was paying 5 dead  (reduce_sum -> row_sum,
                                #          broadcast, sub, mul, reduce_sum -> row_m2) plus
                                #          the ~14-op Welford chain, and keeping only 2;
                                #     layer was paying 2 dead (mul, reduce_sum -> row_sq),
                                #          and keeping 5.
                                # Safe in both directions: the three rms lines read x32
                                # BEFORE the layer branch re-centres it, and the only
                                # buffer they clobber is ``work``, which the layer branch
                                # overwrites with broadcast(tile_mean) before reading it.
                                # Dropping unreachable work is bit-exact by construction,
                                # so this does NOT re-open the R200 section 5 question.
                                if kind == "rms":
                                    T.tile.mul(work, x32, x32)
                                    T.reduce_sum(work, row_sq, dim=-1)
                                    T.tile.add(stat2, stat2, row_sq)
                                else:
                                    T.reduce_sum(x32, row_sum, dim=-1)
                                    if not has_n_tail or nt < n_tiles - 1:
                                        T.tile.fill(tile_count, T.cast(float(block_n), "float32"))
                                    else:
                                        T.tile.fill(
                                            tile_count,
                                            T.cast(float(n - (n_tiles - 1) * block_n), "float32"),
                                        )
                                    T.tile.div(tile_mean, row_sum, tile_count)
                                    T.tile.broadcast(work, tile_mean)
                                    T.tile.sub(x32, x32, work)
                                    # Padded columns must not contribute to M2.
                                    if has_n_tail and nt == n_tiles - 1:
                                        for r in T.serial(sub_m):
                                            for c in T.serial(block_n):
                                                if c >= n - (n_tiles - 1) * block_n:
                                                    x32[r, c] = 0.0
                                    T.tile.mul(work, x32, x32)
                                    T.reduce_sum(work, row_m2, dim=-1)
                                    if single_n_tile:
                                        # Bit-exact shortcut, see ``single_n_tile``.
                                        T.copy(tile_mean, stat)
                                        T.copy(row_m2, m2)
                                        T.copy(tile_count, count)
                                    else:
                                        T.tile.sub(delta, tile_mean, stat)
                                        T.tile.add(new_count, count, tile_count)
                                        T.tile.div(ratio, tile_count, new_count)
                                        T.tile.mul(correction, delta, ratio)
                                        T.tile.add(stat, stat, correction)
                                        T.tile.mul(correction, delta, delta)
                                        T.tile.mul(ratio, count, tile_count)
                                        T.tile.div(ratio, ratio, new_count)
                                        T.tile.mul(correction, correction, ratio)
                                        T.tile.add(row_m2, row_m2, correction)
                                        T.tile.add(m2, m2, row_m2)
                                        T.tile.add(count, count, tile_count)

                            inv_n = T.cast(1.0 / n, "float32")
                            if kind != "rms":
                                T.copy(stat, mean)
                            if kind == "rms":
                                # RMSNorm keeps its original E[x^2] path.  The
                                # Welford state is only needed by LayerNorm.
                                T.tile.mul(denom, stat2, inv_n)
                                T.tile.add(denom, denom, T.cast(eps, "float32"))
                            else:
                                T.tile.div(denom, m2, count)
                                T.tile.add(denom, denom, T.cast(eps, "float32"))
                            T.tile.rsqrt(inv, denom)
                            # Ascend's vector rsqrt is intentionally fast and
                            # approximate.  One Newton step restores fp32
                            # accuracy before the final dtype cast.
                            T.tile.mul(refine, inv, inv)
                            T.tile.mul(refine, refine, denom)
                            T.tile.mul(refine, refine, 0.5)
                            T.tile.fill(denom, 1.5)
                            T.tile.sub(denom, denom, refine)
                            T.tile.mul(inv, inv, denom)
                            T.tile.broadcast(bc, inv if kind == "rms" else inv)

                            for nt in T.serial(n_tiles):
                                if has_n_tail:
                                    T.tile.fill(src, 0.0)
                                if not has_n_tail or nt < n_tiles - 1:
                                    if not has_m_tail:
                                        T.copy(A[row_base, nt * block_n], src)
                                    else:
                                        for r in T.serial(sub_m):
                                            if row_base + r < m:
                                                T.copy(A[row_base + r, nt * block_n], src[r, :])
                                else:
                                    for r in T.serial(sub_m):
                                        if row_base + r < m:
                                            for c in T.serial(block_n):
                                                if c < n - nt * block_n:
                                                    src[r, c] = A[row_base + r, nt * block_n + c]
                                if need_cast:
                                    T.tile.cast(x32, src, "CAST_NONE", sub_m * block_n)
                                else:
                                    T.copy(src, x32)
                                if kind == "layer":
                                    T.tile.broadcast(work, mean)
                                    T.tile.sub(x32, x32, work)
                                T.tile.mul(x32, x32, bc)
                                # R200: this was a per-ELEMENT scalar GM read of
                                # the whole weight (and bias) vector, re-run by
                                # every task -- ``n * n_tasks`` scalar iterations,
                                # and the single dominant cost of this template.
                                # Differential on llama-405b-decode / n=16384:
                                # RMSNorm 1173.7 -> 542.4 us at unchanged
                                # geometry (2.16x), LayerNorm 1834.4 -> 652.5 us
                                # (2.81x, it loads two vectors).  See
                                # R200-data/probe_vecweight_decode.json.
                                if not has_n_tail or nt < n_tiles - 1:
                                    T.copy(W[nt * block_n], wub)
                                    if kind == "layer":
                                        T.copy(Bias[nt * block_n], bub)
                                else:
                                    # Only the ragged last tile keeps the scalar
                                    # path; a vector copy would read past W.
                                    for c in T.serial(block_n):
                                        if c < n - nt * block_n:
                                            wub[c] = W[nt * block_n + c]
                                            if kind == "layer":
                                                bub[c] = Bias[nt * block_n + c]
                                if need_cast:
                                    T.tile.cast(w32, wub, "CAST_NONE", block_n)
                                    T.tile.broadcast(work, w32)
                                    T.tile.mul(x32, x32, work)
                                    if kind == "layer":
                                        T.tile.cast(b32, bub, "CAST_NONE", block_n)
                                        T.tile.broadcast(work, b32)
                                        T.tile.add(x32, x32, work)
                                    T.tile.cast(src, x32, "CAST_RINT", sub_m * block_n)
                                    T.copy(src, B[row_base, nt * block_n])
                                else:
                                    T.tile.broadcast(work, wub)
                                    T.tile.mul(x32, x32, work)
                                    if kind == "layer":
                                        T.tile.broadcast(work, bub)
                                        T.tile.add(x32, x32, work)
                                    T.copy(x32, B[row_base, nt * block_n])
        return main

    return factory()


def _build_row_norm(x, weight, bias, *, eps, kind, op_name):
    if x is None or weight is None:
        raise ValueError(f"{op_name} requires tensor inputs")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"{op_name} supports floating dtypes, got {x.dtype}")
    x_shape = tuple(int(v) for v in x.shape)
    n = int(x_shape[-1])
    if tuple(weight.shape) != (n,):
        raise ValueError(f"{op_name} expects weight shape ({n},), got {tuple(weight.shape)}")
    if weight.dtype != x.dtype:
        raise ValueError(f"{op_name} weight must match input dtype")
    if kind == "layer":
        if bias is None or tuple(bias.shape) != (n,) or bias.dtype != x.dtype:
            raise ValueError(f"{op_name} bias must match weight shape and dtype")
    m = math.prod(x_shape) // n
    kernel = _compile_norm(m, n, _dtype_name(x.dtype), kind, float(eps))
    def launch(inp, scale=weight, shift=bias):
        out = kernel(inp.contiguous().reshape(m, n), scale.contiguous(), shift.contiguous() if shift is not None else scale.contiguous())
        return out[:m, :n].reshape(x_shape)

    return launch


@register("RMSNormFwdOp")
def build_rms_norm(x, weight, *, normalized_shape=None, eps=1e-6):
    if normalized_shape is not None and tuple(int(v) for v in normalized_shape) != (int(x.shape[-1]),):
        raise ValueError("RMSNormFwdOp currently supports one trailing normalized axis")
    return _build_row_norm(x, weight, None, eps=eps, kind="rms", op_name="RMSNormFwdOp")


@register("LayerNormFwdOp")
def build_layer_norm(x, weight, bias, *, normalized_shape=None, eps=1e-5):
    if normalized_shape is not None and tuple(int(v) for v in normalized_shape) != (int(x.shape[-1]),):
        raise ValueError("LayerNormFwdOp currently supports one trailing normalized axis")
    return _build_row_norm(x, weight, bias, eps=eps, kind="layer", op_name="LayerNormFwdOp")


def _unsupported(name):
    @register(name)
    def builder(*args, **kwargs):
        raise NotImplementedError(f"{name} is outside the verified row-wise two-pass template")
    return builder


# BatchNormBwdOp is owned by families.normalization_spatial.py.  Keeping its
# fail-closed stub here would register the same name twice during package import.


@lru_cache(maxsize=128)
def _compile(m: int, n: int, dtype_name: str, kind: str = "softmax"):
    # Small/non-divisible probe shapes get one exact-width tile. Production
    # workloads use the bounded 256-column tile and therefore exercise the
    # two-pass scan over many chunks.
    # Vector instructions require a 32-element aligned extent.  Keep small
    # probe shapes in one tile while padding only the UB extent; tail accesses
    # below are still guarded by the real ``n``.
    itemsize = 2 if dtype_name in ("float16", "bfloat16") else 4
    # --- Geometry (R200) -------------------------------------------------
    # UB census: (sub_m, block_n) -> src[itemsize] + x32[4] + max_2d[4]
    #                                + sum_2d[4];  (sub_m,) -> 6 fp32 scalars.
    # ``grain=32`` keeps the old "small probe shapes get one exact-width tile"
    # behaviour reachable; the vector extent requirement is 32 elements.
    plan = plan_rowwise_norm(
        m, n, itemsize,
        bytes_per_cell=itemsize + 12,
        bytes_per_col=0,
        bytes_per_row=6 * 4,
        grain=32 if n <= 512 else None,
    )
    sub_m = plan["sub_m"]
    block_m = plan["block_m"]
    block_n = plan["block_n"]
    m_tiles = plan["m_tiles"]
    n_tiles = plan["n_tiles"]
    m_pad = plan["m_pad"]
    n_pad = plan["n_pad"]
    has_m_tail = plan["has_m_tail"]
    has_n_tail = plan["has_n_tail"]
    launch_blocks = plan["launch_blocks"]
    grid_repeats = plan["grid_repeats"]
    calc_dtype = "float32"
    need_cast = dtype_name != calc_dtype
    pad_value = {
        "float16": -65504.0,
        "bfloat16": -3.38953139e38,
        "float32": -3.402823466e38,
    }[dtype_name]

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
            A: T.Tensor([m, n], dtype_name),
            B: T.Tensor([m_pad, n_pad], dtype_name),
        ):
            with T.Kernel(launch_blocks, is_npu=True) as (cid, vid):
                src = T.alloc_ub((sub_m, block_n), dtype_name)
                x32 = T.alloc_ub((sub_m, block_n), calc_dtype)
                tile_max = T.alloc_ub((sub_m,), calc_dtype)
                prev_max = T.alloc_ub((sub_m,), calc_dtype)
                tile_sum = T.alloc_ub((sub_m,), calc_dtype)
                prev_sum = T.alloc_ub((sub_m,), calc_dtype)
                tmp = T.alloc_ub((sub_m,), calc_dtype)
                max_2d = T.alloc_ub((sub_m, block_n), calc_dtype)
                sum_2d = T.alloc_ub((sub_m, block_n), calc_dtype)
                log_sum = T.alloc_ub((sub_m,), calc_dtype)

                with T.Scope("V"):
                    for grid_repeat in T.serial(grid_repeats):
                        logical_cid = cid + grid_repeat * launch_blocks
                        if logical_cid < m_tiles:
                            row_base = logical_cid * block_m + vid * sub_m
                            T.tile.fill(prev_max, -T.infinity(calc_dtype))
                            T.tile.fill(prev_sum, 0.0)

                            # Pass 1: online max and exp-sum. Padded columns are -inf,
                            # so they contribute zero without a host-side pad/copy.
                            for nt in T.serial(n_tiles):
                                if has_n_tail:
                                    T.tile.fill(src, pad_value)
                                if not has_n_tail:
                                    # R200: the m-tail predicate used to be
                                    # missing here, so a full sub_m-row vector
                                    # copy read up to sub_m-1 rows PAST the end
                                    # of A whenever n divided block_n but m did
                                    # not divide block_m (e.g. SoftmaxFwdOp
                                    # lm-head-logits, m=4 sub_m=64: 60 rows x
                                    # 102400 cols of out-of-bounds GM).
                                    if not has_m_tail:
                                        T.copy(A[row_base, nt * block_n], src)
                                    else:
                                        for r in T.serial(sub_m):
                                            if row_base + r < m:
                                                T.copy(A[row_base + r, nt * block_n], src[r, :])
                                elif nt < n_tiles - 1:
                                    if not has_m_tail:
                                        T.copy(A[row_base, nt * block_n], src)
                                    else:
                                        for r in T.serial(sub_m):
                                            if row_base + r < m:
                                                T.copy(A[row_base + r, nt * block_n], src[r, :])
                                else:
                                    for r in T.serial(sub_m):
                                        if row_base + r < m:
                                            for c in T.serial(block_n):
                                                if c < n - nt * block_n:
                                                    src[r, c] = A[row_base + r, nt * block_n + c]

                                if need_cast:
                                    T.tile.cast(x32, src, "CAST_NONE", sub_m * block_n)
                                else:
                                    T.copy(src, x32)
                                T.reduce_max(x32, tile_max, dim=-1)
                                T.tile.max(tile_max, prev_max, tile_max)
                                T.tile.sub(tmp, prev_max, tile_max)
                                T.tile.exp(tmp, tmp)
                                T.tile.mul(tmp, prev_sum, tmp)
                                T.tile.broadcast(max_2d, tile_max)
                                T.tile.sub(x32, x32, max_2d)
                                T.tile.exp(x32, x32)
                                T.reduce_sum(x32, tile_sum, dim=-1)
                                T.tile.add(prev_sum, tile_sum, tmp)
                                T.copy(tile_max, prev_max)

                            # Pass 2: normalize each input tile using the final row
                            # statistics. All output writes are vector T.copy operations.
                            T.tile.broadcast(max_2d, prev_max)
                            T.tile.broadcast(sum_2d, prev_sum)
                            for nt in T.serial(n_tiles):
                                if has_n_tail:
                                    T.tile.fill(src, pad_value)
                                if not has_n_tail:
                                    # R200: the m-tail predicate used to be
                                    # missing here, so a full sub_m-row vector
                                    # copy read up to sub_m-1 rows PAST the end
                                    # of A whenever n divided block_n but m did
                                    # not divide block_m (e.g. SoftmaxFwdOp
                                    # lm-head-logits, m=4 sub_m=64: 60 rows x
                                    # 102400 cols of out-of-bounds GM).
                                    if not has_m_tail:
                                        T.copy(A[row_base, nt * block_n], src)
                                    else:
                                        for r in T.serial(sub_m):
                                            if row_base + r < m:
                                                T.copy(A[row_base + r, nt * block_n], src[r, :])
                                elif nt < n_tiles - 1:
                                    if not has_m_tail:
                                        T.copy(A[row_base, nt * block_n], src)
                                    else:
                                        for r in T.serial(sub_m):
                                            if row_base + r < m:
                                                T.copy(A[row_base + r, nt * block_n], src[r, :])
                                else:
                                    for r in T.serial(sub_m):
                                        if row_base + r < m:
                                            for c in T.serial(block_n):
                                                if c < n - nt * block_n:
                                                    src[r, c] = A[row_base + r, nt * block_n + c]

                                if need_cast:
                                    T.tile.cast(x32, src, "CAST_NONE", sub_m * block_n)
                                else:
                                    T.copy(src, x32)
                                if kind == "log_softmax":
                                    T.tile.sub(x32, x32, max_2d)
                                    T.tile.ln(log_sum, prev_sum)
                                    T.tile.broadcast(sum_2d, log_sum)
                                    T.tile.sub(x32, x32, sum_2d)
                                else:
                                    T.tile.sub(x32, x32, max_2d)
                                    T.tile.exp(x32, x32)
                                    T.tile.div(x32, x32, sum_2d)
                                if need_cast:
                                    T.tile.cast(src, x32, "CAST_RINT", sub_m * block_n)
                                if need_cast:
                                    T.copy(src, B[row_base, nt * block_n])
                                else:
                                    T.copy(x32, B[row_base, nt * block_n])

        return main

    return factory()


def build_softmax(x, *, dim=None):
    """Build a callable preserving the public tensor shape and dtype."""
    if x is None:
        raise ValueError("SoftmaxFwdOp requires an input tensor")
    shape = tuple(int(v) for v in x.shape)
    ndim = len(shape)
    axis = _implicit_dim(ndim) if dim is None else int(dim)
    if axis < 0:
        axis += ndim
    if axis < 0 or axis >= ndim:
        raise ValueError(f"softmax dim {dim!r} out of range for rank {ndim}")
    order = tuple(i for i in range(ndim) if i != axis) + (axis,)
    inverse = tuple(order.index(i) for i in range(ndim))
    m = math.prod(shape[i] for i in range(ndim) if i != axis)
    n = shape[axis]
    kernel = _compile(m, n, _dtype_name(x.dtype), "softmax")

    def launch(inp):
        view = inp if order == tuple(range(ndim)) else inp.permute(order)
        result = kernel(view.reshape(m, n).contiguous())[:m, :n]
        result = result.reshape(tuple(shape[i] for i in order))
        return result if inverse == tuple(range(ndim)) else result.permute(inverse)

    return launch


def build_log_softmax(x, *, dim=None):
    """Build log-softmax with the same online-max two-pass traversal."""
    if x is None:
        raise ValueError("LogSoftmaxFwdOp requires an input tensor")
    shape = tuple(int(v) for v in x.shape)
    ndim = len(shape)
    axis = _implicit_dim(ndim) if dim is None else int(dim)
    if axis < 0:
        axis += ndim
    if axis < 0 or axis >= ndim:
        raise ValueError(f"log_softmax dim {dim!r} out of range for rank {ndim}")
    order = tuple(i for i in range(ndim) if i != axis) + (axis,)
    inverse = tuple(order.index(i) for i in range(ndim))
    m = math.prod(shape[i] for i in range(ndim) if i != axis)
    n = shape[axis]
    kernel = _compile(m, n, _dtype_name(x.dtype), "log_softmax")

    def launch(inp):
        view = inp if order == tuple(range(ndim)) else inp.permute(order)
        result = kernel(view.reshape(m, n).contiguous())[:m, :n]
        result = result.reshape(tuple(shape[i] for i in order))
        return result if inverse == tuple(range(ndim)) else result.permute(inverse)

    return launch


@register("SoftmaxFwdOp")
def build_softmax_registered(x, *, dim=None):
    return build_softmax(x, dim=dim)


@register("LogSoftmaxFwdOp")
def build_log_softmax_registered(x, *, dim=None):
    return build_log_softmax(x, dim=dim)


__all__ = ["build_log_softmax", "build_softmax", "build_softmax_registered"]
