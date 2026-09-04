"""Reusable Ascend vector template for unary elementwise operators.

The kernel is deliberately shape-specialized, just like the binary template:
the input is flattened only as a metadata view and every block owns a bounded
tile.  Both vector contexts participate in the address calculation.
"""

import math
from functools import lru_cache

import tilelang
import tilelang.language as T
import torch

from .common import grid_repeat_count, launch_block_count


FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
PREDICATE_DTYPES = (torch.uint8,)
INT_DTYPES = (
    torch.bool,
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
)
MAX_BLOCK_COUNT = 65535

# Same VECCALC watermark the binary template reuses: the lowering reserves
# 196352 bytes and the shipped geometry there proved 180224 of them safe.
UB_BUDGET_BYTES = 180224

# ``T.tile.compare`` emits a packed one-bit mask and dav-2201 has a 128-lane
# native repeat, so every tile must stay a whole number of mask blocks.  This
# is the *granularity*, not the size -- reading it as the size is what pinned
# the whole family at 128 elements per block (R194 / PROJECT_STATE 13.49).
MASK_GRAIN = 128

# R194 probe (exp, 16M elements, card 2): bandwidth stops improving past a
# 16384-element tile, and the 22528 point measured slightly slower.
TILE_HARD_CAP = 16384

# R194 probe: launch-block count is the dominant cost -- one block per tile
# (the shipped behaviour) spends ~0.11 us per block and caps exp at 15.7 GB/s,
# while a grid-stride loop over 48-192 blocks reaches 200-696 GB/s at the same
# tile.  48-192 is a plateau; 384 starts to fall off and 1536 collapses.
LAUNCH_BLOCK_CAP = 48

_ELEMENT_BYTES = {
    "float16": 2, "bfloat16": 2, "float32": 4,
    "int8": 1, "int16": 2, "int32": 4, "int64": 8,
    "uint8": 1, "bool": 1,
}

# Which scratch buffers each expression actually reads or writes.  Anything not
# listed here keeps a one-mask-block allocation so it does not consume tile
# budget; the branch that would touch it is not emitted for that op.  Derived
# by reading every ``op_kind`` branch in ``_compile_unary`` -- keep in sync.
_NEEDS_TMP2 = frozenset({"rsqrt", "floor", "ceil", "round", "trunc", "erf", "sign"})
_NEEDS_TMP3 = frozenset({"rsqrt", "floor", "ceil", "round", "trunc", "erf", "sign"})
_NEEDS_TMP4 = frozenset({"floor", "ceil", "round", "trunc", "erf"})
_NEEDS_I32 = frozenset({"floor", "ceil", "round", "trunc"})
_NEEDS_MASK = frozenset({
    "floor", "ceil", "round", "trunc", "erf", "sign",
    "isnan", "isinf", "isfinite", "logical_not",
})
_NEEDS_ZERO = frozenset({"neg", "erf"})
# A third packed mask, only for the rounding four: it holds "the int32 round
# trip produced zero", which is what the signed-zero repair selects on.
_NEEDS_MASK3 = _NEEDS_I32


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def _supported_message(
    op_name: str, dtype: torch.dtype, supported: tuple[torch.dtype, ...]
) -> str:
    names = ", ".join(str(item) for item in supported)
    return f"{op_name} does not support dtype {dtype}; supported dtypes are [{names}], but received input dtype {dtype}"


@lru_cache(maxsize=128)
def _compile_unary(n_total: int, input_dtype: str, output_dtype: str, op_kind: str):
    # Ascend vector transcendental instructions operate on fp32 reliably for
    # BF16.  Keep the source/destination buffers in their public dtypes and use
    # fp32 scratch for composed expressions and predicates.
    bool_logical = op_kind == "logical_not" and input_dtype == "uint8"
    fp32_unary = op_kind in {"floor", "ceil", "round", "trunc", "erf", "rsqrt"}
    compute_dtype = (
        "float32"
        if input_dtype == "bfloat16"
        or op_kind == "reciprocal"
        and input_dtype not in {"float16", "float32"}
        or bool_logical
        or fp32_unary
        else input_dtype
    )
    needs_compute_cast = compute_dtype != input_dtype
    is_predicate = output_dtype == "uint8"
    integral_threshold = {
        "float16": 2048.0,
        "bfloat16": 256.0,
        "float32": 8388608.0,
    }.get(input_dtype, 0.0)

    # --- Geometry (R194) -------------------------------------------------
    # Two things decide throughput here and the shipped template got both
    # wrong: the tile was pinned to the mask granularity instead of the UB
    # budget, and every block handled exactly one tile, so the block count
    # grew with the tensor and ~0.11 us of per-block dispatch dominated
    # everything.  Allocate the tile from the buffers this expression really
    # uses, then cover the tensor with a bounded grid-stride loop.
    needs_tmp2 = op_kind in _NEEDS_TMP2
    needs_tmp3 = op_kind in _NEEDS_TMP3
    needs_tmp4 = op_kind in _NEEDS_TMP4
    needs_i32 = op_kind in _NEEDS_I32
    needs_mask = op_kind in _NEEDS_MASK
    needs_zero = op_kind in _NEEDS_ZERO
    needs_mask3 = op_kind in _NEEDS_MASK3
    _compute_bytes = _ELEMENT_BYTES[compute_dtype]
    # Eighths of a byte per element, because the two packed masks cost 1/8
    # of a byte each and integer arithmetic must stay exact.
    live_eighths = 8 * (
        _ELEMENT_BYTES[input_dtype]
        + _ELEMENT_BYTES[output_dtype]
        + _compute_bytes                                  # x_calc
        + _compute_bytes                                  # tmp
        + (_compute_bytes if needs_tmp2 else 0)
        + (_compute_bytes if needs_tmp3 else 0)
        + (_compute_bytes if needs_tmp4 else 0)
        + (4 if needs_i32 else 0)
        + (_compute_bytes if needs_zero else 0)
    ) + (2 if needs_mask else 0) + (1 if needs_mask3 else 0)
    ub_cap = (UB_BUDGET_BYTES * 8) // live_eighths
    # Never widen past what the tensor needs: a tile larger than half the
    # tensor only lengthens the scalar tail loop, and small shapes are
    # already at parity with the vendor because both sides are launch-bound.
    needed = max(MASK_GRAIN, math.ceil(n_total / 2))
    ceiling = min(ub_cap, TILE_HARD_CAP, needed)
    ceiling = max(MASK_GRAIN, ceiling - ceiling % MASK_GRAIN)
    # Prefer a tile that divides the tensor.  A single ragged tile falls into
    # the per-lane scalar tail below, and that one tile is on the critical
    # path of the whole launch: measured, fp32 exp at 16M went 2.65x slower
    # (113 us -> 300 us) purely because 745 blocks x 22528 overshot 2**24 by
    # one partial tile, while the fp16 geometry divided exactly and paid
    # nothing (R194 / PROJECT_STATE 13.50).  Give up at most half the tile
    # width chasing a divisor -- below that the narrower DMA costs more than
    # the tail.
    tile = ceiling
    candidate = ceiling
    while candidate >= max(MASK_GRAIN, ceiling // 2):
        if n_total % (2 * candidate) == 0:
            tile = candidate
            break
        candidate -= MASK_GRAIN
    scratch2 = tile if needs_tmp2 else MASK_GRAIN
    scratch3 = tile if needs_tmp3 else MASK_GRAIN
    scratch4 = tile if needs_tmp4 else MASK_GRAIN
    scratch_i32 = tile if needs_i32 else MASK_GRAIN
    scratch_mask = tile if needs_mask else MASK_GRAIN
    scratch_mask3 = tile if needs_mask3 else MASK_GRAIN
    scratch_zero = tile if needs_zero else MASK_GRAIN
    block_total = tile * 2
    logical_blocks = max(1, math.ceil(n_total / block_total))
    launch_blocks = launch_block_count(min(logical_blocks, LAUNCH_BLOCK_CAP))
    grid_repeats = grid_repeat_count(logical_blocks, launch_blocks)

    @tilelang.jit(out_idx=[-1])
    def kernel():
        @T.prim_func
        def main(
            X: T.Tensor((n_total,), input_dtype),
            Y: T.Tensor((n_total,), output_dtype),
        ):
            with T.Kernel(launch_blocks, is_npu=True) as (cid, vid):
                x_ub = T.alloc_ub((tile,), input_dtype)
                x_calc = T.alloc_ub((tile,), compute_dtype)
                y_ub = T.alloc_ub((tile,), output_dtype)
                tmp = T.alloc_ub((tile,), compute_dtype)
                tmp2 = T.alloc_ub((scratch2,), compute_dtype)
                tmp3 = T.alloc_ub((scratch3,), compute_dtype)
                tmp4 = T.alloc_ub((scratch4,), compute_dtype)
                rounded_i32 = T.alloc_ub((scratch_i32,), "int32")
                # Compare emits a packed one-bit mask (8 lanes per byte).
                mask = T.alloc_ub((scratch_mask // 8,), "uint8")
                mask2 = T.alloc_ub((scratch_mask // 8,), "uint8")
                mask3 = T.alloc_ub((scratch_mask3 // 8,), "uint8")
                zero = T.alloc_ub((scratch_zero,), compute_dtype)

                with T.Scope("V"):
                    # A bounded grid-stride loop: ``launch_blocks`` blocks walk the
                    # tensor instead of one block per tile.  Per-block dispatch is
                    # ~0.11 us and used to be the whole cost (R194 / 13.49); a loop
                    # iteration is nearly free by comparison.
                    for grid_repeat in T.serial(grid_repeats):
                        logical_cid = cid + grid_repeat * launch_blocks
                        if logical_cid < logical_blocks:
                            start = logical_cid * block_total + vid * tile
                            full = start + tile <= n_total
                            if full:
                                T.copy(X[start], x_ub)
                            else:
                                if input_dtype in {"int64", "uint8"}:
                                    for lane in T.serial(tile):
                                        idx = start + lane
                                        if idx < n_total:
                                            x_ub[lane] = X[idx]
                                        else:
                                            x_ub[lane] = 0
                                else:
                                    T.tile.fill(x_ub, 0)
                                    for lane in T.serial(tile):
                                        idx = start + lane
                                        if idx < n_total:
                                            x_ub[lane] = X[idx]
                            T.barrier_all()

                            if op_kind == "bitwise_not" or bool_logical:
                                pass
                            elif needs_compute_cast:
                                T.tile.cast(x_calc, x_ub, "CAST_NONE", tile)
                            else:
                                # A UB assignment is a metadata-level alias in the
                                # frontend; the following operations overwrite only tmp
                                # buffers and never mutate the input tensor.
                                T.copy(x_ub, x_calc)

                            if needs_zero:
                                T.tile.fill(zero, 0.0)
                            if op_kind == "abs":
                                T.tile.abs(tmp, x_calc)
                            elif op_kind == "neg":
                                T.tile.sub(tmp, zero, x_calc)
                            elif op_kind == "exp":
                                T.tile.exp(tmp, x_calc)
                            elif op_kind == "log":
                                T.tile.ln(tmp, x_calc)
                            elif op_kind == "log1p":
                                T.tile.add(tmp, x_calc, 1.0)
                                T.tile.ln(tmp, tmp)
                            elif op_kind == "expm1":
                                T.tile.exp(tmp, x_calc)
                                T.tile.sub(tmp, tmp, 1.0)
                            elif op_kind == "sqrt":
                                T.tile.sqrt(tmp, x_calc)
                            elif op_kind == "rsqrt":
                                T.tile.rsqrt(tmp, x_calc)
                                # Ascend's vector rsqrt is a fast approximation.  Keep
                                # both Newton corrections in fp32 scratch before the
                                # public dtype cast.
                                for _ in T.serial(2):
                                    T.tile.mul(tmp2, tmp, tmp)
                                    T.tile.mul(tmp2, tmp2, x_calc)
                                    T.tile.mul(tmp2, tmp2, 0.5)
                                    T.tile.fill(tmp3, 1.5)
                                    T.tile.sub(tmp3, tmp3, tmp2)
                                    T.tile.mul(tmp, tmp, tmp3)
                            elif op_kind == "reciprocal":
                                T.tile.reciprocal(tmp, x_calc)
                            elif op_kind == "sin":
                                T.tile.sin(tmp, x_calc)
                            elif op_kind == "cos":
                                T.tile.cos(tmp, x_calc)
                            elif op_kind == "sigmoid":
                                T.tile.sigmoid(tmp, x_calc)
                            elif op_kind == "tanh":
                                T.tile.mul(tmp, x_calc, 2.0)
                                T.tile.sigmoid(tmp, tmp)
                                T.tile.mul(tmp, tmp, 2.0)
                                T.tile.sub(tmp, tmp, 1.0)
                            elif op_kind in {"floor", "ceil", "round", "trunc"}:
                                # The rounding mode is effective on float-to-int casts.
                                # Values outside this dtype-specific interval are already
                                # integral in the source format, so preserve them instead
                                # of risking integer overflow during the conversion.
                                #
                                # R194: the two selects below used to be
                                # ``T.serial(tile)`` lane loops over the packed
                                # masks, which made the whole expression scalar --
                                # 3.0 GB/s against a vendor baseline sitting on the
                                # HBM plateau.  That cost is proportional to the
                                # element count, so no tile or grid change moves it;
                                # only vectorizing the selects does.  Semantics are
                                # preserved exactly, signed-zero repair included.
                                T.tile.abs(tmp, x_calc)
                                T.tile.compare(mask, tmp, integral_threshold, "LT")
                                # Cast ``x_calc`` straight through.  The shipped code
                                # first parked out-of-range lanes at 0.0 to keep the
                                # int32 conversion from overflowing, but whatever the
                                # conversion produces for those lanes is discarded by
                                # the final select, so the clamp only cost a vector op.
                                # These expressions are vector-bound (msprof: floor
                                # aiv_vec_ratio 0.729), so op count is the lever.
                                if op_kind == "floor":
                                    T.tile.cast(rounded_i32, x_calc, "CAST_FLOOR", tile)
                                elif op_kind == "ceil":
                                    T.tile.cast(rounded_i32, x_calc, "CAST_CEIL", tile)
                                elif op_kind == "round":
                                    T.tile.cast(rounded_i32, x_calc, "CAST_RINT", tile)
                                else:
                                    T.tile.cast(rounded_i32, x_calc, "CAST_TRUNC", tile)
                                T.tile.cast(tmp3, rounded_i32, "CAST_NONE", tile)
                                # ``x * 0`` carries the sign of x, so a rounded
                                # result of zero recovers -0.0 where the int32
                                # round trip dropped the sign.
                                T.tile.mul(tmp4, x_calc, 0.0)
                                T.tile.compare(mask3, tmp3, 0.0, "EQ")
                                T.tile.select(
                                    tmp2, mask3, tmp4, tmp3, "VSEL_TENSOR_TENSOR_MODE"
                                )
                                # The shipped code also excluded ``|x| == 0`` from the
                                # rounded branch.  That exclusion is redundant given the
                                # signed-zero repair above: for x = +-0.0 the repair
                                # already yields +-0.0, so both branches agree.  Dropping
                                # it removes a compare and a mask AND (verified bitwise
                                # against the pre-change implementation).
                                T.tile.select(
                                    tmp, mask, tmp2, x_calc, "VSEL_TENSOR_TENSOR_MODE"
                                )
                            elif op_kind == "erf":
                                # Abramowitz and Stegun 7.1.26, evaluated in fp32:
                                # max approximation error <= 1.5e-7.
                                T.tile.abs(tmp, x_calc)
                                T.tile.mul(tmp2, tmp, 0.3275911)
                                T.tile.add(tmp2, tmp2, 1.0)
                                T.tile.reciprocal(tmp3, tmp2)
                                # The hardware reciprocal is an estimate. Two Newton
                                # steps recover fp32 accuracy before Horner evaluation.
                                T.tile.mul(tmp4, tmp2, tmp3)
                                T.tile.mul(tmp4, tmp4, -1.0)
                                T.tile.add(tmp4, tmp4, 2.0)
                                T.tile.mul(tmp3, tmp3, tmp4)
                                T.tile.mul(tmp4, tmp2, tmp3)
                                T.tile.mul(tmp4, tmp4, -1.0)
                                T.tile.add(tmp4, tmp4, 2.0)
                                T.tile.mul(tmp3, tmp3, tmp4)
                                T.tile.fill(tmp2, 1.061405429)
                                T.tile.mul(tmp2, tmp2, tmp3)
                                T.tile.add(tmp2, tmp2, -1.453152027)
                                T.tile.mul(tmp2, tmp2, tmp3)
                                T.tile.add(tmp2, tmp2, 1.421413741)
                                T.tile.mul(tmp2, tmp2, tmp3)
                                T.tile.add(tmp2, tmp2, -0.284496736)
                                T.tile.mul(tmp2, tmp2, tmp3)
                                T.tile.add(tmp2, tmp2, 0.254829592)
                                T.tile.mul(tmp2, tmp2, tmp3)
                                T.tile.mul(tmp4, tmp, tmp)
                                T.tile.sub(tmp4, zero, tmp4)
                                T.tile.exp(tmp4, tmp4)
                                T.tile.mul(tmp2, tmp2, tmp4)
                                T.tile.fill(tmp4, 1.0)
                                T.tile.sub(tmp2, tmp4, tmp2)
                                T.tile.sub(tmp3, zero, tmp2)
                                # R194: was two nested per-lane selects.  ``tmp4`` is
                                # dead here (it last held the 1.0 fill), so it can
                                # carry the sign-resolved value.
                                T.tile.compare(mask, x_calc, 0.0, "LT")
                                T.tile.select(
                                    tmp4, mask, tmp3, tmp2, "VSEL_TENSOR_TENSOR_MODE"
                                )
                                T.tile.compare(mask2, x_calc, 0.0, "EQ")
                                T.tile.select(
                                    tmp, mask2, x_calc, tmp4, "VSEL_TENSOR_TENSOR_MODE"
                                )
                            elif op_kind == "identity":
                                T.copy(x_ub, y_ub)
                            elif op_kind == "bitwise_not":
                                # AscendC's vector Not accepts only narrow integer storage
                                # on dav-2201.  Keep int32/int64 semantics in UB-local
                                # scalar expressions and still use a single GM copy per
                                # full tile.
                                if input_dtype in {"int32", "int64"}:
                                    for lane in T.serial(tile):
                                        y_ub[lane] = ~x_ub[lane]
                                else:
                                    T.tile.bitwise_not(y_ub, x_ub)
                            elif bool_logical:
                                for lane in T.serial(tile):
                                    y_ub[lane] = T.if_then_else(x_ub[lane] == 0, 1, 0)
                            elif op_kind == "logical_not":
                                T.tile.compare(mask, x_calc, 0.0, "EQ")
                                for lane in T.serial(tile):
                                    y_ub[lane] = (mask[lane // 8] >> (lane % 8)) & 1
                            elif op_kind == "isnan":
                                # NaN does not satisfy AscendC's ordinary NE predicate;
                                # derive it as ``not finite and not infinite``.
                                T.tile.abs(tmp, x_calc)
                                T.tile.compare(mask, tmp, T.infinity(compute_dtype), "LT")
                                T.tile.compare(mask2, tmp, T.infinity(compute_dtype), "EQ")
                                for lane in T.serial(tile):
                                    finite = (mask[lane // 8] >> (lane % 8)) & 1
                                    infinite = (mask2[lane // 8] >> (lane % 8)) & 1
                                    y_ub[lane] = T.if_then_else(
                                        finite == 0, T.if_then_else(infinite == 0, 1, 0), 0
                                    )
                            elif op_kind in {"isinf", "isfinite"}:
                                T.tile.abs(tmp, x_calc)
                                T.tile.compare(
                                    mask,
                                    tmp,
                                    T.infinity(compute_dtype),
                                    "EQ" if op_kind == "isinf" else "LT",
                                )
                                if op_kind == "isinf":
                                    for lane in T.serial(tile):
                                        y_ub[lane] = (mask[lane // 8] >> (lane % 8)) & 1
                                else:
                                    for lane in T.serial(tile):
                                        y_ub[lane] = (mask[lane // 8] >> (lane % 8)) & 1
                            elif op_kind == "sign":
                                # R194: was a per-lane loop building two 0/1 tiles.
                                # ``VSEL_TENSOR_SCALAR_MODE`` needs a Buffer for src0,
                                # so keep a ones tile in tmp3 and select against 0.0.
                                T.tile.compare(mask, x_calc, 0.0, "GT")
                                T.tile.compare(mask2, x_calc, 0.0, "LT")
                                T.tile.fill(tmp3, 1.0)
                                T.tile.select(
                                    tmp, mask, tmp3, 0.0, "VSEL_TENSOR_SCALAR_MODE"
                                )
                                T.tile.select(
                                    tmp2, mask2, tmp3, 0.0, "VSEL_TENSOR_SCALAR_MODE"
                                )
                                T.tile.sub(tmp, tmp, tmp2)
                            else:
                                raise ValueError(
                                    f"unsupported unary kernel operation {op_kind!r}"
                                )

                            if op_kind not in {
                                "identity",
                                "bitwise_not",
                                "logical_not",
                                "isnan",
                                "isinf",
                                "isfinite",
                            }:
                                if is_predicate:
                                    T.tile.cast(y_ub, tmp, "CAST_RINT", tile)
                                elif needs_compute_cast:
                                    T.tile.cast(y_ub, tmp, "CAST_RINT", tile)
                                else:
                                    T.copy(tmp, y_ub)
                            T.barrier_all()
                            if full:
                                T.copy(y_ub, Y[start])
                            else:
                                for lane in T.serial(tile):
                                    idx = start + lane
                                    if idx < n_total:
                                        Y[idx] = y_ub[lane]

        return main

    return kernel()


def build_unary_kernel(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    *,
    op_kind: str,
    supported_dtypes: tuple[torch.dtype, ...],
    output_dtype: torch.dtype | None = None,
    op_name: str,
):
    if dtype not in supported_dtypes:
        raise TypeError(_supported_message(op_name, dtype, supported_dtypes))
    if output_dtype is None:
        output_dtype = dtype
    if op_kind == "reciprocal" and dtype in INT_DTYPES:
        output_dtype = torch.float32
    n_total = math.prod(tuple(shape))
    kernel_input_dtype = torch.uint8 if dtype == torch.bool else dtype
    kernel_output_dtype = torch.uint8 if output_dtype == torch.bool else output_dtype
    compiled = _compile_unary(
        n_total,
        _dtype_name(kernel_input_dtype),
        _dtype_name(kernel_output_dtype),
        op_kind,
    )

    def invoke(input_tensor: torch.Tensor):
        if tuple(input_tensor.shape) != tuple(shape):
            raise ValueError(
                f"{op_name} kernel shape mismatch: expected input={tuple(shape)}, received input={tuple(input_tensor.shape)}"
            )
        flat = input_tensor if input_tensor.ndim == 1 else input_tensor.reshape(-1)
        if dtype == torch.bool:
            flat = flat.view(torch.uint8)
        # The kernel is declared with out_idx=[-1], so TileLang allocates and
        # returns the output. Passing an explicit tensor here would make it an
        # extra input, as documented by the out_idx contract.
        result = compiled(flat)
        if output_dtype == torch.bool:
            result = result.view(torch.bool)
        return result if tuple(shape) == (result.numel(),) else result.reshape(shape)

    return invoke
