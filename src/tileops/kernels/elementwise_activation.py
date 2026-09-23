"""Ascend vector kernels for the activation batch in T033.

The implementation keeps all arithmetic in UB and uses vector GM transfers for
full tiles.  The scalar loop is only used to express the activation formula in
UB; it never performs scalar GM stores.
"""

import math
from functools import lru_cache

import tilelang
import tilelang.language as T
import torch

from .common import grid_repeat_count, launch_block_count
from .elementwise_binary import (
    _broadcast_strides,
    _is_contiguous,
    _offset_expr,
    ub_fits,
)


FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

# --- Geometry (R195, recipe from R194 / PROJECT_STATE 13.50) --------------
# Same VECCALC watermark the binary and unary templates use: the lowering
# reserves 196352 bytes and the shipped binary geometry proved 180224 safe.
UB_BUDGET_BYTES = 180224

# ``T.tile.compare`` emits a packed one-bit mask and dav-2201 has a 128-lane
# native repeat, so a tile must stay a whole number of mask blocks.  This is a
# *granularity*, not a size.
MASK_GRAIN = 128

# R194 probe (card 2): bandwidth stops improving past a 16384-element tile.
TILE_HARD_CAP = 16384

# R194 probe: per-block dispatch is ~0.11 us and used to be the entire cost.
# 48-192 launch blocks is a plateau; one block per tile collapses to 1/13.
LAUNCH_BLOCK_CAP = 48

_ELEMENT_BYTES = {"float16": 2, "bfloat16": 2, "float32": 4}

# Which scratch buffers each expression actually reads or writes.  Derived by
# reading every ``op_kind`` branch in ``_compile_activation`` below -- keep in
# sync.  An ``op_kind`` that is *not* in ``_KNOWN_KINDS`` falls back to
# allocating everything at full width, so an unlisted branch can never index
# past a shrunken buffer.
_NEEDS_TMP = frozenset({
    "gelu_none", "gelu_tanh", "gelu_and_mul", "gelu_tanh_and_mul",
    "hardswish", "leaky_relu", "elu", "mish", "selu", "softplus", "lerp",
    "nan_to_num", "atan2",
})
_NEEDS_TMP2 = frozenset({"mish", "softplus", "nan_to_num", "atan2"})
_NEEDS_ZERO = frozenset({"mish", "softplus", "atan2"})
# Packed compare masks: the vectorised nan_to_num and the atan2 quadrant
# resolution.  T263: atan2 lives in this template rather than in
# ``elementwise_binary_batch`` precisely because the mask allocation here is
# already MASK_GRAIN-aligned and there are seven fp32 tiles to spend.
_NEEDS_MASK = frozenset({"nan_to_num", "atan2"})
_KNOWN_KINDS = frozenset({
    "relu", "silu", "silu_and_mul",
    "gelu_none", "gelu_tanh", "gelu_and_mul", "gelu_tanh_and_mul",
    "hardswish", "hardsigmoid", "hardtanh", "leaky_relu", "elu", "mish",
    "selu", "softplus", "clamp", "clamp_min", "clamp_max", "clamp_scalar",
    "lerp", "nan_to_num",
    # T263: relu6 is the fixed-bound hardtanh(0, 6); no extra scratch.
    "relu6", "atan2",
})

_GATED_KINDS = frozenset({"silu_and_mul", "gelu_and_mul", "gelu_tanh_and_mul"})
_GENERATED_KINDS = frozenset({"alibi", "sinusoidal"})


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def _shape_info(shapes: tuple[tuple[int, ...], ...]):
    out_shape = tuple(shapes[0])
    for shape in shapes[1:]:
        out_shape = tuple(torch.broadcast_shapes(out_shape, tuple(shape)))
    strides = tuple(_broadcast_strides(shape, out_shape) for shape in shapes)
    return out_shape, strides


def _round_down_grain(value: int) -> int:
    return max(MASK_GRAIN, int(value) - int(value) % MASK_GRAIN)


def _round_up_grain(value: int) -> int:
    value = max(1, int(value))
    return ((value + MASK_GRAIN - 1) // MASK_GRAIN) * MASK_GRAIN


def _largest_divisor_at_most(value: int, ceiling: int) -> int:
    """Largest divisor of ``value`` that is <= ``ceiling`` (>= 1 always exists)."""
    value = max(1, int(value))
    ceiling = max(1, int(ceiling))
    if ceiling >= value:
        return value
    best = 1
    limit = int(math.isqrt(value))
    for d in range(1, limit + 1):
        if value % d:
            continue
        if d <= ceiling and d > best:
            best = d
        other = value // d
        if other <= ceiling and other > best:
            best = other
    return best


def _scalar_path_tile(out_numel: int, ceiling: int) -> int:
    """Tile width for a per-lane gather -- the *opposite* rule from the vector path.

    When the load is ``for lane in T.serial(tile)`` the cost of one vector
    context is ``tile`` iterations no matter how many of those lanes are live,
    so a wide tile does not buy a wider DMA, it just lengthens the critical
    path.  What matters is spreading the lanes over as many contexts as the
    launch plateau allows, i.e. the narrowest tile that still keeps the grid
    at one pass.

    Measured (R195-data/probe_gated_graph.log, graph-replay regime, one
    process, ``silu_and_mul`` on the 26 KB ``tail-fp16`` shape whose prime gate
    width forces this path):

        tile 2176 (UB rule)  100.1 us
        tile 2048 (shipped)   89.4 us
        tile  256             64.0 us
        tile  128             63.3 us     <- what this function returns

    All five variants are bit-identical.  Note the effect is invisible in the
    eager regime -- its launch floor on this shape is ~160 us and swallows the
    whole 40 us spread (R195-data/probe_gated_tail.log).
    """
    want = math.ceil(out_numel / (2 * LAUNCH_BLOCK_CAP))
    want = max(MASK_GRAIN, math.ceil(want / MASK_GRAIN) * MASK_GRAIN)
    return max(MASK_GRAIN, min(ceiling, want))


def _pick_tile(ceiling: int, out_numel: int, gate_width: int | None):
    """Return ``(tile, gated_vector_load)`` for one UB ceiling.

    Two separate reasons to give up tile width, both worth far more than the
    width itself (R194 2.3 measured 2.65x for the first one alone):

    * a tile that does not divide the tensor leaves one ragged tile, and that
      tile falls into the per-lane scalar tail on the critical path;
    * for the gated ops the source rows are twice as wide as the output rows,
      so a tile that divides the gate width keeps every tile inside one row
      and lets the gather become two ordinary ``T.copy`` calls instead of a
      per-lane loop.

    Give up at most half the width chasing either -- below that the narrower
    DMA costs more than what it buys.

    🚨 R274 step 1: "at most half the width" was the wrong stopping rule for the
    GATED ops, because what is on the other side of it is not a narrower DMA, it
    is the ``for lane in T.serial(tile)`` per-lane gather -- the same failure
    class R263 measured on the broadcast entry point (30-150x) and R269 measured
    on conv's im2col (113-135 cycle per scalar GM read).  A gate width with no
    grain-aligned divisor (the ``tail-fp16`` probe's ``gate_width = 257`` is
    prime) fell off the end of this loop into that gather.  So when the aligned
    search fails, take the largest *unaligned* divisor instead: the tile is then
    still wholly inside one output row, the two loads are still ordinary
    ``T.copy`` calls, and the only thing given up is that the tile is no longer a
    multiple of ``MASK_GRAIN`` -- which costs nothing here because no gated kind
    allocates a packed compare mask (``_NEEDS_MASK``), and the buffers are padded
    up to the grain by ``_round_up_grain`` so a vector op that rounds its extent
    to a whole 32-byte block cannot walk into the next buffer.
    """
    ceiling = _round_down_grain(ceiling)
    floor = max(MASK_GRAIN, ceiling // 2)
    if gate_width is not None:
        candidate = ceiling
        while candidate >= floor:
            if gate_width % candidate == 0:
                return candidate, True
            candidate -= MASK_GRAIN
        unaligned = _largest_divisor_at_most(gate_width, ceiling)
        if unaligned > 1:
            return unaligned, True
    candidate = ceiling
    while candidate >= floor:
        if out_numel % (2 * candidate) == 0:
            return candidate, False
        candidate -= MASK_GRAIN
    return ceiling, False


@lru_cache(maxsize=256)
def _compile_activation(
    input_numels: tuple[int, ...],
    out_shape: tuple[int, ...],
    strides: tuple[tuple[int, ...], ...],
    dtype: str,
    op_kind: str,
    params: tuple[float | None, ...],
):
    out_numel = math.prod(out_shape)
    input_count = len(input_numels)
    padded_numels = tuple(input_numels) + (1,) * (3 - input_count)
    a_numel, b_numel, c_numel = padded_numels
    # Composed activations must not round intermediate values to fp16/bf16.
    # Keep the complete expression in fp32 and cast exactly once at writeback.
    compute_dtype = "float32"
    params = tuple(params) + (0.0,) * (3 - len(params))
    generated_seq = max(1, int(params[0]))
    generated_dim = max(2, int(params[1]))
    all_contiguous = all(_is_contiguous(out_shape, item) for item in strides)

    gated = op_kind in _GATED_KINDS
    generated = op_kind in _GENERATED_KINDS
    gate_width = int(out_shape[-1]) if gated else None

    # --- Live UB accounting (R195) ---------------------------------------
    # The shipped template pinned ``native_tile = 2048`` and allocated all
    # eleven buffers at that width for every op.  Both halves were wrong: the
    # tile is a UB-budget question, and most ops touch four of the eleven.
    # Anything outside ``_KNOWN_KINDS`` keeps every buffer at full width, so
    # an unlisted branch can never index past a shrunken allocation.
    conservative = op_kind not in _KNOWN_KINDS
    needs_b = conservative or input_count > 1 or gated
    # T263: atan2 passes two tensors but its expression needs a third fp32
    # tile, so it opts into the ``c`` allocation as scratch.
    needs_c = conservative or input_count > 2 or op_kind == "atan2"
    needs_tmp = conservative or op_kind in _NEEDS_TMP
    needs_tmp2 = conservative or op_kind in _NEEDS_TMP2
    needs_zero = conservative or op_kind in _NEEDS_ZERO
    needs_mask = conservative or op_kind in _NEEDS_MASK

    element_bytes = _ELEMENT_BYTES[dtype]
    # Eighths of a byte per element: a packed compare mask costs 1/8 byte.
    live_eighths = 8 * (
        element_bytes                                   # x_ub
        + element_bytes                                 # y_ub
        + 4                                             # x32
        + 4                                             # y32
        + ((element_bytes + 4) if needs_b else 0)       # b_ub + b32
        + ((element_bytes + 4) if needs_c else 0)       # c_ub + c32
        + (4 if needs_zero else 0)
        + (4 if needs_tmp else 0)
        + (4 if needs_tmp2 else 0)
    ) + (2 if needs_mask else 0)                        # mask + mask2
    ub_cap = (UB_BUDGET_BYTES * 8) // live_eighths
    # Never widen past what the tensor needs: a tile wider than half the
    # tensor only lengthens the scalar tail, and the small shapes are already
    # at parity because both sides are launch-bound.
    needed = max(MASK_GRAIN, math.ceil(out_numel / 2))

    # ``vectorized`` selects the per-lane fallback below.  ``nan_to_num`` used
    # to live here and measured 9.6 GB/s against a vendor baseline at 890
    # GB/s; it is expressed with compares and selects now (R195 step 2).
    vectorized = op_kind not in {"alibi", "sinusoidal", "where"}

    def build(cap: int):
        ceiling = min(cap, TILE_HARD_CAP, needed)
        tile, gated_vector = _pick_tile(ceiling, out_numel, gate_width)
        # Which load path this signature will actually take.  Everything that
        # is not a contiguous whole-tile ``T.copy`` drops into the per-lane
        # gather below, and that path wants a narrow tile, not a wide one.
        scalar_load = (
            generated
            or (gated and not gated_vector)
            or (not gated and not all_contiguous)
        )
        if scalar_load:
            tile = _scalar_path_tile(out_numel, _round_down_grain(ceiling))
        # R274: ``tile`` is the number of LIVE lanes; ``span`` is how wide every
        # buffer is actually allocated.  They differ only on the unaligned gated
        # divisor introduced in ``_pick_tile`` -- everywhere else ``tile`` is
        # already a multiple of MASK_GRAIN and ``span == tile``, so the emitted
        # artifact for every previously-compiling signature is byte-identical.
        # The padding is what makes the unaligned tile safe: a ``T.tile.*`` call
        # addresses whole 32-byte blocks, so an operand of 257 fp32 lanes is
        # executed over 264, and without the pad those 7 lanes are the next
        # buffer.
        span = _round_up_grain(tile)
        scratch_b = span if needs_b else MASK_GRAIN
        scratch_c = span if needs_c else MASK_GRAIN
        scratch_zero = span if needs_zero else MASK_GRAIN
        scratch_tmp = span if needs_tmp else MASK_GRAIN
        scratch_tmp2 = span if needs_tmp2 else MASK_GRAIN
        scratch_mask = span if needs_mask else MASK_GRAIN
        block_total = tile * 2
        logical_blocks = max(1, math.ceil(out_numel / block_total))
        launch_blocks = launch_block_count(min(logical_blocks, LAUNCH_BLOCK_CAP))
        grid_repeats = grid_repeat_count(logical_blocks, launch_blocks)

        @tilelang.jit(out_idx=[-1])
        def kernel():
            @T.prim_func
            def main(
                A: T.Tensor((a_numel,), dtype),
                B: T.Tensor((b_numel,), dtype),
                C: T.Tensor((c_numel,), dtype),
                Y: T.Tensor((out_numel,), dtype),
            ):
                with T.Kernel(launch_blocks, is_npu=True) as (cid, vid):
                    x_ub = T.alloc_ub((span,), dtype)
                    y_ub = T.alloc_ub((span,), dtype)
                    x32 = T.alloc_ub((span,), compute_dtype)
                    y32 = T.alloc_ub((span,), compute_dtype)
                    b_ub = T.alloc_ub((scratch_b,), dtype)
                    c_ub = T.alloc_ub((scratch_c,), dtype)
                    b32 = T.alloc_ub((scratch_b,), compute_dtype)
                    c32 = T.alloc_ub((scratch_c,), compute_dtype)
                    zero = T.alloc_ub((scratch_zero,), compute_dtype)
                    tmp = T.alloc_ub((scratch_tmp,), compute_dtype)
                    tmp2 = T.alloc_ub((scratch_tmp2,), compute_dtype)
                    # Compare emits a packed one-bit mask (8 lanes per byte).
                    mask = T.alloc_ub((scratch_mask // 8,), "uint8")
                    mask2 = T.alloc_ub((scratch_mask // 8,), "uint8")

                    if needs_zero:
                        # Loop invariant: nothing below writes ``zero``.
                        T.tile.fill(zero, 0.0)

                    # A bounded grid-stride loop: ``launch_blocks`` blocks walk
                    # the tensor instead of one block per tile.  Per-block
                    # dispatch is ~0.11 us and used to be the dominant cost
                    # (R194 / PROJECT_STATE 13.49-13.50).
                    for grid_repeat in T.serial(grid_repeats):
                        logical_cid = cid + grid_repeat * launch_blocks
                        if logical_cid < logical_blocks:
                            start = logical_cid * block_total + vid * tile
                            full = start + tile <= out_numel

                            if gated and gated_vector:
                                # ``tile`` divides the gate width, so this tile
                                # sits inside one output row and the gather is
                                # two ordinary contiguous copies instead of a
                                # per-lane loop.  ``out_numel`` is then a whole
                                # multiple of ``tile``, so a tile is either
                                # entirely live or entirely past the end.
                                if full:
                                    gate_src = (start // gate_width) * (
                                        gate_width * 2
                                    ) + start % gate_width
                                    T.copy(
                                        A[gate_src : gate_src + tile],
                                        x_ub if span == tile else x_ub[0:tile],
                                    )
                                    T.copy(
                                        A[
                                            gate_src
                                            + gate_width : gate_src
                                            + gate_width
                                            + tile
                                        ],
                                        b_ub if span == tile else b_ub[0:tile],
                                    )
                                else:
                                    T.tile.fill(x_ub, 0)
                                    T.tile.fill(b_ub, 0)
                            elif not gated and not generated and full and all_contiguous:
                                T.copy(A[start : start + tile], x_ub)
                                if input_count > 1:
                                    T.copy(B[start : start + tile], b_ub)
                                if input_count > 2:
                                    T.copy(C[start : start + tile], c_ub)
                            else:
                                T.tile.fill(x_ub, 0)
                                if input_count > 1 or gated:
                                    T.tile.fill(b_ub, 0)
                                if input_count > 2:
                                    T.tile.fill(c_ub, 0)
                                for lane in T.serial(tile):
                                    idx = start + lane
                                    if idx < out_numel:
                                        if gated:
                                            row = idx // gate_width
                                            col = idx % gate_width
                                            input_idx = row * (gate_width * 2) + col
                                            x_ub[lane] = A[input_idx]
                                            b_ub[lane] = A[input_idx + gate_width]
                                        else:
                                            x_ub[lane] = A[
                                                _offset_expr(idx, out_shape, strides[0])
                                            ]
                                            if input_count > 1:
                                                b_ub[lane] = B[
                                                    _offset_expr(
                                                        idx, out_shape, strides[1]
                                                    )
                                                ]
                                            if input_count > 2:
                                                c_ub[lane] = C[
                                                    _offset_expr(
                                                        idx, out_shape, strides[2]
                                                    )
                                                ]
                            T.barrier_all()

                            if dtype != "float32":
                                T.tile.cast(x32, x_ub, "CAST_NONE", tile)
                                if input_count > 1 or gated:
                                    T.tile.cast(b32, b_ub, "CAST_NONE", tile)
                                if input_count > 2:
                                    T.tile.cast(c32, c_ub, "CAST_NONE", tile)
                            else:
                                T.copy(x_ub, x32)
                                # R195: the shipped code tested ``input_count > 1``
                                # here while the cast branch above tests
                                # ``input_count > 1 or gated``.  A gated op passes
                                # one tensor, so fp32 gated left ``b32``
                                # uninitialised and multiplied by allocator
                                # contents.  No manifest case is fp32 gated, so
                                # this never showed up in coverage.
                                if input_count > 1 or gated:
                                    T.copy(b_ub, b32)
                                if input_count > 2:
                                    T.copy(c_ub, c32)

                            if op_kind == "relu":
                                T.tile.max(y32, x32, 0.0)
                            elif op_kind in {
                                "gelu_none",
                                "gelu_tanh",
                                "gelu_and_mul",
                                "gelu_tanh_and_mul",
                            }:
                                # tanh(z) = 2 * sigmoid(2*z) - 1.  The tanh GELU
                                # approximation stays in fp32 until final
                                # writeback.  It also implements gelu_none on
                                # dav-2201, whose vector backend has no erf
                                # intrinsic; the error against the TileOPs
                                # Torch-NPU contract is verified below 1e-3.
                                T.tile.mul(tmp, x32, x32)
                                T.tile.mul(tmp, tmp, x32)
                                T.tile.mul(tmp, tmp, 0.044715)
                                T.tile.add(tmp, tmp, x32)
                                T.tile.mul(tmp, tmp, 1.5957691216057308)
                                T.tile.sigmoid(tmp, tmp)
                                T.tile.mul(tmp, tmp, 2.0)
                                T.tile.sub(tmp, tmp, 1.0)
                                T.tile.add(tmp, tmp, 1.0)
                                T.tile.mul(y32, x32, tmp)
                                T.tile.mul(y32, y32, 0.5)
                                if op_kind in {"gelu_and_mul", "gelu_tanh_and_mul"}:
                                    T.tile.mul(y32, y32, b32)
                            elif op_kind in {"silu", "silu_and_mul"}:
                                T.tile.sigmoid(y32, x32)
                                T.tile.mul(y32, y32, x32)
                                if op_kind == "silu_and_mul":
                                    T.tile.mul(y32, y32, b32)
                            elif op_kind == "hardswish":
                                T.tile.add(tmp, x32, 3.0)
                                T.tile.max(tmp, tmp, 0.0)
                                T.tile.min(tmp, tmp, 6.0)
                                T.tile.mul(y32, x32, tmp)
                                T.tile.mul(y32, y32, 0.16666666666666666)
                            elif op_kind == "hardsigmoid":
                                T.tile.add(y32, x32, 3.0)
                                T.tile.max(y32, y32, 0.0)
                                T.tile.min(y32, y32, 6.0)
                                T.tile.mul(y32, y32, 0.16666666666666666)
                            elif op_kind == "atan2":
                                # atan2(num, den).  ``x32`` is the numerator
                                # (manifest ``input``) and ``b32`` the
                                # denominator (``other``), matching
                                # ``torch.atan2(input, other)``.
                                #
                                # Three stages, all whole-tile vector ops:
                                #   1. q = num / den, then atan(|q|) with the
                                #      same [0, 1] range reduction and
                                #      degree-6-in-z^2 minimax polynomial the
                                #      unary ``atan`` uses (|err| <= 5.8e-7).
                                #   2. restore the sign of q.
                                #   3. add +-pi where the denominator is
                                #      negative, sign taken from the numerator.
                                # ``den == 0`` needs no special case: q becomes
                                # +-inf, 1/|q| becomes 0, and the reduced branch
                                # returns exactly +-pi/2.  ``num == den == 0``
                                # yields NaN here where torch returns 0 -- see
                                # docs/reports/R263.md 4.
                                T.tile.div(tmp, x32, b32)
                                T.tile.abs(tmp2, tmp)
                                T.tile.compare(mask, tmp2, 1.0, "GT")
                                T.tile.fill(c32, 1.0)
                                T.tile.div(zero, c32, tmp2)
                                T.tile.select(
                                    c32, mask, zero, tmp2, "VSEL_TENSOR_TENSOR_MODE"
                                )
                                T.tile.mul(zero, c32, c32)
                                T.tile.fill(tmp2, 0.008006898229)
                                T.tile.mul(tmp2, tmp2, zero)
                                T.tile.add(tmp2, tmp2, -0.03744297094)
                                T.tile.mul(tmp2, tmp2, zero)
                                T.tile.add(tmp2, tmp2, 0.08435407574)
                                T.tile.mul(tmp2, tmp2, zero)
                                T.tile.add(tmp2, tmp2, -0.1351217486)
                                T.tile.mul(tmp2, tmp2, zero)
                                T.tile.add(tmp2, tmp2, 0.198873209)
                                T.tile.mul(tmp2, tmp2, zero)
                                T.tile.add(tmp2, tmp2, -0.3332701333)
                                T.tile.mul(tmp2, tmp2, zero)
                                T.tile.add(tmp2, tmp2, 0.9999994166)
                                T.tile.mul(tmp2, tmp2, c32)
                                T.tile.fill(zero, 1.5707963267948966)
                                T.tile.sub(zero, zero, tmp2)
                                T.tile.select(
                                    c32, mask, zero, tmp2, "VSEL_TENSOR_TENSOR_MODE"
                                )
                                T.tile.compare(mask, tmp, 0.0, "LT")
                                T.tile.mul(zero, c32, -1.0)
                                T.tile.select(
                                    tmp2, mask, zero, c32, "VSEL_TENSOR_TENSOR_MODE"
                                )
                                T.tile.compare(mask, x32, 0.0, "LT")
                                T.tile.fill(zero, 3.141592653589793)
                                T.tile.mul(c32, zero, -1.0)
                                T.tile.select(
                                    tmp, mask, c32, zero, "VSEL_TENSOR_TENSOR_MODE"
                                )
                                T.tile.add(zero, tmp2, tmp)
                                T.tile.compare(mask2, b32, 0.0, "LT")
                                T.tile.select(
                                    y32, mask2, zero, tmp2, "VSEL_TENSOR_TENSOR_MODE"
                                )
                            elif op_kind == "relu6":
                                # relu6(x) = min(max(x, 0), 6): the same two
                                # vector ops hardtanh below emits, with the
                                # bounds fixed by the operator instead of read
                                # from params.
                                T.tile.max(y32, x32, 0.0)
                                T.tile.min(y32, y32, 6.0)
                            elif op_kind == "hardtanh":
                                T.tile.max(y32, x32, params[0])
                                T.tile.min(y32, y32, params[1])
                            elif op_kind == "leaky_relu":
                                T.tile.max(y32, x32, 0.0)
                                T.tile.min(tmp, x32, 0.0)
                                T.tile.mul(tmp, tmp, params[0])
                                T.tile.add(y32, y32, tmp)
                            elif op_kind == "elu":
                                T.tile.min(tmp, x32, 0.0)
                                T.tile.exp(tmp, tmp)
                                T.tile.sub(tmp, tmp, 1.0)
                                T.tile.mul(tmp, tmp, params[0])
                                T.tile.max(y32, x32, 0.0)
                                T.tile.add(y32, y32, tmp)
                            elif op_kind == "mish":
                                T.tile.abs(tmp, x32)
                                T.tile.sub(tmp, zero, tmp)
                                T.tile.exp(tmp, tmp)
                                T.tile.add(tmp, tmp, 1.0)
                                T.tile.ln(tmp, tmp)
                                T.tile.max(tmp2, x32, 0.0)
                                T.tile.add(tmp, tmp, tmp2)
                                T.tile.mul(tmp, tmp, 2.0)
                                T.tile.sigmoid(tmp, tmp)
                                T.tile.mul(tmp, tmp, 2.0)
                                T.tile.sub(tmp, tmp, 1.0)
                                T.tile.mul(y32, x32, tmp)
                            elif op_kind == "selu":
                                T.tile.min(tmp, x32, 0.0)
                                T.tile.exp(tmp, tmp)
                                T.tile.sub(tmp, tmp, 1.0)
                                T.tile.mul(tmp, tmp, 1.6732632423543772)
                                T.tile.max(y32, x32, 0.0)
                                T.tile.add(y32, y32, tmp)
                                T.tile.mul(y32, y32, 1.0507009873554805)
                            elif op_kind == "softplus":
                                T.tile.mul(tmp2, x32, params[0])
                                T.tile.abs(tmp, tmp2)
                                T.tile.sub(tmp, zero, tmp)
                                T.tile.exp(tmp, tmp)
                                T.tile.add(tmp, tmp, 1.0)
                                T.tile.ln(tmp, tmp)
                                T.tile.max(tmp2, tmp2, 0.0)
                                T.tile.add(y32, tmp2, tmp)
                                T.tile.mul(y32, y32, 1.0 / params[0])
                            elif op_kind == "clamp":
                                T.copy(x32, y32)
                                if input_count > 1:
                                    T.tile.max(y32, y32, b32)
                                if input_count > 2:
                                    T.tile.min(y32, y32, c32)
                            elif op_kind == "clamp_min":
                                T.tile.max(y32, x32, b32)
                            elif op_kind == "clamp_max":
                                T.tile.min(y32, x32, b32)
                            elif op_kind == "clamp_scalar":
                                T.tile.max(y32, x32, params[0])
                                T.tile.min(y32, y32, params[1])
                            elif op_kind == "lerp":
                                T.tile.sub(tmp, b32, x32)
                                T.tile.mul(tmp, tmp, c32)
                                T.tile.add(y32, x32, tmp)
                            elif op_kind == "nan_to_num":
                                # R195: was a per-lane ``T.serial(tile)`` chain
                                # of ``T.isnan`` / infinity comparisons, i.e.
                                # scalar over the whole tensor -- 9.6 GB/s while
                                # the vendor baseline sits at 890 GB/s.  The
                                # order below matters and reproduces the scalar
                                # ``if/elif`` chain exactly:
                                #
                                # An unordered compare returns false on
                                # dav-2201 (measured, R195-data/
                                # probe_nan_predicate.log), so ``nan != +inf``
                                # is false and the +inf select would claim NaN
                                # lanes.  Doing the NaN substitution *last*
                                # makes that harmless: NaN overwrites whatever
                                # the infinity steps left behind, which is what
                                # the scalar ``if T.isnan(x)`` head did.
                                # ``|x| <= inf`` is exactly "not NaN" in one
                                # vector op, because an unordered compare
                                # returns false for every mode (measured).
                                T.tile.abs(tmp, x32)
                                T.tile.compare(
                                    mask, tmp, T.infinity(compute_dtype), "LE"
                                )
                                # ``tmp`` is dead from here on.
                                T.tile.compare(
                                    mask2, x32, T.infinity(compute_dtype), "NE"
                                )
                                T.tile.select(
                                    tmp, mask2, x32, params[1],
                                    "VSEL_TENSOR_SCALAR_MODE",
                                )
                                T.tile.compare(
                                    mask2, x32, -T.infinity(compute_dtype), "NE"
                                )
                                T.tile.select(
                                    tmp2, mask2, tmp, params[2],
                                    "VSEL_TENSOR_SCALAR_MODE",
                                )
                                T.tile.select(
                                    y32, mask, tmp2, params[0],
                                    "VSEL_TENSOR_SCALAR_MODE",
                                )

                            if not vectorized:
                                for lane in T.serial(tile):
                                    idx = start + lane
                                    x = x32[lane]
                                    if generated:
                                        if op_kind == "alibi":
                                            head = idx // (generated_seq * generated_seq)
                                            row = (idx // generated_seq) % generated_seq
                                            col = idx % generated_seq
                                            slope = T.exp(
                                                -5.545177444479562
                                                * (head + 1.0)
                                                / generated_dim
                                            )
                                            y32[lane] = -slope * T.abs(row - col)
                                        else:
                                            row = idx // generated_dim
                                            col = idx % generated_dim
                                            pair = col // 2
                                            angle = row * T.exp(
                                                -T.log(10000.0)
                                                * (2.0 * pair / generated_dim)
                                            )
                                            y32[lane] = T.if_then_else(
                                                (col % 2) == 0, T.sin(angle), T.cos(angle)
                                            )
                                    elif op_kind == "where":
                                        y32[lane] = T.if_then_else(
                                            b32[lane] != 0, c32[lane], x
                                        )

                            if dtype != "float32":
                                T.tile.cast(y_ub, y32, "CAST_RINT", tile)
                            else:
                                T.copy(y32, y_ub)
                            T.barrier_all()
                            if full:
                                T.copy(
                                    y_ub if span == tile else y_ub[0:tile],
                                    Y[start : start + tile],
                                )
                            else:
                                if start < out_numel:
                                    T.copy(
                                        y_ub[0 : out_numel - start], Y[start:out_numel]
                                    )

            return main

        return kernel(), tile, gated_vector

    # ``ub_fits`` reads the emitted UB offsets back: some intrinsics take an
    # implicit scratch buffer that never appears in the Python source, so the
    # declared buffers fitting the budget does not prove the artifact does.
    cap = ub_cap
    compiled = None
    while True:
        compiled, tile, gated_vector = build(cap)
        if ub_fits(compiled.get_kernel_source()):
            return compiled
        if tile <= MASK_GRAIN:
            break
        cap = max(MASK_GRAIN, tile // 2)
    raise ValueError(
        f"activation {op_kind}/{dtype} does not fit the unified buffer at any tile"
    )


def build_activation_kernel(
    shapes: tuple[tuple[int, ...], ...],
    dtype: torch.dtype,
    *,
    op_kind: str,
    params: tuple[float | None, ...] = (),
    op_name: str,
    output_shape: tuple[int, ...] | None = None,
):
    supported_kinds = {
        "atan2",
        "clamp",
        "clamp_max",
        "clamp_min",
        "clamp_scalar",
        "elu",
        "gelu_and_mul",
        "gelu_none",
        "gelu_tanh",
        "gelu_tanh_and_mul",
        "hardsigmoid",
        "hardswish",
        "hardtanh",
        "leaky_relu",
        "lerp",
        "mish",
        "nan_to_num",
        "relu",
        "relu6",
        "selu",
        "silu",
        "silu_and_mul",
        "softplus",
    }
    if op_kind not in supported_kinds:
        raise NotImplementedError(
            f"{op_name} remains fail-closed: activation expression {op_kind!r} "
            "does not yet have a validated vector writeback path on Ascend"
        )
    if dtype not in FLOAT_DTYPES:
        names = ", ".join(str(item) for item in FLOAT_DTYPES)
        raise TypeError(
            f"{op_name} does not support dtype {dtype}; supported dtypes are [{names}], received input dtype {dtype}"
        )
    shapes = tuple(tuple(shape) for shape in shapes)
    if output_shape is None:
        out_shape, strides = _shape_info(shapes)
    else:
        out_shape = tuple(output_shape)
        if op_kind in {"silu_and_mul", "gelu_and_mul", "gelu_tanh_and_mul"}:
            strides = (_broadcast_strides(out_shape, out_shape),)
        else:
            strides = tuple(_broadcast_strides(shape, out_shape) for shape in shapes)
    input_numels = tuple(math.prod(shape) for shape in shapes)
    compiled = _compile_activation(
        input_numels, out_shape, strides, _dtype_name(dtype), op_kind, tuple(params)
    )

    def invoke(*tensors):
        if len(tensors) != len(shapes):
            raise ValueError(
                f"{op_name} expects {len(shapes)} tensor inputs, received {len(tensors)}"
            )
        for tensor, shape in zip(tensors, shapes):
            if tuple(tensor.shape) != shape:
                raise ValueError(
                    f"{op_name} kernel shape mismatch: expected {shape}, received {tuple(tensor.shape)}"
                )
        flat = [
            tensor if tensor.ndim == 1 else tensor.reshape(-1) for tensor in tensors
        ]
        dummy = torch.empty((1,), dtype=dtype, device=flat[0].device)
        while len(flat) < 3:
            flat.append(dummy)
        # ``out_idx=[-1]`` makes the JIT wrapper allocate and return Y.  Passing
        # a fourth tensor here treats it as an extra input and leaves that
        # tensor untouched, which previously masked the real kernel result
        # with allocator contents (NaN on the non-divisible probe).
        result = compiled(*flat)
        return (
            result
            if tuple(out_shape) == (result.numel(),)
            else result.reshape(out_shape)
        )

    invoke.compiled = compiled

    return invoke
