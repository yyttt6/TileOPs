"""Mixed-domain elementwise kernels.

These paths deliberately keep the input roles separate.  In particular, a
masked-fill mask is byte-backed bool storage while the selected value keeps the
input dtype, and PReLU indexes a channel vector rather than using ordinary
trailing-axis broadcasting.
"""

import math
from functools import lru_cache

import tilelang
import tilelang.language as T
import torch

from .common import grid_repeat_count, launch_block_count
from .elementwise_binary import _broadcast_strides, _is_contiguous, _offset_expr, ub_fits


_MASKED_DTYPES = (
    torch.bool,
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.float16,
    torch.bfloat16,
    torch.float32,
)
_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_INTEGER_DTYPES = (torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)

# --- Geometry (R195, recipe from R194 / PROJECT_STATE 13.50) --------------
# All three kernels in this file shipped with the same two defects the unary
# template had: ``native_tile = 2048`` with no grid-stride loop, so the block
# count grew with the tensor and ~0.11 us of per-block dispatch dominated.
# On top of that every one of them ran its *main* path as a per-lane scalar
# loop, which is the 3-15 GB/s fingerprint in PROJECT_STATE 13.49.
UB_BUDGET_BYTES = 180224
MASK_GRAIN = 128
TILE_HARD_CAP = 16384
LAUNCH_BLOCK_CAP = 48

# PReLU emits one ``T.tile.fill`` per packed channel run at parse time; past a
# handful of them the unrolled statements cost more than the gather they
# replace, so the geometry falls back instead of widening further.
_MAX_UNIT_FILLS = 16

_ELEMENT_BYTES = {
    "float16": 2, "bfloat16": 2, "float32": 4,
    "int8": 1, "int16": 2, "int32": 4, "int64": 8, "uint8": 1, "bool": 1,
}

# Measured on dav-2201, one compile per (intrinsic, dtype) pair -- see
# ``R195-data/probe_mixed_intrinsics.log`` for the raw output.  Anything not
# listed here fails to compile, so those paths keep the shipped per-lane loop.
#
#   T.tile.select(VSEL_TENSOR_TENSOR): float16, float32 only.  bfloat16 and
#       every integer storage type are rejected by the AscendC header, so the
#       float masked-fill runs its select in fp32 (which is also what the
#       shipped scalar path did, keeping the result bit-identical).
#   T.tile.bitwise_and / bitwise_or: *compile* for uint8/int8/int16/int32/
#       int64, but only int8/uint8/int16 are correct.  On int32 half the
#       elements come back untouched and on int64 three quarters do -- the
#       lowering counts 16-bit lanes and passes the element count straight
#       through, so a 4-byte operand only ever covers the first half of the
#       buffer.  **A compile that succeeds is not a correctness check**: the
#       first probe here filled both operands with a constant and read
#       ``out[0]``, and it reported int32/int64 as OK.
#   T.tile.bitwise_xor: int16 only; it does not compile at any other width.
#       Same narrow-integer restriction R194 hit with ``T.tile.bitwise_not``.
#   Consequence: a bitwise op only runs on the vector path after the operands
#   are reinterpreted down to a verified width -- uint8 for AND/OR (any byte
#   count) and int16 for XOR (even byte counts only).
_SELECT_DTYPES = frozenset({"float16", "float32"})
_BITWISE_OK = {
    "and": frozenset({"uint8", "int8", "int16"}),
    "or": frozenset({"uint8", "int8", "int16"}),
    "xor": frozenset({"int16"}),
}


def _dtype_name(dtype: torch.dtype) -> str:
    return "uint8" if dtype == torch.bool else str(dtype).replace("torch.", "")


def _shape_info(a_shape, b_shape):
    out_shape = tuple(torch.broadcast_shapes(tuple(a_shape), tuple(b_shape)))
    return (
        tuple(a_shape),
        tuple(b_shape),
        out_shape,
        _broadcast_strides(tuple(a_shape), out_shape),
        _broadcast_strides(tuple(b_shape), out_shape),
    )


def _round_down_grain(value: int) -> int:
    return max(MASK_GRAIN, int(value) - int(value) % MASK_GRAIN)


def _pick_tile(ceiling: int, out_numel: int) -> int:
    """Prefer a tile that divides the tensor; give up at most half the width.

    A single ragged tile drops into the per-lane tail and that tile is on the
    critical path of the whole launch -- R194 2.3 measured 2.65x for exactly
    this on the unary template.
    """
    ceiling = _round_down_grain(ceiling)
    candidate = ceiling
    while candidate >= max(MASK_GRAIN, ceiling // 2):
        if out_numel % (2 * candidate) == 0:
            return candidate
        candidate -= MASK_GRAIN
    return ceiling


def _grid(tile: int, out_numel: int):
    block_total = tile * 2
    logical_blocks = max(1, math.ceil(out_numel / block_total))
    launch_blocks = launch_block_count(min(logical_blocks, LAUNCH_BLOCK_CAP))
    return block_total, logical_blocks, launch_blocks, grid_repeat_count(
        logical_blocks, launch_blocks
    )


def masked_fill_plan(out_shape, data_strides, mask_strides, in_dtype):
    """Return ``(vector, storage)`` for one masked-fill signature.

    A masked fill performs no arithmetic -- it only chooses between two bit
    patterns -- so the kernel runs on whatever storage type ``T.tile.select``
    accepts at the operand's width.  ``T.tile.select`` takes float16 and
    float32 and rejects bfloat16 and every integer type on dav-2201
    (measured, ``R195-data/probe_mixed_intrinsics.log``), so bfloat16 is
    reinterpreted as float16: same width, and moving bits never inspects them.

    That is also what makes the result bit-exact.  The shipped scalar path
    promoted fp16/bf16 through fp32 and back, which is lossless for every
    finite value and for +-inf but *not* for a NaN payload -- measured, a
    bf16 ``0x7fc0`` comes back as ``0x7fff`` through ``CAST_RINT``.  Selecting
    on the raw bits preserves the input exactly, which is what
    ``Tensor.masked_fill`` promises.

    An integer dtype or a broadcast operand keeps the shipped per-lane path.
    """
    contiguous = _is_contiguous(out_shape, data_strides) and _is_contiguous(
        out_shape, mask_strides
    )
    if not contiguous or in_dtype not in {"float16", "bfloat16", "float32"}:
        return False, in_dtype
    return True, ("float32" if in_dtype == "float32" else "float16")


@lru_cache(maxsize=128)
def _compile_masked_fill(
    data_numel: int,
    mask_numel: int,
    value_numel: int,
    out_shape: tuple[int, ...],
    data_strides: tuple[int, ...],
    mask_strides: tuple[int, ...],
    in_dtype: str,
    tensor_value: bool,
):
    out_numel = math.prod(out_shape)
    # R195: the shipped kernel had no UB at all -- every element was a scalar
    # GM read and a scalar GM store inside ``for lane in T.serial(tile)``, at
    # 15 GB/s against a vendor baseline on the HBM plateau (PROJECT_STATE
    # 13.49).
    vector, storage = masked_fill_plan(out_shape, data_strides, mask_strides, in_dtype)
    element_bytes = _ELEMENT_BYTES[storage]
    if vector:
        # a_ub + c_ub + v_ub in the select storage, the int8 mask tile and its
        # fp16 cast (compare needs a float source; int8 -> fp32 does not lower
        # on dav-2201, int8 -> fp16 does and is exact over the whole int8
        # range), plus one packed compare mask at 1/8 byte per element.
        live_eighths = 8 * (3 * element_bytes + 1 + 2) + 1
    else:
        live_eighths = 8 * MASK_GRAIN
    ub_cap = (UB_BUDGET_BYTES * 8) // live_eighths
    needed = max(MASK_GRAIN, math.ceil(out_numel / 2))

    def build(cap: int):
        if vector:
            tile = _pick_tile(min(cap, TILE_HARD_CAP, needed), out_numel)
        else:
            tile = 2048
        scratch = tile if vector else MASK_GRAIN
        block_total, logical_blocks, launch_blocks, grid_repeats = _grid(tile, out_numel)

        @tilelang.jit(out_idx=[-1])
        def kernel():
            @T.prim_func
            def main(
                A: T.Tensor((data_numel,), storage),
                # Keep the mask storage signed.  The mask is bool at the public
                # boundary, but TVM's block-bound analysis can otherwise form a
                # negative sentinel while indexing a uint8 buffer from the
                # unsigned Kernel context ("cannot make uint from ... -1").
                M: T.Tensor((mask_numel,), "int8"),
                V: T.Tensor((value_numel,), storage),
                C: T.Tensor((out_numel,), storage),
            ):
                with T.Kernel(launch_blocks, is_npu=True) as (cid, vid):
                    a_ub = T.alloc_ub((scratch,), storage)
                    c_ub = T.alloc_ub((scratch,), storage)
                    v_ub = T.alloc_ub((scratch,), storage)
                    m_ub = T.alloc_ub((scratch,), "int8")
                    m16 = T.alloc_ub((scratch,), "float16")
                    packed = T.alloc_ub((scratch // 8,), "uint8")
                    with T.Scope("V"):
                        if vector:
                            # The fill value is loop invariant: one scalar GM
                            # read for the whole launch.
                            T.tile.fill(v_ub, V[0])
                        for grid_repeat in T.serial(grid_repeats):
                            logical_cid = cid + grid_repeat * launch_blocks
                            if logical_cid < logical_blocks:
                                start = logical_cid * block_total + vid * tile
                                if vector:
                                    full = start + tile <= out_numel
                                    if full:
                                        T.copy(A[start], a_ub)
                                        T.copy(M[start], m_ub)
                                    else:
                                        # ``T.tile.fill`` lowers to AscendC
                                        # Duplicate, which rejects int8 storage
                                        # ("current api support dtype
                                        # combination is ... half / bfloat16_t
                                        # / int16_t / uint16_t / int32_t /
                                        # uint32_t / float"), so the mask tail
                                        # is zeroed lane by lane.
                                        T.tile.fill(a_ub, 0)
                                        for lane in T.serial(tile):
                                            idx = start + lane
                                            if idx < out_numel:
                                                a_ub[lane] = A[idx]
                                                m_ub[lane] = M[idx]
                                            else:
                                                m_ub[lane] = 0
                                    T.barrier_all()
                                    T.tile.cast(m16, m_ub, "CAST_NONE", tile)
                                    # "EQ 0" is "keep the data", so no mask
                                    # negation is needed -- ``bitwise_not``
                                    # rejects uint8 packed masks on dav-2201.
                                    T.tile.compare(packed, m16, 0.0, "EQ")
                                    T.tile.select(
                                        c_ub, packed, a_ub, v_ub,
                                        "VSEL_TENSOR_TENSOR_MODE",
                                    )
                                    T.barrier_all()
                                    if full:
                                        T.copy(c_ub, C[start])
                                    else:
                                        if start < out_numel:
                                            T.copy(c_ub[0:out_numel - start],
                                                   C[start:out_numel])
                                else:
                                    for lane in T.serial(tile):
                                        idx = start + lane
                                        if idx < out_numel:
                                            data = A[
                                                _offset_expr(idx, out_shape, data_strides)
                                            ]
                                            mask = M[
                                                _offset_expr(idx, out_shape, mask_strides)
                                            ]
                                            if in_dtype in {"float16", "bfloat16"}:
                                                data32 = T.cast(data, "float32")
                                                value32 = T.cast(V[0], "float32")
                                                chosen = T.if_then_else(
                                                    mask != 0, value32, data32
                                                )
                                                C[idx] = T.cast(chosen, in_dtype)
                                            else:
                                                C[idx] = T.if_then_else(
                                                    mask != 0, V[0], data
                                                )

            return main

        return kernel(), tile

    cap = ub_cap
    while True:
        compiled, tile = build(cap)
        if ub_fits(compiled.get_kernel_source()):
            return compiled
        if tile <= MASK_GRAIN or not vector:
            break
        cap = max(MASK_GRAIN, tile // 2)
    raise ValueError(
        f"masked_fill/{in_dtype} does not fit the unified buffer at any tile"
    )


def _pack_channel_units(
    inner_size: int, channels: int, cap: int, weight_bytes: int
) -> tuple[int, int]:
    """Return ``(units_per_tile, tile)`` for the PReLU channel geometry.

    A *unit* is one channel's contiguous run of ``inner_size`` output elements:
    every element in a unit shares one weight, so a whole unit can be produced
    with a single ``T.tile.fill`` instead of ``inner_size`` scalar loads out of
    the weight vector.  Packing whole units keeps the DMA as wide as the UB
    budget allows while keeping every fill region at a constant offset.

    Three constraints, all of which return ``(0, 0)`` -- meaning "keep the
    per-lane weight gather" -- when they cannot be met:

    * a tile must stay a whole number of mask blocks, because the ``x > 0``
      compare emits a packed mask;
    * a unit is written through a UB buffer *region*, and the Ascend vector
      ops fault on a region that is not 32-byte aligned (same constraint the
      binary template's packed broadcast has to respect);
    * the fills are emitted one per unit at parse time, so a tiny
      ``inner_size`` would unroll into hundreds of statements and lose to the
      gather it is replacing.
    """
    if channels <= 1 or inner_size <= 0 or inner_size > cap:
        return 0, 0
    if (inner_size * weight_bytes) % 32:
        return 0, 0
    for units in range(min(cap // inner_size, _MAX_UNIT_FILLS), 0, -1):
        if (units * inner_size) % MASK_GRAIN == 0:
            return units, units * inner_size
    return 0, 0


@lru_cache(maxsize=128)
def _compile_prelu(
    input_numel: int,
    weight_numel: int,
    out_shape: tuple[int, ...],
    input_dtype: str,
    weight_dtype: str,
    output_dtype: str,
    channels: int,
    inner_size: int,
):
    # R195: the shipped kernel ran two per-lane loops on the main path -- the
    # weight gather ``w_ub[lane] = W[(idx // inner) % channels]`` and the
    # ``x > 0 ? x : w*x`` select -- and measured 4.4 GB/s against a vendor
    # baseline at 271 GB/s.  Both are gone here: the weight tile is a small
    # number of constant-offset fills (every element of one channel run shares
    # a weight) and the select is a compare plus a VSEL.
    single_weight = channels <= 1
    in_bytes = _ELEMENT_BYTES[input_dtype]
    w_bytes = _ELEMENT_BYTES[weight_dtype]
    out_bytes = _ELEMENT_BYTES[output_dtype]
    # x_ub + y_ub + w_ub (only wide on the scalar fallback) + x32/w32/t32
    # fp32 scratch, plus one packed compare mask at 1/8 byte per element.
    live_eighths = 8 * (in_bytes + out_bytes + w_bytes + 4 + 4 + 4) + 1
    ub_cap = (UB_BUDGET_BYTES * 8) // live_eighths
    needed = max(MASK_GRAIN, math.ceil(input_numel / 2))
    cap0 = min(ub_cap, TILE_HARD_CAP, needed)

    def build(cap: int):
        cap = _round_down_grain(cap)
        if single_weight:
            units, tile, weight_vector = 0, _pick_tile(cap, input_numel), True
        else:
            units, packed_tile = _pack_channel_units(
                inner_size, channels, cap, w_bytes
            )
            if units:
                tile, weight_vector = packed_tile, True
            else:
                tile, weight_vector = _pick_tile(cap, input_numel), False
        block_total, logical_blocks, launch_blocks, grid_repeats = _grid(
            tile, input_numel
        )

        @tilelang.jit(out_idx=[2])
        def kernel():
            @T.prim_func
            def main(
                A: T.Tensor((input_numel,), input_dtype),
                W: T.Tensor((weight_numel,), weight_dtype),
                C: T.Tensor((input_numel,), output_dtype),
            ):
                with T.Kernel(launch_blocks, is_npu=True) as (cid, vid):
                    x_ub = T.alloc_ub((tile,), input_dtype)
                    y_ub = T.alloc_ub((tile,), output_dtype)
                    w_ub = T.alloc_ub((tile,), weight_dtype)
                    x32 = T.alloc_ub((tile,), "float32")
                    w32 = T.alloc_ub((tile,), "float32")
                    t32 = T.alloc_ub((tile,), "float32")
                    packed = T.alloc_ub((tile // 8,), "uint8")
                    with T.Scope("V"):
                        for grid_repeat in T.serial(grid_repeats):
                            logical_cid = cid + grid_repeat * launch_blocks
                            if logical_cid < logical_blocks:
                                start = logical_cid * block_total + vid * tile
                                full = start + tile <= input_numel
                                if full:
                                    T.copy(A[start], x_ub)
                                else:
                                    T.tile.fill(x_ub, 0)
                                    for lane in T.serial(tile):
                                        if start + lane < input_numel:
                                            x_ub[lane] = A[start + lane]
                                # The weight tile is built in the *weight*
                                # dtype and cast once.  A scalar bf16 -> fp32
                                # conversion is rejected by bisheng ("not
                                # support bf16 type cast"); the tile-level cast
                                # is the supported form.
                                if single_weight:
                                    T.tile.fill(w_ub, W[0])
                                elif weight_vector:
                                    # ``tile`` is a whole number of channel
                                    # runs, so every run is a constant-offset
                                    # region carrying exactly one weight.
                                    for unit in range(units):
                                        base = start + unit * inner_size
                                        T.tile.fill(
                                            w_ub[
                                                unit * inner_size : (unit + 1)
                                                * inner_size
                                            ],
                                            W[(base // inner_size) % channels],
                                        )
                                else:
                                    T.tile.fill(w_ub, 0)
                                    for lane in T.serial(tile):
                                        idx = start + lane
                                        if idx < input_numel:
                                            w_ub[lane] = W[
                                                (idx // inner_size) % channels
                                            ]
                                T.barrier_all()
                                # R195: the shipped code cast unconditionally,
                                # including float32 -> float32.  That lowering
                                # does not move the data, so every fp32 PReLU
                                # read uninitialised UB and returned NaN.  No
                                # manifest case is fp32 (the three PreluFwdOp
                                # cases are fp16/bf16), so coverage never saw
                                # it.  A same-dtype move is a ``T.copy``.
                                if input_dtype == "float32":
                                    T.copy(x_ub, x32)
                                else:
                                    T.tile.cast(x32, x_ub, "CAST_NONE", tile)
                                if weight_dtype == "float32":
                                    T.copy(w_ub, w32)
                                else:
                                    T.tile.cast(w32, w_ub, "CAST_NONE", tile)
                                # y = x > 0 ? x : w * x.  An unordered compare
                                # returns false on dav-2201 (measured), so NaN
                                # takes the ``w * x`` branch exactly as the
                                # shipped scalar ``x > 0.0`` test did.
                                T.tile.mul(t32, w32, x32)
                                T.tile.compare(packed, x32, 0.0, "GT")
                                T.tile.select(
                                    w32, packed, x32, t32, "VSEL_TENSOR_TENSOR_MODE"
                                )
                                if output_dtype == "float32":
                                    T.copy(w32, y_ub)
                                else:
                                    T.tile.cast(y_ub, w32, "CAST_RINT", tile)
                                T.barrier_all()
                                if full:
                                    T.copy(y_ub, C[start])
                                else:
                                    for lane in T.serial(tile):
                                        idx = start + lane
                                        if idx < input_numel:
                                            C[idx] = y_ub[lane]

            return main

        return kernel(), tile

    cap = cap0
    while True:
        compiled, tile = build(cap)
        if ub_fits(compiled.get_kernel_source()):
            return compiled
        if tile <= MASK_GRAIN:
            break
        cap = max(MASK_GRAIN, tile // 2)
    raise ValueError(f"prelu/{input_dtype} does not fit the unified buffer at any tile")


@lru_cache(maxsize=128)
def _compile_bitwise(
    a_numel: int,
    b_numel: int,
    out_numel: int,
    out_shape: tuple[int, ...],
    a_strides: tuple[int, ...],
    b_strides: tuple[int, ...],
    dtype: str,
    op_kind: str,
):
    # R195: the shipped kernel was a per-lane GM->GM loop with no UB at all
    # (14-15 GB/s against a vendor baseline above 1000 GB/s).  Vectorise when
    # both operands are contiguous and the storage type has a vector bitwise
    # op; a broadcast operand or an unsupported dtype keeps the scalar path.
    contiguous = _is_contiguous(out_shape, a_strides) and _is_contiguous(
        out_shape, b_strides
    )
    vector = contiguous and dtype in _BITWISE_OK[op_kind]
    element_bytes = _ELEMENT_BYTES[dtype]
    live_eighths = 8 * (3 * element_bytes) if vector else 8 * MASK_GRAIN
    ub_cap = (UB_BUDGET_BYTES * 8) // live_eighths
    needed = max(MASK_GRAIN, math.ceil(out_numel / 2))

    def build(cap: int):
        if vector:
            tile = _pick_tile(min(cap, TILE_HARD_CAP, needed), out_numel)
        else:
            tile = 2048
        block_total, logical_blocks, launch_blocks, grid_repeats = _grid(tile, out_numel)

        @tilelang.jit(out_idx=[2])
        def kernel():
            @T.prim_func
            def main(
                A: T.Tensor((a_numel,), dtype),
                B: T.Tensor((b_numel,), dtype),
                C: T.Tensor((out_numel,), dtype),
            ):
                with T.Kernel(launch_blocks, is_npu=True) as (cid, vid):
                    a_ub = T.alloc_ub((tile if vector else MASK_GRAIN,), dtype)
                    b_ub = T.alloc_ub((tile if vector else MASK_GRAIN,), dtype)
                    c_ub = T.alloc_ub((tile if vector else MASK_GRAIN,), dtype)
                    with T.Scope("V"):
                        for grid_repeat in T.serial(grid_repeats):
                            logical_cid = cid + grid_repeat * launch_blocks
                            if logical_cid < logical_blocks:
                                start = logical_cid * block_total + vid * tile
                                if vector:
                                    full = start + tile <= out_numel
                                    if full:
                                        T.copy(A[start], a_ub)
                                        T.copy(B[start], b_ub)
                                    else:
                                        # R195 续轮：这里原本是 ``T.serial(tile)``
                                        # 的逐 lane 补齐。那个循环的成本是
                                        # **O(tile)，与有效 lane 数无关**，而它
                                        # 落在整个 launch 的关键路径上 ——
                                        # `tail-nondivisible` (4097 x int32) 就是
                                        # 这样从 86.3 µs 退到 141.5 µs 的。
                                        # 切片形式的 T.copy 会把长度钳到
                                        # ``out_numel - start``（生成源码里是一个
                                        # 三元表达式，见 src_activation_relu_5000
                                        # _before.cce:41），所以残缺 tile 也能一次
                                        # DMA 搬完。尾部 lane 留着旧值不要紧：
                                        # 它们既不参与写回，也不影响按位运算。
                                        if start < out_numel:
                                            T.copy(A[start:out_numel],
                                                   a_ub[0:out_numel - start])
                                            T.copy(B[start:out_numel],
                                                   b_ub[0:out_numel - start])
                                    T.barrier_all()
                                    if op_kind == "and":
                                        T.tile.bitwise_and(c_ub, a_ub, b_ub)
                                    elif op_kind == "or":
                                        T.tile.bitwise_or(c_ub, a_ub, b_ub)
                                    else:
                                        T.tile.bitwise_xor(c_ub, a_ub, b_ub)
                                    T.barrier_all()
                                    if full:
                                        T.copy(c_ub, C[start])
                                    else:
                                        if start < out_numel:
                                            T.copy(c_ub[0:out_numel - start],
                                                   C[start:out_numel])
                                else:
                                    for lane in T.serial(tile):
                                        idx = start + lane
                                        if idx < out_numel:
                                            a = A[_offset_expr(idx, out_shape, a_strides)]
                                            b = B[_offset_expr(idx, out_shape, b_strides)]
                                            if op_kind == "and":
                                                C[idx] = T.bitwise_and(a, b)
                                            elif op_kind == "or":
                                                C[idx] = T.bitwise_or(a, b)
                                            else:
                                                C[idx] = T.bitwise_xor(a, b)

            return main

        return kernel(), tile

    cap = ub_cap
    while True:
        compiled, tile = build(cap)
        if ub_fits(compiled.get_kernel_source()):
            return compiled
        if tile <= MASK_GRAIN or not vector:
            break
        cap = max(MASK_GRAIN, tile // 2)
    raise ValueError(f"bitwise {op_kind}/{dtype} does not fit the unified buffer")



def build_masked_fill_kernel(data_shape, mask_shape, data_dtype, *, value_tensor=None, value=0):
    if data_dtype not in _MASKED_DTYPES:
        raise TypeError(f"MaskedFill does not support dtype {data_dtype}")
    if value_tensor is not None and tuple(value_tensor.shape) != ():
        raise ValueError(f"MaskedFillFwdOp value must be 0-D, got shape {tuple(value_tensor.shape)}")
    data_shape, mask_shape, out_shape, data_strides, mask_strides = _shape_info(data_shape, mask_shape)
    storage = _dtype_name(data_dtype)
    compiled = _compile_masked_fill(
        math.prod(data_shape), math.prod(mask_shape), 1, out_shape,
        data_strides, mask_strides, storage, value_tensor is not None,
    )
    # The vector path selects on raw bits, so bfloat16 rides in a float16
    # buffer (``T.tile.select`` has no bfloat16 lowering).  Reinterpret on the
    # way in and back on the way out; nothing in between reads the value.
    vector, kernel_storage = masked_fill_plan(
        out_shape, data_strides, mask_strides, storage
    )
    reinterpret = torch.float16 if vector and storage == "bfloat16" else None

    def invoke(data, mask, value_tensor_arg=None):
        if tuple(data.shape) != data_shape or tuple(mask.shape) != mask_shape:
            raise ValueError("MaskedFill kernel shape mismatch")
        data_flat = data.reshape(-1)
        if data_dtype == torch.bool:
            data_flat = data_flat.view(torch.uint8)
        # Bool tensors are byte-backed; use signed int8 only for the kernel
        # ABI to avoid the TVM uint-index boundary-analysis bug.
        mask_flat = mask.reshape(-1).view(torch.int8)
        if value_tensor is not None:
            if value_tensor_arg is None or tuple(value_tensor_arg.shape) != ():
                raise ValueError("MaskedFillFwdOp expects a 0-D value tensor")
            value_flat = value_tensor_arg.reshape(-1)
            if data_dtype == torch.bool:
                value_flat = value_flat.view(torch.uint8)
        else:
            value_flat = torch.full((1,), value, dtype=data.dtype, device=data.device)
            if data_dtype == torch.bool:
                value_flat = value_flat.view(torch.uint8)
        if reinterpret is not None:
            data_flat = data_flat.view(reinterpret)
            value_flat = value_flat.view(reinterpret)
        result = compiled(data_flat, mask_flat, value_flat)
        if reinterpret is not None:
            result = result.view(data_dtype)
        if data_dtype == torch.bool:
            result = result.view(torch.bool)
        return result if tuple(out_shape) == (result.numel(),) else result.reshape(out_shape)

    invoke.compiled = compiled
    return invoke


def build_prelu_kernel(input_shape, weight_shape, input_dtype, *, weight_dtype=None):
    if input_dtype not in _FLOAT_DTYPES:
        raise TypeError(f"PreluFwdOp does not support dtype {input_dtype}")
    weight_dtype = input_dtype if weight_dtype is None else weight_dtype
    if weight_dtype not in _FLOAT_DTYPES:
        raise TypeError(f"PreluFwdOp does not support weight dtype {weight_dtype}")
    input_shape, weight_shape = tuple(input_shape), tuple(weight_shape)
    if len(weight_shape) > 1:
        raise ValueError("PreluFwdOp weight must be 0-D or 1-D")
    channels = 1 if len(weight_shape) == 0 else int(weight_shape[0])
    if channels != 1 and (len(input_shape) < 2 or channels != int(input_shape[1])):
        raise ValueError(f"PreluFwdOp weight length {channels} does not match channel axis of {input_shape}")
    inner_size = math.prod(input_shape[2:]) if len(input_shape) >= 2 else 1
    compiled = _compile_prelu(
        math.prod(input_shape), math.prod(weight_shape), input_shape,
        _dtype_name(input_dtype), _dtype_name(weight_dtype), _dtype_name(input_dtype), channels, inner_size,
    )

    def invoke(data, weight):
        if tuple(data.shape) != input_shape or tuple(weight.shape) != weight_shape:
            raise ValueError("PreluFwdOp kernel shape mismatch")
        result = compiled(data.reshape(-1), weight.reshape(-1))
        return result if tuple(input_shape) == (result.numel(),) else result.reshape(input_shape)

    invoke.compiled = compiled
    return invoke


def build_bitwise_kernel(a_shape, b_shape, dtype, *, op_kind):
    if dtype not in _INTEGER_DTYPES:
        raise TypeError(f"Bitwise operation does not support dtype {dtype}")
    a_shape, b_shape, out_shape, a_strides, b_strides = _shape_info(a_shape, b_shape)
    storage = _dtype_name(dtype)
    out_numel = math.prod(out_shape)
    itemsize = torch.empty((), dtype=dtype).element_size()
    byte_numel = out_numel * itemsize
    # R195: an AND/OR/XOR is bit-parallel, so when neither operand broadcasts
    # the whole tensor can be reinterpreted down to a width where the vector
    # intrinsic is *verified* correct (see ``_BITWISE_OK``): uint8 for AND/OR,
    # int16 for XOR.  Without this an int32 AND silently leaves half the
    # elements untouched -- it compiles, it just does not compute.
    reinterpret = None
    if a_shape == out_shape and b_shape == out_shape:
        if op_kind in {"and", "or"}:
            reinterpret, r_storage, r_numel = torch.uint8, "uint8", byte_numel
        elif byte_numel % 2 == 0:
            reinterpret, r_storage, r_numel = torch.int16, "int16", byte_numel // 2

    if reinterpret is not None:
        compiled = _compile_bitwise(
            r_numel, r_numel, r_numel, (r_numel,), (1,), (1,), r_storage, op_kind
        )
    else:
        compiled = _compile_bitwise(
            math.prod(a_shape), math.prod(b_shape), out_numel, out_shape,
            a_strides, b_strides, storage, op_kind,
        )

    def invoke(a, b):
        if tuple(a.shape) != a_shape or tuple(b.shape) != b_shape:
            raise ValueError("Bitwise kernel shape mismatch")
        a_flat = a.reshape(-1).view(torch.uint8) if dtype == torch.bool else a.reshape(-1)
        b_flat = b.reshape(-1).view(torch.uint8) if dtype == torch.bool else b.reshape(-1)
        if reinterpret is not None:
            a_flat = a_flat.view(reinterpret)
            b_flat = b_flat.view(reinterpret)
        result = compiled(a_flat, b_flat)
        if reinterpret is not None:
            result = result.view(torch.uint8 if dtype == torch.bool else dtype)
        if dtype == torch.bool:
            result = result.view(torch.bool)
        return result if tuple(out_shape) == (result.numel(),) else result.reshape(out_shape)

    invoke.compiled = compiled
    return invoke
