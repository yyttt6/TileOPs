"""Shared Ascend AIV max-pool-with-indices kernel.

Fixed 1D/2D/3D pooling shares one NCDHW coordinate template. Adaptive 2D
pooling uses the same reduction and dual-output writeback with adaptive bins.
"""

from functools import lru_cache
import math

import tilelang
import tilelang.language as T
import torch

from .common import grid_repeat_count, launch_block_count
from .pool import _dtype_name, _output_dim
from .pool_avg import _normalize_fixed_pool


_VEC = 2
_TILE = 64
_INDEX_SENTINEL = -7
_VALUE_SENTINEL = 123.0


def _adaptive_output_size(output_size, h_in, w_in):
    if output_size is None:
        result = (h_in, w_in)
    elif isinstance(output_size, int) and not isinstance(output_size, bool):
        result = (output_size, output_size)
    elif isinstance(output_size, (tuple, list)) and len(output_size) == 2:
        result = (
            h_in if output_size[0] is None else output_size[0],
            w_in if output_size[1] is None else output_size[1],
        )
    else:
        raise TypeError(
            "output_size must be an int, None, or a length-2 tuple/list, "
            f"got {output_size!r}"
        )
    if not all(isinstance(item, int) and not isinstance(item, bool) for item in result):
        raise TypeError(
            f"output_size entries must be positive ints or None, got {output_size!r}"
        )
    if min(result) <= 0:
        raise ValueError(f"output_size entries must be positive, got {output_size!r}")
    return int(result[0]), int(result[1])


def _max_adaptive_extent(input_size, output_size):
    return max(
        ((index + 1) * input_size + output_size - 1) // output_size
        - (index * input_size) // output_size
        for index in range(output_size)
    )


@lru_cache(maxsize=256)
def _compile_pool_indices(
    input_shape,
    output_shape,
    kernel,
    stride,
    padding,
    dilation,
    dtype_name,
    adaptive,
    reduction,
):
    ndim = len(input_shape) - 2
    input_dims = (1,) * (3 - ndim) + input_shape[2:]
    output_dims = (1,) * (3 - ndim) + output_shape[2:]
    kernel = (1,) * (3 - ndim) + kernel
    stride = (1,) * (3 - ndim) + stride
    padding = (0,) * (3 - ndim) + padding
    dilation = (1,) * (3 - ndim) + dilation
    total = math.prod(output_shape)
    input_total = math.prod(input_shape)
    logical_blocks = math.ceil(total / (_VEC * _TILE))
    launch_blocks = launch_block_count(logical_blocks)
    repeats = grid_repeat_count(logical_blocks, launch_blocks)
    padded_total = logical_blocks * _VEC * _TILE

    @tilelang.jit(
        out_idx=[],
        pass_configs={
            tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
            tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
        },
    )
    def factory():
        @T.prim_func
        def main(
            A: T.Tensor((input_total,), dtype_name),
            values_gm: T.Tensor((padded_total,), dtype_name),
            indices_gm: T.Tensor((padded_total,), "int64"),
        ):
            with T.Kernel(launch_blocks, is_npu=True) as (cid, vid):
                gathered = T.alloc_ub((_TILE,), dtype_name)
                gathered_fp32 = T.alloc_ub((_TILE,), "float32")
                best_values = T.alloc_ub((_TILE,), "float32")
                sum_values = T.alloc_ub((_TILE,), "float32")
                value_output = T.alloc_ub((_TILE,), dtype_name)
                best_indices = T.alloc_ub((_TILE,), "int64")
                first_valid = T.alloc_ub((_TILE,), "int32")

                with T.Scope("V"):
                    for repeat in T.serial(repeats):
                        logical_cid = cid + repeat * launch_blocks
                        if logical_cid < logical_blocks:
                            start = logical_cid * (_VEC * _TILE) + vid * _TILE
                            T.tile.fill(best_values, -T.infinity("float32"))
                            T.tile.fill(sum_values, 0.0)
                            T.tile.fill(first_valid, 1)
                            for lane in T.serial(_TILE):
                                best_indices[lane] = 0

                            for tap in T.serial(math.prod(kernel)):
                                T.tile.fill(gathered, 0)
                                for lane in T.serial(_TILE):
                                    output_index = start + lane
                                    if output_index < total:
                                        ow = output_index % output_dims[2]
                                        output_row = output_index // output_dims[2]
                                        oh = output_row % output_dims[1]
                                        output_plane = output_row // output_dims[1]
                                        od = output_plane % output_dims[0]
                                        channel_batch = output_plane // output_dims[0]
                                        channel = channel_batch % input_shape[1]
                                        batch = channel_batch // input_shape[1]
                                        kw = tap % kernel[2]
                                        tap_row = tap // kernel[2]
                                        kh = tap_row % kernel[1]
                                        kd = tap_row // kernel[1]
                                        adaptive_start_h = (
                                            oh * input_dims[1]
                                        ) // output_dims[1]
                                        adaptive_end_h = (
                                            (oh + 1) * input_dims[1]
                                            + output_dims[1]
                                            - 1
                                        ) // output_dims[1]
                                        adaptive_start_w = (
                                            ow * input_dims[2]
                                        ) // output_dims[2]
                                        adaptive_end_w = (
                                            (ow + 1) * input_dims[2]
                                            + output_dims[2]
                                            - 1
                                        ) // output_dims[2]
                                        id_ = T.if_then_else(
                                            adaptive,
                                            0,
                                            od * stride[0]
                                            - padding[0]
                                            + kd * dilation[0],
                                        )
                                        ih = T.if_then_else(
                                            adaptive,
                                            adaptive_start_h + kh,
                                            oh * stride[1]
                                            - padding[1]
                                            + kh * dilation[1],
                                        )
                                        iw = T.if_then_else(
                                            adaptive,
                                            adaptive_start_w + kw,
                                            ow * stride[2]
                                            - padding[2]
                                            + kw * dilation[2],
                                        )
                                        valid = T.if_then_else(
                                            adaptive,
                                            ih < adaptive_end_h and iw < adaptive_end_w,
                                            id_ >= 0
                                            and id_ < input_dims[0]
                                            and ih >= 0
                                            and ih < input_dims[1]
                                            and iw >= 0
                                            and iw < input_dims[2],
                                        )
                                        if valid:
                                            input_offset = (
                                                (
                                                    (batch * input_shape[1] + channel)
                                                    * input_dims[0]
                                                    + id_
                                                )
                                                * input_dims[1]
                                                + ih
                                            ) * input_dims[2] + iw
                                            gathered[lane] = A[input_offset]

                                if dtype_name == "float32":
                                    T.copy(gathered, gathered_fp32)
                                else:
                                    T.tile.cast(
                                        gathered_fp32,
                                        gathered,
                                        "CAST_NONE",
                                        _TILE,
                                    )

                                for lane in T.serial(_TILE):
                                    output_index = start + lane
                                    if output_index < total:
                                        ow = output_index % output_dims[2]
                                        output_row = output_index // output_dims[2]
                                        oh = output_row % output_dims[1]
                                        output_plane = output_row // output_dims[1]
                                        od = output_plane % output_dims[0]
                                        kw = tap % kernel[2]
                                        tap_row = tap // kernel[2]
                                        kh = tap_row % kernel[1]
                                        kd = tap_row // kernel[1]
                                        adaptive_start_h = (
                                            oh * input_dims[1]
                                        ) // output_dims[1]
                                        adaptive_end_h = (
                                            (oh + 1) * input_dims[1]
                                            + output_dims[1]
                                            - 1
                                        ) // output_dims[1]
                                        adaptive_start_w = (
                                            ow * input_dims[2]
                                        ) // output_dims[2]
                                        adaptive_end_w = (
                                            (ow + 1) * input_dims[2]
                                            + output_dims[2]
                                            - 1
                                        ) // output_dims[2]
                                        id_ = T.if_then_else(
                                            adaptive,
                                            0,
                                            od * stride[0]
                                            - padding[0]
                                            + kd * dilation[0],
                                        )
                                        ih = T.if_then_else(
                                            adaptive,
                                            adaptive_start_h + kh,
                                            oh * stride[1]
                                            - padding[1]
                                            + kh * dilation[1],
                                        )
                                        iw = T.if_then_else(
                                            adaptive,
                                            adaptive_start_w + kw,
                                            ow * stride[2]
                                            - padding[2]
                                            + kw * dilation[2],
                                        )
                                        valid = T.if_then_else(
                                            adaptive,
                                            ih < adaptive_end_h and iw < adaptive_end_w,
                                            id_ >= 0
                                            and id_ < input_dims[0]
                                            and ih >= 0
                                            and ih < input_dims[1]
                                            and iw >= 0
                                            and iw < input_dims[2],
                                        )
                                        if valid:
                                            value = gathered_fp32[lane]
                                            flat_index = (
                                                T.cast(id_, "int64") * input_dims[1]
                                                + T.cast(ih, "int64")
                                            ) * input_dims[2] + T.cast(iw, "int64")
                                            if reduction == "avg":
                                                sum_values[lane] += value
                                            else:
                                                if T.isnan(value):
                                                    best_values[lane] = value
                                                    best_indices[lane] = T.cast(
                                                        flat_index, "int64"
                                                    )
                                                elif not T.isnan(best_values[lane]) and (
                                                    first_valid[lane] != 0
                                                    or value > best_values[lane]
                                                ):
                                                    best_values[lane] = value
                                                    best_indices[lane] = T.cast(
                                                        flat_index, "int64"
                                                    )
                                                    first_valid[lane] = 0

                            if reduction == "avg":
                                for lane in T.serial(_TILE):
                                    output_index = start + lane
                                    if output_index < total:
                                        ow = output_index % output_dims[2]
                                        output_row = output_index // output_dims[2]
                                        oh = output_row % output_dims[1]
                                        start_h = (
                                            oh * input_dims[1]
                                        ) // output_dims[1]
                                        end_h = (
                                            (oh + 1) * input_dims[1]
                                            + output_dims[1]
                                            - 1
                                        ) // output_dims[1]
                                        start_w = (
                                            ow * input_dims[2]
                                        ) // output_dims[2]
                                        end_w = (
                                            (ow + 1) * input_dims[2]
                                            + output_dims[2]
                                            - 1
                                        ) // output_dims[2]
                                        divisor = (end_h - start_h) * (
                                            end_w - start_w
                                        )
                                        best_values[lane] = sum_values[lane] / T.cast(
                                            divisor, "float32"
                                        )

                            if dtype_name == "float32":
                                T.copy(
                                    best_values,
                                    values_gm[start : start + _TILE],
                                )
                            else:
                                T.tile.cast(
                                    value_output,
                                    best_values,
                                    "CAST_RINT",
                                    _TILE,
                                )
                                T.copy(
                                    value_output,
                                    values_gm[start : start + _TILE],
                                )
                            T.copy(
                                best_indices,
                                indices_gm[start : start + _TILE],
                            )

        return main

    return factory()



# ---------------------------------------------------------------------------
# Adaptive value-only gather path (R197 round 2)
# ---------------------------------------------------------------------------
# Adaptive bins are `lo = (o*IN)//OUT`, `hi = ((o+1)*IN + OUT-1)//OUT`, so the
# window origin is not an arithmetic progression and `arith_progression` cannot
# build the offset table.  But the coordinate depends only on `(ow, kw)` -- not
# on `oh`, not on the slice, not on the batch -- so the table is a COMPILE-TIME
# CONSTANT of size `OW*KW`, which is at most 63 entries for every adaptive case
# in the manifest.  A scalar loop of that length costs ~7 us per vector core
# (§6 fact A); a scalar loop over LANES, which is what the shipped template
# runs, does not scale and is why these operators sit at 0.03-0.09.
#
# The bins have unequal extents, so taps past a bin's `hi` carry a large
# negative sentinel; the `clamp_min(off, 0)` the tap loop needs anyway lands
# them on `plane[0]`, which holds the reduction identity.
#
# Adaptive outputs are tiny (1x1 .. 7x7), so one slice is only 16-112 lanes and
# the tap loop would be pure call overhead.  G consecutive (n, c) planes are
# therefore staged together -- they are adjacent in GM, so it stays ONE DMA --
# and the whole group shares one lane vector.  Offsets for group member g are
# the single-slice table plus `g * H*W*esz`, which is one vector add per (g, kw)
# at table-build time.

_ADAPT_UB_BUDGET = 180224
_ADAPT_LANE_ALIGN = {2: 16, 4: 8}
_ADAPT_SENTINEL = 1 << 20
_ADAPT_PASS_CFG = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}
_ADAPT_ESZ = {"float16": 2, "bfloat16": 2, "float32": 4}
_ADAPT_G_OVERRIDE = None   # set by probe_adaptive_G.py only
_ADAPT_LANE_TARGET = 32    # measured; see _AdaptiveGeometry


def _adapt_align_up(value, grain):
    return ((value + grain - 1) // grain) * grain


class _AdaptiveGeometry:
    """Compile-time geometry of the adaptive gather path, or `fits = False`."""

    def __init__(self, h_in, w_in, out_h, out_w, planes, dtype_name, reduction):
        esz = _ADAPT_ESZ[dtype_name]
        grain = 32 // esz
        self.H, self.W, self.OH, self.OW = h_in, w_in, out_h, out_w
        self.planes, self.dtype_name, self.esz, self.grain = planes, dtype_name, esz, grain
        self.reduction = reduction
        self.KH = _max_adaptive_extent(h_in, out_h)
        self.KW = _max_adaptive_extent(w_in, out_w)
        self.SLOT = out_h * out_w        # output positions per plane
        self.PLANE_ELEMS = h_in * w_in
        self.fits = False

        # Lane order is (pos, g), NOT (g, oh, ow).  With the group index
        # innermost, the offsets for a fixed output position across the group
        # are an ARITHMETIC PROGRESSION (stride = H*W*esz), so one
        # `arith_progression` builds a whole row and no per-`ow` padding is
        # needed.  The old (g, oh, ow) order had to pad `ow` out to
        # `OWP = align_up(OW, 16)`, which on a 1x1 adaptive output meant 15 of
        # every 16 lanes were idle -- that is what held AdaptiveAvgPool2d at
        # 0.50 on `resnet-global` (R197 §16.3).  Now only the group dimension
        # is padded, and it is padded to at most 63 out of `G`.
        per_slot_bytes = (
            (self.KH + self.KW) * 4                       # rowbase + colbase
            + 4 * 2                                       # off_i32 + off_u32
            + esz * 2                                     # got + out
            + 4 * 2                                       # got_f32 + acc
            + (4 if reduction == "avg" else 0)            # lane_div
        )
        fixed = 4 * grain * esz + 16 * 64 * 4
        # Every UB region offset must be a 32-byte multiple, so the group row
        # is padded to 8 int32 lanes.  (The argmax variant would additionally
        # need 256-byte operands for `T.tile.compare` -- §13.55.3 -- i.e. a
        # 64-lane pad; that is not imposed here because the value-only path
        # uses no `compare`.)
        gp_align = 8
        # G trades vector length against parallelism, and the optimum is not
        # "as large as fits": units = planes/G caps the launch blocks, so a big
        # G starves the machine.  Swept over every divisor on all three manifest
        # shapes (probe_adaptive_G.log); the measured best is reproduced exactly
        # by "grow G only until the lane vector reaches _ADAPT_LANE_TARGET, then
        # stop and spend the rest on units":
        #   resnet-global (SLOT=1)  best G=32  (G=1 is 10.4x slower, G=512 1.2x)
        #   spp-6x6      (SLOT=36)  best G=1   (grouping only removes units)
        #   nondiv-7x7   (SLOT=49)  best G=1
        best = None
        cands = [c for c in range(1, min(planes, 4096) + 1) if planes % c == 0]
        if _ADAPT_G_OVERRIDE is not None:
            cands = [c for c in cands if c == _ADAPT_G_OVERRIDE]
        fitting = []
        for cand in cands:
            gp = _adapt_align_up(cand, gp_align)
            lanes = self.SLOT * gp
            ub = fixed + cand * self.PLANE_ELEMS * esz + lanes * per_slot_bytes
            if ub > _ADAPT_UB_BUDGET:
                continue
            fitting.append((cand, gp, lanes, ub))
            if lanes >= _ADAPT_LANE_TARGET:
                best = (cand, gp, lanes, ub)
                break
        if best is None and fitting:
            best = fitting[-1]
        if best is None:
            # nothing fits: leave `fits` False and give the attributes the
            # dispatcher reads sane values so it can fall through cleanly.
            self.G, self.GP, self.LANES = 1, gp_align, self.SLOT * gp_align
            self.PLANE = self.PLANE_ELEMS + 2 * grain
            self.DATA_OFF, self.MAX_OFF = grain, 0
            self.NGROUP = self.UNITS = planes
            self.logical_blocks = self.launch_blocks = self.grid_repeats = 1
            self.ub_bytes = 1 << 30
            return
        self.G, self.GP, self.LANES, ub = best
        self.PLANE = self.G * self.PLANE_ELEMS + 2 * grain
        self.DATA_OFF = grain
        self.MAX_OFF = (self.DATA_OFF + self.G * self.PLANE_ELEMS - 1) * esz
        self.NGROUP = planes // self.G
        self.UNITS = self.NGROUP
        self.ub_bytes = ub
        self.logical_blocks = max(1, math.ceil(self.UNITS / _VEC))
        self.launch_blocks = min(self.logical_blocks, 24)
        self.grid_repeats = max(
            1, (self.logical_blocks + self.launch_blocks - 1) // self.launch_blocks)
        self.fits = self.ub_bytes <= _ADAPT_UB_BUDGET


@lru_cache(maxsize=128)
def _compile_adaptive_gather(h_in, w_in, out_h, out_w, planes, dtype_name, reduction):
    g = _AdaptiveGeometry(h_in, w_in, out_h, out_w, planes, dtype_name, reduction)
    if not g.fits:
        return None

    H, W, OH, OW = g.H, g.W, g.OH, g.OW
    KH, KW, ESZ, GRAIN = g.KH, g.KW, g.esz, g.grain
    G, GP, SLOT, LANES = g.G, g.GP, g.SLOT, g.LANES
    PLANE, DATA_OFF, MAX_OFF = g.PLANE, g.DATA_OFF, g.MAX_OFF
    PE = g.PLANE_ELEMS
    UNITS, LAUNCH, REPEATS, LOGICAL = (
        g.UNITS, g.launch_blocks, g.grid_repeats, g.logical_blocks)
    BIG = _ADAPT_SENTINEL
    IS_F32 = dtype_name == "float32"
    IDENT = 0.0 if reduction == "avg" else -T.infinity("float32")
    input_total = planes * PE

    @tilelang.jit(out_idx=[-1], pass_configs=_ADAPT_PASS_CFG)
    def factory():
        @T.prim_func
        def main(A: T.Tensor((input_total,), dtype_name),
                 C: T.Tensor((UNITS * LANES,), dtype_name)):
            with T.Kernel(LAUNCH, is_npu=True) as (cid, vid):
                plane = T.alloc_ub((PLANE,), dtype_name)
                colbase = T.alloc_ub((KW * LANES,), "int32")
                rowbase = T.alloc_ub((KH * LANES,), "int32")
                grow = T.alloc_ub((GP,), "int32")
                growf = T.alloc_ub((GP,), "float32")
                off_i32 = T.alloc_ub((LANES,), "int32")
                off_u32 = T.alloc_ub((LANES,), "uint32")
                got = T.alloc_ub((LANES,), dtype_name)
                got_f32 = T.alloc_ub((LANES,), "float32")
                acc = T.alloc_ub((LANES,), "float32")
                out = T.alloc_ub((LANES,), dtype_name)
                lane_div = T.alloc_ub((LANES if reduction == "avg" else 1,), "float32")

                with T.Scope("V"):
                    # ---- adaptive tables ------------------------------------
                    # Lane order is (pos, g).  For a fixed output position the
                    # group dimension is an arithmetic progression of stride
                    # PE*esz, so each row is ONE vector call -- no scalar loop
                    # over lanes anywhere.  `pos` runs over OH*OW <= 49, and the
                    # bin arithmetic is evaluated at Python/TIR scalar level.
                    # A statement-level `if` is used rather than an
                    # `if_then_else` scalar operand: passing the latter to
                    # T.tile.add makes the dispatcher try to take a buffer
                    # pointer of it ("Unexpected type for access_ptr first
                    # argument: tir.NE").
                    for kw in T.serial(KW):
                        for pos in T.serial(SLOT):
                            ow = pos % OW
                            lo = (ow * W) // OW
                            hi = ((ow + 1) * W + OW - 1) // OW
                            iw = lo + kw
                            if iw < hi:
                                T.tile.fill(grow, iw * ESZ)
                            else:
                                T.tile.fill(grow, -BIG)
                            T.copy(grow, colbase[kw * LANES + pos * GP:
                                                 kw * LANES + pos * GP + GP])
                    for kh in T.serial(KH):
                        for pos in T.serial(SLOT):
                            oh = pos // OW
                            lo = (oh * H) // OH
                            hi = ((oh + 1) * H + OH - 1) // OH
                            ih = lo + kh
                            if ih < hi:
                                # group member g reads plane g: + g*PE*esz
                                T.tile.arith_progression(
                                    grow, (DATA_OFF + ih * W) * ESZ, PE * ESZ, GP)
                            else:
                                T.tile.fill(grow, -BIG)
                            T.copy(grow, rowbase[kh * LANES + pos * GP:
                                                 kh * LANES + pos * GP + GP])
                    if reduction == "avg":
                        for pos in T.serial(SLOT):
                            ow = pos % OW
                            oh = pos // OW
                            cw = ((ow + 1) * W + OW - 1) // OW - (ow * W) // OW
                            ch = ((oh + 1) * H + OH - 1) // OH - (oh * H) // OH
                            T.tile.fill(growf, T.cast(cw * ch, "float32"))
                            T.copy(growf, lane_div[pos * GP: pos * GP + GP])

                    for rep in T.serial(REPEATS):
                        lcid = cid + rep * LAUNCH
                        if lcid < LOGICAL:
                            unit = lcid * _VEC + vid
                            if unit < UNITS:
                                # G consecutive planes are adjacent in GM, so the
                                # whole group is one DMA.
                                T.tile.fill(plane[0:GRAIN], IDENT)
                                T.copy(A[unit * G * PE: unit * G * PE + G * PE],
                                       plane[DATA_OFF:DATA_OFF + G * PE])
                                # T.tile.gather reads `plane` through a computed
                                # pointer; auto-sync does not see it (§13.55.1).
                                T.barrier_all()
                                T.tile.fill(acc, IDENT)
                                for tap in T.serial(KH * KW):
                                    kh = tap // KW
                                    kw = tap % KW
                                    T.tile.add(off_i32,
                                               colbase[kw * LANES: kw * LANES + LANES],
                                               rowbase[kh * LANES: kh * LANES + LANES])
                                    T.tile.clamp_min(off_i32, off_i32, 0, LANES)
                                    T.tile.clamp_max(off_i32, off_i32, MAX_OFF, LANES)
                                    T.reinterpretcast(off_u32, off_i32, "uint32_t")
                                    T.tile.gather(got, plane, off_u32, 0)
                                    if IS_F32:
                                        T.copy(got, got_f32)
                                    else:
                                        T.tile.cast(got_f32, got, "CAST_NONE", LANES)
                                    if reduction == "avg":
                                        T.tile.add(acc, acc, got_f32)
                                    else:
                                        T.tile.max(acc, acc, got_f32)
                                if reduction == "avg":
                                    T.tile.div(acc, acc, lane_div)
                                if IS_F32:
                                    T.copy(acc, out)
                                else:
                                    T.tile.cast(out, acc, "CAST_RINT", LANES)
                                T.barrier_all()
                                T.copy(out, C[unit * LANES: (unit + 1) * LANES])
        return main

    compiled = factory()
    compiled.geometry = g
    return compiled


def _make_launch(compiled, output_shape, logical_blocks, path_kind):
    total = math.prod(output_shape)
    padded_total = logical_blocks * _VEC * _TILE

    def launch_into(x, values, indices):
        if values.numel() != padded_total or indices.numel() != padded_total:
            raise ValueError(
                f"explicit pool outputs must each contain {padded_total} elements"
            )
        if values.dtype != x.dtype or indices.dtype != torch.int64:
            raise TypeError("pool outputs must use input dtype and torch.int64")
        compiled(x, values, indices)
        return (
            values[:total].view(*output_shape),
            indices[:total].view(*output_shape),
        )

    def launch(x):
        values = torch.empty((padded_total,), dtype=x.dtype, device=x.device)
        indices = torch.empty((padded_total,), dtype=torch.int64, device=x.device)
        return launch_into(x, values, indices)

    launch.launch_into = launch_into
    launch.output_shape = output_shape
    launch.logical_blocks = logical_blocks
    launch.launch_blocks = launch_block_count(logical_blocks)
    launch.padded_total = padded_total
    launch.path_kind = path_kind
    launch.compiled = compiled
    launch.value_sentinel = _VALUE_SENTINEL
    launch.index_sentinel = _INDEX_SENTINEL
    return launch


def build_max_pool_indices_kernel(
    input_shape,
    dtype,
    *,
    ndim,
    kernel_size,
    stride=None,
    padding=0,
    dilation=1,
    ceil_mode=False,
):
    if len(input_shape) != ndim + 2:
        layout = {1: "NCL", 2: "NCHW", 3: "NCDHW"}[ndim]
        raise ValueError(
            f"MaxPool{ndim}dIndicesFwdOp expects a {ndim + 2}D {layout} input, "
            f"got {input_shape}"
        )
    input_shape = tuple(int(item) for item in input_shape)
    if min(input_shape) <= 0:
        raise ValueError(f"pooling requires non-empty dimensions, got {input_shape}")
    kernel, stride, padding, dilation = _normalize_fixed_pool(
        ndim,
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode,
    )
    output_dims = tuple(
        _output_dim(size, k, step, pad, dil, ceil_mode)
        for size, k, step, pad, dil in zip(
            input_shape[2:], kernel, stride, padding, dilation
        )
    )
    if min(output_dims) <= 0:
        raise ValueError(
            f"pooling output dimensions must be positive, got {output_dims}"
        )
    output_shape = input_shape[:2] + output_dims
    logical_blocks = math.ceil(math.prod(output_shape) / (_VEC * _TILE))
    compiled = _compile_pool_indices(
        input_shape,
        output_shape,
        kernel,
        stride,
        padding,
        dilation,
        _dtype_name(dtype),
        False,
        "max",
    )
    return _make_launch(compiled, output_shape, logical_blocks, f"max{ndim}d_indices")


def build_adaptive_max_pool2d_indices_kernel(input_shape, dtype, *, output_size):
    if len(input_shape) != 4:
        raise ValueError(
            f"AdaptiveMaxPool2dIndicesFwdOp expects a 4D NCHW input, got {input_shape}"
        )
    input_shape = tuple(int(item) for item in input_shape)
    if min(input_shape) <= 0:
        raise ValueError(f"pooling requires non-empty dimensions, got {input_shape}")
    out_h, out_w = _adaptive_output_size(output_size, input_shape[2], input_shape[3])
    output_shape = input_shape[:2] + (out_h, out_w)
    kernel = (
        _max_adaptive_extent(input_shape[2], out_h),
        _max_adaptive_extent(input_shape[3], out_w),
    )
    logical_blocks = math.ceil(math.prod(output_shape) / (_VEC * _TILE))
    compiled = _compile_pool_indices(
        input_shape,
        output_shape,
        kernel,
        (1, 1),
        (0, 0),
        (1, 1),
        _dtype_name(dtype),
        True,
        "max",
    )
    return _make_launch(
        compiled,
        output_shape,
        logical_blocks,
        "adaptive_max2d_indices",
    )


def build_adaptive_pool2d_kernel(
    input_shape, dtype, *, output_size, reduction
):
    if reduction not in {"avg", "max"}:
        raise ValueError(f"unsupported adaptive pool reduction {reduction!r}")
    if len(input_shape) != 4:
        raise ValueError(
            f"Adaptive{reduction.title()}Pool2dFwdOp expects a 4D NCHW input, "
            f"got {input_shape}"
        )
    input_shape = tuple(int(item) for item in input_shape)
    if min(input_shape) <= 0:
        raise ValueError(f"pooling requires non-empty dimensions, got {input_shape}")
    out_h, out_w = _adaptive_output_size(
        output_size, input_shape[2], input_shape[3]
    )
    output_shape = input_shape[:2] + (out_h, out_w)
    kernel = (
        _max_adaptive_extent(input_shape[2], out_h),
        _max_adaptive_extent(input_shape[3], out_w),
    )
    # --- fast path: one vector gather per window tap (R197) -------------------
    dtype_name = _dtype_name(dtype)
    planes = input_shape[0] * input_shape[1]
    adapt = _AdaptiveGeometry(input_shape[2], input_shape[3], out_h, out_w,
                              planes, dtype_name, reduction)
    if adapt.fits:
        compiled = _compile_adaptive_gather(
            input_shape[2], input_shape[3], out_h, out_w, planes, dtype_name,
            reduction)
        ngroup, gsz, gp, slot = adapt.NGROUP, adapt.G, adapt.GP, adapt.SLOT

        def launch(x):
            # kernel writes (unit, pos, gp); the group index is innermost, so
            # recovering (plane, oh, ow) needs one transpose.  The output is
            # tiny for every adaptive shape, so this costs far less than the
            # lane padding the (pos, g) order removes.
            raw = compiled(x.reshape(-1))
            return (raw.view(ngroup, slot, gp)[:, :, :gsz]
                    .permute(0, 2, 1).reshape(*output_shape))

        launch.output_shape = output_shape
        launch.logical_blocks = adapt.logical_blocks
        launch.launch_blocks = adapt.launch_blocks
        launch.path_kind = f"adaptive_{reduction}2d_gather"
        launch.geometry = adapt
        launch.compiled = compiled
        return launch

    logical_blocks = math.ceil(math.prod(output_shape) / (_VEC * _TILE))
    compiled = _compile_pool_indices(
        input_shape,
        output_shape,
        kernel,
        (1, 1),
        (0, 0),
        (1, 1),
        _dtype_name(dtype),
        True,
        reduction,
    )
    dual_launch = _make_launch(
        compiled,
        output_shape,
        logical_blocks,
        f"adaptive_{reduction}2d",
    )

    def launch(x):
        return dual_launch(x)[0]

    launch.output_shape = output_shape
    launch.logical_blocks = dual_launch.logical_blocks
    launch.launch_blocks = dual_launch.launch_blocks
    launch.padded_total = dual_launch.padded_total
    launch.path_kind = dual_launch.path_kind
    launch.compiled = compiled
    return launch


__all__ = [
    "build_adaptive_pool2d_kernel",
    "build_adaptive_max_pool2d_indices_kernel",
    "build_max_pool_indices_kernel",
]
