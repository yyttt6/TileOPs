"""Shared Ascend AIV fixed-window pooling for avg 1D/2D/3D and max 3D.

Two paths.

``_compile_gather_pool`` (R197) is the fast one.  Its unit of work is one output
slice ``(n, c, od)``; for every depth tap it pulls the corresponding input plane
into UB with one DMA and turns each ``(kh, kw)`` tap into a single vector
``T.tile.gather`` over a precomputed byte-offset table plus one vector reduce.
Nothing in the per-element path is scalar.

``_compile_fixed_pool`` is the original per-lane path, kept as the fallback for
geometries whose UB working set does not fit (a single output row wider than the
unified buffer, e.g. ``AvgPool1d`` on a 32000-long signal).  R197 removed its
duplicated scalar reduction pass; see the comment there.

Why this shape: on dav-2201 a scalar loop iteration that writes UB costs about
100 ns whatever its body contains, and the original template ran
``outputs x window_size x 2`` of them, which is the whole reason the pool family
sat at 0.1-6 GB/s.  Measurements in ``docs/reports/R197.md``.
"""

from functools import lru_cache
import math

import tilelang
import tilelang.language as T

from .common import grid_repeat_count, launch_block_count
from .pool import _dtype_name, _output_dim


_VEC = 2
_TILE = 64


def _spatial_tuple(name, value, ndim):
    if isinstance(value, int) and not isinstance(value, bool):
        result = (value,) * ndim
    elif isinstance(value, (tuple, list)) and len(value) == ndim:
        result = tuple(value)
    else:
        raise TypeError(
            f"{name} must be an int or a length-{ndim} tuple/list, got {value!r}"
        )
    if not all(isinstance(item, int) and not isinstance(item, bool) for item in result):
        raise TypeError(f"{name} entries must be ints, got {value!r}")
    return tuple(int(item) for item in result)


def _normalize_fixed_pool(
    ndim,
    kernel_size,
    stride,
    padding,
    dilation,
    ceil_mode,
    *,
    divisor_override=None,
):
    kernel = _spatial_tuple("kernel_size", kernel_size, ndim)
    stride = kernel if stride is None else _spatial_tuple("stride", stride, ndim)
    padding = _spatial_tuple("padding", padding, ndim)
    dilation = _spatial_tuple("dilation", dilation, ndim)
    if any(item <= 0 for item in kernel):
        raise ValueError("kernel_size entries must be positive")
    if any(item <= 0 for item in stride):
        raise ValueError("stride entries must be positive")
    if any(item <= 0 for item in dilation):
        raise ValueError("dilation entries must be positive")
    if any(pad < 0 or pad > size // 2 for pad, size in zip(padding, kernel)):
        raise ValueError(
            "padding must be non-negative and at most half the kernel size"
        )
    if not isinstance(ceil_mode, bool):
        raise TypeError("ceil_mode must be bool")
    if divisor_override is not None and (
        not isinstance(divisor_override, int) or isinstance(divisor_override, bool)
    ):
        raise TypeError("divisor_override must be an int or None")
    if divisor_override == 0:
        raise ValueError("divisor_override must not be zero")
    return kernel, stride, padding, dilation


UB_BUDGET_BYTES = 180224
# Weak lever here -- +-20%, not the 13x R194 saw on the unary family, because
# pool is not block-dispatch bound.
#
# Picking the value is a cautionary tale.  An offline sweep (8 values x 6
# geometries, probe_launch_cap*.log) says 48 is the global optimum: normalised
# per geometry it averages 1.003 vs 1.016 for 24.  But that probe times each
# call with a host timer in the EAGER regime, and the headline number is the
# GRAPH regime.  Measured through the official harness, 24 beats 48 on three of
# the four operators (AvgPool2d 0.979 vs 0.887, AvgPool3d 1.274 vs 1.255,
# MaxPool2d 0.690 vs 0.682; only MaxPool3d prefers 48, 0.828 vs 0.840).
# The harness is the authority, so: 24.  Do not re-derive this from the probe.
# cap=32 is reproducibly the worst setting on every geometry tried, in both
# regimes, and has no explanation yet.
LAUNCH_BLOCK_CAP = 24
_VEC = 2
_SENTINEL_SCALE = 1 << 20      # > any legal byte offset into the UB plane
def _lane_align(esz):
    """Lane-row padding that keeps every table's `oh` row 32-byte aligned.

    The offset tables are int32 (8 elements per 32 bytes) and the value tables
    are the input dtype (32/esz per 32 bytes), so the row length has to be a
    multiple of both.  Hard-coding 16 wasted 12% of the lanes on float32.
    """
    return 16 if esz == 2 else 8

_GATHER_PASS_CFG = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

_ESZ = {"float16": 2, "bfloat16": 2, "float32": 4}


def _align_up(value, grain):
    return ((value + grain - 1) // grain) * grain


class _GatherGeometry:
    """Compile-time geometry of the fast path, or ``None`` if it does not fit."""

    def __init__(self, input_dims, output_dims, kernel, stride, padding, dilation,
                 batch_channels, dtype_name, reduction, count_include_pad,
                 divisor_override):
        d_in, h_in, w_in = input_dims
        od, oh, ow = output_dims
        esz = _ESZ[dtype_name]
        grain = 32 // esz

        self.D, self.H, self.W = d_in, h_in, w_in
        self.OD, self.OH, self.OW = od, oh, ow
        self.KD, self.KH, self.KW = kernel
        self.SD, self.SH, self.SW = stride
        self.PD, self.PH, self.PW = padding
        self.DD, self.DH, self.DW = dilation
        self.NC = batch_channels
        self.dtype_name = dtype_name
        self.esz = esz
        self.grain = grain
        self.reduction = reduction
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override

        # --- UB plane: PADT identity rows, H data rows, PADB identity rows -----
        # The UB row stride is padded to a 32-byte multiple.  Every UB buffer
        # region offset must be 32-byte aligned (probe_ub_align.log), and that
        # includes the region the plane load's tail over-write has to repair.
        # With WS aligned, every row's destination is aligned and the bytes the
        # over-write clobbers are the row's own right padding, which no legal
        # gather offset can reach (columns outside [0,W) carry the sentinel).
        # The plane can stay contiguous (WS == W, one flat DMA) whenever the
        # DATA REGION is a whole number of 32-byte blocks -- which is weaker than
        # requiring each ROW to be one.  `step` rows always span a whole number
        # of blocks, so a PADT that is a multiple of `step` keeps the data
        # region's start aligned, and `H*W*esz % 32 == 0` keeps its end aligned.
        # This matters a lot: in percase_round2.csv every case that fell back to
        # the row-by-row load is 3-4x slower than an equivalent one that did not
        # (vgg-block float32 flat = 107 us vs vgg-block float16 row = 422 us, on
        # the same shape with MORE bytes to move).
        step = 32 // math.gcd(w_in * esz, 32)
        if (h_in * w_in * esz) % 32 == 0:
            self.mode = "pad"          # materialised identity rows above/below
            self.WS = w_in
            self.PADT = _align_up(max(self.PH, 1), step)
        else:
            # The data region is not a whole number of 32-byte blocks, so no
            # choice of PADT lines both its ends up and the "pad" mode has to
            # fall back to H separate row DMAs -- which costs 3-4x (§5.1a).
            # "sum" mode drops the padding rows entirely: the plane is
            # [GRAIN identity][H*W data][GRAIN slack], so the data always starts
            # at an aligned offset and the copy's tail lands in slack nothing
            # reads.  Vertical validity then has to move into the table, and it
            # is kept additive -- a row table indexed by kh plus a column table
            # indexed by kw -- so the cost is (KH+KW) tables, not KH*KW.
            self.mode = "sum"
            self.WS = w_in
            self.PADT = 0
        self.flat_load = True
        if self.mode == "pad":
            max_ih = (oh - 1) * self.SH - self.PH + (self.KH - 1) * self.DH
            self.PADB = max(1, max_ih + 1 - h_in, -(-grain // self.WS))
            self.HPAD = self.PADT + h_in + self.PADB
            self.PLANE = self.HPAD * self.WS + grain   # + slack for the copy tail
            self.DATA_OFF = self.PADT * self.WS
        else:
            self.PADB = 0
            self.HPAD = h_in
            self.DATA_OFF = grain                      # one aligned identity block
            self.PLANE = grain + h_in * w_in + grain   # + slack for the copy tail

        # --- lane tables -------------------------------------------------------
        self.OWP = _align_up(ow, _lane_align(esz))
        self.LANES = oh * self.OWP
        self.OUT_SLICE = oh * ow
        # Every UB buffer-region offset must be a 32-byte multiple (measured:
        # probe_ub_align.log -- an unaligned T.copy or T.tile.add faults with
        # ADDR_MISALIGN).  So the lane rows stay padded to OWP all the way into
        # GM, and the host slices `[..., :OW]` back off.  When OWP == OW that
        # slice is free; otherwise it costs one torch copy of the output.
        self.needs_host_unpad = self.OWP != ow

        self.UNITS = batch_channels * od
        self.logical_blocks = max(1, math.ceil(self.UNITS / _VEC))
        self.launch_blocks = min(self.logical_blocks, LAUNCH_BLOCK_CAP)
        self.grid_repeats = max(
            1, (self.logical_blocks + self.launch_blocks - 1) // self.launch_blocks)

        # --- UB accounting -----------------------------------------------------
        lanes, owp = self.LANES, self.OWP
        ub = self.PLANE * esz                       # plane
        if self.mode == "pad":
            ub += self.KW * lanes * 4               # base_all (int32 byte offsets)
        else:
            ub += (self.KW + self.KH) * lanes * 4   # colbase + rowbase
        ub += lanes * 4                             # off_i32
        ub += lanes * 4                             # off_u32
        ub += lanes * esz                           # got
        ub += lanes * 4                             # got_f32
        ub += lanes * 4                             # acc
        ub += lanes * esz                           # out
        if reduction == "avg":
            ub += lanes * 4                         # lane_div
            ub += lanes * 4                         # div_f
        ub += 4 * owp * 4                           # colo / tmp0 / tmp1 / scr
        if reduction == "avg":
            ub += 2 * owp * 4                       # cw_f / scr_f
        self.ub_bytes = ub
        self.fits = ub <= UB_BUDGET_BYTES

    # ---- compile-time column geometry, used to build the sentinel vectorially --
    def col_bytes(self, kw):
        """(first_value, diff) of the byte-offset progression for column tap kw."""
        return (kw * self.DW - self.PW) * self.esz, self.SW * self.esz


def _identity(reduction, dtype_name):
    # T.infinity("bfloat16") raises ("Cannot decide infinity for type bfloat16")
    # and a Python float("-inf") lowers to CUDART_INF, which bisheng rejects.
    # A float32 -inf scalar duplicated into the buffer works for all three
    # supported dtypes (verified: probe_bf16_inf.log).
    return 0.0 if reduction == "avg" else -T.infinity("float32")


@lru_cache(maxsize=256)
def _compile_gather_pool(input_dims, output_dims, kernel, stride, padding, dilation,
                      batch_channels, dtype_name, reduction, count_include_pad,
                      divisor_override):
    g = _GatherGeometry(input_dims, output_dims, kernel, stride, padding, dilation,
                 batch_channels, dtype_name, reduction, count_include_pad,
                 divisor_override)
    if not g.fits:
        return None

    D, H, W = g.D, g.H, g.W
    OD, OH, OW, OWP = g.OD, g.OH, g.OW, g.OWP
    KD, KH, KW = g.KD, g.KH, g.KW
    SD, SH, SW = g.SD, g.SH, g.SW
    PD, PH, PW = g.PD, g.PH, g.PW
    DD, DH, DW = g.DD, g.DH, g.DW
    ESZ, GRAIN = g.esz, g.grain
    LANES, PLANE, DATA_OFF, PADT = g.LANES, g.PLANE, g.DATA_OFF, g.PADT
    OUT_SLICE = g.OUT_SLICE
    LAUNCH, REPEATS, LOGICAL = g.launch_blocks, g.grid_repeats, g.logical_blocks
    UNITS = g.UNITS
    WS = g.WS
    MODE = g.mode
    SUM_MODE = MODE == "sum"
    if SUM_MODE:
        MAX_OFF = (DATA_OFF + H * W - 1) * ESZ
        ROW_TABLE = KH * LANES          # rowbase, indexed by kh
        COL_TABLE = KW * LANES          # colbase, indexed by kw
    else:
        MAX_OFF = (g.HPAD * WS - 1) * ESZ
        ROW_TABLE = 1                   # unused; TVMScript needs it declared
        COL_TABLE = KW * LANES
    BIG = _SENTINEL_SCALE
    IDENT = _identity(reduction, dtype_name)
    IS_F32 = dtype_name == "float32"
    input_total = batch_channels * D * H * W
    USE_OVERRIDE = divisor_override is not None
    OVERRIDE = 1 if divisor_override is None else divisor_override
    W_BYTES = W * ESZ           # column-overflow bound (real data width)
    ROW_BYTES = WS * ESZ        # UB row stride
    H_HI, H_LO = (H + PH, -PH) if count_include_pad else (H, 0)
    D_HI, D_LO = (D + PD, -PD) if count_include_pad else (D, 0)
    DIV_LANES = LANES if reduction == "avg" else 1
    DIV_OWP = OWP if reduction == "avg" else 1

    @tilelang.jit(out_idx=[-1], pass_configs=_GATHER_PASS_CFG)
    def factory():
        @T.prim_func
        def main(A: T.Tensor((input_total,), dtype_name),
                 C: T.Tensor((UNITS * LANES,), dtype_name)):
            with T.Kernel(LAUNCH, is_npu=True) as (cid, vid):
                plane = T.alloc_ub((PLANE,), dtype_name)
                base_all = T.alloc_ub((COL_TABLE,), "int32")
                rowbase = T.alloc_ub((ROW_TABLE,), "int32")
                off_i32 = T.alloc_ub((LANES,), "int32")
                off_u32 = T.alloc_ub((LANES,), "uint32")
                got = T.alloc_ub((LANES,), dtype_name)
                got_f32 = T.alloc_ub((LANES,), "float32")
                acc = T.alloc_ub((LANES,), "float32")
                out = T.alloc_ub((LANES,), dtype_name)
                # max needs no divisor.  These cannot be allocated inside an
                # `if` -- TVMScript does not carry a buffer binding out of a
                # Python `if`, so a later `if reduction == "avg"` block would
                # fail with "Undefined variable".  Allocating them at length 1
                # is what keeps the kernel inside the budget its own accounting
                # predicts (MaxPool2d/resnet-stem/float32 missed by 1.4 KB).
                lane_div = T.alloc_ub((DIV_LANES,), "float32")
                div_f = T.alloc_ub((DIV_LANES,), "float32")
                colo = T.alloc_ub((OWP,), "int32")
                tmp0 = T.alloc_ub((OWP,), "int32")
                tmp1 = T.alloc_ub((OWP,), "int32")
                scr = T.alloc_ub((OWP,), "int32")
                cw_f = T.alloc_ub((DIV_OWP,), "float32")
                scr_f = T.alloc_ub((DIV_OWP,), "float32")

                with T.Scope("V"):
                    # ================= plane-independent tables ================
                    # base_all[kw][oh][ow] = byte offset of input element
                    #   (oh*SH - PH + PADT, ow*SW - PW + kw*DW)
                    # with a large negative sentinel where the column is outside
                    # the input.  Built entirely with vector ops: a scalar loop
                    # over LANES would cost ~100 ns per lane.
                    for kw in T.serial(KW):
                        T.tile.arith_progression(
                            colo, (kw * DW - PW) * ESZ, SW * ESZ, OWP)
                        # tmp0 = -1 where the column underflows, else 0
                        T.tile.clamp_max(tmp0, colo, 0, OWP)
                        T.tile.clamp_min(tmp0, tmp0, -1, OWP)
                        # tmp1 = +1 where the column overflows, else 0
                        T.tile.add(tmp1, colo, -(W_BYTES - 1))
                        T.tile.clamp_min(tmp1, tmp1, 0, OWP)
                        T.tile.clamp_max(tmp1, tmp1, 1, OWP)
                        T.tile.sub(tmp0, tmp0, tmp1)
                        T.tile.mul(tmp0, tmp0, BIG)
                        T.tile.add(colo, colo, tmp0)
                        for oh in T.serial(OH):
                            if SUM_MODE:
                                # column part only; the row part is added per tap
                                T.copy(colo, base_all[kw * LANES + oh * OWP:
                                                      kw * LANES + oh * OWP + OWP])
                            else:
                                T.tile.add(scr, colo,
                                           (oh * SH - PH + PADT) * ROW_BYTES)
                                T.copy(scr, base_all[kw * LANES + oh * OWP:
                                                     kw * LANES + oh * OWP + OWP])

                    if SUM_MODE:
                        # rowbase[kh][oh][*] = DATA_OFF + ih*W  (bytes), or a
                        # sentinel when the row is outside the input.  Adding a
                        # sentinel row to a sentinel column stays far negative,
                        # so `clamp_min(off, 0)` lands both on plane[0], which is
                        # the identity block.
                        for kh in T.serial(KH):
                            for oh in T.serial(OH):
                                ih = oh * SH - PH + kh * DH
                                in_range = T.if_then_else(
                                    ih >= 0, T.if_then_else(ih < H, 1, 0), 0)
                                T.tile.fill(scr, T.if_then_else(
                                    in_range != 0,
                                    (DATA_OFF + ih * W) * ESZ, -BIG))
                                T.copy(scr, rowbase[kh * LANES + oh * OWP:
                                                    kh * LANES + oh * OWP + OWP])

                    # per-lane divisor = count_h(oh) * count_w(ow); the depth
                    # factor is a per-slice scalar applied later.
                    if reduction == "avg":
                        if USE_OVERRIDE:
                            T.tile.fill(lane_div, T.cast(OVERRIDE, "float32"))
                        else:
                            T.tile.arith_progression(
                                cw_f, T.cast(-PW, "float32"), T.cast(SW, "float32"), OWP)
                            T.tile.add(scr_f, cw_f, T.cast(KW, "float32"))
                            if count_include_pad:
                                T.tile.clamp_max(scr_f, scr_f,
                                                 T.cast(W + PW, "float32"), OWP)
                                T.tile.clamp_min(cw_f, cw_f,
                                                 T.cast(-PW, "float32"), OWP)
                            else:
                                T.tile.clamp_max(scr_f, scr_f, T.cast(W, "float32"), OWP)
                                T.tile.clamp_min(cw_f, cw_f, T.cast(0, "float32"), OWP)
                            T.tile.sub(cw_f, scr_f, cw_f)
                            T.tile.clamp_min(cw_f, cw_f, T.cast(0, "float32"), OWP)
                            for oh in T.serial(OH):
                                # limits picked at Python level: TVMScript will not
                                # let one name be bound in two branches of an `if`.
                                start_h = oh * SH - PH
                                end_h = start_h + KH
                                ch = T.max(T.min(end_h, H_HI) - T.max(start_h, H_LO), 0)
                                T.tile.mul(scr_f, cw_f, T.cast(ch, "float32"))
                                T.copy(scr_f, lane_div[oh * OWP: oh * OWP + OWP])

                    # ========================= main loop =======================
                    for rep in T.serial(REPEATS):
                        lcid = cid + rep * LAUNCH
                        if lcid < LOGICAL:
                            unit = lcid * _VEC + vid
                            if unit < UNITS:
                                nc = unit // OD
                                od = unit % OD
                                if reduction == "avg":
                                    T.tile.fill(acc, 0.0)
                                else:
                                    T.tile.fill(acc, -T.infinity("float32"))

                                for kd in T.serial(KD):
                                    id_ = od * SD - PD + kd * DD
                                    if id_ >= 0 and id_ < D:
                                        gm = (nc * D + id_) * H * W
                                        if SUM_MODE:
                                            # plane[0:GRAIN] is the identity block
                                            # every sentinel lane clamps onto; the
                                            # copy's 32-byte tail over-write lands
                                            # in the trailing slack, which no legal
                                            # offset reaches (MAX_OFF stops at the
                                            # end of the data).
                                            T.tile.fill(plane[0:GRAIN], IDENT)
                                            T.copy(A[gm:gm + H * W],
                                                   plane[DATA_OFF:DATA_OFF + H * W])
                                        else:
                                            T.tile.fill(plane, IDENT)
                                            T.copy(A[gm:gm + H * W],
                                                   plane[DATA_OFF:DATA_OFF + H * W])
                                            # the copy rounds its destination up
                                            # to 32 bytes; repair the identity it
                                            # clobbered (aligned because WS is)
                                            T.tile.fill(
                                                plane[DATA_OFF + H * W:
                                                      DATA_OFF + H * W + GRAIN], IDENT)
                                        T.barrier_all()
                                        for tap in T.serial(KH * KW):
                                            kh = tap // KW
                                            kw = tap % KW
                                            if SUM_MODE:
                                                # column table + row table; both
                                                # can be the sentinel
                                                T.tile.add(
                                                    off_i32,
                                                    base_all[kw * LANES:
                                                             kw * LANES + LANES],
                                                    rowbase[kh * LANES:
                                                            kh * LANES + LANES])
                                            else:
                                                T.tile.add(
                                                    off_i32,
                                                    base_all[kw * LANES:
                                                             kw * LANES + LANES],
                                                    kh * DH * ROW_BYTES)
                                            T.tile.clamp_min(off_i32, off_i32, 0, LANES)
                                            T.tile.clamp_max(off_i32, off_i32,
                                                             MAX_OFF, LANES)
                                            T.reinterpretcast(off_u32, off_i32,
                                                              "uint32_t")
                                            T.tile.gather(got, plane, off_u32, 0)
                                            if IS_F32:
                                                T.copy(got, got_f32)
                                            else:
                                                T.tile.cast(got_f32, got,
                                                            "CAST_NONE", LANES)
                                            if reduction == "avg":
                                                T.tile.add(acc, acc, got_f32)
                                            else:
                                                T.tile.max(acc, acc, got_f32)

                                if reduction == "avg":
                                    if USE_OVERRIDE:
                                        T.copy(lane_div, div_f)
                                    else:
                                        start_d = od * SD - PD
                                        end_d = start_d + KD
                                        cd = T.max(T.min(end_d, D_HI)
                                                   - T.max(start_d, D_LO), 0)
                                        T.tile.mul(div_f, lane_div,
                                                   T.cast(cd, "float32"))
                                        T.tile.clamp_min(div_f, div_f, 1.0, LANES)
                                    T.tile.div(acc, acc, div_f)

                                if IS_F32:
                                    T.copy(acc, out)
                                else:
                                    T.tile.cast(out, acc, "CAST_RINT", LANES)
                                T.copy(out, C[unit * LANES: (unit + 1) * LANES])
        return main

    compiled = factory()
    compiled.geometry = g
    return compiled



# ---------------------------------------------------------------------------
# Chunked 1-D path (R197 round 2)
# ---------------------------------------------------------------------------
# `_compile_gather_pool` stages one whole (n, c, od) input plane in UB.  For a
# 1-D signal the "plane" IS the whole row, so AvgPool1d/long-temporal
# (L = 32000, OL = 2000) needs 768 KB and falls off the fast path entirely --
# it was the last operator in the family still running the per-lane scalar
# kernel, at ratio 0.0068.
#
# Here the OUTPUT row is cut into chunks of `OWC` lanes and only that chunk's
# input window is staged.  Two things make it cheap:
#
#   * the window start is clamped to `[0, W - SPAN]` so the DMA length is a
#     compile-time constant; the leftover `r = iw0 - gm_start` is a RUNTIME
#     SCALAR that folds into the per-tap offset constant the tap loop already
#     adds -- so handling the edges costs nothing;
#   * `SPAN` is rounded up to a 32-byte multiple, so the copy lands exactly on
#     a boundary and cannot over-write the identity padding behind it (§3.4).
#
# Out-of-range columns need no mask: the window is flanked by identity padding
# wide enough for the maximum overhang on each side, and a lane whose global
# column is outside [0, W) lands in that padding by construction.
#
# Restricted to OH == OD == 1 so that the GM output stays (nc, chunk, ow_local)
# and the host can recover (n, c, ow) with one contiguous slice.


class _ChunkGeometry:
    """Compile-time geometry of the chunked 1-D path, or `fits = False`."""

    def __init__(self, w_in, ow, kw, sw, pw, dw, batch_channels, dtype_name,
                 reduction, count_include_pad, divisor_override):
        esz = _ESZ[dtype_name]
        grain = 32 // esz
        self.W, self.OW = w_in, ow
        self.KW, self.SW, self.PW, self.DW = kw, sw, pw, dw
        self.NC = batch_channels
        self.dtype_name, self.esz, self.grain = dtype_name, esz, grain
        self.reduction = reduction
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override
        self.applicable = True
        self.fits = False

        max_iw = (ow - 1) * sw - pw + (kw - 1) * dw
        self.LPAD = _align_up(max(pw, 1), grain)
        rpad = max(grain, max_iw - w_in + 1 + grain)
        self.RPAD = _align_up(rpad, grain)

        # Largest chunk width whose whole working set fits.  Descending so the
        # DMA stays as wide as UB allows (the family is DMA-count bound).
        best = None
        owc = _align_up(ow, 16)
        while owc >= 16:
            span = (owc - 1) * sw + (kw - 1) * dw + 1
            span_alloc = _align_up(span, grain)
            if span_alloc > w_in:
                owc -= 16
                continue
            plane = self.LPAD + span_alloc + self.RPAD + grain
            lanes = owc                      # OH == 1, so LANES == OWP == owc
            ub = plane * esz
            ub += kw * lanes * 4             # colbase
            ub += lanes * 4 * 2              # off_i32 + off_u32
            ub += lanes * esz * 2            # got + out
            ub += lanes * 4 * 2              # got_f32 + acc
            if reduction == "avg":
                ub += lanes * 4 * 3          # cw_f + div_f + scratch
            ub += 4 * lanes * 4              # colo / tmp / scr
            if ub <= UB_BUDGET_BYTES:
                best = (owc, span_alloc, plane, ub)
                break
            owc -= 16
        if best is None:
            self.applicable = False
            self.ub_bytes = None
            return

        self.OWC, self.SPAN, self.PLANE, self.ub_bytes = best
        self.DATA_OFF = self.LPAD
        self.LANES = self.OWC
        self.NCHUNK = math.ceil(ow / self.OWC)
        self.UNITS = batch_channels * self.NCHUNK
        self.logical_blocks = max(1, math.ceil(self.UNITS / _VEC))
        self.launch_blocks = min(self.logical_blocks, LAUNCH_BLOCK_CAP)
        self.grid_repeats = max(
            1, (self.logical_blocks + self.launch_blocks - 1) // self.launch_blocks)
        self.fits = True


@lru_cache(maxsize=128)
def _compile_chunk_pool(w_in, ow, kw, sw, pw, dw, batch_channels, dtype_name,
                        reduction, count_include_pad, divisor_override):
    g = _ChunkGeometry(w_in, ow, kw, sw, pw, dw, batch_channels, dtype_name,
                       reduction, count_include_pad, divisor_override)
    if not g.fits:
        return None

    W, OW, KW, SW, PW, DW = g.W, g.OW, g.KW, g.SW, g.PW, g.DW
    ESZ, GRAIN = g.esz, g.grain
    OWC, SPAN, PLANE, DATA_OFF = g.OWC, g.SPAN, g.PLANE, g.DATA_OFF
    LANES, NCHUNK, UNITS = g.LANES, g.NCHUNK, g.UNITS
    LAUNCH, REPEATS, LOGICAL = g.launch_blocks, g.grid_repeats, g.logical_blocks
    MAX_OFF = (PLANE - GRAIN - 1) * ESZ
    IDENT = _identity(reduction, dtype_name)
    IS_F32 = dtype_name == "float32"
    input_total = batch_channels * w_in
    USE_OVERRIDE = divisor_override is not None
    OVERRIDE = 1 if divisor_override is None else divisor_override
    W_HI, W_LO = (W + PW, -PW) if count_include_pad else (W, 0)
    LAST_START = max(0, w_in - SPAN)
    DIV_LANES = LANES if reduction == "avg" else 1

    @tilelang.jit(out_idx=[-1], pass_configs=_GATHER_PASS_CFG)
    def factory():
        @T.prim_func
        def main(A: T.Tensor((input_total,), dtype_name),
                 C: T.Tensor((UNITS * LANES,), dtype_name)):
            with T.Kernel(LAUNCH, is_npu=True) as (cid, vid):
                plane = T.alloc_ub((PLANE,), dtype_name)
                colbase = T.alloc_ub((KW * LANES,), "int32")
                off_i32 = T.alloc_ub((LANES,), "int32")
                off_u32 = T.alloc_ub((LANES,), "uint32")
                got = T.alloc_ub((LANES,), dtype_name)
                got_f32 = T.alloc_ub((LANES,), "float32")
                acc = T.alloc_ub((LANES,), "float32")
                out = T.alloc_ub((LANES,), dtype_name)
                colo = T.alloc_ub((LANES,), "int32")
                rbuf = T.alloc_ub((GRAIN,), "int32")
                cw_f = T.alloc_ub((DIV_LANES,), "float32")
                div_f = T.alloc_ub((DIV_LANES,), "float32")
                end_f = T.alloc_ub((DIV_LANES,), "float32")

                with T.Scope("V"):
                    # chunk-relative column offsets; the edges are handled by the
                    # identity padding, so there is no sentinel and no mask.
                    for kw_i in T.serial(KW):
                        T.tile.arith_progression(
                            colo, (DATA_OFF + kw_i * DW) * ESZ, SW * ESZ, LANES)
                        T.copy(colo, colbase[kw_i * LANES: kw_i * LANES + LANES])
                    if reduction == "avg":
                        # chunk-relative start_w; the chunk term is added per unit
                        T.tile.arith_progression(
                            cw_f, T.cast(-PW, "float32"), T.cast(SW, "float32"), LANES)

                    for rep in T.serial(REPEATS):
                        lcid = cid + rep * LAUNCH
                        if lcid < LOGICAL:
                            unit = lcid * _VEC + vid
                            if unit < UNITS:
                                nc = unit // NCHUNK
                                chunk = unit % NCHUNK
                                iw0 = chunk * OWC * SW - PW
                                gm_start = T.max(T.min(iw0, LAST_START), 0)
                                # `r` folds into the per-tap offset constant, so
                                # clamping the window start costs nothing -- but
                                # it has to reach T.tile.add as a BufferLoad.
                                # A non-constant scalar operand is lowered as
                                # `x.GetValue(0)`, so a plain scalar (T.alloc_var
                                # or a bare expression) produces ill-formed code
                                # and bisheng reports "no matching function for
                                # call to 'Adds'".
                                rbuf[0] = (iw0 - gm_start) * ESZ

                                T.tile.fill(plane, IDENT)
                                T.copy(A[nc * W + gm_start: nc * W + gm_start + SPAN],
                                       plane[DATA_OFF:DATA_OFF + SPAN])
                                # T.tile.gather reads `plane` through a computed
                                # pointer, which auto-sync does not see (§13.55.1).
                                T.barrier_all()

                                if reduction == "avg":
                                    T.tile.fill(acc, 0.0)
                                else:
                                    T.tile.fill(acc, -T.infinity("float32"))
                                for kw_i in T.serial(KW):
                                    T.tile.add(off_i32,
                                               colbase[kw_i * LANES:
                                                       kw_i * LANES + LANES],
                                               rbuf[0])
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
                                    if USE_OVERRIDE:
                                        T.tile.fill(div_f, T.cast(OVERRIDE, "float32"))
                                    else:
                                        T.tile.add(div_f, cw_f,
                                                   T.cast(chunk * OWC * SW, "float32"))
                                        T.tile.add(end_f, div_f, T.cast(KW, "float32"))
                                        T.tile.clamp_max(end_f, end_f,
                                                         T.cast(W_HI, "float32"), LANES)
                                        T.tile.clamp_min(div_f, div_f,
                                                         T.cast(W_LO, "float32"), LANES)
                                        T.tile.sub(div_f, end_f, div_f)
                                        T.tile.clamp_min(div_f, div_f, 1.0, LANES)
                                    T.tile.div(acc, acc, div_f)

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


@lru_cache(maxsize=256)
def _compile_fixed_pool(
    input_shape,
    output_shape,
    kernel,
    stride,
    padding,
    dilation,
    dtype_name,
    reduction,
    count_include_pad,
    divisor_override,
):
    ndim = len(kernel)
    input_dims = (1,) * (3 - ndim) + input_shape[2:]
    output_dims = (1,) * (3 - ndim) + output_shape[2:]
    kernel = (1,) * (3 - ndim) + kernel
    stride = (1,) * (3 - ndim) + stride
    padding = (0,) * (3 - ndim) + padding
    dilation = (1,) * (3 - ndim) + dilation
    total = math.prod(output_shape)
    window_size = math.prod(kernel)
    input_total = math.prod(input_shape)
    divisor_override_value = 1 if divisor_override is None else divisor_override
    use_divisor_override = divisor_override is not None
    logical_blocks = math.ceil(total / (_VEC * _TILE))
    launch_blocks = launch_block_count(logical_blocks)
    repeats = grid_repeat_count(logical_blocks, launch_blocks)
    padded_total = logical_blocks * _VEC * _TILE
    _FALLBACK_IDENTITY = _identity(reduction, dtype_name)

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
            tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
        },
    )
    def factory():
        @T.prim_func
        def main(
            A: T.Tensor((input_total,), dtype_name),
            C: T.Tensor((padded_total,), dtype_name),
        ):
            with T.Kernel(launch_blocks, is_npu=True) as (cid, vid):
                gathered = T.alloc_ub((_TILE,), dtype_name)
                gathered_fp32 = T.alloc_ub((_TILE,), "float32")
                sum_fp32 = T.alloc_ub((_TILE,), "float32")
                max_fp32 = T.alloc_ub((_TILE,), "float32")
                result_fp32 = T.alloc_ub((_TILE,), "float32")
                result = T.alloc_ub((_TILE,), dtype_name)

                with T.Scope("V"):
                    for repeat in T.serial(repeats):
                        logical_cid = cid + repeat * launch_blocks
                        if logical_cid < logical_blocks:
                            start = logical_cid * (_VEC * _TILE) + vid * _TILE
                            T.tile.fill(sum_fp32, 0.0)
                            T.tile.fill(max_fp32, -T.infinity("float32"))
                            T.tile.fill(result_fp32, 0.0)

                            for tap in T.serial(window_size):
                                T.tile.fill(gathered, _FALLBACK_IDENTITY)
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
                                        id_ = (
                                            od * stride[0]
                                            - padding[0]
                                            + kd * dilation[0]
                                        )
                                        ih = (
                                            oh * stride[1]
                                            - padding[1]
                                            + kh * dilation[1]
                                        )
                                        iw = (
                                            ow * stride[2]
                                            - padding[2]
                                            + kw * dilation[2]
                                        )
                                        valid = (
                                            id_ >= 0
                                            and id_ < input_dims[0]
                                            and ih >= 0
                                            and ih < input_dims[1]
                                            and iw >= 0
                                            and iw < input_dims[2]
                                        )
                                        if valid:
                                            input_offset = (
                                                (
                                                    (
                                                        batch * input_shape[1]
                                                        + channel
                                                    )
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
                                # R197: the reduction used to be a second
                                # per-lane scalar pass that re-decoded every
                                # NCDHW coordinate purely to re-derive `valid`.
                                # Pre-filling `gathered` with the reduction
                                # identity instead makes the whole pass one
                                # vector op: an invalid lane contributes 0.0 to
                                # the sum and -inf to the max, which is exactly
                                # what skipping it did.  `T.tile.max` propagates
                                # NaN the same way the explicit isnan branch did
                                # (measured, R197 §4).
                                if reduction == "avg":
                                    T.tile.add(sum_fp32, sum_fp32, gathered_fp32)
                                else:
                                    T.tile.max(max_fp32, max_fp32, gathered_fp32)

                            for lane in T.serial(_TILE):
                                output_index = start + lane
                                if output_index < total:
                                    ow = output_index % output_dims[2]
                                    output_row = output_index // output_dims[2]
                                    oh = output_row % output_dims[1]
                                    output_plane = output_row // output_dims[1]
                                    od = output_plane % output_dims[0]
                                    start_d = od * stride[0] - padding[0]
                                    start_h = oh * stride[1] - padding[1]
                                    start_w = ow * stride[2] - padding[2]
                                    end_d = start_d + kernel[0]
                                    end_h = start_h + kernel[1]
                                    end_w = start_w + kernel[2]
                                    valid_d = T.max(
                                        T.min(end_d, input_dims[0])
                                        - T.max(start_d, 0),
                                        0,
                                    )
                                    valid_h = T.max(
                                        T.min(end_h, input_dims[1])
                                        - T.max(start_h, 0),
                                        0,
                                    )
                                    valid_w = T.max(
                                        T.min(end_w, input_dims[2])
                                        - T.max(start_w, 0),
                                        0,
                                    )
                                    padded_d = T.max(
                                        T.min(
                                            end_d,
                                            input_dims[0] + padding[0],
                                        )
                                        - T.max(start_d, -padding[0]),
                                        0,
                                    )
                                    padded_h = T.max(
                                        T.min(
                                            end_h,
                                            input_dims[1] + padding[1],
                                        )
                                        - T.max(start_h, -padding[1]),
                                        0,
                                    )
                                    padded_w = T.max(
                                        T.min(
                                            end_w,
                                            input_dims[2] + padding[2],
                                        )
                                        - T.max(start_w, -padding[2]),
                                        0,
                                    )
                                    valid_count = valid_d * valid_h * valid_w
                                    padded_count = padded_d * padded_h * padded_w
                                    auto_count = T.if_then_else(
                                        count_include_pad,
                                        padded_count,
                                        valid_count,
                                    )
                                    divisor = T.if_then_else(
                                        use_divisor_override,
                                        divisor_override_value,
                                        T.max(auto_count, 1),
                                    )
                                    if reduction == "avg":
                                        result_fp32[lane] = sum_fp32[lane] / T.cast(
                                            divisor, "float32"
                                        )
                                    else:
                                        result_fp32[lane] = max_fp32[lane]

                            if dtype_name == "float32":
                                T.copy(
                                    result_fp32,
                                    C[start : start + _TILE],
                                )
                            else:
                                T.tile.cast(
                                    result,
                                    result_fp32,
                                    "CAST_RINT",
                                    _TILE,
                                )
                                T.copy(result, C[start : start + _TILE])

        return main

    return factory()


def _build_fixed_pool(
    input_shape,
    dtype,
    *,
    ndim,
    reduction,
    kernel_size,
    stride,
    padding,
    dilation,
    ceil_mode,
    count_include_pad=False,
    divisor_override=None,
):
    if len(input_shape) != ndim + 2:
        layout = {1: "NCL", 2: "NCHW", 3: "NCDHW"}[ndim]
        raise ValueError(
            f"{reduction.title()}Pool{ndim}dFwdOp expects a {ndim + 2}D "
            f"{layout} input, got {input_shape}"
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
        divisor_override=divisor_override,
    )
    if not isinstance(count_include_pad, bool):
        raise TypeError("count_include_pad must be bool")
    output_dims = tuple(
        _output_dim(size, k, step, pad, dil, ceil_mode)
        for size, k, step, pad, dil in zip(
            input_shape[2:], kernel, stride, padding, dilation
        )
    )
    if min(output_dims) <= 0:
        raise ValueError(
            f"pooling calculated output size must be greater than zero, got {output_dims}"
        )
    output_shape = input_shape[:2] + output_dims
    dtype_name = _dtype_name(dtype)

    # --- fast path: one vector gather per window tap (R197) -------------------
    lead = (1,) * (3 - ndim)
    in3 = lead + input_shape[2:]
    out3 = lead + output_dims
    k3 = lead + kernel
    s3 = lead + stride
    p3 = (0,) * (3 - ndim) + padding
    d3 = lead + dilation
    batch_channels = input_shape[0] * input_shape[1]
    geometry = _GatherGeometry(
        in3, out3, k3, s3, p3, d3, batch_channels, dtype_name, reduction,
        count_include_pad, divisor_override,
    )
    if geometry.fits:
        compiled = _compile_gather_pool(
            in3, out3, k3, s3, p3, d3, batch_channels, dtype_name, reduction,
            count_include_pad, divisor_override,
        )
        units, oh, owp, ow = (
            geometry.UNITS, geometry.OH, geometry.OWP, geometry.OW)

        def launch(x):
            # Lane rows stay padded to OWP all the way into GM because every UB
            # buffer-region offset has to be a 32-byte multiple; the slice is
            # free whenever OWP == OW.
            raw = compiled(x.reshape(-1))
            return raw.view(units, oh, owp)[:, :, :ow].reshape(*output_shape)

        launch.path_kind = f"{reduction}{ndim}d_gather"
        launch.output_shape = output_shape
        launch.logical_blocks = geometry.logical_blocks
        launch.launch_blocks = geometry.launch_blocks
        launch.geometry = geometry
        launch.compiled = compiled
        return launch

    # --- chunked 1-D path: one output row wider than UB can stage -------------
    if ndim == 1 and geometry.OH == 1 and geometry.OD == 1:
        chunk_geo = _ChunkGeometry(
            in3[2], out3[2], k3[2], s3[2], p3[2], d3[2], batch_channels,
            dtype_name, reduction, count_include_pad, divisor_override,
        )
        if chunk_geo.fits:
            compiled = _compile_chunk_pool(
                in3[2], out3[2], k3[2], s3[2], p3[2], d3[2], batch_channels,
                dtype_name, reduction, count_include_pad, divisor_override,
            )
            nc_count = batch_channels
            wide = chunk_geo.NCHUNK * chunk_geo.LANES
            ow_out = output_dims[0]

            def launch(x):
                raw = compiled(x.reshape(-1))
                return raw.view(nc_count, wide)[:, :ow_out].reshape(*output_shape)

            launch.path_kind = f"{reduction}1d_chunk"
            launch.output_shape = output_shape
            launch.logical_blocks = chunk_geo.logical_blocks
            launch.launch_blocks = chunk_geo.launch_blocks
            launch.geometry = chunk_geo
            launch.compiled = compiled
            return launch

    # --- fallback: per-lane path, for rows too wide to stage in UB ------------
    compiled = _compile_fixed_pool(
        input_shape,
        output_shape,
        kernel,
        stride,
        padding,
        dilation,
        dtype_name,
        reduction,
        count_include_pad,
        divisor_override,
    )
    total = math.prod(output_shape)

    def launch(x):
        return compiled(x)[:total].view(*output_shape)

    launch.path_kind = f"{reduction}{ndim}d_generic"
    launch.gather_ub_bytes = geometry.ub_bytes
    launch.output_shape = output_shape
    launch.logical_blocks = math.ceil(total / (_VEC * _TILE))
    launch.launch_blocks = launch_block_count(launch.logical_blocks)
    launch.compiled = compiled
    return launch


def build_avg_pool_kernel(
    input_shape,
    dtype,
    *,
    ndim,
    kernel_size,
    stride=None,
    padding=0,
    ceil_mode=False,
    count_include_pad=True,
    divisor_override=None,
):
    return _build_fixed_pool(
        input_shape,
        dtype,
        ndim=ndim,
        reduction="avg",
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=1,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        divisor_override=divisor_override,
    )


def build_max_pool3d_kernel(
    input_shape,
    dtype,
    *,
    kernel_size,
    stride=None,
    padding=0,
    dilation=1,
    ceil_mode=False,
):
    return _build_fixed_pool(
        input_shape,
        dtype,
        ndim=3,
        reduction="max",
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
    )


__all__ = ["build_avg_pool_kernel", "build_max_pool3d_kernel"]
