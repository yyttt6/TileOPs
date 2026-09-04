"""Shared Ascend launch limits for TileLang kernels."""

# CANN 8.5 accepts one-dimensional blockDim values in [1, 65535].
MAX_BLOCK_COUNT = 65535


def launch_block_count(logical_block_count: int) -> int:
    """Return a legal blockDim while preserving at least one launch block."""
    return max(1, min(int(logical_block_count), MAX_BLOCK_COUNT))


def grid_repeat_count(logical_block_count: int, launch_blocks: int) -> int:
    """Number of grid-stride passes needed to cover every logical block."""
    return max(1, (int(logical_block_count) + int(launch_blocks) - 1) // int(launch_blocks))


# --- Geometry for row-wise normalization templates (R200) -------------------
#
# Every row-wise normalization kernel in this tree (families.two_pass.
# _compile_norm / _compile, kernels.normalization_spatial._compile_fused_add_norm
# / _compile_ada_norm) used to hardcode ``block_m = 128`` (so ``sub_m = 64``) and
# ``block_n = 128``.  R200 measured what that costs:
#
#   * ``block_n = 128`` makes every GM->UB transfer a ``sub_m``-way strided read
#     of only ``128 * itemsize`` bytes, and multiplies the per-tile op chain by
#     ``n / 128``.
#   * ``launch_blocks = ceil(m / 128)`` leaves 47 of the 48 vector cores idle for
#     the decode shapes (m = 1), which is where every ``ratio_min`` in the family
#     came from.
#
#: UB waterline shared with elementwise_binary.py / R183.  ``ub_size`` is
#: 196608 on ascend910b1 but the usable arena is smaller (R194 saw a runtime
#: blow-up at 198656 against a 196352 arena), so budget below it.
UB_BUDGET_BYTES = 180224

#: ⚠️ SEPARATE, LOWER waterline for the normalization templates.
#:
#: 180224 is the elementwise waterline (R183/R194) and it does NOT hold here.
#: R200 measured the failure: SoftmaxFwdOp / lm-head-logits / float32
#: (m=4, n=102400) planned sub_m=8 x block_n=1280, a DECLARED 163864 bytes --
#: comfortably under 180224 -- and tilelang's own allocator refused it::
#:
#:     ascend_memory_planning.cc:767
#:     TVMError: Memory allocation failed for: sum
#:
#: i.e. exactly T195's warning that "the declared buffers summing under budget
#: is not the same as it actually fitting": ``T.reduce_sum`` / ``T.reduce_max``
#: carry scratch that never appears in the Python allocation list.  32 KB is
#: reserved for it here.  ⚠️ 32 KB is a fitted margin, not a measured scratch
#: size -- the real requirement is unknown and probably intrinsic-dependent.
UB_BUDGET_NORM_BYTES = 147456

#: ``T.tile.compare``/``T.tile.cast`` want 128-lane native repeats, so every
#: tile width stays a multiple of this.  It is a GRANULARITY, not a size
#: (R194 2.2 -- the same comment trap that cost the unary family 3.5x).
TILE_GRAIN = 128

#: Hard cap on the column width of a row-wise normalization tile.  1280 rather
#: than 1024 so that dit-xl's n = 1152 fits in a single tile; one-process-per-
#: width probes clear every intrinsic these templates use up to 4096
#: (docs/reports/R200-data/probe_broadcast_limit.log), and the UB budget binds
#: well below that anyway once ROW_EXTENT_GRAIN rows have to fit.
TILE_WIDTH_CAP = 1280

#: ⚠️ MINIMUM number of rows per vector lane, and the granularity of it.
#:
#: The row-wise templates run ``T.tile.{add,sub,mul,div,rsqrt,fill}`` directly on
#: their ``(sub_m,)`` fp32 statistic buffers (row_sum, mean, m2, inv, ...).  An
#: Ascend vector instruction addresses whole 32-byte blocks, so a ``(sub_m,)``
#: fp32 operand with ``sub_m < 8`` encodes a zero-block repeat and the core
#: raises::
#:
#:     aicore exception 507015 ... there is an exception of fftsplus aivector
#:     error ... errorStr: VEC supports illegal configurations in commands
#:
#: R200 hit this at sub_m in {1, 2, 4} and never at sub_m in {8, 16, 64}
#: (probe_rowwise: 64x128 / 16x512 / 8x1024 ok, 4x2048 / 2x4096 / 1x4096 crash;
#: it was NOT the column width, which one-process-per-width probes clear to
#: 4096 for copy / fill / mul / reduce_sum / reduce_max / broadcast).
#:
#: ⚠️ The cost is real: with ``_VEC = 2`` this forces ``block_m >= 16``, so an
#: m = 1 decode row cannot be spread over more than one block.  Fixing THAT
#: needs the ``(sub_m,)`` statistics padded to 8 lanes independently of how many
#: rows a lane owns; R200 did not do it.  See R200 for the open item.
ROW_EXTENT_GRAIN = 8

#: ⚠️ SEPARATE, MUCH HIGHER width cap for the ``(1, tile)`` spatial kernels.
#:
#: ``TILE_WIDTH_CAP = 1280`` above was fitted to the ROW-WISE template, whose
#: real constraint turned out to be ``sub_m >= 8`` (R200 §8.1) -- nothing to do
#: with the column width.  A one-row buffer has no such coupling, and R200 round
#: 2 measured it directly (one process per width,
#: docs/reports/R200-data/r2/probe_spatial_tile.log): a ``(1, tile)`` spatial
#: kernel is correct at tile = 1024 / 2048 / 4096 / 8192 / 12288 in fp16 and at
#: 1024 / 4096 / 8192 in fp32.  8192 is the largest width verified in BOTH.
#:
#: This is the lever for BatchNormFwdOp/large-spatial, which walked 1024 chunks
#: per pass at tile=1024 and walks 128 at tile=8192.
SPATIAL_TILE_WIDTH_CAP = 8192

#: Element granularity for the ``(1, tile)`` spatial chunks.  Those kernels hold
#: both a dtype-width and an fp32 buffer of the chunk, and an Ascend vector
#: instruction addresses whole 32-byte blocks, so the tile has to be a multiple
#: of 32 bytes in the WIDEST element it is used with: 32/2 = 16 for fp16/bf16
#: and 32/4 = 8 for fp32, i.e. 16 covers both.
#:
#: Unlike the row-wise templates this is 16 rather than 128 on purpose: 128 is
#: the packed-mask native repeat, and these kernels have no packed masks.  The
#: difference matters -- ``spatial = 784`` (a 28x28 feature map, and the case
#: that sets InstanceNormFwdOp's ratio_min) has NO divisor that is a multiple of
#: 128, but 784 itself is a multiple of 16.
SPATIAL_ELEM_GRAIN = 16

#: ``vector_core_cnt`` from the ascend910b1 platform config.  ``T.Kernel(N)``
#: with an ``(cid, vid)`` binding gives ``N`` blocks x 2 vector lanes, so
#: ``N = 48`` oversubscribes 2x; R194 measured 48-192 as a plateau for the
#: elementwise templates, so stay at the low end of it.
VECTOR_CORE_COUNT = 48
LAUNCH_BLOCK_CAP = 48


def plan_rowwise_norm(m, n, itemsize, bytes_per_cell, bytes_per_col=0,
                      bytes_per_row=0, grain=TILE_GRAIN,
                      width_cap=TILE_WIDTH_CAP,
                      launch_cap=LAUNCH_BLOCK_CAP,
                      ub_budget=UB_BUDGET_NORM_BYTES):
    """Choose ``(sub_m, block_n, launch_blocks, grid_repeats)`` for a row-wise
    normalization kernel over an ``(m, n)`` row-major input.

    ``bytes_per_cell`` is the UB cost of one ``(sub_m, block_n)`` element summed
    over every 2-D buffer the kernel allocates; ``bytes_per_col`` the cost of one
    column summed over the ``(block_n,)`` buffers; ``bytes_per_row`` the cost of
    one row summed over the ``(sub_m,)`` buffers.  Callers must derive these by
    reading their own allocation list -- guessing one wrong is a silent UB
    overflow (R194 2.2 / T195 step 1.1).

    Returns a dict so callers can log the whole plan.
    """
    m = max(1, int(m))
    n = max(1, int(n))
    grain = TILE_GRAIN if grain is None else max(1, int(grain))

    # --- column width: as wide as the broadcast unit cap and UB allow, and
    #     preferably an exact divisor of n so the scalar n-tail path is dead.
    n_grained = ((n + grain - 1) // grain) * grain
    hi = min(int(width_cap), max(grain, n_grained))
    # UB must admit the HARD row floor at this width, not just one row -- the
    # floor cannot be traded away (see ROW_EXTENT_GRAIN), so the width is what
    # gives.
    while hi > grain and (ROW_EXTENT_GRAIN * (hi * bytes_per_cell + bytes_per_row)
                          + hi * bytes_per_col) > ub_budget:
        hi -= grain
    block_n = hi
    if n <= hi:
        # One padded tile covers the row: no n-tail loop at all.
        block_n = n_grained if n_grained <= hi else hi
    else:
        # Search down for an exact divisor of n, over the WHOLE range rather
        # than R194's "at most half the width".  The rule is different here
        # because the penalty is different: a ragged tile in the unary template
        # falls into a scalar tail costing 2.65x, but in these templates it
        # falls into a ``for r in range(sub_m): for c in range(block_n)`` DOUBLE
        # scalar loop -- 8 * 1024 iterations for one tile.  Giving up width to
        # avoid it is nearly always right (n=1152 -> 384 x 3 rather than
        # 1024 + a ragged 128).
        block_n = hi
        w = hi
        while w >= grain:
            if n % w == 0:
                block_n = w
                break
            w -= grain

    # --- rows per vector lane: fill the machine first, then spend the UB that
    #     is left.  Powers of two only, so ``block_m = 2 * sub_m`` divides m
    #     whenever m is a power of two -- that kills the guarded per-row tail
    #     copy loop, which is the m = 1 decode disaster.
    ub_room = ub_budget - block_n * bytes_per_col
    per_row_cost = block_n * bytes_per_cell + bytes_per_row
    sub_m_ub = max(1, ub_room // per_row_cost) if per_row_cost > 0 else 1
    sub_m_occ = max(1, m // (2 * int(launch_cap)))
    # ROW_EXTENT_GRAIN is a HARD floor, not a preference: below it the vector
    # ops on the (sub_m,) fp32 statistics buffers are illegally encoded and the
    # core faults.  If UB cannot even hold the floor the caller has asked for a
    # tile that does not exist -- say so rather than emit a faulting kernel.
    sub_m = ROW_EXTENT_GRAIN
    while sub_m * 2 <= sub_m_occ and sub_m * 2 <= sub_m_ub:
        sub_m *= 2
    if sub_m > sub_m_ub:
        raise NotImplementedError(
            f"plan_rowwise_norm({m=}, {n=}): UB budget {ub_budget} admits only "
            f"sub_m={sub_m_ub} rows per lane at block_n={block_n}, below the "
            f"hard floor ROW_EXTENT_GRAIN={ROW_EXTENT_GRAIN}"
        )

    block_m = sub_m * 2
    m_tiles = (m + block_m - 1) // block_m
    launch_blocks = launch_block_count(min(m_tiles, int(launch_cap)))
    grid_repeats = grid_repeat_count(m_tiles, launch_blocks)
    n_tiles = (n + block_n - 1) // block_n
    return {
        "sub_m": sub_m,
        "block_m": block_m,
        "block_n": block_n,
        "m_tiles": m_tiles,
        "n_tiles": n_tiles,
        "m_pad": m_tiles * block_m,
        "n_pad": n_tiles * block_n,
        "has_m_tail": (m % block_m) != 0,
        "has_n_tail": (n % block_n) != 0,
        "launch_blocks": launch_blocks,
        "grid_repeats": grid_repeats,
        "ub_bytes": sub_m * block_n * bytes_per_cell + block_n * bytes_per_col
                    + sub_m * bytes_per_row,
    }


def plan_spatial_tile(spatial, bytes_per_cell, bytes_fixed=0,
                      grain=SPATIAL_ELEM_GRAIN, width_cap=SPATIAL_TILE_WIDTH_CAP,
                      ub_budget=UB_BUDGET_NORM_BYTES):
    """Pick the chunk width for the ``(1, tile)`` spatial normalization kernels
    (``_compile_group`` / ``_compile_instance_infer`` / ``_compile_batch`` /
    ``_compile_batch_bwd`` in ``normalization_spatial.py``).

    History, because the rule reversed twice and the reasons matter:

    * pre-R200: ``_TILE = 128`` for every shape, so ``spatial = 1024*1024``
      walked 8192 chunks per pass.
    * R200 round 1: widened it, but the ragged chunk fell into a
      ``for j in T.serial(tile)`` scalar load that iterates ``tile`` times
      REGARDLESS of how short the tail is -- so a wide tile with a short tail
      was the worst case, and GroupNorm went 0.113 -> 0.038.  The rule was
      then narrowed to exact divisors only.
    * R200 round 2 (this): the ragged load is vectorised
      (``_ragged_gm_to_ub``: 32-byte-aligned vector prefix + at most
      ``32/itemsize - 1`` scalar iterations), so a ragged tail costs ~4 scalar
      iterations instead of ~1024.  The divisor constraint is therefore no
      longer worth paying width for, and this maximises width again.

    A divisor is still mildly preferable (no ragged chunk at all), so it wins
    when it costs at most half the width.

    ``bytes_per_cell`` is the UB cost of one chunk element summed over every
    ``(1, tile)`` buffer; ``bytes_fixed`` covers the scalar ``(1,)`` buffers.
    """
    spatial = max(1, int(spatial))
    grain = max(1, int(grain))
    up = ((spatial + grain - 1) // grain) * grain
    hi = min(int(width_cap), up)
    hi -= hi % grain
    while hi > grain and hi * bytes_per_cell + bytes_fixed > ub_budget:
        hi -= grain
    hi = max(grain, hi)
    # An exact divisor removes the ragged chunk entirely; take the largest one
    # that does not cost more than half the width.
    w = hi
    floor_w = max(grain, hi // 2)
    while w >= floor_w:
        if spatial % w == 0:
            return w
        w -= grain
    return hi
