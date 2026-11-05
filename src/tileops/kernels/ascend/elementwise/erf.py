"""Ascend Erf kernel: ``y = erf(x)``, via the ``tilelang.ascend`` dialect.

Registers an additional implementation of ``ErfFwdOp``'s ``UnaryElementwiseFwdInterface``
under the ``"ascend"`` dispatch key, beside the in-tree CUDA ``ErfFwdKernel``
(``tileops.kernels.elementwise.math_unary.ErfFwdKernel``). No existing file is edited to do
this: ``tileops.backend.register_kernel_type`` joins this class to every ``ErfFwdOp`` instance
constructed after this module is imported (``docs/ascend950/dev_process.md`` §1).

Design: ``docs/ascend950/dev/ErfFwdOp/dsh_20261020_schedule.md``. The fp32 path uses an
op-local five-coefficient exp/rational body, ``sign(x) * (1 - P(t) * t * exp(-a*a))``
with ``a = abs(x)`` and ``t = 1 / (1 + 0.3275911*a)``, explicit ``x == 0`` returning the
signed zero, ``abs(x) >= 5`` returning exact +/-1 and the shared ``keep_nan`` guard. It
passed the isolated request-005 prototype on a 262,328-point corpus against same-NPU
``torch.erf`` and CPython ``math.erf`` at the original 1e-5 contract (max sampled
absolute error 4.84e-7); that is a bounded device gate, not a universal proof. The fp16 and bf16 paths keep the
degree-8/degree-6 clamped polynomial in fp32 arithmetic. The shared ``_erf.py`` and
``_nan.py`` are read-only; only ``keep_nan`` is reused. The degree-5/6 interpolations in
the old draft are not used.

Tiling mirrors ``SinFwdOp``/``CosFwdOp`` (identical manifest case shape: 16M x
{bfloat16, float16, float32}, 256M x {bfloat16, float16}, no decode case):
``num_blocks=64`` (live-queried vector-core count 72, floored to the nearest power of two,
same calibrated value reused unmodified by five prior siblings), wide tile (16384) for
fp16/bf16 and 8192 for float32 shipped as round 1's primary candidate from the start (per
schedule §2.1 -- this lever independently fixed an initial-measurement miss on
``SqrtFwdOp``/``RsqrtFwdOp``/``CosFwdOp``/``SinFwdOp``, four consecutive hits with zero
misses on this exact family shape). ``num_stages`` is measured, not assumed -- see
``round_1_result.md`` §4 for the comparison between 2 and 3 under the official
``measure_tileops_ascend.py`` protocol and the final shipped choice.

Round 2 (``docs/ascend950/dev/ErfFwdOp/round_2_schedule.md``/``round_2_result.md``) diagnosed the
remaining 256M-float16/256M-bfloat16 ~6% miss with ``msprof_detail`` pipe-occupancy evidence:
``aiv_vec`` (vector-ALU) is saturated at 97-99.8% in every measured (size, dtype) combination,
i.e. this op is genuinely VF-compute-bound, not DMA-bound, at both 16M and 256M. A ``num_blocks``
resweep (128 instead of 64) was implemented, measured under the official protocol, and found to
give **no improvement** (within normal run-to-run noise) -- reverted, this file is unchanged in
behavior from round 1. No safe lever within this file's scope (tile/``num_stages``/``num_blocks``)
closes this gap; round_2_result.md records that historical diagnosis-backed non-improvement.
The old float32 ~69% miss motivated the op-local rational body now used here; it is no longer
a shared ``T.erf`` path.

Importing this module when ``tilelang.ascend`` is unavailable (no Ascend toolchain, a CUDA-only
process) is a no-op: the Ascend-specific imports, the kernel class, and the registration call are
wrapped in one ``try/except ImportError`` block, per ``docs/ascend950/dev_process.md`` §1.

This file does not get a subpackage ``__init__.py`` with ``__all__`` re-exports, which
``TileOPs/.claude/rules/code-style.md`` otherwise requires for every ``src/tileops/kernels/*``
subpackage: ``docs/ascend950/dev_process.md`` §1 explicitly asks every op's worker to self-
register from its own file and *not* build a shared aggregator file under ``kernels/ascend/``,
since several ops' workers write into this same new directory concurrently and a shared
``__init__.py`` would be a file they all contend for. ``kernels/ascend`` and
``kernels/ascend/elementwise`` are therefore left as implicit (PEP 420) namespace packages.
"""

import functools

import torch

try:
    import tilelang
    import tilelang.ascend.language as T
    from tilelang.carver.arch.driver import get_num_vector_cores

    import tileops.backend
    from tileops.kernels.elementwise._nan import keep_nan
    from tileops.kernels.elementwise.call_spec import (
        ElementwiseCall,
        UnaryElementwiseFwdInterface,
    )
    from tileops.kernels.kernel_base import Entry, Kernel

    def _floor_pow2(n: int) -> int:
        """Largest power of two ``<= n`` (``n >= 1``)."""
        p = 1
        while p * 2 <= n:
            p *= 2
        return p

    @functools.lru_cache(maxsize=1)
    def _calibrated_num_blocks(fallback: int = 64) -> int:
        """Persistent-grid block count for the large, bandwidth-bound path.

        Verbatim reuse of ``sin.py``/``cos.py``'s already-calibrated, already-hardware-
        validated helper (round_1_schedule.md §2.2): queries this SKU's real vector-core count
        (``get_num_vector_cores`` -- ``vector_core_num=72`` on ``Ascend950DT_9592``, not a
        power of two), then floors to the nearest power of two, landing on 64.

        Every ``n_total`` in scope for this op (the 2^20-element correctness fixture, the
        2^24/2^28-element benchmark cases) is a pure power of two, while 72=2^3x3^2 cannot
        divide any of them evenly. Six independent siblings
        (``ReluFwdOp``/``BitwiseNotFwdOp``/``TanhFwdOp``/``SqrtFwdOp``/``SinFwdOp``/
        ``CosFwdOp``) all measured 64 beating 128 under the official protocol for this exact
        cheap-unary-op shape -- reused here, not re-swept.

        Falls back to ``fallback`` if the live query raises or returns a falsy value at
        runtime, rather than crashing kernel build. Deliberately **not** evaluated at module
        import time -- ``get_num_vector_cores`` queries the real device, so this is only
        called lazily, the first time a persistent-path kernel is actually built (always
        under an ``npu-guard`` reservation already). ``lru_cache`` makes it a one-time query
        per process either way.
        """
        try:
            queried = get_num_vector_cores(device_id=0)
        except Exception:
            queried = None
        if not queried:
            return fallback
        return _floor_pow2(int(queried))

    # Tile size for the memory-bound, large-n persistent-grid path (schedule doc §2.1). Round 1
    # shipped the schedule's proposed wide-for-16-bit-only config first (tile=16384 fp16/bf16,
    # tile=8192 fp32), then measured num_stages in {1, 2, 3} under the official protocol
    # (round_1_result.md §4 has the full table). Summary: num_stages=2 is the best of the three --
    # num_stages=3 is ~1% slower everywhere, and a fourth candidate (doubling tile again, to 32768
    # fp16/bf16 / 16384 float32, which only fits the 216KiB UB budget at num_stages=1 -- no
    # pipeline double-buffering) measured *worse on every one of the 5 cases*, including flipping
    # the two cases that passed at num_stages=2 into misses. That rules out the "more independent
    # calls in flight hides latency" mechanism that worked for `sin`/`cos`'s tile-widening lever --
    # for `erf`, losing compute/DMA overlap costs more than the wider tile gains, the opposite
    # trade from what tile-widening bought those siblings. num_stages=2 is shipped here.
    #
    # Even at num_stages=2, float32 misses its target by a large, stable margin (~69-72%,
    # essentially unchanged across all three num_stages candidates tested -- 0.100834/0.102618/
    # 0.110485 ms, all far over the 0.059735ms target), while fp16/bf16 pass 16M comfortably and
    # miss 256M by only ~6%. float32's tile is already at its UB-budget ceiling for num_stages=2
    # (2 buffers * 2 stages * 4 bytes * 8192 = 131072 bytes; 16384 would need 262144 bytes, over
    # the 221184-byte/216KiB envelope), so no further tile-widening lever is available for it
    # within this UB budget. This is consistent with `_erf.py`'s own module docstring, which
    # states the float32 path's ``erff`` "costs roughly twice the polynomial's instruction count"
    # -- the old float32 native ``T.erf`` call was the heavier compute body; the current
    # fp32 path is the op-local rational body gated by the request-005 prototype.
    _TILE_BY_DTYPE = {"float16": 16384, "bfloat16": 16384, "float32": 8192}
    _NUM_STAGES_BY_DTYPE = {"float16": 2, "bfloat16": 2, "float32": 2}

    # Op-local 16-bit coefficients from the 2026-10-20 CPU screen, highest power
    # first.  fp32 uses the separate five-coefficient rational body below; this
    # degree reduction is only for the 16-bit storage dtypes and is gated by the
    # original exhaustive comparator.
    _CLAMP = 3.6
    _FP16_POLY = (
        10.346251487731934,
        -32.11433792114258,
        42.01287841796875,
        -28.760366439819336,
        11.102203369140625,
        -2.2027556896209717,
        0.3365305960178375,
        0.12998349964618683,
        0.27783286571502686,
    )
    _BF16_POLY = (
        6.515710353851318,
        -14.437643051147461,
        12.623186111450195,
        -4.944900989532471,
        1.0117061138153076,
        0.07986576855182648,
        0.27838456630706787,
    )
    _LOCAL_POLY = {"float16": _FP16_POLY, "bfloat16": _BF16_POLY}

    def _local_polynomial(wide, dtype):
        """Clamped/scaled Horner in fp32 with the selected 16-bit coefficients."""
        coeffs = _LOCAL_POLY[dtype]
        one = T.cast(1.0, "float32")
        clamped = T.min(T.max(wide, T.cast(-_CLAMP, "float32")), T.cast(_CLAMP, "float32"))
        scaled = clamped * T.cast(1.0 / _CLAMP, "float32")
        w = one - scaled * scaled
        acc = T.cast(coeffs[0], "float32")
        for coeff in coeffs[1:]:
            acc = acc * w + T.cast(coeff, "float32")
        return T.min(T.max(clamped * acc, -one), one)

    def _local_erf(x, dtype):
        """op-local 16-bit erf, preserving the shared NaN contract."""
        wide = T.cast(x, "float32")
        return keep_nan(wide, lambda wide: _local_polynomial(wide, dtype))

    # fp32-only five-coefficient exp/rational candidate from the saved CPU
    # screen; this is not the 16-bit polynomial and is used only after a real
    # isolated prototype passes the original 1e-5 gate.
    _FP32_P = 0.3275911
    _FP32_COEFFS = (
        1.061405429,
        -1.453152027,
        1.421413741,
        -0.284496736,
        0.254829592,
    )

    def _fp32_body(w):
        zero = T.cast(0.0, "float32")
        one = T.cast(1.0, "float32")
        five = T.cast(5.0, "float32")
        a = T.abs(w)
        t = one / (one + T.cast(_FP32_P, "float32") * a)
        acc = T.cast(_FP32_COEFFS[0], "float32")
        for coeff in _FP32_COEFFS[1:]:
            acc = acc * t + T.cast(coeff, "float32")
        residual = (acc * t) * T.exp(-(a * a))
        signed = T.if_then_else(w >= zero, one, -one) * (one - residual)
        saturated = T.if_then_else(a >= five, T.if_then_else(w >= zero, one, -one), signed)
        return T.if_then_else(w == zero, w, saturated)

    def _fp32_erf(x):
        wide = T.cast(x, "float32")
        body = _fp32_body(wide)
        return T.if_then_else(T.isnan(wide), wide, body)

    # Round 2 (docs/ascend950/dev/ErfFwdOp/round_2_schedule.md SS2/SS4): msprof_detail pipe-
    # occupancy diagnosis on the unmodified round-1 kernel found aiv_vec (the vector-ALU pipe)
    # saturated at 97.0-99.8% in *every* measured (size, dtype) combination -- this op's
    # fp16/bf16 path is VF-compute-bound, not DMA-bound, at both 16M and 256M. The 256M cases'
    # ~6% miss (16M passes with real margin at the identical tile/num_stages) is best explained
    # by the baseline's own throughput scaling more favorably with size than this kernel's does
    # (round_2_schedule.md SS3's arithmetic: baseline scales 14.36x from 16M->256M, this kernel
    # scales 15.60x, both sub-linear/overhead-amortizing but the baseline's asymptote is reached
    # faster) -- i.e. a small asymptotic compute-cost gap, not a tiling artifact.
    #
    # A num_blocks resweep (128 instead of 64, same tile/num_stages, so block_span and therefore
    # the per-block iteration count is unchanged -- isolates "more parallel blocks" as the only
    # variable) was diagnosed via msprof_detail and found flat-to-very-slightly-worse (+0.2%
    # dur_us, aiv_vec unchanged at ~99.6-99.7%): doubling the logical block count did not unlock
    # any additional concurrent vector-ALU throughput, so this SKU's realized concurrently-
    # schedulable width for this persistent-kernel shape appears to already top out at (or
    # before) 64, independent of get_num_vector_cores()'s reported 72. A second candidate
    # (num_blocks=128, tile=8192, num_stages=3) was a clear 7% regression (no DMA-pipe slack for
    # deeper pipelining to hide behind, since aiv_mte2 already sits at 25-29%).
    #
    # Per round_2_schedule.md SS4, the num_blocks=128 resweep (the lower-risk, merely-flat
    # candidate on the msprof_detail diagnostic pass above) was still carried through the
    # *official* measure_tileops_ascend.py protocol for the 256M-fp16/bf16 cases only (never
    # decide from msprof_detail's cache-flushed wall-clock alone, dev_process.md SS7.5) --
    # **measured no improvement** (256M-float16 1.015528ms, 256M-bfloat16 1.015724ms, both
    # ~0.1-0.17% *worse* than round 1's 1.014045ms/1.013990ms, within this machine's normal
    # run-to-run noise under concurrent-worker contention, not a real regression either -- see
    # round_2_result.md SS2 for the full table). Reverted: this file always uses
    # ``_calibrated_num_blocks()`` for every dtype/size, byte-identical to round 1's shipped
    # behavior. Left here as a documented negative result (dev_process.md SS7.5's convention,
    # e.g. TanhFwdOp round 2's rejected native-fp16 path) so a later round doesn't re-try the
    # same num_blocks resweep without first reading why it didn't help: msprof_detail pipe
    # occupancy (SS2 above) showed aiv_vec already saturated at 97-99.8% in every config,
    # insensitive to block count -- this SKU's realized concurrently-schedulable width for this
    # persistent-kernel shape tops out at (or before) 64, independent of
    # ``get_num_vector_cores()``'s reported 72, so oversubscribing to 128 blocks bought no
    # additional concurrent vector-ALU throughput. The old float32 native-``T.erf`` path has
    # been replaced by the separately gated op-local rational body.

    def _tile_for_dtype(dtype_str: str) -> int:
        return _TILE_BY_DTYPE[dtype_str]

    def _stages_for_dtype(dtype_str: str) -> int:
        return _NUM_STAGES_BY_DTYPE[dtype_str]

    # Below this many elements, a persistent multi-block grid and T.Pipelined staging would add
    # fixed per-launch overhead with no bandwidth-bound tail to amortize it against: run the
    # whole call on a single block instead, one shot, no pipelining. No decode-shaped manifest
    # case exists for this op (schedule doc §0/§2.5). Fixed at the narrowest per-dtype tile
    # (independent of which dtype's persistent tile is in play) so the small-shape routing
    # decision does not shift underneath the M3 fixtures as the persistent tile is tuned.
    _SINGLE_BLOCK_THRESHOLD = min(_TILE_BY_DTYPE.values())

    def _dtype_str(dtype: torch.dtype) -> str:
        return str(dtype).split(".")[-1]

    @T.macro
    def _erf_body(x_ub, y_ub, n: int, threads: int, dtype: str) -> None:
        """``y_ub[:n] = erf(x_ub[:n])`` over ``threads`` lanes.

        float32 uses ``_fp32_erf``: the five-coefficient exp/rational body with exact
        +/-0, |x|>=5 saturation and the shared ``keep_nan`` guard. float16 and
        bfloat16 use ``_local_erf``: the fp32-promoted clamp (+/-3.6),
        scale-before-square, Horner and clip (+/-1) with degree-8/degree-6 coefficients.
        A ``@T.macro`` is required because TileLang rewrites only
        ``@T.prim_func``/``@T.macro`` bodies into trace-builder calls.
        """
        with T.SimtVF(threads=threads):
            for i in T.Parallel(n):
                value = _fp32_erf(x_ub[i]) if dtype == "float32" else _local_erf(x_ub[i], dtype)
                y_ub[i] = T.Cast(dtype, value)

    @functools.lru_cache(maxsize=64)
    def _make_erf_single_block(n_total: int, dtype: str):
        """One block, one tile, no pipelining: correctness-first fallback for a small shape
        that does not divide ``_calibrated_num_blocks()`` evenly."""

        @T.prim_func
        def main(X: T.Tensor((n_total,), dtype), Y: T.Tensor((n_total,), dtype)):
            with T.Kernel(1):
                x_ub = T.alloc_shared((n_total,), dtype)
                y_ub = T.alloc_shared((n_total,), dtype)
                T.copy(X, x_ub)
                _erf_body(x_ub, y_ub, n_total, threads=min(128, n_total), dtype=dtype)
                T.copy(y_ub, Y)

        return tilelang.compile(main, target="ascend", out_idx=-1)

    @functools.lru_cache(maxsize=64)
    def _make_erf_small_multi_block(n_total: int, dtype: str, num_blocks: int):
        """Multi-block, one-shot grid for a small shape that divides ``num_blocks`` evenly
        (schedule doc §2.5): spreads a small, correctness-only shape across the calibrated core
        count rather than collapsing to a single core, per the "decode/small shape shouldn't be
        single-core" lesson (dev_process.md §7.5) -- cheap to do right here even though this
        op's own manifest cases never reach this path.

        No ``T.Pipelined`` loop: each of ``num_blocks`` blocks does exactly one GM->UB copy,
        one compute-body call, one UB->GM copy. Requires ``n_total`` to divide ``num_blocks``
        exactly; the caller only reaches this builder for a shape that does.
        """
        assert n_total % num_blocks == 0, (
            f"_make_erf_small_multi_block requires n_total ({n_total}) to divide "
            f"num_blocks ({num_blocks}) exactly; the caller routes other shapes elsewhere"
        )
        chunk = n_total // num_blocks

        @T.prim_func
        def main(X: T.Tensor((n_total,), dtype), Y: T.Tensor((n_total,), dtype)):
            with T.Kernel(num_blocks) as bx:
                x_ub = T.alloc_shared((chunk,), dtype)
                y_ub = T.alloc_shared((chunk,), dtype)
                begin = bx * chunk
                T.copy(X[begin : begin + chunk], x_ub)
                _erf_body(x_ub, y_ub, chunk, threads=min(2048, chunk), dtype=dtype)
                T.copy(y_ub, Y[begin : begin + chunk])

        return tilelang.compile(main, target="ascend", out_idx=-1)

    @functools.lru_cache(maxsize=64)
    def _make_erf_persistent(n_total: int, dtype: str, num_blocks: int, tile: int, num_stages: int):
        """Persistent multi-block grid with ``T.Pipelined`` staging (schedule doc §2); the
        primary path, serving all 5 manifest cases.

        UB budget: 2 buffers (x_ub, y_ub) only -- the polynomial's intermediate values (``acc``,
        ``w``, ``scaled``, ``clamped``) are per-lane registers inside ``T.SimtVF``, not
        UB-resident, so they do not add to the tile-sizing formula (schedule §2.1). At
        ``tile=16384``/fp16/bf16/``num_stages<=3``: ``2 * 3 * 2 * 16384 = 196,608`` bytes, at the
        edge of (not exceeding) ``example_vecadd.py``'s own hardware-validated 3-buffer/fp32/
        196,608-byte footprint at the same stage count; at ``tile=8192``/fp32: ``2 * 3 * 4 * 8192
        = 196,608`` bytes, the same bound.

        Requires ``n_total`` to divide ``num_blocks * tile`` exactly; the caller only reaches
        this builder for a shape that does, and routes every other shape to
        ``_make_erf_generic_single_block`` instead.
        """
        block_span = num_blocks * tile
        assert n_total % block_span == 0, (
            f"_make_erf_persistent requires n_total ({n_total}) to divide "
            f"num_blocks*tile ({block_span}) exactly; the caller routes other shapes elsewhere"
        )
        iterations = n_total // block_span

        @T.prim_func
        def main(X: T.Tensor((n_total,), dtype), Y: T.Tensor((n_total,), dtype)):
            with T.Kernel(num_blocks) as bx:
                x_ub = T.alloc_shared((tile,), dtype)
                y_ub = T.alloc_shared((tile,), dtype)
                T.annotate_buffer_versions({x_ub: num_stages, y_ub: num_stages})
                for step in T.Pipelined(iterations, num_stages=num_stages):
                    begin = (step * num_blocks + bx) * tile
                    T.copy(X[begin : begin + tile], x_ub)
                    _erf_body(x_ub, y_ub, tile, threads=min(2048, tile), dtype=dtype)
                    T.copy(y_ub, Y[begin : begin + tile])

        return tilelang.compile(main, target="ascend", out_idx=-1)

    @functools.lru_cache(maxsize=64)
    def _make_erf_generic_single_block(n_total: int, dtype: str, tile: int, num_stages: int):
        """Correctness-first fallback for an ``n_total`` the persistent tiling does not evenly
        divide. One block; a main loop over full tile-sized chunks, plus one smaller,
        statically-sized tail chunk. Not performance-tuned: none of this op's 5 manifest
        cases reaches this function this round (all 5 divide ``block_span`` exactly) -- but
        two of the M3 correctness fixtures do (``MathFixture``'s n=1,048,576 and
        ``ErfRoundingWorkload``'s n=65,536 for fp16/bf16, schedule §2.5), so it must still be
        UB-budget-safe, not just compile-clean.

        ``num_stages`` is a parameter, not hardcoded, **on purpose**: an earlier version of this
        file hardcoded ``num_stages=2`` here (verbatim from ``sin.py``, whose own tile never
        exceeded 16384) while the primary persistent builder's tile/``num_stages`` pair was
        tuned independently. Widening the persistent tile to 32768 (fp16/bf16) without updating
        this fallback's pipeline depth to match blew the UB budget for exactly the two fixtures
        above: ``2 buffers * 2 stages * 2 bytes * 32768 = 262144`` bytes, over the
        221,184-byte (216 KiB) envelope, and failed on real hardware with
        ``RuntimeError: aclrtLaunchKernelWithHostArgs failed ... dyn_ubuf_bytes=262144 ...
        the sum of dynamic shared memory and kernel shared memory exceeds the maximum limit``
        (ACL error 107000) -- not a flaky failure, a hard UB-capacity violation, caught by M3
        re-running against the real fixtures, not assumed safe from the manifest cases alone.
        Taking the same per-dtype ``num_stages`` the caller already computed for the primary
        tile keeps the two builders' UB usage consistent by construction.
        """
        full_tiles, remainder = divmod(n_total, tile)

        @T.prim_func
        def main(X: T.Tensor((n_total,), dtype), Y: T.Tensor((n_total,), dtype)):
            with T.Kernel(1):
                if full_tiles:
                    x_ub = T.alloc_shared((tile,), dtype)
                    y_ub = T.alloc_shared((tile,), dtype)
                    T.annotate_buffer_versions({x_ub: num_stages, y_ub: num_stages})
                    for step in T.Pipelined(full_tiles, num_stages=num_stages):
                        begin = step * tile
                        T.copy(X[begin : begin + tile], x_ub)
                        _erf_body(x_ub, y_ub, tile, threads=min(2048, tile), dtype=dtype)
                        T.copy(y_ub, Y[begin : begin + tile])
                if remainder:
                    xt_ub = T.alloc_shared((remainder,), dtype)
                    yt_ub = T.alloc_shared((remainder,), dtype)
                    begin = full_tiles * tile
                    T.copy(X[begin : begin + remainder], xt_ub)
                    _erf_body(xt_ub, yt_ub, remainder, threads=min(128, remainder), dtype=dtype)
                    T.copy(yt_ub, Y[begin : begin + remainder])

        return tilelang.compile(main, target="ascend", out_idx=-1)

    @functools.lru_cache(maxsize=128)
    def _build_with_tail(n_total: int, dtype_str: str):
        """Bounded UB tail; only valid elements cross the global-memory boundary."""
        tile = _tile_for_dtype(dtype_str)
        stages = _stages_for_dtype(dtype_str)
        # Leave room for both tail buffers alongside pipelined full tiles.
        itemsize = 4 if dtype_str == "float32" else 2
        while (2 * tile * stages + 2 * ((n_total % tile + 63) // 64 * 64)) * itemsize > 248 * 1024:
            tile //= 2
        full, tail = divmod(n_total, tile)
        grid = min(64, full) if full else 1
        padded_tail = (tail + 63) // 64 * 64

        @T.prim_func
        def main(X: T.Tensor((n_total,), dtype_str), Y: T.Tensor((n_total,), dtype_str)):
            with T.Kernel(grid) as bx:
                if full:
                    x = T.alloc_shared((tile,), dtype_str)
                    y = T.alloc_shared((tile,), dtype_str)
                    T.annotate_buffer_versions({x: stages, y: stages})
                    for step in T.Pipelined((full + grid - 1) // grid, num_stages=stages):
                        block = step * grid + bx
                        if block < full:
                            begin = block * tile
                            T.copy(X[begin : begin + tile], x)
                            _erf_body(x, y, tile, min(tile, 2048), dtype_str)
                            T.copy(y, Y[begin : begin + tile])
                if tail:  # noqa: SIM102 - preserve validated TileLang DSL structure
                    if bx == 0:
                        xt = T.alloc_shared((padded_tail,), dtype_str)
                        yt = T.alloc_shared((padded_tail,), dtype_str)
                        T.copy(X[full * tile : n_total], xt[:tail])
                        _erf_body(xt, yt, padded_tail, min(padded_tail, 2048), dtype_str)
                        T.copy(yt[:tail], Y[full * tile : n_total])

        return tilelang.compile(main, target="ascend", out_idx=-1)

    def _make_erf_kernel(n_total: int, dtype: str):
        """Pick the tiling for ``n_total`` elements: small multi-block, single-block,
        persistent, or fallback."""
        if n_total % 64:
            return _build_with_tail(n_total, dtype)
        if n_total <= _SINGLE_BLOCK_THRESHOLD:
            num_blocks = _calibrated_num_blocks()
            if n_total % num_blocks == 0:
                return _make_erf_small_multi_block(n_total, dtype, num_blocks)
            return _make_erf_single_block(n_total, dtype)
        tile = _tile_for_dtype(dtype)
        num_blocks = _calibrated_num_blocks()
        block_span = num_blocks * tile
        if n_total % block_span == 0:
            return _make_erf_persistent(n_total, dtype, num_blocks, tile, _stages_for_dtype(dtype))
        return _build_with_tail(n_total, dtype)

    class AscendErfKernel(Kernel, UnaryElementwiseFwdInterface):
        """Erf on Ascend NPUs: ``y = erf(x)``; op-local fp32 rational and 16-bit fits.

        float32 uses the five-coefficient exp/rational body, with explicit signed-zero
        preservation, exact |x|>=5 saturation and ``keep_nan``; it is gated by the
        original 1e-5 device contract. float16/bfloat16 use the op-local clamped
        degree-8/degree-6 polynomial evaluated in fp32, preserving the shared +/-3.6
        clamp, scale-before-square, +/-1 clip and ``keep_nan`` contract. The 16-bit
        coefficients are the root-selected CPU-screen values and their device gate is
        the original exhaustive ``ErfRoundingWorkload`` comparator. See
        ``docs/ascend950/dev/ErfFwdOp/dsh_20261020_schedule.md``.

        No ``refusal()`` override: the manifest admits exactly ``{float16, bfloat16,
        float32}`` for this op and has no integer-dtype sibling kernel to route around
        (``ErfFwdOp.kernel_types = {"erf": ErfFwdKernel}``, a single key -- unlike
        ``AbsFwdOp``/``NegFwdOp``/``SignFwdOp``, which pair with an ``*IntFwdKernel``), so the
        base class's permissive default (serves every call) is already correct and complete.
        """

        devices = frozenset({"npu"})

        @classmethod
        def entry_for(cls, call: ElementwiseCall) -> Entry:
            return call, lambda: cls(call.n_total, call.dtype)

        def __init__(self, n_total: int, dtype: torch.dtype):
            super().__init__()
            self.n_total = n_total
            self.dtype = dtype
            self._compiled_fn = _make_erf_kernel(n_total, _dtype_str(dtype)) if n_total else None

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            if (
                input.device.type != "npu"
                or input.dtype != self.dtype
                or input.numel() != self.n_total
            ):
                raise ValueError("input must match compiled NPU kernel")
            if not self.n_total:
                return torch.empty_like(input)
            flat = input.contiguous().reshape(-1)
            result = self._compiled_fn(flat)
            return result.reshape(input.shape)

    tileops.backend.register_kernel_type("ErfFwdOp", "ascend", AscendErfKernel)

except ImportError:
    pass
