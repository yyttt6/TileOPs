"""Ascend Cos kernel: ``y = cos(x)``, via the ``tilelang.ascend`` dialect.

Registers an additional implementation of ``CosFwdOp``'s ``UnaryElementwiseFwdInterface``
under the ``"ascend"`` dispatch key, beside the in-tree CUDA ``CosFwdKernel``
(``tileops.kernels.elementwise.math_unary.CosFwdKernel``). No existing file is edited to do
this: ``tileops.backend.register_kernel_type`` joins this class to every ``CosFwdOp`` instance
constructed after this module is imported (``docs/ascend950/dev_process.md`` §1).

Design: ``docs/ascend950/dev/CosFwdOp/round_1_schedule.md``. This op has the identical cost
shape to ``TanhFwdOp``/``SqrtFwdOp``/``SinFwdOp`` (1 load + 1 cheap transcendental + 1 store per
element, memory-bound, 2 UB buffers) and reuses their already-calibrated persistent-grid tiling
verbatim: ``num_blocks`` from the live-queried, floored-to-power-of-2 vector core count (64 on
this SKU), ``tile=8192`` uniformly across all three dtypes, ``num_stages=2``,
``threads=min(2048, tile)=2048``.

Manifest confirmed (``elementwise_unary_math.yaml``): ``signature.forall.T =
DType[float16 | bfloat16 | float32]`` only, no integer type in the union, and
``CosFwdOp.kernel_types = {"cos": CosFwdKernel}`` has no ``"cos_int"`` companion (unlike
``AbsFwdOp``/``NegFwdOp``/``SignFwdOp``) -- trig-of-an-integer has no int-output interpretation
``torch.cos`` itself defines, so there is no int-dtype branch to add here and no ``refusal()``
override is needed (the default, serving every call, is correct and complete). 5 manifest
cases, all prefill-sized (16M = 4096x4096 x {bfloat16, float16, float32}, 256M = 16384x16384 x
{bfloat16, float16}); no decode-shaped (``[1, N]``) case exists, so this file does not implement
the small-``n_total`` core-spreading machinery ``ReluFwdOp``/``GeluFwdOp``/``SiluFwdOp`` need
for their own decode cases.

Math body mirrors ``CosFwdKernel.op_func`` (``tileops/kernels/elementwise/math_unary.py``):
compute in fp32 regardless of storage dtype, cast back at the boundary
(``code-style.md``'s overflow-prone-math promotion rule). This is also the only path that
compiles for ``bfloat16`` at all: a fresh compile-only probe (schedule doc §1) confirmed the
project's known bf16 transcendental gap (``dev_process.md`` §7.5, first established for
``tanh``/``sqrt``, re-confirmed for ``sin``) applies to ``cos`` too -- a direct, uncast ``T.cos``
call on a ``bfloat16`` buffer fails with ``InternalError: Unresolved call ir.Op(...,
name="tirx.cos", ...)``, while ``float16`` *does* have a compiling native alternative (``hcos``,
see ``_cos_body_native16`` below). Round 1 ships the uniform fp32-promote body for all three
dtypes without wiring in the native path by default: ``TanhFwdOp`` round 2 measured the
structurally-identical-shaped native ``htanh`` path as 3.4-8% *slower* than the promote path on
real hardware, despite compiling to leaner code, so compiling is not evidence ``hcos`` would be
a win here either -- see ``_cos_body_native16``'s docstring.

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

        Verbatim reuse of ``tanh.py``/``sqrt.py``/``sin.py``'s already-calibrated,
        already-hardware-validated helper (round_1_schedule.md §3): queries this SKU's real
        vector-core count (``get_num_vector_cores`` -- ``vector_core_num=72`` on
        ``Ascend950DT_9592``, not a power of two), then floors to the nearest power of two,
        landing on 64.

        Every ``n_total`` in scope for this op (the 2^20-element correctness fixture, the
        2^24/2^28-element benchmark cases) is a pure power of two, while 72=2^3x3^2 cannot
        divide any of them evenly. ``ReluFwdOp``/``BitwiseNotFwdOp``/``TanhFwdOp``/``SqrtFwdOp``/
        ``SinFwdOp`` all independently measured 64 beating 128 under the official protocol for
        this exact cheap-unary-op shape -- reused here, not re-swept.

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

    # Tile size and stage count for the memory-bound, large-n persistent-grid path (schedule
    # doc §3), reused unmodified from tanh.py/sqrt.py/sin.py's already-calibrated values;
    # num_blocks itself is resolved lazily by _calibrated_num_blocks() (see its docstring).
    # num_stages=3 is already a closed question for this cheap-unary-op family (ReluFwdOp
    # round 2, TanhFwdOp round 1 both independently found it measurably worse under the
    # official protocol) -- not re-tried here as a first move.
    _TILE = 8192
    _NUM_STAGES = 2

    # Round 1 M4 (uniform tile=8192 for all three dtypes): float32 comfortably passed its 80%
    # target (16M: best_baseline/measured = 0.928), but float16/bfloat16 missed at *both*
    # manifest sizes by an almost size-invariant ~27% (16M: 0.734/0.733, 256M: 0.728/0.728 for
    # float16/bfloat16 respectively) -- round_1_result.md §4 (ratios there spell out the exact
    # numbers). This is the "fp16/bf16 miss while fp32 passes comfortably" signature
    # round_1_schedule.md §4 attributes to the 16-bit pack/unpack conversion tax
    # (``TanhFwdOp``'s mechanism), not the §3 precedent pattern (that precedent is "16M fine,
    # only 256M misses") -- but round_1_schedule.md §1 still prescribes trying this same
    # tile-widening lever first, before treating a float16 miss as evidence for the
    # deprioritized native-``hcos`` path. Applying it here per that instruction, to both 2-byte
    # dtypes (both miss, unlike ``SqrtFwdOp``'s float16-only asymmetry) -- round_1_result.md §4
    # has the before/after measurement that justifies this split (float32 excluded: it already
    # passes comfortably at 8192, and ``SqrtFwdOp`` round 1 showed blindly widening a
    # non-missing dtype can regress it).
    _WIDE_TILE = 16384
    _WIDE_TILE_DTYPES = frozenset({"float16", "bfloat16"})

    def _tile_for_dtype(dtype_str: str) -> int:
        return _WIDE_TILE if dtype_str in _WIDE_TILE_DTYPES else _TILE

    # Below this many elements, a persistent multi-block grid and T.Pipelined staging would add
    # fixed per-launch overhead with no bandwidth-bound tail to amortize it against: run the
    # whole call on a single block instead, one shot, no pipelining. No decode-shaped manifest
    # case exists for this op (schedule doc §0/§3), so this threshold only needs to cover the
    # correctness fixture's small n_total (1_048_576 for test_cos, which is itself still larger
    # than this threshold and so uses the persistent path -- this builder only serves an even
    # smaller n_total than any fixture currently exercises).
    _SINGLE_BLOCK_THRESHOLD = _TILE

    def _dtype_str(dtype: torch.dtype) -> str:
        return str(dtype).split(".")[-1]

    @T.macro
    def _cos_body_native16(x_ub, y_ub, n: int, threads: int) -> None:
        """``y_ub[:n] = cos(x_ub[:n])`` with no fp32 promotion -- ``float16``, UNUSED.

        **Not used by any builder below. Kept only as a documented, compile-verified
        candidate for a future measurement-gated experiment (round_1_schedule.md §1/§3);
        do not wire this in as the shipped path without new hardware evidence that it wins.**

        A compile-only probe (round_1_schedule.md §1) found that a direct, uncast ``T.cos``
        call on a ``float16`` buffer at this kernel's real tile geometry (``tile=8192``,
        ``threads=2048``, 4 elements/thread) compiles to 4 native ``hcos()`` calls operating in
        place on vectorized ``half2``-packed registers -- one 128-bit vectorized load, one
        128-bit vectorized store, no ``half2<->float2`` unpack/repack and no local-array
        staging -- structurally identical in shape to ``TanhFwdOp`` round 2's
        ``_tanh_body_native16`` finding for ``htanh`` and ``SinFwdOp``'s ``_sin_body_native16``
        finding for ``hsin``.

        That structural leanness did **not** translate into a measured win for ``tanh``:
        round 2 (``docs/ascend950/dev/TanhFwdOp/round_2_result.md``) found the native ``htanh``
        path 3.4-8% *slower* than the fp32-promote path on real hardware, reproducibly, at both
        AIV-cycle and wall-clock granularity -- the hardware 16-bit transcendental instruction
        itself apparently costs more cycles per call than the toolchain's software ``tanhf``,
        enough to outweigh the saved conversion/staging instructions. There is no a priori
        reason ``hcos`` vs. ``cosf`` resolves differently, and this round's schedule explicitly
        defers this macro to a round-2+-only lever, to be spent only if ``float16`` misses its
        target *and* the tile-widening lever (round_1_schedule.md §3) doesn't close the gap on
        its own (round_1_schedule.md §1). ``float16`` therefore stays on
        ``_cos_body_promote_f32`` below unless round 1 M4 evidence says otherwise.
        """
        with T.SimtVF(threads=threads):
            for i in T.Parallel(n):
                y_ub[i] = T.cos(x_ub[i])

    @T.macro
    def _cos_body_promote_f32(x_ub, y_ub, n: int, threads: int) -> None:
        """``y_ub[:n] = cos(x_ub[:n])``, compiler-vectorized over ``threads`` lanes.

        Mirrors ``CosFwdKernel.op_func`` (``kernels/elementwise/math_unary.py``) exactly:
        promote to fp32 (code-style.md's overflow-prone-math rule), call ``T.cos``, cast back
        to the storage dtype at the boundary. Used for **every** dtype (round_1_schedule.md
        §2): it is the only path that compiles for ``bfloat16`` at all (direct ``T.cos`` on a
        bf16 buffer fails to compile, ``InternalError: Unresolved call
        ir.Op(..., name="tirx.cos", ...)`` -- the same bf16 transcendental gap already
        confirmed for ``tanh``/``sqrt``/``sin``, ``dev_process.md`` §7.5), it is a no-op
        promotion for ``float32`` (compute dtype already equals storage dtype), and for
        ``float16`` it is kept over the compiling-but-unvalidated native ``hcos`` alternative
        (``_cos_body_native16``) per ``TanhFwdOp`` round 2's hard-won lesson that a
        structurally leaner native 16-bit path is not evidence of a real speed win -- see that
        macro's docstring.

        No domain-edge-case handling is needed here: ``cos`` is defined and bounded in
        ``[-1, 1]`` for every finite real input, and no ``test_cos_edge`` fixture exists to
        probe a restricted domain the way ``sqrt``/``log``/``reciprocal`` have
        (round_1_schedule.md §0).

        A ``@T.macro``, not a plain Python helper (code-style.md: "sub-routines use
        ``@T.macro``, never nested ``prim_func``"): TileLang's eager builder rewrites only
        ``@T.prim_func``/``@T.macro`` bodies into its trace-builder calls, so an undecorated
        helper's ``for i in T.Parallel(n):`` fails with ``TypeError: 'ForFrame' object is not
        iterable``.
        """
        with T.SimtVF(threads=threads):
            for i in T.Parallel(n):
                y_ub[i] = T.Cast(x_ub.dtype, T.cos(T.Cast("float32", x_ub[i])))

    @functools.lru_cache(maxsize=64)
    def _make_cos_single_block(n_total: int, dtype: str):
        """One block, one tile, no pipelining: correctness-first fallback for a small shape
        that does not divide ``_calibrated_num_blocks()`` evenly."""

        @T.prim_func
        def main(X: T.Tensor((n_total,), dtype), Y: T.Tensor((n_total,), dtype)):
            with T.Kernel(1):
                x_ub = T.alloc_shared((n_total,), dtype)
                y_ub = T.alloc_shared((n_total,), dtype)
                T.copy(X, x_ub)
                _cos_body_promote_f32(x_ub, y_ub, n_total, threads=min(128, n_total))
                T.copy(y_ub, Y)

        return tilelang.compile(main, target="ascend", out_idx=-1)

    @functools.lru_cache(maxsize=64)
    def _make_cos_small_multi_block(n_total: int, dtype: str, num_blocks: int):
        """Multi-block, one-shot grid for a small shape that divides ``num_blocks`` evenly
        (schedule doc §3): spreads a small, correctness-only shape across the calibrated core
        count rather than collapsing to a single core, per the "decode/small shape shouldn't be
        single-core" lesson (dev_process.md §7.5) -- cheap to do right here even though this
        op's own manifest cases never reach this path.

        No ``T.Pipelined`` loop: each of ``num_blocks`` blocks does exactly one GM->UB copy,
        one compute-body call, one UB->GM copy. Requires ``n_total`` to divide ``num_blocks``
        exactly; the caller only reaches this builder for a shape that does.
        """
        assert n_total % num_blocks == 0, (
            f"_make_cos_small_multi_block requires n_total ({n_total}) to divide "
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
                _cos_body_promote_f32(x_ub, y_ub, chunk, threads=min(2048, chunk))
                T.copy(y_ub, Y[begin : begin + chunk])

        return tilelang.compile(main, target="ascend", out_idx=-1)

    @functools.lru_cache(maxsize=64)
    def _make_cos_persistent(n_total: int, dtype: str, num_blocks: int, tile: int, num_stages: int):
        """Persistent multi-block grid with ``T.Pipelined`` staging (schedule doc §3); the
        primary path, serving all 5 manifest cases.

        UB budget: 2 buffers (x_ub, y_ub) -- ``2 x 2 x itemsize x 8192`` is 131,072 bytes at
        fp32 and 65,536 bytes at fp16/bf16, both comfortably inside ``example_vecadd.py``'s own
        hardware-validated 3-buffer/fp32/196,608-byte footprint at the same tile/stage count
        (``tanh.py``'s reasoning, directly reusable since the buffer count and dtype set are
        identical), so no per-dtype tile widening is needed up front.

        Requires ``n_total`` to divide ``num_blocks * tile`` exactly; the caller only reaches
        this builder for a shape that does, and routes every other shape to
        ``_make_cos_generic_single_block`` instead.
        """
        block_span = num_blocks * tile
        assert n_total % block_span == 0, (
            f"_make_cos_persistent requires n_total ({n_total}) to divide "
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
                    _cos_body_promote_f32(x_ub, y_ub, tile, threads=min(2048, tile))
                    T.copy(y_ub, Y[begin : begin + tile])

        return tilelang.compile(main, target="ascend", out_idx=-1)

    @functools.lru_cache(maxsize=64)
    def _make_cos_generic_single_block(n_total: int, dtype: str, tile: int):
        """Correctness-first fallback for an ``n_total`` the persistent tiling does not evenly
        divide. One block; a main loop over full tile-sized chunks, plus one smaller,
        statically-sized tail chunk. Not performance-tuned: none of this op's 5 manifest
        cases reaches this function this round (all 5 divide ``block_span`` exactly), so it
        is written but unexercised by M3/M4 -- see the round-1 result report.
        """
        full_tiles, remainder = divmod(n_total, tile)

        @T.prim_func
        def main(X: T.Tensor((n_total,), dtype), Y: T.Tensor((n_total,), dtype)):
            with T.Kernel(1):
                if full_tiles:
                    x_ub = T.alloc_shared((tile,), dtype)
                    y_ub = T.alloc_shared((tile,), dtype)
                    T.annotate_buffer_versions({x_ub: 2, y_ub: 2})
                    for step in T.Pipelined(full_tiles, num_stages=2):
                        begin = step * tile
                        T.copy(X[begin : begin + tile], x_ub)
                        _cos_body_promote_f32(x_ub, y_ub, tile, threads=min(2048, tile))
                        T.copy(y_ub, Y[begin : begin + tile])
                if remainder:
                    xt_ub = T.alloc_shared((remainder,), dtype)
                    yt_ub = T.alloc_shared((remainder,), dtype)
                    begin = full_tiles * tile
                    T.copy(X[begin : begin + remainder], xt_ub)
                    _cos_body_promote_f32(xt_ub, yt_ub, remainder, threads=min(128, remainder))
                    T.copy(yt_ub, Y[begin : begin + remainder])

        return tilelang.compile(main, target="ascend", out_idx=-1)

    @functools.lru_cache(maxsize=128)
    def _build_with_tail(n_total: int, dtype_str: str):
        """Bounded UB tail; only valid elements cross the global-memory boundary."""
        tile = _tile_for_dtype(dtype_str)
        # Leave room for both tail buffers alongside pipelined full tiles.
        itemsize = 4 if dtype_str == "float32" else 2
        while (
            2 * tile * _NUM_STAGES + 2 * ((n_total % tile + 63) // 64 * 64)
        ) * itemsize > 248 * 1024:
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
                    T.annotate_buffer_versions({x: _NUM_STAGES, y: _NUM_STAGES})
                    for step in T.Pipelined((full + grid - 1) // grid, num_stages=_NUM_STAGES):
                        block = step * grid + bx
                        if block < full:
                            begin = block * tile
                            T.copy(X[begin : begin + tile], x)
                            _cos_body_promote_f32(x, y, tile, min(tile, 2048))
                            T.copy(y, Y[begin : begin + tile])
                if tail:  # noqa: SIM102 - preserve validated TileLang DSL structure
                    if bx == 0:
                        xt = T.alloc_shared((padded_tail,), dtype_str)
                        yt = T.alloc_shared((padded_tail,), dtype_str)
                        T.copy(X[full * tile : n_total], xt[:tail])
                        _cos_body_promote_f32(xt, yt, padded_tail, min(padded_tail, 2048))
                        T.copy(yt[:tail], Y[full * tile : n_total])

        return tilelang.compile(main, target="ascend", out_idx=-1)

    def _make_cos_kernel(n_total: int, dtype: str):
        """Pick the tiling for ``n_total`` elements: small multi-block, single-block,
        persistent, or fallback.

        The persistent path's ``tile`` is dtype-dependent (``_tile_for_dtype``): float16 and
        bfloat16 use the widened 16384 (round 1 M4 lever, see ``_WIDE_TILE_DTYPES``'s comment),
        float32 keeps 8192 since it already passes comfortably there.
        """
        if n_total % 64:
            return _build_with_tail(n_total, dtype)
        if n_total <= _SINGLE_BLOCK_THRESHOLD:
            num_blocks = _calibrated_num_blocks()
            if n_total % num_blocks == 0:
                return _make_cos_small_multi_block(n_total, dtype, num_blocks)
            return _make_cos_single_block(n_total, dtype)
        tile = _tile_for_dtype(dtype)
        num_blocks = _calibrated_num_blocks()
        block_span = num_blocks * tile
        if n_total % block_span == 0:
            return _make_cos_persistent(n_total, dtype, num_blocks, tile, _NUM_STAGES)
        return _build_with_tail(n_total, dtype)

    class AscendCosKernel(Kernel, UnaryElementwiseFwdInterface):
        """Cos on Ascend NPUs: ``y = cos(x)``, promoted to fp32 then cast back.

        One load, one cheap transcendental call, one store per element; see
        ``docs/ascend950/dev/CosFwdOp/round_1_schedule.md`` §3 for the base tiling rationale
        (``num_blocks``/``num_stages``/``threads`` are an unmodified reuse of
        ``TanhFwdOp``/``SqrtFwdOp``/``SinFwdOp``'s calibrated values). A fresh class per
        ``docs/ascend950/dev_process.md`` §1 -- it does not subclass the CUDA-locked
        ``UnaryKernel``/``FloatUnaryKernel`` family, which imports ``tilelang.language`` (the
        CUDA dialect facade).

        Round 1 M4's first measurement (uniform ``tile=8192`` for all three dtypes) found
        ``float32`` comfortably clearing its 80% target (ratio 0.928) at both the conversion-tax
        and bandwidth level, but ``float16``/``bfloat16`` missing at **both** manifest sizes by
        an almost size-invariant ~27% (16M: ratio 0.733-0.734; 256M: ratio 0.728, both dtypes)
        -- the ``TanhFwdOp``-style "16-bit pack/unpack conversion tax dominates" signature
        (``round_1_schedule.md`` §4), not the narrower "16M fine, only 256M misses" pattern
        ``SqrtFwdOp``/``RsqrtFwdOp`` saw. Applying ``round_1_schedule.md`` §1/§3's prescribed
        tile-widening lever to both 2-byte dtypes (``_WIDE_TILE_DTYPES``, 8192->16384; ``float32``
        left untouched at 8192 since it already passed) closed the gap at every case: re-measured
        ratios 0.877-0.880 across all four previously-missing cases, with ``float32`` unaffected
        (still 0.928) -- see ``round_1_result.md`` for the full before/after table. All 5
        manifest cases pass as shipped.

        No ``refusal()`` override: the manifest admits exactly ``{float16, bfloat16,
        float32}`` for this op and has no integer-dtype sibling kernel to route around
        (``CosFwdOp.kernel_types = {"cos": CosFwdKernel}``, a single key -- unlike
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
            self._compiled_fn = _make_cos_kernel(n_total, _dtype_str(dtype)) if n_total else None

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

    tileops.backend.register_kernel_type("CosFwdOp", "ascend", AscendCosKernel)

except ImportError:
    pass
