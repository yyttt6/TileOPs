"""Ascend Hardsigmoid kernel: ``y = clamp(x + 3, 0, 6) / 6``, via the ``tilelang.ascend``
dialect.

Registers an additional implementation of ``HardsigmoidFwdOp``'s
``UnaryElementwiseFwdInterface`` under the ``"ascend"`` dispatch key, beside the in-tree CUDA
``HardsigmoidFwdKernel`` (``tileops.kernels.elementwise.activations.HardsigmoidFwdKernel``). No
existing file is edited: ``tileops.backend.register_kernel_type`` joins this class to every
``HardsigmoidFwdOp`` instance constructed after this module is imported
(``docs/ascend950/dev_process.md`` §1). ``inplace`` is handled entirely by the op layer
(``_UnaryActivationMixin.forward``, which copies the kernel's fresh buffer back into ``input``
when asked); this kernel's ``forward`` always returns a new buffer, exactly like
``AscendReluKernel``/``relu.py``. An empty call never reaches this kernel: ``Op._run_call``
short-circuits a zero-element call to ``_empty_result`` before any kernel is built
(``ops/op_base.py`` empty branch), so no builder here ever has to launch a zero-thread grid.

Follows ``docs/ascend950/dev/HardsigmoidFwdOp/resume_20261019_schedule.md`` and its
``resume_20261019_review.md`` (which supersede ``round_1_schedule.md``). Two tiling paths:

1. **One-shot grid (``_make_hardsigmoid_oneshot``), the hot path.** Each block does exactly one
   GM->UB copy, one ``_hardsigmoid_body`` call, one UB->GM copy -- no device loop, nothing to
   pipeline. It serves every shape whose *per-block* chunk fits ``_ONESHOT_MAX_CHUNK``: the two
   manifest cases (7680, 30720 elements, 24/32 blocks -> chunk 320/960) and every M3 shape to
   date (64 -> chunk 1, 768 -> chunk 12, 1,048,576 -> 64 blocks -> chunk 16384). Its UB
   footprint is two chunk-wide buffers with no double-buffering, so the fp32 worst case at the
   cap is ``2 * 16384 * 4 B = 128 KiB`` -- exactly what the 1,048,576-element M3 case already
   runs at, and comfortably under this SKU's measured ~256 KiB UB overflow point
   (``dev_process.md`` §7.5, BitwiseNotFwdOp). The earlier claim that every footprint stayed
   "well under a tenth of the UB budget" was wrong (schedule §1.3) and is removed.
2. **Bounded-UB grid (``_make_hardsigmoid_bounded``), the correctness fallback.** When the
   per-block chunk would exceed ``_ONESHOT_MAX_CHUNK``, pouring it into one shot would blow the
   UB -- a prime ``n_total`` above the vector-core count picks a single block, e.g.
   ``n_total=65537`` -> 1 block -> a 65537-wide chunk, ``2 * 65537 * 4 B ~= 512 KiB`` in fp32,
   far past the overflow point (schedule §1.3, review §4.1). Such a chunk is instead walked as a
   ``T.Pipelined`` main loop of fixed ``_BOUNDED_TILE``-wide steps plus one static tail of
   ``chunk % _BOUNDED_TILE`` elements, so the UB footprint depends on the tile, not on
   ``n_total``: fp32 holds ``2 * 8192 * 4 B * 2 stages = 128 KiB`` in the main loop plus a tail
   of at most ``2 * 8191 * 4 B ~= 64 KiB``, a <=192 KiB worst-case peak still under the ~256
   KiB overflow point. This mirrors ``relu.py``'s ``_make_relu_generic_single_block`` tail
   idiom, extended across this op's multi-block grid through the per-block ``base`` offset; the
   tile/stage values themselves are the ones ``relu.py``'s persistent path has run on hardware.
   Non-manifest shapes are correctness-only (no latency target, schedule §1.3/§4.1), so a prime
   collapsing to a single block is accepted here -- unlike the hidden single-core *performance*
   regression ``dev_process.md`` §7.5 "Batch 6" warns about for a shape that *does* have a
   target, which does not apply to this op's correctness-only fallback.

``num_blocks`` is shape-specific and keyed per ``n_total``: the small manifest case (7680) keeps
24 blocks while the deep one (30720) uses 32. Those per-shape choices come from the prior Claude
Opus 4.8 round-2 candidate matrix (16/24/32 blocks, each a full 4-case formal M4, reviewed in
``docs/ascend950/dev/HardsigmoidFwdOp/opus48_round2_review.md``): at 32 blocks the small case
regressed ~20--27% versus 24, while the deep case gained ~5--9%. This file therefore keeps each
shape's own best measured configuration; the dsh round-1 same-hash M3, production-Op boundary
probe and formal 4-case M4 that re-verify the mixed mapping are recorded in
``docs/ascend950/dev/HardsigmoidFwdOp/dsh_round1_result.md``. The earlier resume M4
(``resume_20261019_result.md`` / ``resume_20261019_review.md``, directory
``m4_20261019T033855.079356Z``) had confirmed the uniform-24 starting point passed all four
graded cases (worst ratio 1.123); that record is history, not this round's measurement. The
missing ``round_1_result.md`` an old comment cited as a measured sweep is not evidence and is no
longer referenced. Every non-manifest shape uses the
generic rule (schedule §3/§4): ``num_blocks`` = the largest divisor of ``n_total`` that does not
exceed this SKU's real vector-core count, queried live via ``get_num_vector_cores``
(``_choose_fallback_num_blocks``) -- always an exact divisor of ``n_total`` by construction, so
every block owns the same ``chunk`` and ``base = bx * chunk`` tiles both builders cleanly.

Both builders take ``num_blocks`` (and the bounded one ``tile``/``num_stages``) as explicit
parameters so the launch configuration is part of each ``functools.lru_cache`` key: a later
change to ``_MANIFEST_NUM_BLOCKS`` then compiles a fresh program rather than reusing a cached
binary built for the old block count (schedule §1.4, "no change-map -> reuse-cache -> paste old
data onto a new config").

**NaN propagation is a tested correctness requirement for this op**
(``tests/ops/test_special_elementwise.py::test_clamp_family_propagates_nan_like_torch``,
the ``"hardsigmoid-nan-input"`` case) -- the manifest's CUDA reference
(``HardsigmoidFwdKernel.op_func``) uses ``nan_max``/``nan_min``
(``tileops.kernels.elementwise._nan``) specifically to get this. This file does **not** port
``nan_max``/``nan_min``: their fp16/bf16 fast path calls the TileLang builtins
``tl.max_nan``/``tl.min_nan``, which ``_nan.py``'s own docstring says lower to CUDA's
``__hmax_nan``/``__hmin_nan`` -- there is no record anywhere in this project of those builtins
having an Ascend lowering, and no Ascend kernel in this directory has ever called ``T.max``/
``T.min``. Instead, ``_hardsigmoid_body`` below builds the clamp from primitives already
verified on this backend elsewhere in this project (``T.if_then_else`` -- ``relu.py``;
comparison operators), written with the comparison direction schedule §1 specifies: each branch
is "out of bounds -> replace with the bound, else -> pass through"
(``shifted > six -> six``, ``upper < zero -> zero``), the mirror image of ``relu.py``'s
direction (``x > zero -> x``, "in range -> keep"). Under IEEE 754 ordered-compare semantics
(`>`/`<` are false whenever either operand is NaN), a NaN operand makes both conditions false,
so NaN falls through every ``T.if_then_else`` unmodified and survives to the final multiply --
no explicit ``T.isnan`` branch is needed for this to work, and ``+Inf``/``-Inf`` saturate to
``1.0``/``0.0`` the same way ``torch.nn.functional.hardsigmoid`` does. Schedule §5.1/review §4.2
flagged this as a real risk to verify on hardware rather than assume (this project's ``_nan.py``
separately warns that ``!=`` behaves unexpectedly on at least one backend, so "comparisons
behave the standard way" is not something to take for granted here) -- the resume M3 run of
``test_clamp_family_propagates_nan_like_torch``, plus this round's exclusive extreme-value probe
(NaN/+-Inf across all three dtypes), records the actual verification.

Importing this module when ``tilelang.ascend`` is unavailable (no Ascend toolchain, a CUDA-only
process) is a no-op: the Ascend-specific imports, the kernel class, and the registration call are
wrapped in one ``try/except ImportError`` block, per ``docs/ascend950/dev_process.md`` §1.

This file does not get a subpackage ``__init__.py`` with ``__all__`` re-exports, which
``TileOPs/.claude/rules/code-style.md`` otherwise requires for every ``src/tileops/kernels/*``
subpackage: ``docs/ascend950/dev_process.md`` §1 explicitly asks every op's worker to
self-register from its own file and *not* build a shared aggregator file under
``kernels/ascend/``, since several ops' workers write into this same new directory
concurrently. ``kernels/ascend`` and ``kernels/ascend/elementwise`` are therefore left as
implicit (PEP 420) namespace packages, matching every sibling file here.
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

    # Fallback vector-core count when the live query is unavailable. Confirmed 72 on this SKU
    # (device Ascend950DT_9592) by multiple independent prior ops in this project
    # (docs/ascend950/dev_process.md §7.5).
    _FALLBACK_VECTOR_CORES = 72
    # T.SimtVF's hard 2048-thread cap (dev_process.md §7.5), confirmed project-wide. A tile
    # wider than this is multiple elements per thread, never one thread per element.
    _MAX_THREADS = 2048

    # Measured num_blocks for the two manifest decode-size cases, keyed per n_total. Chosen from
    # the prior Claude Opus 4.8 round-2 candidate matrix (16/24/32 blocks, each measured as a
    # full 4-case formal M4 against the fixed v2 baseline): 24 wins the 7680 small case (32
    # blocks regressed ~20--27% there) and 32 wins the 30720 deep case (~5--9% faster than 24);
    # 16 was not better for either shape beyond noise. Kept per n_total, not shared, since
    # 30720 = 4*7680's optimum is demonstrably not equal to 7680's.
    _MANIFEST_NUM_BLOCKS = {
        7680: 24,
        30720: 32,
    }

    # Per-block element count at or below which one GM->UB->compute->UB->GM shot per block keeps
    # the UB footprint safe. The one-shot builder allocates two chunk-wide buffers with no
    # double-buffering, so fp32 at this cap is 2 * 16384 * 4 B = 128 KiB -- exactly the footprint
    # the 1,048,576-element M3 case (64 blocks, chunk 16384) already runs at, and well under this
    # SKU's ~256 KiB UB overflow point (dev_process.md §7.5, BitwiseNotFwdOp). A per-block chunk
    # larger than this (only a non-manifest shape whose largest divisor <= the vector-core count
    # still leaves a big quotient -- a prime n_total picks a single block) routes to the bounded
    # builder instead (schedule §1.3/§4, review §4.1). Both the dispatcher and the choice of the
    # bounded tile agree on this threshold.
    _ONESHOT_MAX_CHUNK = 16384

    # Bounded-fallback tiling for a per-block chunk above _ONESHOT_MAX_CHUNK. _BOUNDED_TILE is
    # the per-tile element count of the main loop; with _BOUNDED_NUM_STAGES=2 double-buffering,
    # fp32 holds 2 * 8192 * 4 B * 2 = 128 KiB in the main loop plus a tail of at most
    # 2 * 8191 * 4 B ~= 64 KiB -- a <=192 KiB worst-case peak that stays under the ~256 KiB
    # overflow point. These tile/stage values are the ones relu.py's persistent path has run on
    # hardware; both the dispatcher (passing them) and the bounded builder (consuming them) agree
    # on them, which is why they are module-level rather than builder locals.
    _BOUNDED_TILE = 8192
    _BOUNDED_NUM_STAGES = 2

    @functools.lru_cache(maxsize=1)
    def _calibrated_vector_cores() -> int:
        """This SKU's real vector-core count, live-queried once per process.

        Deliberately **not** evaluated at module import time -- merely importing this module
        must stay a no-NPU-touch no-op (dev_process.md §4); this is only called lazily, the
        first time a kernel for a non-manifest shape is actually built, always under an
        ``npu-guard`` reservation already. Falls back to ``_FALLBACK_VECTOR_CORES`` if the
        live query raises or returns a falsy value, rather than crashing kernel build.
        """
        try:
            queried = get_num_vector_cores(device_id=0)
        except Exception:
            queried = None
        if not queried:
            return _FALLBACK_VECTOR_CORES
        return int(queried)

    def _choose_fallback_num_blocks(n_total: int, vector_cores: int) -> int:
        """The largest divisor of ``n_total`` that does not exceed ``vector_cores``.

        Generic correctness-first rule for any shape outside the two manifest cases
        (schedule §3/§4): takes the largest divisor of ``n_total`` *itself* (not of
        ``n_total // tile``), so it never silently collapses a non-prime shape to one core the
        way the quotient-factoring formula ``dev_process.md`` §7.5 "Batch 6" warns about. It is
        always an exact divisor of ``n_total`` by construction, so each block owns an equal
        ``chunk``. A prime ``n_total`` above ``vector_cores`` legitimately returns 1 (e.g.
        65537 -> 1); that is a correctness-only fallback with no latency target, and the bounded
        builder keeps its single-block UB in budget.
        """
        upper = min(n_total, vector_cores)
        for candidate in range(upper, 0, -1):
            if n_total % candidate == 0:
                return candidate
        return 1  # unreachable: 1 always divides n_total

    def _num_blocks_for(n_total: int) -> int:
        manifest_choice = _MANIFEST_NUM_BLOCKS.get(n_total)
        if manifest_choice is not None:
            return manifest_choice
        return _choose_fallback_num_blocks(n_total, _calibrated_vector_cores())

    def _dtype_str(dtype: torch.dtype) -> str:
        return str(dtype).split(".")[-1]

    @T.macro
    def _hardsigmoid_body(x_ub, y_ub, n: int, threads: int) -> None:
        """``y_ub[:n] = clamp(x_ub[:n] + 3, 0, 6) / 6``, NaN-propagating, in fp32 compute.

        Mirrors ``HardsigmoidFwdKernel.op_func``'s math exactly (promote to fp32, add 3,
        two-sided clamp, multiply by the precomputed reciprocal ``1/6`` rather than a true
        divide) but builds the clamp from ``T.if_then_else`` + ordered comparisons instead of
        ``nan_max``/``nan_min`` -- see this module's docstring for why. Both branches are
        written "out of bounds -> bound, else -> pass through"
        (``shifted > six -> six``, ``upper < zero -> zero``): under IEEE ordered-compare
        semantics a NaN ``x`` makes every comparison false, so NaN threads through both
        ``T.if_then_else`` calls unmodified and survives the final multiply by
        ``one_sixth``, while ``+Inf``/``-Inf`` saturate to ``1.0``/``0.0``. This direction is
        NOT the same as ``relu.py``'s (``x > zero -> x``, "in range -> keep", which would
        instead collapse NaN to the bound and lose it), and swapping it back is the single
        highest-risk way to regress this op's NaN test, so do not "simplify" this to match
        relu.py's direction.

        Always promotes to fp32 (``code-style.md``'s cast-to-fp32-and-back rule) even for an
        fp32 input tile, matching the CUDA reference's "compute in fp32 for every dtype"
        choice (schedule §1) -- the cast is then a no-op, not a special case to avoid.

        A ``@T.macro``, not a plain Python helper (code-style.md: "sub-routines use
        ``@T.macro``, never nested ``prim_func``") -- an undecorated helper's
        ``for i in T.Parallel(n):`` fails with ``TypeError: 'ForFrame' object is not
        iterable`` under TileLang's eager builder (dev_process.md §7.5). It is called from both
        builders' main and tail tiles, so ``threads`` is capped at ``_MAX_THREADS`` by the
        caller, never derived from ``n`` alone.
        """
        with T.SimtVF(threads=threads):
            for i in T.Parallel(n):
                x32 = T.cast(x_ub[i], "float32")
                three = T.cast(3.0, "float32")
                six = T.cast(6.0, "float32")
                zero = T.cast(0.0, "float32")
                one_sixth = T.cast(1.0 / 6.0, "float32")
                shifted = x32 + three
                upper = T.if_then_else(shifted > six, six, shifted)
                clamped = T.if_then_else(upper < zero, zero, upper)
                y_ub[i] = T.cast(clamped * one_sixth, y_ub.dtype)

    @functools.lru_cache(maxsize=64)
    def _make_hardsigmoid_oneshot(n_total: int, dtype: str, num_blocks: int):
        """One-shot multi-block grid: each of ``num_blocks`` blocks does exactly one GM->UB
        copy, one ``_hardsigmoid_body`` call, one UB->GM copy -- no ``T.Pipelined`` loop
        (schedule §1/§4: every per-block chunk this builder serves is small enough that a single
        shot covers it, so there is nothing to pipeline).

        The caller (``_make_hardsigmoid_kernel``) only reaches this builder when the per-block
        ``chunk`` fits ``_ONESHOT_MAX_CHUNK``; a larger chunk routes to ``_make_hardsigmoid_
        bounded`` instead. ``num_blocks`` always divides ``n_total`` exactly (manifest value or
        the divisor rule) so this builder never needs a remainder/tail chunk. It is an explicit
        parameter -- part of this ``lru_cache`` key -- so a later ``_MANIFEST_NUM_BLOCKS`` change
        compiles a fresh program rather than reusing a binary built for the old block count
        (schedule §1.4).
        """
        assert n_total % num_blocks == 0, (
            f"_make_hardsigmoid_oneshot requires num_blocks ({num_blocks}) to divide n_total "
            f"({n_total}) exactly; _num_blocks_for should only ever return a divisor"
        )
        chunk = n_total // num_blocks

        @T.prim_func
        def main(X: T.Tensor((n_total,), dtype), Y: T.Tensor((n_total,), dtype)):
            with T.Kernel(num_blocks) as bx:
                x_ub = T.alloc_shared((chunk,), dtype)
                y_ub = T.alloc_shared((chunk,), dtype)
                begin = bx * chunk
                T.copy(X[begin : begin + chunk], x_ub)
                _hardsigmoid_body(x_ub, y_ub, chunk, threads=min(_MAX_THREADS, chunk))
                T.copy(y_ub, Y[begin : begin + chunk])

        return tilelang.compile(main, target="ascend", out_idx=-1)

    @functools.lru_cache(maxsize=64)
    def _make_hardsigmoid_bounded(
        n_total: int, dtype: str, num_blocks: int, tile: int, num_stages: int
    ):
        """Bounded-UB multi-block grid for a per-block chunk too large for one shot.

        Each of ``num_blocks`` blocks owns ``chunk = n_total // num_blocks`` contiguous
        elements at ``base = bx * chunk`` and walks them as full-width ``tile`` steps in a
        ``T.Pipelined`` main loop plus one static tail of ``chunk % tile`` elements. The UB
        footprint is therefore a function of ``tile`` (double-buffered main loop + a single-shot
        tail), not of ``chunk`` or ``n_total`` -- which is the whole point: it keeps a prime
        ``n_total``'s single-block chunk (e.g. 65537 -> 1 block -> chunk 65537) in budget
        instead of allocating a 65537-wide UB buffer (schedule §1.3, review §4.1). Mirrors
        ``relu.py``'s ``_make_relu_generic_single_block`` tail idiom, extended across this op's
        multi-block grid via the per-block ``base`` offset.

        All build parameters are explicit so the launch configuration is part of this
        ``lru_cache`` key (schedule §1.4). ``n_total % num_blocks == 0`` holds by construction,
        so ``chunk`` is uniform across blocks and ``base = bx * chunk`` tiles exactly. ``n_total
        = full_tiles * tile + remainder`` per block, so the main loop and the tail together
        cover ``[base, base + chunk)`` with no overlap and no gap; ``n_total`` is a Python int at
        build time, so both the loop trip count and the tail width are ordinary ints baked into
        the program (no dynamic-extent DMA).
        """
        assert n_total % num_blocks == 0, (
            f"_make_hardsigmoid_bounded requires num_blocks ({num_blocks}) to divide n_total "
            f"({n_total}) exactly; _num_blocks_for should only ever return a divisor"
        )
        chunk = n_total // num_blocks
        full_tiles, remainder = divmod(chunk, tile)

        @T.prim_func
        def main(X: T.Tensor((n_total,), dtype), Y: T.Tensor((n_total,), dtype)):
            with T.Kernel(num_blocks) as bx:
                base = bx * chunk
                if full_tiles:
                    x_ub = T.alloc_shared((tile,), dtype)
                    y_ub = T.alloc_shared((tile,), dtype)
                    T.annotate_buffer_versions({x_ub: num_stages, y_ub: num_stages})
                    for step in T.Pipelined(full_tiles, num_stages=num_stages):
                        begin = base + step * tile
                        T.copy(X[begin : begin + tile], x_ub)
                        _hardsigmoid_body(x_ub, y_ub, tile, threads=min(_MAX_THREADS, tile))
                        T.copy(y_ub, Y[begin : begin + tile])
                if remainder:
                    xt_ub = T.alloc_shared((remainder,), dtype)
                    yt_ub = T.alloc_shared((remainder,), dtype)
                    begin = base + full_tiles * tile
                    T.copy(X[begin : begin + remainder], xt_ub)
                    _hardsigmoid_body(xt_ub, yt_ub, remainder, threads=min(_MAX_THREADS, remainder))
                    T.copy(yt_ub, Y[begin : begin + remainder])

        return tilelang.compile(main, target="ascend", out_idx=-1)

    def _make_hardsigmoid_kernel(n_total: int, dtype: str):
        """Pick the tiling for ``n_total`` elements: the one-shot grid when each block's chunk
        fits ``_ONESHOT_MAX_CHUNK`` (every manifest and M3 shape to date), else the bounded-UB
        grid. ``num_blocks`` divides ``n_total`` exactly, so ``chunk`` is the uniform per-block
        element count that decides the route."""
        num_blocks = _num_blocks_for(n_total)
        chunk = n_total // num_blocks
        if chunk <= _ONESHOT_MAX_CHUNK:
            return _make_hardsigmoid_oneshot(n_total, dtype, num_blocks)
        return _make_hardsigmoid_bounded(
            n_total, dtype, num_blocks, _BOUNDED_TILE, _BOUNDED_NUM_STAGES
        )

    class AscendHardsigmoidKernel(Kernel, UnaryElementwiseFwdInterface):
        """Hardsigmoid on Ascend NPUs: ``y = clamp(x + 3, 0, 6) / 6``, NaN-propagating.

        A fresh class per ``docs/ascend950/dev_process.md`` §1 -- it does not subclass the
        CUDA-locked ``UnaryKernel``/``FloatUnaryKernel`` family, which imports
        ``tilelang.language`` (the CUDA dialect facade) and whose ``devices={"cuda"}`` would
        reject NPU tensors outright. Accepts exactly the manifest's float domain
        (``float16``/``bfloat16``/``float32``, schedule §1) -- there is no int-dtype sibling
        kernel for this op (it is a pure float activation, unlike ``AbsFwdOp``/``NegFwdOp``/
        ``SignFwdOp``'s ``*IntFwdKernel`` siblings), so the default (accept-everything)
        ``refusal()`` is correct as-is and is not overridden.
        """

        devices = frozenset({"npu"})

        @classmethod
        def entry_for(cls, call: ElementwiseCall) -> Entry:
            return call, lambda: cls(call.n_total, call.dtype)

        def __init__(self, n_total: int, dtype: torch.dtype):
            super().__init__()
            self.n_total = n_total
            self.dtype = dtype
            self._compiled_fn = _make_hardsigmoid_kernel(n_total, _dtype_str(dtype))

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            """Return ``hardsigmoid(input)`` as a new buffer; ``inplace`` is the op layer's
            job (``_UnaryActivationMixin.forward``), not this kernel's."""
            flat = input.reshape(-1)
            result = self._compiled_fn(flat)
            return result.reshape(input.shape)

    tileops.backend.register_kernel_type("HardsigmoidFwdOp", "ascend", AscendHardsigmoidKernel)

except ImportError:
    pass
