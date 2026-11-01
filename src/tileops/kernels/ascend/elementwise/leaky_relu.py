"""Ascend LeakyReLU kernel: ``y = x if x > 0 else negative_slope * x``, via the
``tilelang.ascend`` dialect.

Round 1 of ``docs/ascend950/dev_process.md``'s Ascend development loop for ``LeakyReluFwdOp``;
see ``docs/ascend950/dev/LeakyReluFwdOp/round_1_schedule.md`` for the full design this file
implements. Registers an additional implementation of ``LeakyReluFwdOp``'s own
``LeakyReluFwdInterface`` (``tileops.kernels.elementwise.call_spec.LeakyReluFwdInterface`` --
**not** the generic ``UnaryElementwiseFwdInterface`` most batch-1/2 unary ops use; schedule doc
§2.1 flags this as the ``ReciprocalFwdOp`` trap class: the two interfaces' ``forward(self,
input)`` signatures look identical, so getting this wrong raises ``"keys implement none of its
kernel interfaces"`` on *every* device/test with no signature-level hint) under the ``"ascend"``
dispatch key, beside the in-tree CUDA ``LeakyReluFwdKernel``
(``tileops.kernels.elementwise.activations.LeakyReluFwdKernel``). No existing file is edited to do
this: ``tileops.backend.register_kernel_type`` joins this class to every ``LeakyReluFwdOp``
instance constructed after this module is imported (``dev_process.md`` §1).

Round 2 (``opus48_round2_schedule.md`` §3 VF-A): the large-n persistent path replaces the
compiler-lowered ``T.SimtVF`` inner loop -- whose per-thread remainder loop was emitted as a
serial loop -- with an explicit 64-lane ``T.SimdVF`` chain (storage -> fp32 -> ordered compare
``x > 0`` -> fp32 scalar multiply -> select -> storage, four chains unrolled per step). The
generic single-block/tail fallback keeps the original SimtVF body so arbitrary ``n`` still has
a correct path. The VF body's bits, NaN/Inf classification and +-0 sign bits were checked
against the fp32-promote contract before it was wired in; the round's correctness and formal
M4 evidence live under ``docs/ascend950/dev/resume_20261019/workers/LeakyReluFwdOp/
dsh_round1/``.

Round-2 continuation (``dsh_round2_schedule.md``): the persistent fp32 arithmetic chain gains a
VF-B form that replaces ``vcmps(x>0) + vmuls + vsel`` with one ``T.simd.vlrelu`` (CANN
``asc_leakyrelu``). Because vlrelu branches on ``x >= 0`` while the contract branches on
``x > 0``, and because an Inf alpha changes ``0*Inf``, VF-B is selected only for a positive,
finite, normal-fp32 ``negative_slope``; negative/zero/NaN/Inf/overflowing/subnormal slopes keep
the verified VF-A cmp-select chain (``_vlrelu_slope_eligible``). The slope remains a cache key,
so the choice is per compiled specialization.
``_slope_cache_key`` augments the builder ``lru_cache`` key with the slope's exact IEEE bits so
``+0.0`` and ``-0.0`` never share a compiled kernel (``functools.lru_cache`` would otherwise
treat them as equal and bake the wrong zero sign into the second specialization).
The zero-slope gate also exposed a TIR constant-fold in the **generic/tail** path
(``0.0 * wide`` -> ``+0.0``, losing signed zero and NaN); ``_make_zero_slope_body``
replaces that multiply with an explicit sign-bit form for +-0.0 slopes; the slope's zero sign
is passed as a Python integer (0 / 0x80000000) and is also an explicit builder cache dimension.
The persistent path's vector intrinsics were already IEEE-correct and are unchanged.

Math: mirrors ``LeakyReluFwdKernel._make_op_func`` (``kernels/elementwise/activations.py``)
exactly -- the negative branch promotes to fp32 for the scalar multiply, then casts back (the
CUDA kernel's own comment: "a large slope would overflow before the product does [in a narrow
dtype], and a small one would lose precision"); the positive branch returns the original
storage-dtype value unchanged. See ``_leaky_relu_body_promote`` below (schedule doc §4.2 Variant
A, the generic/tail body) and ``_leaky_relu_body_native`` (§4.2 Variant B, an H3 diagnostic
variant that skips the fp32 round trip -- built and measured but never wired into
``AscendLeakyReluKernel``. The opus48 round measured both this native variant and a branchless
``max/min`` reformulation on device against Variant A at the production tiling; both were slightly
*slower* and the native one lost bit-exactness (max_abs_err 3.05e-5 at slope 0.01) -- evidence in
``docs/ascend950/dev/resume_20261019/workers/LeakyReluFwdOp/round_opus48/probe_round1.json`` and
``probe_body_ganfeat.json``. Those are finite negative results for those two bodies, **not** a
DMA-bound proof: all of them shared the same compiler lowering (the per-thread inner loop was
emitted as a serial loop), so the vector-pipeline cost was never isolated; the round-1 review
(``opus48_review.md`` §6) explicitly rejects the earlier "DMA-bound" conclusion. (The earlier
``round_1_result.md`` the prior docstring cited was never found on disk; those "measured" claims
are superseded by this round's logged measurements, the only A/B numbers to rely on.)

**``negative_slope`` is a real cache-key dimension, not a constant** (schedule doc §2.3): unlike
``ReluFwdOp``, this op has a per-instance scalar baked into the compiled program body at trace
time, and the shared test suite genuinely exercises two distinct values -- ``0.01`` (manifest
default, every M4 case, ``ActivationFixture``'s ``test_leaky_relu``/
``test_leaky_relu_under_tuning``) and ``0.2`` (``tests/ops/test_elementwise_compile.py``'s
parametric ``torch.compile(fullgraph=True)`` smoke case, at ``n=2**20``). Every
``functools.lru_cache``'d builder below takes ``negative_slope`` as an explicit, hashable
parameter so the two values never silently collide on the same compiled artifact (the shared
test suite's own shapes never collide by accident -- ``2**20`` for ``0.2`` vs. ``2**23``/``2**24``
for ``0.01`` -- ``negative_slope`` is carried as an explicit cache key so the two never collide).

Tiling design (schedule resume §4.1, opus48-measured): persistent grid, ``num_blocks=64``,
dtype-aware ``tile`` (16384 for 16-bit, 8192 for fp32), ``num_stages=3``, ``threads=min(2048,
tile)``. These are from this round's own on-device sweep under the official 1 GiB-flush protocol,
not a blind copy of ``ReluFwdOp`` -- see the ``_TILE_16BIT`` / ``_NUM_STAGES`` comment block for
the per-config ratios and the UB ceiling that bounds tile at 16384 (two UB buffers x tile x 2 B x
3 stages = 192 KiB; tile 32768 / stages 4 OOM at 256 KiB, EE1017). ``num_blocks`` past 64
regressed here too, matching ``ReluFwdOp``'s earlier finding. The op moves 2 bytes per element in
and out and does one trivial compare/select, so bandwidth plausibly dominates, but that is a
hypothesis rather than an established bound (offline-bandwidth analysis is not a substitute for
the vector-pipe isolation this file's round-2 VF body now provides). The ``manifest_reference``
baseline (torch_npu's native kernel) is ~1.5x our achieved bandwidth; see
``docs/ascend950/dev/LeakyReluFwdOp/opus48_result.md`` for the remaining gap analysis.

**Generic fallback path is a round-1 requirement, not deferred** (schedule doc §3's boxed
warning, §4.4): ``tests/ops/test_activation.py::test_leaky_relu_under_tuning`` hardcodes
``n_total=4096*7+13=28685`` (``28685 = 5 * 5737``, no factor of 2 at all) under the ``smoke``
mark, which this op's own authoritative M3 gate collects -- unlike ``ReluFwdOp`` round 1, which
could legitimately defer tail-handling as unexercised, this op cannot. ``_make_leaky_relu_kernel``
below routes any ``n_total`` that does not divide ``num_blocks * tile`` exactly to
``_make_leaky_relu_generic_single_block`` (single block, small-tile main loop + one tail pass),
analogous to ``relu.py``'s ``_make_relu_generic_single_block`` but actually exercised this round.

This file does not get a subpackage ``__init__.py`` with ``__all__`` re-exports, which
``TileOPs/.claude/rules/code-style.md`` otherwise requires for every ``src/tileops/kernels/*``
subpackage: per ``dev_process.md`` §1, every op's worker self-registers from its own file and does
*not* build a shared aggregator under ``kernels/ascend/``, since several ops' workers write into
this same directory concurrently and a shared ``__init__.py`` would be a file they all contend
for. ``kernels/ascend`` and ``kernels/ascend/elementwise`` are implicit (PEP 420) namespace
packages.

``inplace`` needs no handling here: ``LeakyReluFwdOp`` inherits ``_UnaryActivationMixin``
(``ops/elementwise/_base.py``), which copies this kernel's freshly-written output back into the
input tensor at the op layer when ``inplace=True`` -- the kernel itself always returns a new
buffer, same as every other activation kernel in this family.

Importing this module when ``tilelang.ascend`` is unavailable (no Ascend toolchain, a CUDA-only
process) is a no-op: the Ascend-specific imports, the kernel class, and the registration call are
wrapped in one ``try/except ImportError`` block, per ``dev_process.md`` §1, mirroring
``relu.py``'s file layout exactly (schedule doc §2.1 asks for this explicitly).
"""

import functools
import math
import struct

import torch

try:
    import tilelang
    import tilelang.ascend.language as T
    from tilelang.carver.arch.driver import get_num_vector_cores

    import tileops.backend
    from tileops.kernels.elementwise.call_spec import LeakyReluCall, LeakyReluFwdInterface
    from tileops.kernels.kernel_base import Entry, Kernel

    def _floor_pow2(n: int) -> int:
        """Largest power of two ``<= n`` (``n >= 1``)."""
        p = 1
        while p * 2 <= n:
            p *= 2
        return p

    def _vlrelu_slope_eligible(slope) -> bool:
        """True only for a positive, finite, normal-fp32 scalar slope.

        ``T.simd.vlrelu`` implements ``dst = src >= 0 ? src : alpha * src`` while this op's
        contract is ``src > 0``. For a *negative* alpha the two differ on the sign of both
        signed zeros (x=+0: contract gives ``alpha*+0 = -0``, vlrelu returns ``+0``; x=-0 is
        the mirror case), so negative slopes keep the verified cmp-select body. Zero, NaN,
        Inf, and slopes whose fp32 conversion overflows or becomes subnormal also keep it: an
        Inf alpha turns the contract's ``0 * Inf`` into NaN while vlrelu leaves ``+0``; a
        zero/subnormal alpha may be flushed by the vector unit. This is a build-time Python
        predicate, so the body choice is fully determined by the already-cached
        ``negative_slope`` key.
        """
        try:
            s = float(slope)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(s) or s <= 0.0:
            return False
        try:
            f32 = struct.unpack("<f", struct.pack("<f", s))[0]
        except OverflowError:
            return False
        return math.isfinite(f32) and f32 >= 2.0 ** -126

    def _slope_cache_key(negative_slope) -> bytes:
        """Exact-bit cache key for ``negative_slope``.

        ``functools.lru_cache`` keys float arguments by ``==``/``hash``, where ``0.0 == -0.0``;
        without this the second of two zero-slope specializations would silently reuse the
        first compiled kernel with the wrong zero sign baked in. ``struct.pack('<d', ...)``
        keeps every float distinct by exact bits and never overflows (unlike the '<f' pack).
        """
        try:
            return struct.pack("<d", float(negative_slope))
        except (TypeError, ValueError, struct.error):
            return repr(negative_slope).encode()

    @functools.lru_cache(maxsize=1)
    def _calibrated_num_blocks(fallback: int = 64) -> int:
        """Persistent-grid block count for the large, bandwidth-bound path.

        Live-queries this SKU's real vector-core count (``get_num_vector_cores``, device 0) and
        floors it to the nearest power of two, exactly mirroring ``relu.py``'s
        ``_calibrated_num_blocks`` (schedule doc §4.1, J3/J4 discipline: query live rather than
        hardcode, even though the expected answer -- ``vector_core_num=72`` on this SKU,
        ``Ascend950DT_9592``, floored to 64 -- was re-confirmed live this round (the probe logged
        ``get_num_vector_cores(0) = 72``). The opus48 sweep also re-measured ``num_blocks`` 64 vs
        128 directly for this op (at tile 16384, fp16 16M): 64 beat 128 (ratio 0.623 vs 0.583),
        matching ``ReluFwdOp``'s earlier finding, so this round keeps 64.

        Deliberately **not** evaluated at module import time: the query touches the real device,
        so merely importing this module must stay a no-NPU-touch no-op (``dev_process.md`` §4).
        ``lru_cache`` makes it a one-time query per process either way. Falls back to
        ``fallback`` if the live query raises or returns a falsy value at runtime.
        """
        try:
            queried = get_num_vector_cores(device_id=0)
        except Exception:
            queried = None
        if not queried:
            return fallback
        return _floor_pow2(int(queried))

    # Tile width and stage count for the large-n persistent-grid path (schedule resume §4.1).
    # Tile width was the dominant *measured* lever in the round-1 sweep
    # (``docs/ascend950/dev/resume_20261019/workers/LeakyReluFwdOp/round_opus48/probe_round1.json``,
    # official 1 GiB-flush protocol via ``tools.measure_baselines._time_reference``): at gan-feat
    # fp16 (16M) the baseline/kernel ratio moved 0.517 (tile 8192, stages 2) -> 0.623 (tile 16384,
    # stages 2) -> 0.646 (tile 16384, stages 3); num_blocks 64 -> 128 *regressed* (0.623 -> 0.583),
    # and neither a branchless ``max/min`` body nor a native-dtype multiply helped
    # (``probe_body_*.json``). That does **not** establish a DMA bound: all of those bodies shared
    # the same unvectorized per-thread lowering, so the vector-pipe cost was never isolated (round-1
    # review §6). The explicit ``T.SimdVF`` body added below is the first body in this file that
    # removes that confound at the same tiling. 16-bit tile is 16384: with two UB
    # buffers at ``_NUM_STAGES`` this is 2*16384*2*3 = 192 KiB, the largest that fits -- tile 32768
    # or stages 4 raises the explicit UB working set to 256 KiB and the launch fails EE1017 ("sum
    # of dynamic shared memory and kernel shared memory exceeds the maximum limit"), measured this
    # round. fp32 keeps 8192 (its own 2*8192*4*3 = 192 KiB two-buffer working set is already at the
    # same ceiling, and fp32 is not a benchmarked case). num_blocks is resolved lazily by
    # ``_calibrated_num_blocks()`` (see its docstring). Both 16-bit tile 16384 and fp32 tile 8192
    # keep the shared ``n=2**20`` correctness test on this persistent path (block_span divides
    # 2**20), so M3 still exercises the production tiling rather than the fallback.
    _TILE_16BIT = 16384
    _TILE_FP32 = 8192
    _NUM_STAGES = 3

    def _tile_for(dtype: str) -> int:
        """Persistent-path tile width for ``dtype`` (16-bit vs fp32 UB budget; see above)."""
        return _TILE_FP32 if dtype in ("float32", "float") else _TILE_16BIT
    # Small-tile width for the generic (non-exact-division) fallback path (schedule doc §4.4).
    # Not performance-tuned: no manifest case reaches this path, only the shared test suite's
    # n_total=28685 smoke shape, which only checks correctness, not latency.
    _TILE_SMALL_FALLBACK = 2048

    def _dtype_str(dtype: torch.dtype) -> str:
        return str(dtype).split(".")[-1]

    @T.macro
    def _leaky_relu_body_promote(x_ub, y_ub, n: int, threads: int, negative_slope: float) -> None:
        """Variant A (default/production, schedule doc §4.2): promote to fp32 for the multiply.

        ``y_ub[:n] = leaky_relu(x_ub[:n], negative_slope)``, compiler-vectorized over ``threads``
        lanes. Mirrors ``LeakyReluFwdKernel._make_op_func`` (``kernels/elementwise/
        activations.py``) exactly: the positive branch returns the raw, un-promoted
        ``x_ub[i]`` (no fp32 round-trip on the common path, nothing to overflow or lose precision
        in a no-op); only the negative branch's multiply runs in fp32 and is cast back to the
        storage dtype at the store (code-style.md's promotion rule for overflow-prone narrow-dtype
        arithmetic). ``negative_slope`` is a plain Python float baked in at trace time -- one
        compiled specialization per distinct value (schedule doc §2.3), not a runtime buffer.

        A ``@T.macro``, not a plain Python helper (code-style.md: "sub-routines use ``@T.macro``,
        never nested ``prim_func``") -- TileLang's eager builder only rewrites ``@T.macro``/
        ``@T.prim_func`` bodies, so an undecorated helper's ``for i in T.Parallel(n):`` fails with
        ``TypeError: 'ForFrame' object is not iterable`` (dev_process.md §7.5).
        """
        with T.SimtVF(threads=threads):
            for i in T.Parallel(n):
                zero_f32 = T.cast(0, "float32")
                slope_f32 = T.cast(negative_slope, "float32")
                wide = T.cast(x_ub[i], "float32")
                neg = T.cast(slope_f32 * wide, y_ub.dtype)
                y_ub[i] = T.if_then_else(wide > zero_f32, x_ub[i], neg)

    def _make_zero_slope_body(zero_sign_bits: int):
        """Build the IEEE ``0*x`` generic/tail body with the slope sign as an integer.

        ``zero_sign_bits`` is 0 for +0.0 and 0x80000000 for -0.0, decided at Python build time
        with ``math.copysign`` (never by reinterpreting a float zero in TIR). The macro closes
        over that integer and XORs it into the operand sign mask, so the two specializations
        differ by an integer constant, not by a float zero bit pattern.
        """

        @T.macro
        def body(x_ub, y_ub, n: int, threads: int, negative_slope: float) -> None:
            """IEEE ``0*x`` fallback for the generic/tail path when ``negative_slope`` is +-0.0.

            The TIR simplifier constant-folds ``0.0 * wide`` to ``+0.0``, which loses both the
            signed zero of ``0 * -x`` and the NaN produced by ``0 * NaN`` / ``0 * +-Inf``. Only
            the generic single-block/tail path uses a plain TIR multiply (the persistent path
            uses the ``vmuls`` intrinsic, which preserves IEEE semantics; the zero-slope gate
            confirmed that), so this body computes the same product from the operand sign bits:
            ``sign(0*x) = sign(x) XOR zero_sign_bits`` for finite ``x``, NaN for NaN/+-Inf.
            """
            with T.SimtVF(threads=threads):
                for i in T.Parallel(n):
                    wide = T.cast(x_ub[i], "float32")
                    sign = ((T.reinterpret(wide, "uint32") & T.uint32(0x80000000))
                            ^ T.uint32(zero_sign_bits))
                    signed_zero = T.reinterpret(sign, "float32")
                    neg = T.if_then_else(
                        T.isnan(wide),
                        wide,
                        T.if_then_else(
                            T.isinf(wide), T.cast(float("nan"), "float32"), signed_zero
                        ),
                    )
                    y_ub[i] = T.if_then_else(
                        wide > T.cast(0, "float32"), x_ub[i], T.cast(neg, y_ub.dtype)
                    )

        return body

    @T.macro
    def _leaky_relu_body_native(x_ub, y_ub, n: int, threads: int, negative_slope: float) -> None:
        """Variant B (H3 diagnostic, schedule doc §4.2): native-dtype multiply, no fp32 round
        trip.

        Built and measured for the H3 A/B question ("does skipping the fp32 promote
        measurably help, since this op is memory-bound and has no transcendental intrinsic to
        compete against") but **not** wired into ``AscendLeakyReluKernel``. The opus48 round
        measured it on device (``round_opus48/probe_round1.json``): at tile 8192 fp16 16M it was
        *slower* than Variant A (29.31 vs 28.69 us) and lost bit-exactness (max_abs_err 3.05e-5 at
        slope 0.01), i.e. a clean negative result -- the H3 prediction ("expect little to no
        difference on a DMA-bound op") held. Kept here, not deleted, as a documented artifact (same
        spirit as ``TanhFwdOp``'s retained, docstring-flagged negative native-vs-promote A/B).
        """
        with T.SimtVF(threads=threads):
            for i in T.Parallel(n):
                zero_native = T.cast(0, x_ub.dtype)
                slope_native = T.cast(negative_slope, x_ub.dtype)
                y_ub[i] = T.if_then_else(
                    x_ub[i] > zero_native, x_ub[i], slope_native * x_ub[i]
                )

    # Explicit-T.SimdVF production body (round-2 schedule §3 VF-A; round-1 review §7 item 2).
    # One 64-lane fp32 vector chain per 64 storage elements, four independent chains per serial
    # step (``T.unroll(4)``) so the compiler can overlap the vld/vcvt/compare/multiply/select
    # latency instead of issuing one register chain at a time. The vector register is 2048 bit:
    # 64 fp32 lanes. For fp16/bf16 the storage->fp32 step is ``vld(..., UNPK_B16)`` (64 elements,
    # one per 32-bit slot) followed by ``vcvt(part=0)``; the store is ``vcvt`` back to the storage
    # dtype and ``vsts(..., PK_B32)`` (64 elements). This is the installed cast idiom
    # (``tilelang/testing/ascend/language/test_tilelang_ascend_simdvf_cast.py``) and was validated
    # bit-for-bit against the fp32-promote contract for all three dtypes, six slopes, both vcvt
    # sat modes and ramp/special patterns by the round's device correctness probe before being
    # wired in here.
    #
    # Semantics kept identical to ``_leaky_relu_body_promote``: ordered ``x > 0`` (NaN takes the
    # negative branch exactly like the SimtVF body), ``negative_slope`` is converted to a float32
    # scalar and never narrowed, and the positive branch is the original storage value.
    # ``threads`` is accepted for call-site compatibility with the SimtVF bodies and is unused:
    # the explicit chain is always 64 lanes wide. The
    # generic single-block fallback below deliberately keeps the SimtVF body: it must run for
    # arbitrary ``n`` (no 64-element chunk guarantee), and it is correctness-only, never graded.
    @T.macro
    def _leaky_relu_body_vf16(x_ub, y_ub, n: int, threads: int, negative_slope: float) -> None:
        """Explicit VF-A body for 16-bit storage dtypes (fp16/bf16), fp32 arithmetic."""
        with T.SimdVF():
            for g in T.serial((n // 64) // 4):
                for s in T.unroll(4, explicit=True):
                    base = (g * 4 + s) * 64
                    x16 = T.simd.vld(x_ub[base], dist="UNPK_B16")
                    xf = T.simd.vcvt(x16, "float32", part=0)
                    m = T.simd.vcmps(xf, T.cast(0, "float32"), op="gt")
                    neg = T.simd.vmuls(xf, T.cast(negative_slope, "float32"))
                    yf = T.simd.vsel(xf, neg, m)
                    y16 = T.simd.vcvt(yf, y_ub.dtype, sat=True)
                    T.simd.vsts(y_ub[base], y16, dist="PK_B32")

    @T.macro
    def _leaky_relu_body_vf_f32(x_ub, y_ub, n: int, threads: int, negative_slope: float) -> None:
        """Explicit VF-A body for float32 storage (no conversion round trip)."""
        with T.SimdVF():
            for g in T.serial((n // 64) // 4):
                for s in T.unroll(4, explicit=True):
                    base = (g * 4 + s) * 64
                    xf = T.simd.vld(x_ub[base])
                    m = T.simd.vcmps(xf, T.cast(0, "float32"), op="gt")
                    neg = T.simd.vmuls(xf, T.cast(negative_slope, "float32"))
                    yf = T.simd.vsel(xf, neg, m)
                    T.simd.vsts(y_ub[base], yf)

    # VF-B: one fp32 ``vlrelu`` replaces the compare+multiply+select chain. Selected only when
    # ``_vlrelu_slope_eligible`` proves the slope's signed-zero/Inf contract is preserved.
    @T.macro
    def _leaky_relu_body_vlrelu16(x_ub, y_ub, n: int, threads: int, negative_slope: float) -> None:
        """VF-B for 16-bit storage: storage -> fp32 -> vlrelu(fp32 alpha) -> storage."""
        with T.SimdVF():
            for g in T.serial((n // 64) // 4):
                for s in T.unroll(4, explicit=True):
                    base = (g * 4 + s) * 64
                    x16 = T.simd.vld(x_ub[base], dist="UNPK_B16")
                    xf = T.simd.vcvt(x16, "float32", part=0)
                    yf = T.simd.vlrelu(xf, T.cast(negative_slope, "float32"))
                    y16 = T.simd.vcvt(yf, y_ub.dtype, sat=True)
                    T.simd.vsts(y_ub[base], y16, dist="PK_B32")

    @T.macro
    def _leaky_relu_body_vlrelu_f32(x_ub, y_ub, n: int, threads: int, negative_slope: float) -> None:
        """VF-B for float32 storage: vld -> vlrelu(fp32 alpha) -> vsts."""
        with T.SimdVF():
            for g in T.serial((n // 64) // 4):
                for s in T.unroll(4, explicit=True):
                    base = (g * 4 + s) * 64
                    xf = T.simd.vld(x_ub[base])
                    yf = T.simd.vlrelu(xf, T.cast(negative_slope, "float32"))
                    T.simd.vsts(y_ub[base], yf)

    def _body_for(variant: str):
        if variant == "promote":
            return _leaky_relu_body_promote
        if variant == "native":
            return _leaky_relu_body_native
        raise ValueError(f"unknown leaky_relu body variant {variant!r}")

    def _generic_body_for(variant: str, negative_slope: float, zero_sign_bits: int):
        """Generic/tail body: the IEEE zero-slope form for a +-0.0 slope, else the usual body.

        The persistent path uses VF-A/VF-B vector intrinsics and is unaffected by the TIR
        ``0.0 * x`` fold; only this generic path needs the explicit signed-zero/NaN form. The
        slope sign enters as a Python integer (0 / 0x80000000), never as a float-zero bitcast.
        """
        if variant == "promote" and float(negative_slope) == 0.0:
            return _make_zero_slope_body(zero_sign_bits)
        return _body_for(variant)

    def _persistent_body_for(dtype: str, variant: str, negative_slope: float):
        """Persistent-path body selection (see ``_vlrelu_slope_eligible``).

        Eligible positive normal-fp32 slopes use the VF-B ``vlrelu`` chain; every other slope
        keeps the fully verified VF-A cmp-select chain. The generic fallback builder below
        does not use this function and keeps its SimtVF body.
        """
        if variant != "promote":
            return _body_for(variant)
        if _vlrelu_slope_eligible(negative_slope):
            return _leaky_relu_body_vlrelu_f32 if dtype == "float32" else _leaky_relu_body_vlrelu16
        return _leaky_relu_body_vf_f32 if dtype == "float32" else _leaky_relu_body_vf16

    @functools.lru_cache(maxsize=64)
    def _make_leaky_relu_persistent(
        n_total: int,
        dtype: str,
        num_blocks: int,
        tile: int,
        num_stages: int,
        negative_slope: float,
        slope_key: bytes,
        zero_sign_bits: int,
        variant: str = "promote",
    ):
        """Persistent multi-block grid with ``T.Pipelined`` staging (schedule doc §4.1).

        Requires ``n_total`` to divide ``num_blocks * tile`` exactly; the caller
        (``_make_leaky_relu_kernel``) only reaches this builder for a shape that does, and routes
        every other shape to ``_make_leaky_relu_generic_single_block`` instead. Both manifest
        workloads (``n=2**24``, ``n=2**23``) and the shared 16-bit ``test_leaky_relu`` /
        ``test_parametric_unary_compile`` shape (``n=2**20``) satisfy this at ``num_blocks=64,
        tile=16384`` (``block_span=2**20``; 2**20 is one iteration, so M3 still covers this path);
        the shared fp32 ``test_leaky_relu`` shape satisfies it at ``tile=8192`` (``block_span=
        2**19``).

        ``negative_slope`` (and ``variant``) are explicit cache-key parameters (schedule doc
        §2.3): copying ``relu.py``'s cache-key shape verbatim -- ``(n_total, dtype, num_blocks,
        tile, num_stages)`` without the scalar -- would let a process that first builds a
        ``negative_slope=0.01`` kernel for some ``(n_total, dtype)`` later silently serve that
        same compiled binary to a ``negative_slope=0.2`` instance at the same shape (wrong slope
        baked in, no crash, no warning).
        """
        block_span = num_blocks * tile
        assert n_total % block_span == 0, (
            f"_make_leaky_relu_persistent requires n_total ({n_total}) to divide "
            f"num_blocks*tile ({block_span}) exactly; the caller routes other shapes elsewhere"
        )
        iterations = n_total // block_span
        # Production ``promote`` on this path is the explicit VF-A body; the retained SimtVF
        # bodies remain reachable only through the diagnostic ``native`` variant (and the generic
        # fallback below keeps SimtVF unconditionally).
        assert tile % 256 == 0, f"VF-A body needs tile divisible by 256, got {tile}"
        body = _persistent_body_for(dtype, variant, negative_slope)

        @T.prim_func
        def main(X: T.Tensor((n_total,), dtype), Y: T.Tensor((n_total,), dtype)):
            with T.Kernel(num_blocks) as bx:
                x_ub = T.alloc_shared((tile,), dtype)
                y_ub = T.alloc_shared((tile,), dtype)
                T.annotate_buffer_versions({x_ub: num_stages, y_ub: num_stages})
                for step in T.Pipelined(iterations, num_stages=num_stages):
                    begin = (step * num_blocks + bx) * tile
                    T.copy(X[begin : begin + tile], x_ub)
                    body(x_ub, y_ub, tile, threads=min(2048, tile), negative_slope=negative_slope)
                    T.copy(y_ub, Y[begin : begin + tile], l2_cache_ctrl=0)

        return tilelang.compile(main, target="ascend", out_idx=-1)

    @functools.lru_cache(maxsize=64)
    def _make_leaky_relu_generic_single_block(
        n_total: int,
        dtype: str,
        tile_small: int,
        negative_slope: float,
        slope_key: bytes,
        zero_sign_bits: int,
        variant: str = "promote",
    ):
        """Correctness-first fallback for an ``n_total`` the persistent tiling does not evenly
        divide (schedule doc §4.4, a round-1 requirement for this op, not deferred).

        One block; a main loop over full ``tile_small``-sized chunks (``T.Pipelined(num_stages=
        2)``), plus one smaller, statically-sized tail chunk, no pipelining. Directly analogous
        to ``relu.py``'s ``_make_relu_generic_single_block``, except this path is actually
        exercised this round by ``tests/ops/test_activation.py::test_leaky_relu_under_tuning``'s
        ``n_total=28685`` shape (``full_tiles=14``, ``remainder=13`` at ``tile_small=2048``). Not
        performance-tuned (single block, no cross-block parallelism): no manifest case reaches
        this function.

        ``n_total`` is already a Python int at kernel-build time (one compiled program per
        distinct ``(n_total, dtype, negative_slope, variant)`` via ``entry_for``), so both the
        main-loop trip count and the tail chunk's width are ordinary Python ints baked into the
        generated program -- no dynamic-extent DMA is needed.
        """
        full_tiles, remainder = divmod(n_total, tile_small)
        body = _generic_body_for(variant, negative_slope, zero_sign_bits)

        @T.prim_func
        def main(X: T.Tensor((n_total,), dtype), Y: T.Tensor((n_total,), dtype)):
            with T.Kernel(1):
                if full_tiles:
                    x_ub = T.alloc_shared((tile_small,), dtype)
                    y_ub = T.alloc_shared((tile_small,), dtype)
                    T.annotate_buffer_versions({x_ub: 2, y_ub: 2})
                    for step in T.Pipelined(full_tiles, num_stages=2):
                        begin = step * tile_small
                        T.copy(X[begin : begin + tile_small], x_ub)
                        body(
                            x_ub,
                            y_ub,
                            tile_small,
                            threads=min(2048, tile_small),
                            negative_slope=negative_slope,
                        )
                        T.copy(y_ub, Y[begin : begin + tile_small])
                if remainder:
                    xt_ub = T.alloc_shared((remainder,), dtype)
                    yt_ub = T.alloc_shared((remainder,), dtype)
                    begin = full_tiles * tile_small
                    T.copy(X[begin : begin + remainder], xt_ub)
                    body(
                        xt_ub,
                        yt_ub,
                        remainder,
                        threads=min(128, remainder),
                        negative_slope=negative_slope,
                    )
                    T.copy(yt_ub, Y[begin : begin + remainder])

        return tilelang.compile(main, target="ascend", out_idx=-1)

    def _make_leaky_relu_kernel(
        n_total: int, dtype: str, negative_slope: float, variant: str = "promote"
    ):
        """Pick the tiling for ``n_total`` elements: persistent grid, or generic fallback."""
        num_blocks = _calibrated_num_blocks()
        tile = _tile_for(dtype)
        block_span = num_blocks * tile
        slope_key = _slope_cache_key(negative_slope)
        zero_sign_bits = (0x80000000 if math.copysign(1.0, float(negative_slope)) < 0.0 else 0)
        if n_total % block_span == 0:
            return _make_leaky_relu_persistent(
                n_total, dtype, num_blocks, tile, _NUM_STAGES, negative_slope, slope_key,
                zero_sign_bits, variant
            )
        return _make_leaky_relu_generic_single_block(
            n_total, dtype, _TILE_SMALL_FALLBACK, negative_slope, slope_key, zero_sign_bits,
            variant
        )

    class AscendLeakyReluKernel(Kernel, LeakyReluFwdInterface):
        """Leaky ReLU on Ascend NPUs: ``y = x if x > 0 else negative_slope * x``.

        One compare, one trivial-cost multiply, one load, one store per element; whether the
        vector pipe or the DMA staging dominates was not established by the round-1 SimtVF
        bodies (they shared one lowering), which is why the persistent path now runs the
        explicit VF-A body above. See ``docs/ascend950/dev/LeakyReluFwdOp/round_1_schedule.md``
        §4 for the tiling rationale
        and §2.3 for why ``negative_slope`` must be (and is) part of every compile-cache key
        below. A fresh class per ``docs/ascend950/dev_process.md`` §1 -- it does not subclass the
        CUDA-locked ``ScalarParamUnaryKernel``/``UnaryKernel`` family, which imports
        ``tilelang.language`` (the CUDA dialect facade).

        Subclasses ``LeakyReluFwdInterface`` (``tileops.kernels.elementwise.call_spec``), **not**
        the generic ``UnaryElementwiseFwdInterface`` -- ``LeakyReluFwdOp`` declares the former as
        its own ``interfaces`` mapping (schedule doc §2.1); subclassing the generic interface
        here would make every ``LeakyReluFwdOp()`` construction raise ``"keys implement none of
        its kernel interfaces"`` on every device, with no signature-level hint.
        """

        devices = frozenset({"npu"})

        @classmethod
        def entry_for(cls, call: LeakyReluCall) -> Entry:
            return call, lambda: cls(call.n_total, call.dtype, call.negative_slope)

        def __init__(self, n_total: int, dtype: torch.dtype, negative_slope: float = 0.01):
            super().__init__()
            self.n_total = n_total
            self.dtype = dtype
            self.negative_slope = negative_slope
            self._compiled_fn = _make_leaky_relu_kernel(
                n_total, _dtype_str(dtype), negative_slope, variant="promote"
            )

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            flat = input.reshape(-1)
            result = self._compiled_fn(flat)
            return result.reshape(input.shape)

    tileops.backend.register_kernel_type("LeakyReluFwdOp", "ascend", AscendLeakyReluKernel)

except ImportError:
    pass
