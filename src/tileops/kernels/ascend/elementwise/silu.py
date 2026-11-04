"""Ascend NPU SiLU forward kernel: ``devices={"npu"}``, outside the CUDA kernel hierarchy.

Round 1 of ``docs/ascend950/dev_process.md``'s Ascend development loop for ``SiluFwdOp``; see
``docs/ascend950/dev/SiluFwdOp/round_1_schedule.md`` for the tiling/schedule design this file
implements. Registers itself under the ``"ascend"`` dispatch key
(``tileops.backend.register_kernel_type``), so importing this module is enough to make
``SiluFwdOp`` route NPU tensors here -- no change to the existing CUDA ``SiluFwdKernel`` or
``SiluFwdOp``.

Round 2 (``docs/ascend950/dev/SiluFwdOp/round_2_schedule.md``): round 1's decode case
(14,336 elements) shipped on ``_build_single_block``'s single-core ``T.Kernel(1)`` path, the
same structural bug ``GeluFwdOp`` round 1 shipped and ``GeluFwdOp`` round 2 fixed by spreading
the identical element count across a 64-core grid. ``_build_decode_multicore`` below ports
that proven fix (``gelu.py``'s ``_build_decode_multicore``), with ``_silu_op`` substituted for
``gelu(...)`` and the core count derived from this file's own ``_choose_num_blocks`` helper
rather than hardcoded.

Does *not* inherit ``FloatUnaryKernel``/``UnaryKernel``
(``tileops/kernels/elementwise/_base.py``): both hardcode CUDA-only behavior
(``devices={"cuda"}``, ``supported_archs=[80, 86, 89, 90]`` read through
``tileops.utils.get_sm_version``, and ``_require_cuda``'s ``tensor.is_cuda`` check) that would
reject every NPU tensor outright.
"""

import functools
from typing import ClassVar, Optional

import torch

from tileops.kernels.elementwise.call_spec import ElementwiseCall, UnaryElementwiseFwdInterface
from tileops.kernels.kernel_base import Entry, Kernel

try:
    import tilelang
    import tilelang.ascend.language as T

    _ASCEND_IMPORT_ERROR: Optional[ImportError] = None
except ImportError as exc:  # pragma: no cover - exercised only off-Ascend
    tilelang = None  # type: ignore[assignment]
    T = None  # type: ignore[assignment]
    _ASCEND_IMPORT_ERROR = exc

__all__ = ["AscendSiluKernel"]

# SIMT UB budget per vector core, in bytes, borrowed from
# ascend_baselines/TileKernels/tile_kernels/config.py:get_max_ub_per_vector_core -- flagged
# unverified on 950 in round_1_schedule.md SS3.1/SS5.5. M3 on real hardware confirmed the
# budget must scale with itemsize: a single hardcoded 16384-element tile (sized for fp16/
# bf16's 2-byte elements) overflowed the device's dynamic-shared-memory limit for float32's
# 4-byte elements ("Checking whether the size of the dynamic shared memory exceeds the limit
# failed", dyn_ubuf_bytes=262144 for 2 buffers x 2 stages x 4 bytes x 16384 elems = 256 KiB).
# See ``_tile_elems_for_dtype``.
_UB_BUDGET_BYTES = 216 * 1024
# 3, not the textbook 2, after an on-device sweep (round 1 M4 iteration) of the prefill case
# across {num_blocks, num_stages}: 3 stages measurably beat 2 at the same tile/block geometry
# (e.g. tile=16384/blocks=64: ~36.5us vs ~41.2us by `do_bench`; both are informal sweep
# numbers, not the authoritative torch_npu.profiler.profile figures in the M4 report, but the
# direction was consistent), and still fits the UB budget at this tile size (2 buffers x 3
# stages x 2 bytes x 16384 = 192 KiB, vs 262144 B = 256 KiB which overflows at 4 stages).
_NUM_STAGES = 3
# Conservative fallback vector-core count when the device property this kernel prefers
# (``cube_core_num``) is unavailable. Only changes how many persistent blocks the multi-tile
# path launches with -- a single block (``_choose_num_blocks`` always finds one) is still a
# correct, just more serial, schedule.
_FALLBACK_VECTOR_CORES = 48
# Thread count the SimtVF region launches with. 2048 is not just the value
# examples/ascend/example_vecadd.py measures with -- M4 tuning on real hardware (round 1)
# confirmed it is a hard Ascend SIMT ceiling: __launch_bounds__ above 2048 fails to compile
# ("Maximum number of SIMT threads returned by LAUNCH_BOUND must be an integer constant in
# range (0, 2048]"), and 512/1024/2048 measured flat (no headroom to trade away by going
# lower). Capped to the region's own element count for the decode/tail paths, which can be
# smaller than 2048.
_MAX_THREADS = 2048
# Round 2 P3 (round_2_schedule.md SS2.4) tried, measured, and discarded a hand-written
# ``T.SimdVF`` compute body for the prefill tiled path (ported from the shipped
# ``tile_kernels.quant.swiglu_forward_asc`` kernel's ``compute_swiglu`` macro -- the one place
# in this environment a working fp16/bf16 SimdVF SiLU computation is known to exist --
# dropping its SwiGLU gate multiply). It compiled and passed M3 (max_abs_err=0) on the first
# try, but measured 97.1/97.2us on the authoritative profiler vs. the shipped ``T.SimtVF``
# path's 31.1/31.1us -- a >3x regression, not an improvement. Root cause (not re-measured in
# isolation, but consistent with the numbers): each 64-element chunk serializes a
# vld/vcvt/vexpdif/vadds/vdiv/vcvt/vsts sequence through ``T.serial`` one register at a time
# (256 chunks per 16384-element tile), paying SFU latency 256 times per tile with no
# hardware-scheduled overlap across iterations; ``T.SimtVF``'s compiler-driven 2048-thread
# vectorization already overlaps that latency automatically for this bandwidth-bound,
# compute-light op. Removed after measurement rather than left dead in this file; see
# round_2_result.md SS2 for the full P1/P2/P3 comparison.


def _silu_op(x):
    """``x / (1 + exp(-x))``, computed in fp32; mirrors ``SiluFwdKernel.op_func``'s algebra.

    ``SiluFwdKernel.op_func`` (``tileops/kernels/elementwise/activations.py``) uses
    ``exp2(-x * log2e)`` for the CUDA MUFU.EX2 instruction; round_1_schedule.md SS0/SS3.3 calls
    for the direct ``exp`` form on Ascend instead, since ``exp``/``exp2`` are both registered
    Ascend hardware intrinsics (no CUDA-specific speed reason to keep the exp2 rewrite), and
    ``SiluFwdKernel`` itself does not reach for ``SigmoidFwdKernel``'s CUDA-only
    ``__fdividef``-based refinements at this op's tolerance (fp16 1e-3 / bf16 1.6e-2).
    """
    one = T.cast(1.0, "float32")
    wide = T.cast(x, "float32")
    return wide / (one + T.exp(-wide))


def _num_vector_cores() -> int:
    """Best-effort AIV (vector) core count of the current NPU device.

    Only changes how many persistent blocks the multi-tile path launches with; a wrong or
    unavailable count degrades the launch geometry, never the result (``_choose_num_blocks``
    always returns a divisor, so every block still runs the same exact iteration count).
    """
    try:
        prop = torch.npu.get_device_properties(torch.npu.current_device())
        cube_cores = getattr(prop, "cube_core_num", None)
        if cube_cores:
            return int(cube_cores) * 2
    except Exception:
        pass
    return _FALLBACK_VECTOR_CORES


def _tile_elems_for_dtype(dtype_str: str) -> int:
    """The largest power-of-two element count keeping the multi-tile path's UB use in budget.

    2 buffers (x_tile, y_tile) x ``_NUM_STAGES`` x ``itemsize`` x ``tile_elems`` must fit
    ``_UB_BUDGET_BYTES``; a power of two keeps the resulting iteration/block-count arithmetic
    in ``_choose_num_blocks`` simple and gives every known benchmark ``n_total`` an exact or
    near-exact tiling. For fp16/bf16 (itemsize 2) this returns 16384 (128 KiB used, ~40%
    headroom); for float32 (itemsize 4) it returns 8192 (also 128 KiB used).
    """
    itemsize = getattr(torch, dtype_str).itemsize
    max_elems = _UB_BUDGET_BYTES // (2 * _NUM_STAGES * itemsize)
    tile = 1
    while tile * 2 <= max_elems:
        tile *= 2
    return tile


def _choose_num_blocks(num_full_tiles: int, vector_cores: int) -> int:
    """The largest divisor of ``num_full_tiles`` that does not exceed ``vector_cores``.

    Guarantees ``num_full_tiles % result == 0``, so every persistent block runs the exact same
    number of iterations -- no per-block tail inside the main loop, regardless of whether
    ``num_full_tiles`` happens to be round (round_1_schedule.md SS3.5: pick a block count the
    iteration count divides evenly, not the reverse). 1 always qualifies, so this never fails
    to return a valid schedule, even for a prime ``num_full_tiles`` larger than every core count.
    """
    cap = max(1, min(num_full_tiles, vector_cores))
    for candidate in range(cap, 0, -1):
        if num_full_tiles % candidate == 0:
            return candidate
    return 1  # unreachable: candidate == 1 always divides num_full_tiles


if T is not None:

    @functools.lru_cache(maxsize=None)
    def _build_single_block(extent: int, dtype_str: str):
        """One block, one untiled GM->UB->compute->UB->GM pass over ``extent`` elements.

        Used both for the decode-sized case (round_1_schedule.md SS3.1/SS3.4: too small to
        pipeline) and, reused unchanged, as the tail epilogue of the tiled path below when
        ``n_total`` is not an exact multiple of the dtype's tile size.
        """

        @T.prim_func
        def main(X: T.Tensor((extent,), dtype_str), Y: T.Tensor((extent,), dtype_str)):
            with T.Kernel(1):
                x_tile = T.alloc_shared((extent,), dtype_str)
                y_tile = T.alloc_shared((extent,), dtype_str)
                T.copy(X, x_tile)
                with T.SimtVF(threads=min(extent, _MAX_THREADS)):
                    for i in T.Parallel(extent):
                        y_tile[i] = _silu_op(x_tile[i])
                T.copy(y_tile, Y)

        return tilelang.compile(main, target="ascend", out_idx=-1)

    @functools.lru_cache(maxsize=None)
    def _build_decode_multicore(extent: int, dtype_str: str, num_cores: int):
        """Grid-of-``num_cores`` SiLU kernel for a small (decode-shaped) ``extent`` that
        divides evenly by ``num_cores``.

        Use whole 64-lane vectors per block and the native SIMD body already used
        by prefill. The caller targets about 2,048 elements per active core;
        14,336 elements use seven blocks, reducing small-workload launch overhead.

        Round 2 fix for the decode case (round_2_schedule.md SS1.2), porting
        ``AscendGeluKernel``'s proven ``_build_decode_multicore`` (``gelu.py:157-190``):
        round 1 ran the *entire* decode-sized extent on one of the device's 72 AIV cores via
        ``_build_single_block``, leaving the rest idle, and missed its target by 1.78x.
        ``GeluFwdOp`` round 2 measured the identical fix (same ``n_total=14,336``, same
        ``bfloat16``, a heavier per-lane body) take its decode case from a 2.02x miss to a
        0.818x pass, so this ports the grid shape, not SiLU's math, which is unchanged
        (``_silu_op``). Every core gets exactly one statically-sized
        ``block = extent // num_cores``-element tile -- no re-looping, hence a plain
        ``T.Kernel(num_cores) as core_id:`` body rather than ``T.Persistent`` (same reasoning
        as the Gelu precedent: with exactly one tile per core, ``T.Persistent``'s general
        re-looping machinery degenerates to exactly what this direct form already expresses).
        """
        assert extent % num_cores == 0, (
            f"_build_decode_multicore requires extent={extent} to divide evenly by "
            f"num_cores={num_cores}"
        )
        block = extent // num_cores
        assert block % 64 == 0, "SIMD decode blocks must contain complete vectors"

        @T.prim_func
        def main(X: T.Tensor((extent,), dtype_str), Y: T.Tensor((extent,), dtype_str)):
            with T.Kernel(num_cores) as core_id:
                x_tile = T.alloc_shared((block,), dtype_str)
                y_tile = T.alloc_shared((block,), dtype_str)
                begin = core_id * block
                T.copy(X[begin : begin + block], x_tile)
                _silu_simd(x_tile, y_tile, block, dtype_str)
                T.copy(y_tile, Y[begin : begin + block])

        return tilelang.compile(main, target="ascend", out_idx=-1)

    @T.macro
    def _silu_simd(x, y, width: int, dtype: str):
        with T.SimdVF():
            mask = T.simd.pset(32)
            for i in range(width // 64):
                value = (
                    T.simd.vld(x[i * 64])
                    if dtype == "float32"
                    else T.simd.vcvt(T.simd.vld(x[i * 64], dist="UNPK_B16"), "float32", part=0)
                )
                negative = T.simd.vmuls(value, -1.0, mask)
                denominator = T.simd.vadds(T.simd.vexp(negative, mask), 1.0, mask)
                result = T.simd.vdiv(value, denominator, mask, precision="ftz_true")
                if dtype == "float32":
                    T.simd.vsts(y[i * 64], result, mask)
                else:
                    T.simd.vsts(y[i * 64], T.simd.vcvt(result, dtype), mask, dist="PK_B32")

    @functools.lru_cache(maxsize=None)
    def _build_tiled(main_n: int, dtype_str: str, num_blocks: int, tile_elems: int):
        """Persistent-block, pipelined main loop over exact multiples of ``tile_elems``.

        ``main_n`` is an exact multiple of ``num_blocks * tile_elems`` by construction (the
        caller, ``_build_silu_kernel``, only ever passes the full-tile portion of ``n_total``
        and a ``num_blocks`` that divides its tile count), so no tail handling is needed inside
        this program itself. ``iterations`` is recomputed from ``main_n`` *inside* the traced
        function body rather than passed in as its own parameter: TileLang's eager
        ``get_type_hints`` (``tilelang/language/eager/builder.py``) only captures a closed-over
        name for a nested ``@T.prim_func`` when that name is also read somewhere in the
        function's actual body, not only in a type annotation (a documented "incomplete and
        buggy" limitation) -- a parameter used solely in ``T.Tensor((main_n,), ...)`` is never
        compiled into a real closure cell and resolves to a ``NameError`` at trace time.
        """

        @T.prim_func
        def main(X: T.Tensor((main_n,), dtype_str), Y: T.Tensor((main_n,), dtype_str)):
            with T.Kernel(num_blocks) as bx:
                iterations = main_n // (num_blocks * tile_elems)
                x_tile = T.alloc_shared((tile_elems,), dtype_str)
                y_tile = T.alloc_shared((tile_elems,), dtype_str)
                T.annotate_buffer_versions({x_tile: _NUM_STAGES, y_tile: _NUM_STAGES})
                for step in T.Pipelined(iterations, num_stages=_NUM_STAGES):
                    begin = (step * num_blocks + bx) * tile_elems
                    T.copy(X[begin : begin + tile_elems], x_tile)
                    _silu_simd(x_tile, y_tile, tile_elems, dtype_str)
                    T.copy(y_tile, Y[begin : begin + tile_elems], l2_cache_ctrl=0)

        return tilelang.compile(main, target="ascend", out_idx=-1)

    @T.macro
    def _silu_body(x, y, width: int):
        with T.SimtVF(threads=min(width, 2048)):
            for i in T.Parallel(width):
                y[i] = _silu_op(x[i])

    @functools.lru_cache(maxsize=128)
    def _build_with_tail(n_total: int, dtype_str: str):
        """Bounded UB tail; only valid elements cross the global-memory boundary."""
        tile = _tile_elems_for_dtype(dtype_str)
        stages = _NUM_STAGES
        # Reserve space for SIMT static scratch as well as full and tail buffers.
        itemsize = 4 if dtype_str == "float32" else 2
        while (
            2 * tile * stages + 2 * ((n_total % tile + 63) // 64 * 64)
        ) * itemsize > _UB_BUDGET_BYTES:
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
                            _silu_body(x, y, tile)
                            T.copy(y, Y[begin : begin + tile])
                if tail:  # noqa: SIM102 - preserve validated TileLang DSL structure
                    if bx == 0:
                        xt = T.alloc_shared((padded_tail,), dtype_str)
                        yt = T.alloc_shared((padded_tail,), dtype_str)
                        T.copy(X[full * tile : n_total], xt[:tail])
                        _silu_body(xt, yt, padded_tail)
                        T.copy(yt[:tail], Y[full * tile : n_total])

        return tilelang.compile(main, target="ascend", out_idx=-1)


class _SiluRunner:
    """Dispatches one flat ``(n_total,)`` NPU tensor to one or two compiled Ascend programs.

    Two programs only when ``n_total`` exceeds the tile size and does not divide it evenly:
    the tiled/pipelined program covers every full tile, and a second single-block program (the
    same builder the small/decode case uses) covers the remainder. This is plain Python control
    flow around two already-compiled kernels, not a runtime branch inside either kernel --
    chosen over an in-kernel conditional epilogue (e.g. ``if bx == 0``) because nothing in this
    codebase or the available Ascend examples demonstrates that pattern working under
    AutoSchedule, and a silent miscompile there would be far more costly than one extra Python
    dispatch + ``torch.cat`` for the (here, never exercised by the 3 benchmark cases) uneven-tail
    case. Each half is independently correct and independently tested via M3.
    """

    __slots__ = ("main_kernel", "tail_kernel", "main_n")

    def __init__(self, main_kernel, tail_kernel, main_n: int):
        self.main_kernel = main_kernel
        self.tail_kernel = tail_kernel
        self.main_n = main_n

    def __call__(self, flat_input: torch.Tensor) -> torch.Tensor:
        if self.tail_kernel is None:
            return self.main_kernel(flat_input)
        tail_part = self.tail_kernel(flat_input[self.main_n :])
        if self.main_kernel is None:
            return tail_part
        main_part = self.main_kernel(flat_input[: self.main_n])
        return torch.cat((main_part, tail_part))


@functools.lru_cache(maxsize=None)
def _build_silu_kernel(n_total: int, dtype_str: str) -> _SiluRunner:
    """Build (or reuse, by ``(n_total, dtype_str)``) the Ascend SiLU program.

    ``n_total <= tile_elems``: choose a core count dividing the number of 64-lane
    vectors, targeting about 2,048 elements per core. Each core processes one tile via
    ``_build_decode_multicore`` (round_2_schedule.md SS1.2 -- round 1's single-core
    ``_build_single_block`` path for this branch missed the decode target by 1.78x, the same
    structural bug ``GeluFwdOp`` round 2 fixed on the identical element count). Falls back to
    ``_build_single_block`` when the chosen core count is one. Irregular vector
    tails are handled separately by ``_build_with_tail``.

    ``n_total > tile_elems``: a persistent-block main loop over ``tile_elems``-sized tiles
    (``tile_elems`` from ``_tile_elems_for_dtype``, itemsize-dependent), plus -- when
    ``n_total`` is not an exact multiple of ``tile_elems`` -- a second, differently-sized
    single-block program for the remainder (see ``_SiluRunner``). Every sub-builder is
    memoized with ``functools.lru_cache``, and ``n_total``/``dtype_str`` are Python-level
    facts fixed per compiled kernel instance (one per distinct ``AscendSiluKernel(n_total,
    dtype)``), so this is Python-level specialization, not a runtime mask: every ``n_total``
    gets a *correct* schedule, not one that assumes even division (round_1_schedule.md SS5
    risk 2: a hardcoded "must divide evenly" kernel is not an acceptable general
    ``UnaryElementwiseFwdInterface`` implementation).
    """
    if n_total <= 0:
        raise ValueError(f"n_total must be positive, got {n_total}")

    tile_elems = _tile_elems_for_dtype(dtype_str)
    if n_total % 64 or (n_total > tile_elems and n_total % tile_elems):
        return _build_with_tail(n_total, dtype_str)
    if n_total <= tile_elems:
        # Small tensors benefit from fewer active cores and whole vector tiles.
        active_cores = min(_num_vector_cores(), max(1, (n_total + 2047) // 2048))
        num_cores = _choose_num_blocks(n_total // 64, active_cores)
        if num_cores > 1:
            return _SiluRunner(
                _build_decode_multicore(n_total, dtype_str, num_cores), None, n_total
            )
        return _SiluRunner(_build_single_block(n_total, dtype_str), None, n_total)

    num_full_tiles = n_total // tile_elems
    tail = n_total - num_full_tiles * tile_elems
    main_n = num_full_tiles * tile_elems
    # Round 2 re-validated this call's own answer (64 for both prefill cases) against every
    # other lever the schedule specified, all measured worse or much worse with the
    # authoritative profiler and none shipped (round_2_result.md SS2 has the full comparison):
    # P1 swept num_blocks in {56, 64, 112} (the divisors of 1792 bracketing the real 72-core
    # count) -- 64 won outright (31.1/31.1us vs 56's 35.3/35.4us and 112's 36.6/36.4us), not
    # flat and not trending toward 72. P2 tried a true 72-core two-pass split anyway (two
    # ``_build_tiled`` calls: 72 blocks x 24 iterations + 64 blocks x 1 iteration) -- far worse
    # (71.1/68.9us), the second kernel launch's own overhead swamping the 8-extra-core gain. P3
    # replaced this call's ``T.SimtVF`` compute body with hand-written ``T.SimdVF`` intrinsics
    # (ported from the shipped ``tile_kernels.quant.swiglu_forward_asc`` kernel) -- also far
    # worse (97.1/97.2us), the per-64-element-chunk SFU sequence paying latency serially that
    # ``T.SimtVF``'s compiler-driven vectorization already overlaps across threads. This single
    # call is therefore still the best known configuration for this op.
    num_blocks = _choose_num_blocks(num_full_tiles, _num_vector_cores())
    main_kernel = _build_tiled(main_n, dtype_str, num_blocks, tile_elems)
    tail_kernel = _build_single_block(tail, dtype_str) if tail else None
    return _SiluRunner(main_kernel, tail_kernel, main_n)


class AscendSiluKernel(Kernel, UnaryElementwiseFwdInterface):
    """Ascend NPU SiLU: ``x / (1 + exp(-x))`` in fp32, mirroring ``SiluFwdKernel.op_func``.

    See ``docs/ascend950/dev/SiluFwdOp/round_1_schedule.md`` for the tiling design. ``devices``
    is NPU only; ``supported_archs`` is left ``None`` -- unlike the CUDA
    ``UnaryKernel``/``FloatUnaryKernel`` hierarchy this class does not inherit, there is no
    Ascend SoC-version gate here to replace the CUDA SM-version one (``Kernel._check_arch``
    skips the check entirely when ``supported_archs is None``).

    Tolerance (inherited from the shared elementwise test tolerance table, not re-derived
    here): fp16 rtol/atol 1e-3, bf16 1.6e-2. Large-magnitude inputs that drive ``exp(-x)`` to
    overflow in fp32 are algebraically benign for this formula (the overflow routes the result
    to (+/-)0 or to ``x`` itself, both correct limits of silu); see round_1_schedule.md SS5
    risk 1 and this op's M3 coverage for the large-|x| cases actually exercised on hardware.
    """

    devices: ClassVar[frozenset[str]] = frozenset({"npu"})
    supported_archs = None

    @classmethod
    def entry_for(cls, call: ElementwiseCall) -> Entry:
        return call, lambda: cls(call.n_total, call.dtype)

    def __init__(self, n_total: int, dtype: torch.dtype) -> None:
        if _ASCEND_IMPORT_ERROR is not None:
            raise ImportError(
                "AscendSiluKernel requires tilelang.ascend, which failed to import: "
                f"{_ASCEND_IMPORT_ERROR}"
            ) from _ASCEND_IMPORT_ERROR
        super().__init__()
        self.n_total = n_total
        self.dtype = dtype
        self.kernel = _build_silu_kernel(n_total, self.dtype_str) if n_total else None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Apply SiLU elementwise; nothing is written in place.

        Args:
            input: ``self.n_total`` elements in ``self.dtype`` on an NPU device, contiguous in
                whatever shape the caller passed.

        Returns:
            A new contiguous tensor on the same device, shaped and typed as ``input``.
        """
        if input.device.type != "npu":
            raise ValueError(
                f"{type(self).__name__} is an Ascend NPU kernel; got {input.device}. "
                "Another target's backend serves other devices."
            )
        if input.dtype != self.dtype:
            raise ValueError("input dtype must match compiled kernel")
        if not self.n_total:
            if input.numel():
                raise ValueError("expected empty input")
            return torch.empty_like(input)
        flat_input = input.contiguous().reshape(-1)
        if flat_input.numel() != self.n_total:
            raise ValueError(
                f"{type(self).__name__} was built for n_total={self.n_total}, got "
                f"{flat_input.numel()} elements"
            )
        flat_output = self.kernel(flat_input)
        return flat_output.reshape(input.shape)


try:
    from tileops.backend import register_kernel_type

    register_kernel_type("SiluFwdOp", "ascend", AscendSiluKernel)
except ImportError:
    pass
