"""
Ascend NPU SELU kernel: ``y = scale * (x if x >= 0 else alpha * (exp(x) - 1))``, via the
``tilelang.ascend`` dialect.

``docs/ascend950/dev_process.md``'s Ascend development loop for ``SeluFwdOp`` (resume
2026-10-19). Round 1: ``docs/ascend950/dev/SeluFwdOp/opus48_result.md`` (+ Planner review
``opus48_review.md``). Round 2 wrote the bounded-UB tail path (``opus48_round2_schedule.md`` /
``opus48_round2_review.md``) so any ``n_total`` is served (round 1 raised ``NotImplementedError``
for a shape that is ``> tile`` and not a multiple of ``num_blocks*tile``); the dsh round-1 run
validated it and re-bound M3/M4 to this source
(``docs/ascend950/dev/SeluFwdOp/dsh_round1_result.md``). Registers an additional implementation of
``SeluFwdOp``'s ``UnaryElementwiseFwdInterface`` under the ``"ascend"`` dispatch key, beside the
in-tree CUDA ``SeluFwdKernel`` (``tileops.kernels.elementwise.activations.SeluFwdKernel``). No
existing file is edited: ``tileops.backend.register_kernel_type`` joins this class to every
``SeluFwdOp`` constructed after this module is imported (``dev_process.md`` §1).

Math: mirrors ``SeluFwdKernel.op_func`` (``kernels/elementwise/activations.py``) except for one
signed-zero detail -- ``scale * T.if_then_else(x32 >= zero, x32, alpha * (T.exp(x32) - one))``
with ``x32`` the fp32 promotion of the input. The predicate is ``>=`` rather than the CUDA
kernel's ``>`` solely so ``-0.0`` takes the positive branch and ``scale * -0.0`` keeps its sign,
matching the same-device ``torch.nn.functional.selu`` oracle; every non-zero finite value, NaN
and +-Inf selects the same branch as ``>``. The dsh round-1 boundary probe measured this
explicitly (``dsh_round1_result.md``). ``alpha``/``scale`` are the SELU constants fixed by the
activation (``alpha = 1.6732632423543772``, ``scale = 1.0507009873554805``), **not** parameters --
SELU has no scalar knob, so this kernel inherits the parameter-free
``UnaryElementwiseFwdInterface`` and keys only on ``(n_total, dtype)`` (``SeluFwdOp`` is a
``_ParamFreeActivationOp``). Both branches carry ``scale``, so neither stays in storage dtype: the
whole select is evaluated in fp32, multiplied by ``scale`` in fp32, and cast to storage dtype
exactly **once** at the store. (The generated C source expresses this as a per-lane ``if/else``;
whether the hardware actually skips the untaken ``exp`` or if-converts it is a codegen detail not
asserted here -- the fp32 range of ``exp`` on the negative branch does not affect the selected
positive result either way.)

Tiling (round 1 structure, generalized in round 2 to the sibling ``sigmoid.py``/``silu.py`` tail
pattern): a flat ``n_total`` is split into a full-tile prefix run by a persistent, pipelined
multi-block program and a ``< tile`` remainder run by a one-block program, combined in Python by
``_SeluRunner`` (``torch.cat``). No in-kernel masking and no single kernel that allocates both a
full and a tail buffer (so the persistent UB budget is never doubled). Per-element UB footprint is
one input tile + one output tile, no scratch (``alpha``/``scale``/``zero``/``one`` are trace-time
scalars). Manifest fast path is unchanged from round 1: 8M/16M both land on ``num_blocks=64`` with
``tail=0`` (``num_full_tiles`` 512/1024, whose largest divisor ``<= 72`` cores is 64), 16-bit tile
``_TILE_ELEMS_16BIT`` (16384 = round-1 winner, 8192 = round-2 adjacent candidate), fp32 tile 8192,
``num_stages=3``; round 1 measured 3 faster than 2 on all four cases (``opus48_result.md`` §5), so
3 is kept.

Round-2 (AutoTuner) extension: ``_build_tiled`` takes ``num_stages`` as an explicit fifth
parameter (default ``_NUM_STAGES=3``) used by both ``annotate_buffer_versions`` and
``T.Pipelined``, so the ``lru_cache`` key really separates pipeline depths instead of mutating a
global. ``_MANIFEST_TILING`` is a fixed lookup for the four measured manifest ``(n_total,dtype)``
pairs only; every other shape keeps the round-1 routing. ``AscendSeluKernel.forward`` serves an
empty tensor (``n_total == 0``) by returning a fresh empty clone before any program is built or
launched -- the manifest ``Shape`` has no non-empty restriction, so this is part of the contract,
not a rejected input. Negative ``n_total`` is still rejected.

Does not inherit ``FloatUnaryKernel`` (``tileops/kernels/elementwise/_base.py``): it hardcodes
CUDA-only behavior (``devices = {"cuda"}``, ``supported_archs`` via ``tileops.utils.get_sm_version``,
a ``tensor.is_cuda`` guard) that would reject every NPU tensor outright.

No subpackage ``__init__.py`` with ``__all__`` re-exports (which ``.claude/rules/code-style.md``
asks for every ``src/tileops/kernels/*`` subpackage): per ``dev_process.md`` §1 every op's worker
self-registers from its own file and does *not* build a shared aggregator under ``kernels/ascend/``
that concurrent workers would contend for. ``kernels/ascend`` and ``kernels/ascend/elementwise``
are implicit (PEP 420) namespace packages.

``inplace`` needs no handling here: ``SeluFwdOp`` inherits ``_UnaryActivationMixin``
(``ops/elementwise/_base.py``), which copies this kernel's fresh output back into the input tensor
at the op layer when ``inplace=True`` -- the kernel always returns a new buffer.
"""

import functools
from typing import ClassVar, Optional

import torch

from tileops.kernels.elementwise.call_spec import ElementwiseCall, UnaryElementwiseFwdInterface
from tileops.kernels.kernel_base import Entry, Kernel

try:
    import tilelang
    import tilelang.ascend.language as T
    from tilelang.carver.arch.driver import get_num_vector_cores

    _ASCEND_IMPORT_ERROR: Optional[ImportError] = None
except ImportError as exc:  # pragma: no cover - exercised only off-Ascend
    tilelang = None  # type: ignore[assignment]
    T = None  # type: ignore[assignment]
    get_num_vector_cores = None  # type: ignore[assignment]
    _ASCEND_IMPORT_ERROR = exc

__all__ = ["AscendSeluKernel"]

# The SELU constants (torch.nn.functional.selu / the CUDA SeluFwdKernel). Fixed by the activation,
# not parameters: the one source of truth both builders bake into the compiled body.
_SELU_ALPHA = 1.6732632423543772
_SELU_SCALE = 1.0507009873554805

# SIMT UB budget per vector core, in bytes.
_UB_BUDGET_BYTES = 216 * 1024
# Pipeline depth of the persistent multi-tile path. Round 1 measured 3 faster than 2 on all four
# manifest cases (opus48_result.md §5); kept at 3. Only the multi-tile builder uses it.
_NUM_STAGES = 3
# Live-queried vector-core count fallback. 72 is this SKU's (Ascend950DT_9592) confirmed count
# (sibling pilots + dev_process.md §7.5); a wrong value only changes launch geometry, never the
# result, because _choose_num_blocks always returns an exact divisor.
_FALLBACK_VECTOR_CORES = 72
# Hard Ascend SIMT ceiling (dev_process.md §7.5): __launch_bounds__ above 2048 fails to compile.
_MAX_THREADS = 2048
# The 16-bit (fp16/bf16) persistent tile size. 16384 is the round-1 winner (192 KiB UB at
# _NUM_STAGES=3); 8192 (96 KiB) is the round-2 adjacent candidate measured per case. fp32 is
# always 8192. Whichever value this constant holds is bound to the final M3/M4 for that source.
_TILE_ELEMS_16BIT = 16384
# Fixed per-(n_total, dtype_str) scheduling for the four manifest cases selected by the round-2
# AutoTuner search plus the project's own formal M4 comparison. Values are
# ``(num_blocks, tile_elems, num_stages)``. A missing key keeps the round-1 routing below, so
# every non-manifest shape (including fp32, small shapes and real tails) is unchanged. Only
# measured configurations may be entered here; this is a static lookup, never runtime tuning.
_MANIFEST_TILING: dict[tuple[int, str], tuple[int, int, int]] = {}


def _num_vector_cores() -> int:
    """Best-effort AIV (vector) core count of device 0.

    Queries ``tilelang.carver.arch.driver.get_num_vector_cores``; falls back to
    ``_FALLBACK_VECTOR_CORES`` only if the query raises or returns a falsy value. Only changes how
    many persistent blocks the multi-tile path launches with -- a wrong or unavailable count
    degrades launch geometry, never the result (``_choose_num_blocks`` always returns an exact
    divisor, so every block runs the same iteration count). Not evaluated at import time: it
    touches the real device, so it is called lazily, under an ``npu-guard`` reservation.
    """
    try:
        queried = get_num_vector_cores(device_id=0)
    except Exception:
        queried = None
    if not queried:
        return _FALLBACK_VECTOR_CORES
    return int(queried)


def _tile_elems_for_dtype(dtype_str: str) -> int:
    """Element count of one UB tile for the persistent path, by dtype.

    fp32 (itemsize 4) uses 8192; fp16/bf16 (itemsize 2) use ``_TILE_ELEMS_16BIT`` (16384 round-1
    winner, 8192 round-2 candidate). Both keep ``2 buffers * _NUM_STAGES * itemsize * tile`` within
    ``_UB_BUDGET_BYTES``: fp16/bf16 at 16384 -> 196608 B (192 KiB), at 8192 -> 98304 B; fp32 at
    8192 -> 196608 B. A power of two keeps the ``num_full_tiles`` / ``_choose_num_blocks``
    arithmetic exact for every manifest shape.
    """
    if getattr(torch, dtype_str).itemsize >= 4:
        return 8192
    return _TILE_ELEMS_16BIT


def _choose_num_blocks(num_full_tiles: int, vector_cores: int) -> int:
    """Largest divisor of ``num_full_tiles`` not exceeding ``vector_cores``.

    Guarantees ``num_full_tiles % result == 0`` so every persistent block runs the same iteration
    count (no per-block tail inside the main loop). For the manifest cases ``num_full_tiles`` is
    512/1024 and the answer is 64; for an awkward (even prime) ``num_full_tiles`` larger than every
    core count it returns 1 (one block, still correct). Ported from ``sigmoid.py``/``silu.py``.
    """
    cap = max(1, min(num_full_tiles, vector_cores))
    for candidate in range(cap, 0, -1):
        if num_full_tiles % candidate == 0:
            return candidate
    return 1  # unreachable: 1 always divides num_full_tiles


if T is not None:

    @T.macro
    def _selu_body(x_ub, y_ub, n: int, threads: int) -> None:
        """``y_ub[:n] = selu(x_ub[:n])``, compiler-vectorized over ``threads`` lanes.

        Mirrors ``SeluFwdKernel.op_func``: the whole select runs in fp32 (``x32`` vs
        ``alpha * (exp(x32) - 1)``), is multiplied by ``scale`` in fp32, and cast to the storage
        dtype exactly once at the store. A ``@T.macro``, not a plain Python helper (code-style.md:
        sub-routines use ``@T.macro``) -- TileLang's eager builder only rewrites ``@T.macro``/
        ``@T.prim_func`` bodies, so an undecorated helper's ``for i in T.Parallel(n):`` raises
        ``TypeError: 'ForFrame' object is not iterable`` (dev_process.md §7.5).
        """
        with T.SimtVF(threads=threads):
            for i in T.Parallel(n):
                zero = T.cast(0.0, "float32")
                one = T.cast(1.0, "float32")
                alpha = T.cast(_SELU_ALPHA, "float32")
                scale = T.cast(_SELU_SCALE, "float32")
                x32 = T.cast(x_ub[i], "float32")
                selected = T.if_then_else(x32 >= zero, x32, alpha * (T.exp(x32) - one))
                y_ub[i] = T.cast(scale * selected, y_ub.dtype)

    @functools.lru_cache(maxsize=None)
    def _build_single_block(extent: int, dtype_str: str):
        """One block, one untiled GM->UB->compute->UB->GM pass over ``extent`` elements.

        Serves two routes: any ``n_total <= tile`` (the small/edge shapes and the 64-element
        inplace-conformance case), and -- reused unchanged -- the ``< tile`` remainder of the
        tiled path when ``n_total`` is not a whole number of tiles (``_SeluRunner``). UB is
        ``2 * extent * itemsize`` (no pipeline buffering): at the largest ``extent`` it ever sees
        (``tile - 1``) that is <= 2*16383*2 = 64 KiB, well within budget.
        """

        @T.prim_func
        def main(X: T.Tensor((extent,), dtype_str), Y: T.Tensor((extent,), dtype_str)):
            with T.Kernel(1):
                x_ub = T.alloc_shared((extent,), dtype_str)
                y_ub = T.alloc_shared((extent,), dtype_str)
                T.copy(X, x_ub)
                _selu_body(x_ub, y_ub, extent, threads=min(extent, _MAX_THREADS))
                T.copy(y_ub, Y)

        return tilelang.compile(main, target="ascend", out_idx=-1)

    @functools.lru_cache(maxsize=None)
    def _build_tiled(
        main_n: int, dtype_str: str, num_blocks: int, tile_elems: int, num_stages: int = _NUM_STAGES
    ):
        """Persistent-block, pipelined main loop over an exact multiple of ``tile_elems``.

        ``main_n`` is an exact multiple of ``num_blocks * tile_elems`` by construction (the caller
        passes only the full-tile prefix and a ``num_blocks`` that divides its tile count), so no
        tail handling is needed inside this program. UB is ``2 * tile_elems * num_stages *
        itemsize`` (<= 216 KiB for every accepted configuration). ``num_stages`` is an explicit
        parameter (default ``_NUM_STAGES=3``) used by both ``annotate_buffer_versions`` and
        ``T.Pipelined``; because it is part of this function's arguments it is also part of the
        ``lru_cache`` key, so different pipeline depths are genuinely different compiled programs.
        """
        block_span = num_blocks * tile_elems
        assert main_n % block_span == 0, (
            f"_build_tiled requires main_n ({main_n}) to divide num_blocks*tile_elems "
            f"({block_span}) exactly; the caller passes only the full-tile prefix"
        )
        assert num_stages >= 1, f"num_stages must be positive, got {num_stages}"
        assert (
            2 * tile_elems * num_stages * getattr(torch, dtype_str).itemsize <= _UB_BUDGET_BYTES
        ), (
            f"UB budget exceeded: 2*{tile_elems}*{num_stages}*"
            f"{getattr(torch, dtype_str).itemsize} > {_UB_BUDGET_BYTES}"
        )
        iterations = main_n // block_span

        @T.prim_func
        def main(X: T.Tensor((main_n,), dtype_str), Y: T.Tensor((main_n,), dtype_str)):
            with T.Kernel(num_blocks) as bx:
                x_ub = T.alloc_shared((tile_elems,), dtype_str)
                y_ub = T.alloc_shared((tile_elems,), dtype_str)
                T.annotate_buffer_versions({x_ub: num_stages, y_ub: num_stages})
                for step in T.Pipelined(iterations, num_stages=num_stages):
                    begin = (step * num_blocks + bx) * tile_elems
                    T.copy(X[begin : begin + tile_elems], x_ub)
                    _selu_body(x_ub, y_ub, tile_elems, threads=min(tile_elems, _MAX_THREADS))
                    T.copy(y_ub, Y[begin : begin + tile_elems])

        return tilelang.compile(main, target="ascend", out_idx=-1)


class _SeluRunner:
    """Dispatches one flat ``(n_total,)`` NPU tensor to one or two compiled Ascend programs.

    One program for ``n_total <= tile`` (single-block) or an exact multiple of ``num_blocks*tile``
    (tiled). Two only when ``n_total > tile`` and is not a whole number of tiles: the tiled program
    covers every full tile ``[:main_n]`` and a single-block program the remainder ``[main_n:]``,
    concatenated here (plain Python dispatch around two already-compiled kernels, not a runtime
    branch inside either). Ported from ``sigmoid.py``'s ``_SigmoidRunner``.
    """

    __slots__ = ("main_kernel", "tail_kernel", "main_n")

    def __init__(self, main_kernel, tail_kernel, main_n: int):
        self.main_kernel = main_kernel
        self.tail_kernel = tail_kernel
        self.main_n = main_n

    def __call__(self, flat_input: torch.Tensor) -> torch.Tensor:
        if self.tail_kernel is None:
            return self.main_kernel(flat_input)
        main_part = self.main_kernel(flat_input[: self.main_n])
        tail_part = self.tail_kernel(flat_input[self.main_n :])
        return torch.cat((main_part, tail_part))


@functools.lru_cache(maxsize=None)
def _build_selu_kernel(n_total: int, dtype_str: str) -> "_SeluRunner":
    """Build (memoized by ``(n_total, dtype_str)``) the Ascend SELU runner for ``n_total``.

    ``n_total <= tile``: one single-block program (correctness path for small/edge shapes and the
    conformance case). ``n_total > tile``: a persistent tiled program over the full-tile prefix
    ``main_n = (n_total // tile) * tile`` with ``num_blocks`` the largest core-bounded divisor of
    the tile count, plus -- when ``n_total`` is not a whole number of tiles -- a single-block
    program for the ``< tile`` remainder, combined by ``_SeluRunner``. Every manifest shape has
    ``tail == 0``.

    A ``(n_total, dtype_str)`` present in ``_MANIFEST_TILING`` uses that measured
    ``(num_blocks, tile_elems, num_stages)`` directly; it must divide ``n_total`` exactly and fit
    the UB budget. Every other shape keeps the round-1 routing above. ``n_total == 0`` is served
    by ``AscendSeluKernel.forward`` before this builder is called; a non-positive value here is an
    internal contract violation.
    """
    if n_total <= 0:
        raise ValueError(f"n_total must be positive, got {n_total}")
    tiling = _MANIFEST_TILING.get((n_total, dtype_str))
    if tiling is not None:
        num_blocks, tile_elems, num_stages = tiling
        itemsize = getattr(torch, dtype_str).itemsize
        if num_blocks <= 0 or tile_elems <= 0 or num_stages <= 0:
            raise ValueError(f"manifest tiling must be positive, got {tiling!r}")
        if n_total % (num_blocks * tile_elems) != 0:
            raise ValueError(
                f"manifest tiling {tiling!r} does not divide n_total={n_total} exactly"
            )
        if 2 * tile_elems * num_stages * itemsize > _UB_BUDGET_BYTES:
            raise ValueError(f"manifest tiling {tiling!r} exceeds the UB budget for {dtype_str}")
        return _SeluRunner(
            _build_tiled(n_total, dtype_str, num_blocks, tile_elems, num_stages), None, n_total
        )
    tile_elems = _tile_elems_for_dtype(dtype_str)
    if n_total <= tile_elems:
        return _SeluRunner(_build_single_block(n_total, dtype_str), None, n_total)
    num_full_tiles = n_total // tile_elems
    main_n = num_full_tiles * tile_elems
    tail = n_total - main_n
    num_blocks = _choose_num_blocks(num_full_tiles, _num_vector_cores())
    main_kernel = _build_tiled(main_n, dtype_str, num_blocks, tile_elems)
    tail_kernel = _build_single_block(tail, dtype_str) if tail else None
    return _SeluRunner(main_kernel, tail_kernel, main_n)


class AscendSeluKernel(Kernel, UnaryElementwiseFwdInterface):
    """SELU on Ascend NPUs: ``y = scale * (x if x >= 0 else alpha * (exp(x) - 1))``.

    ``alpha = 1.6732632423543772``, ``scale = 1.0507009873554805``. See
    ``docs/ascend950/dev/SeluFwdOp/opus48_round2_schedule.md`` for the tiling/tail design and
    ``docs/ascend950/dev/SeluFwdOp/dsh_round2_result.md`` for the round-2 AutoTuner selection.
    ``devices`` is NPU only; ``supported_archs`` is ``None`` (no Ascend SoC-version gate; the CUDA
    ``FloatUnaryKernel`` family this class does not inherit carries the SM-version one).

    Implements the parameter-free ``UnaryElementwiseFwdInterface`` (request ``ElementwiseCall``):
    SELU's ``alpha``/``scale`` are activation constants, not call facts, so nothing beyond
    ``(n_total, dtype)`` enters the build identity.

    An empty tensor (``n_total == 0``) builds no program: ``forward`` returns a fresh empty clone
    before any launch, preserving the op layer's inplace copy-back contract. Negative
    ``n_total`` is rejected at construction.

    Tolerance (inherited from the shared elementwise test tolerance table, not re-derived here):
    fp16 1e-3, bf16 1.6e-2, fp32 1e-5 -- ``SeluFwdOp`` is not special-cased in
    ``ElementwiseWorkload.verification``, so it gets the default.
    """

    devices: ClassVar[frozenset[str]] = frozenset({"npu"})
    supported_archs = None

    @classmethod
    def entry_for(cls, call: ElementwiseCall) -> Entry:
        return call, lambda: cls(call.n_total, call.dtype)

    def __init__(self, n_total: int, dtype: torch.dtype) -> None:
        if _ASCEND_IMPORT_ERROR is not None:
            raise ImportError(
                "AscendSeluKernel requires tilelang.ascend, which failed to import: "
                f"{_ASCEND_IMPORT_ERROR}"
            ) from _ASCEND_IMPORT_ERROR
        if n_total < 0:
            raise ValueError(f"n_total must be non-negative, got {n_total}")
        super().__init__()
        self.n_total = n_total
        self.dtype = dtype
        # An empty tensor has no program to compile or launch; forward handles it before use.
        self.kernel = _build_selu_kernel(n_total, self.dtype_str) if n_total else None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Apply SELU elementwise; nothing is written in place.

        Args:
            input: ``self.n_total`` elements in ``self.dtype`` on an NPU device, contiguous in
                whatever shape the caller passed (the op layer makes it contiguous).

        Returns:
            A new contiguous tensor on the same device, shaped and typed as ``input``. For an
            empty tensor, a fresh empty clone with the same shape/dtype/device is returned without
            compiling or launching any program; the op layer's ``inplace=True`` copy-back still
            returns the input object.
        """
        if input.device.type != "npu":
            raise ValueError(
                f"{type(self).__name__} is an Ascend NPU kernel; got {input.device}. "
                "Another target's backend serves other devices."
            )
        flat_input = input.reshape(-1)
        if flat_input.numel() != self.n_total:
            raise ValueError(
                f"{type(self).__name__} was built for n_total={self.n_total}, got "
                f"{flat_input.numel()} elements"
            )
        if self.n_total == 0:
            return input.clone()
        flat_output = self.kernel(flat_input)
        return flat_output.reshape(input.shape)


try:
    from tileops.backend import register_kernel_type

    register_kernel_type("SeluFwdOp", "ascend", AscendSeluKernel)
except ImportError:
    pass
