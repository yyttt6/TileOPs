"""Ascend NPU Reciprocal forward kernel: ``y = 1/x``, via the ``tilelang.ascend`` dialect.

Round 1 of ``docs/ascend950/dev_process.md``'s Ascend development loop for ``ReciprocalFwdOp``;
see ``docs/ascend950/dev/ReciprocalFwdOp/round_1_schedule.md`` for the full design this file
implements. Registers an additional implementation of ``ReciprocalFwdOp``'s own
``ReciprocalFwdInterface`` under the ``"ascend"`` dispatch key
(``tileops.backend.register_kernel_type``), beside the in-tree CUDA ``ReciprocalFwdKernel``
(``tileops.kernels.elementwise.math_unary.ReciprocalFwdKernel``). No existing file is edited.

**Interface choice is load-bearing, not cosmetic**: ``ReciprocalFwdOp`` overrides ``interfaces``
to ``{ELEMENTWISE: ReciprocalFwdInterface}``, not the generic ``UnaryElementwiseFwdInterface``
every batch-1-3 sibling (``SigmoidFwdOp``, ``RsqrtFwdOp``, ...) uses -- the two are sibling
classes (neither extends the other) with an identical abstract ``forward(self, input)`` shape, so
a wrong choice here would not be caught by any signature check, only by ``op_base.py``'s
``issubclass`` filter, which would silently leave this kernel's key "unassigned" and raise
``ValueError: ReciprocalFwdOp keys implement none of its kernel interfaces`` at
``ReciprocalFwdOp()`` construction time, for every test in the repo, on every device, the instant
this module is imported anywhere in the process (round_1_schedule.md §1). Hence
``AscendReciprocalKernel(Kernel, ReciprocalFwdInterface)`` below, not
``UnaryElementwiseFwdInterface``.

Math: plain fp32 ``1.0 / x``, cast in, cast out -- no CUDA-specific PTX tricks. The in-tree CUDA
``ReciprocalFwdKernel.op_func`` reaches for ``float32_quotient`` (``kernels/elementwise/
_division.py``), which dispatches to ``approx_fdiv``/``full_range_fdiv`` (``__fdividef``-based
CUDA PTX builtins) for fp16/bf16 storage -- both are CUDA-only speed tricks that exist to
compensate for an *approximate* divide instruction CUDA chose for throughput; Ascend has no
such instruction (round_1_schedule.md §2 confirms, via ``intrin_rule_ascend.cc``'s registered
intrinsic table, that reciprocal/division is not in Ascend's transcendental dispatch at all --
``1/x`` lowers through plain binary-operator codegen). This matches ``SigmoidFwdOp``'s
already-validated precedent (``sigmoid.py``'s ``_sigmoid_op``: ``one / (one + T.exp(-wide))``,
no divide-compensation trick, which passed bit-exact against ``torch_npu``'s own on-device
``torch.sigmoid`` across all 65536 fp16/bf16 codes) -- reused directly here, with no prior
Ascend op in this project having divided by a literal zero (``1/0``, ``1/-0``, ``1/+-inf``, NaN
payloads are all genuinely new ground for this project, exercised by
``tests/ops/test_unary_math.py::test_16bit_reciprocal_matches_torch_bit_for_bit``). See this
op's ``round_1_result.md`` for what that test actually found on hardware.

Int-dtype handling mirrors the CUDA kernel's own ``refusal``/``entry_for``/``forward`` trio
(round_1_schedule.md §1.1): there is no separate ``ReciprocalIntFwdKernel`` to port (unlike
``AbsFwdOp``/``NegFwdOp``/``SignFwdOp``'s ``*IntFwdKernel(TorchFallbackKernel)`` pattern) --
``ReciprocalFwdKernel`` handles ints itself by redirecting them to the *same* fp32 TileLang
program a float32 call uses, substituting only the compute dtype in ``entry_for`` and converting
the tensor with a plain ``.to()`` in ``forward``. This file reproduces exactly that redirect,
with no new NPU code path for integers.

Tiling scaffolding (``_tile_elems_for_dtype``, ``_calibrated_num_blocks``, ``_choose_num_blocks``,
the single-block/persistent builder pair) is the same shape ``sigmoid.py``/``rsqrt.py`` already
validated on this hardware for this op family's cost profile (one load, one divide, one store
per element, no decode-sized manifest case). One wrinkle this op's test surface exposes that no
prior sibling schedule needed: ``test_16bit_reciprocal_matches_torch_bit_for_bit`` drives a flat
``1 << 16 = 65536``-element fp16/bf16 tensor, which is larger than the single-block cutoff
(``tile=16384``) but does not divide the calibrated persistent block span (``64 * 16384 =
1,048,576``) evenly. Forcing it through an uncapped 65536-element single block would cost
``2 buffers x 65536 elements x 2 bytes = 256 KiB`` of UB -- exactly ``BitwiseNotFwdOp``'s
confirmed UB overflow point, not safely under it (round_1_schedule.md §5.1). Instead,
``_choose_num_blocks`` (ported verbatim from ``sigmoid.py``'s already-proven helper, not
reimplemented) is applied to every ``n_total > tile`` shape rather than assuming the calibrated
core count divides evenly: for 65536 elements it searches divisors of
``num_full_tiles = 65536 // 16384 = 4`` and lands on ``num_blocks=4`` (block span exactly 65536,
one iteration, 192 KiB UB -- the same safe footprint every other persistent build here uses),
while still landing on ``num_blocks=64`` for every power-of-two manifest/correctness shape that
already divides evenly (16M, 256M, and the 1,048,576-element ``test_reciprocal`` fixture).

Importing this module when ``tilelang.ascend`` is unavailable (no Ascend toolchain, a CUDA-only
process) is a no-op: the Ascend-specific imports are guarded by one ``try/except ImportError``
(``_ASCEND_IMPORT_ERROR``), re-raised lazily only when ``AscendReciprocalKernel.__init__`` is
actually reached, mirroring every sibling Ascend kernel file in this directory.

This file does not get a subpackage ``__init__.py`` with ``__all__`` re-exports -- same
deliberate omission every sibling file in this directory documents (several ops' workers write
into this same directory concurrently; a shared aggregator file would be contended,
``docs/ascend950/dev_process.md`` §1).
"""

import functools
from typing import ClassVar, Optional

import torch

from tileops.kernels.elementwise.call_spec import ElementwiseCall, ReciprocalFwdInterface
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

__all__ = ["AscendReciprocalKernel"]

# The integral element types the manifest admits, and the float storage types it admits
# unchanged. Defined locally (plain tuples of ``torch.dtype``) rather than imported from
# ``kernels/elementwise/math_unary.py`` -- keeps this file's only cross-package dependency the
# ``tilelang.ascend`` try/except block, per round_1_schedule.md §1.1.
INT_DTYPES = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

# SIMT UB budget per vector core, in bytes. Reused unchanged from every sibling elementwise
# Ascend kernel file in this directory (``sigmoid.py``, ``rsqrt.py``, ...): this op's UB
# footprint (2 buffers, same two per-dtype tile sizes) is identical in shape to theirs.
_UB_BUDGET_BYTES = 216 * 1024
# cast/divide/cast is a short body, comparable to SigmoidFwdOp's longer cast/negate/exp/add/
# divide/cast chain and heavier than RsqrtFwdOp's/ReluFwdOp's single-intrinsic-or-compare body;
# round_1_schedule.md §5.2 keeps this at Sigmoid/Silu/Exp's converged 3 rather than Rsqrt/Relu's
# 2 (and flags a quick 2-vs-3 A/B as optional, not mandatory, for round 1).
_NUM_STAGES = 3
# Fallback vector-core count when the live query below is unavailable. Confirmed 72 on this SKU
# (device ``Ascend950DT_9592``) by multiple independent prior ops in this batch
# (``docs/ascend950/dev_process.md`` §7.5).
_FALLBACK_VECTOR_CORES = 72
# T.SimtVF's hard 2048-thread cap (dev_process.md §7.5), confirmed project-wide; tile can exceed
# it, so threads stays independent of tile with T.Parallel spreading tile's iterations across
# lanes.
_MAX_THREADS = 2048


def _floor_pow2(n: int) -> int:
    """Largest power of two ``<= n`` (``n >= 1``)."""
    p = 1
    while p * 2 <= n:
        p *= 2
    return p


@functools.lru_cache(maxsize=1)
def _calibrated_num_blocks() -> int:
    """Persistent-grid block count for the pipelined, large-``n`` path.

    Live-queries this SKU's real vector-core count (``get_num_vector_cores``), floored to the
    nearest power of two, rather than hardcoding the figure outright -- cheap, one ``npu-guard``
    reservation, and a defensive check in case this session's hardware differs.
    ``ReluFwdOp``/``GeluFwdOp``/``SigmoidFwdOp``/``RsqrtFwdOp`` all independently land on 64 here
    (``floor_pow2(72) == 64``) for the same one-load/one-store-per-element cost shape this op
    shares (round_1_schedule.md §5.2).

    Deliberately **not** evaluated at module import time -- merely importing this module must
    stay a no-NPU-touch no-op. ``lru_cache`` makes it a one-time query per process either way.
    Falls back to ``_FALLBACK_VECTOR_CORES`` if the live query raises or returns a falsy value.
    """
    try:
        queried = get_num_vector_cores(device_id=0)
    except Exception:
        queried = None
    if not queried:
        return _FALLBACK_VECTOR_CORES
    return _floor_pow2(int(queried))


def _tile_elems_for_dtype(dtype_str: str) -> int:
    """The largest power-of-two element count keeping the persistent path's UB use in budget.

    2 buffers (x_ub, y_ub) x ``_NUM_STAGES`` x ``itemsize`` x ``tile_elems`` must fit
    ``_UB_BUDGET_BYTES``. Ported from ``sigmoid.py``'s identical derivation: for fp16/bf16
    (itemsize 2) this returns 16384 (192 KiB used); for float32 (itemsize 4) it returns 8192
    (also 192 KiB used) -- matching round_1_schedule.md §5.1's targets exactly.
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
    number of iterations -- no per-block tail inside the main loop. Ported verbatim from
    ``sigmoid.py``'s already-proven helper (round_1_schedule.md §5.1 explicitly directs reusing
    this, not reimplementing it): for this op's 5 manifest cases and the
    1,048,576-element correctness fixture, ``num_full_tiles`` is always a power of two that
    divides 64 cleanly, so the search lands on 64; for the one shape that does not --
    ``test_16bit_reciprocal_matches_torch_bit_for_bit``'s 65536-element fp16/bf16 tensor, where
    ``num_full_tiles = 65536 // 16384 = 4`` -- it correctly lands on 4 instead. 1 always
    qualifies, so this never fails to return a valid schedule.
    """
    cap = max(1, min(num_full_tiles, vector_cores))
    for candidate in range(cap, 0, -1):
        if num_full_tiles % candidate == 0:
            return candidate
    return 1  # unreachable: candidate == 1 always divides num_full_tiles


def _dtype_str(dtype: torch.dtype) -> str:
    return str(dtype).split(".")[-1]


if T is not None:

    @T.macro
    def _reciprocal_body(x_ub, y_ub, n: int, dtype_str: str, threads: int) -> None:
        """``y_ub[:n] = 1 / x_ub[:n]``, computed in fp32, compiler-vectorized over ``threads``.

        Plain fp32 ``one / x``, no CUDA-style divide-compensation trick (see module docstring).
        A ``@T.macro``, not a plain Python helper (code-style.md: "sub-routines use
        ``@T.macro``, never nested ``prim_func``") -- an undecorated helper's
        ``for i in T.Parallel(n):`` fails with ``TypeError: 'ForFrame' object is not iterable``
        (dev_process.md §7.5), since it would call the raw context-manager object instead of
        going through the builder's loop lowering.
        """
        with T.SimtVF(threads=threads):
            for i in T.Parallel(n):
                one = T.cast(1.0, "float32")
                y_ub[i] = T.cast(one / T.cast(x_ub[i], "float32"), dtype_str)

    @functools.lru_cache(maxsize=64)
    def _build_single_block(n_total: int, dtype_str: str):
        """One block, one untiled GM->UB->compute->UB->GM pass over ``n_total`` elements.

        Correctness-only fallback for every shape at or below ``_tile_elems_for_dtype
        (dtype_str)``: the roofline/validation fixtures' 4-element calls, the int-promotion
        tests' 64/256-element calls, ``test_reciprocal_edge``'s 4096-element fp32 call, and the
        64-element float16 case in ``test_compiled_non_contiguous_input_matches_eager``. None of
        this round's 5 benchmarked manifest cases reaches this path.
        """

        @T.prim_func
        def main(X: T.Tensor((n_total,), dtype_str), Y: T.Tensor((n_total,), dtype_str)):
            with T.Kernel(1):
                x_ub = T.alloc_shared((n_total,), dtype_str)
                y_ub = T.alloc_shared((n_total,), dtype_str)
                T.copy(X, x_ub)
                _reciprocal_body(x_ub, y_ub, n_total, dtype_str, threads=min(n_total, _MAX_THREADS))
                T.copy(y_ub, Y)

        return tilelang.compile(main, target="ascend", out_idx=-1)

    @functools.lru_cache(maxsize=64)
    def _build_persistent(n_total: int, dtype_str: str, num_blocks: int, tile: int, num_stages: int):
        """Persistent multi-block grid with ``T.Pipelined`` staging.

        Requires ``n_total`` to divide ``num_blocks * tile`` exactly; the caller
        (``_make_reciprocal_kernel``) only reaches this builder for a shape that does (via
        ``_choose_num_blocks``, which always finds such a divisor). Serves all 5 benchmarked
        manifest cases, the 1,048,576-element ``test_reciprocal`` fixture (all 3 dtypes), and
        the 65536-element exhaustive bit-test fixture (``num_blocks=4`` there, not 64 --
        round_1_schedule.md §5.1).
        """
        block_span = num_blocks * tile
        assert n_total % block_span == 0, (
            f"_build_persistent requires n_total ({n_total}) to divide num_blocks*tile "
            f"({block_span}) exactly; the caller routes other shapes elsewhere"
        )
        iterations = n_total // block_span
        threads = min(tile, _MAX_THREADS)

        @T.prim_func
        def main(X: T.Tensor((n_total,), dtype_str), Y: T.Tensor((n_total,), dtype_str)):
            with T.Kernel(num_blocks) as bx:
                x_ub = T.alloc_shared((tile,), dtype_str)
                y_ub = T.alloc_shared((tile,), dtype_str)
                T.annotate_buffer_versions({x_ub: num_stages, y_ub: num_stages})
                for step in T.Pipelined(iterations, num_stages=num_stages):
                    begin = (step * num_blocks + bx) * tile
                    T.copy(X[begin : begin + tile], x_ub)
                    _reciprocal_body(x_ub, y_ub, tile, dtype_str, threads)
                    T.copy(y_ub, Y[begin : begin + tile])

        return tilelang.compile(main, target="ascend", out_idx=-1)

    @functools.lru_cache(maxsize=128)
    def _build_with_tail(n_total: int, dtype_str: str):
        """Bounded UB tail; only valid elements cross the global-memory boundary."""
        tile = _tile_elems_for_dtype(dtype_str)
        # Leave room for both tail buffers alongside pipelined full tiles.
        itemsize = 4 if dtype_str == "float32" else 2
        while (2 * tile * _NUM_STAGES + 2 * ((n_total % tile + 63) // 64 * 64)) * itemsize > 248 * 1024:
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
                            T.copy(X[begin:begin + tile], x)
                            _reciprocal_body(x, y, tile, dtype_str, min(tile, _MAX_THREADS))
                            T.copy(y, Y[begin:begin + tile])
                if tail:
                    if bx == 0:
                        xt = T.alloc_shared((padded_tail,), dtype_str)
                        yt = T.alloc_shared((padded_tail,), dtype_str)
                        T.copy(X[full * tile:n_total], xt[:tail])
                        _reciprocal_body(xt, yt, padded_tail, dtype_str, min(padded_tail, _MAX_THREADS))
                        T.copy(yt[:tail], Y[full * tile:n_total])
        return tilelang.compile(main, target="ascend", out_idx=-1)

    def _make_reciprocal_kernel(n_total: int, dtype_str: str):
        """Keep calibrated aligned kernels; handle arbitrary tails in padded UB."""
        if n_total <= 0:
            raise ValueError(f"n_total must be positive, got {n_total}")

        tile = _tile_elems_for_dtype(dtype_str)
        if n_total % 64:
            return _build_with_tail(n_total, dtype_str)
        if n_total <= tile:
            return _build_single_block(n_total, dtype_str)

        if n_total % tile != 0:
            return _build_with_tail(n_total, dtype_str)
        num_full_tiles = n_total // tile
        num_blocks = _choose_num_blocks(num_full_tiles, _calibrated_num_blocks())
        return _build_persistent(n_total, dtype_str, num_blocks, tile, _NUM_STAGES)


class AscendReciprocalKernel(Kernel, ReciprocalFwdInterface):
    """Reciprocal on Ascend NPUs: ``y = 1/x``, plain fp32 divide, cast back to the storage dtype.

    Implements ``ReciprocalFwdInterface`` (**not** ``UnaryElementwiseFwdInterface`` -- see this
    module's docstring for why that distinction is load-bearing for ``ReciprocalFwdOp``'s own
    ``interfaces`` override). ``devices`` is NPU only; ``supported_archs`` is left ``None`` --
    unlike the CUDA ``UnaryKernel``/``FloatUnaryKernel`` hierarchy this class does not inherit,
    there is no Ascend SoC-version gate here to replace the CUDA SM-version one
    (``Kernel._check_arch`` skips the check entirely when ``supported_archs is None``).

    Integral inputs are this kernel's business too, mirroring the in-tree CUDA
    ``ReciprocalFwdKernel``'s own design: it has no separate integer kernel, so ``refusal``
    accepts every manifest dtype, ``entry_for`` substitutes ``torch.float32`` as the compute
    dtype for an integral call, and ``forward`` converts the tensor with a plain ``.to()`` before
    running it through the same fp32 TileLang program a float32 call uses -- no new NPU code
    path for integers.

    Tolerance for the shared float32/float16/bfloat16 cases is inherited from the shared
    elementwise test tolerance table (not re-derived here); the fp16/bf16 cases additionally
    face ``tests/ops/test_unary_math.py::test_16bit_reciprocal_matches_torch_bit_for_bit``, which
    requires an exact bit-for-bit match (``torch.equal`` on the integer view, NaN payloads
    included) against ``torch_npu``'s own on-device ``torch.reciprocal`` at all 65536 16-bit
    codes of each dtype -- see this module's docstring for why a plain fp32 divide is the
    correct translation of this op's math on this hardware, and this op's own
    ``round_1_result.md`` for the measured outcome.
    """

    devices: ClassVar[frozenset[str]] = frozenset({"npu"})
    supported_archs = None

    @classmethod
    def refusal(cls, call: ElementwiseCall) -> "str | None":
        if call.dtype in FLOAT_DTYPES or call.dtype in INT_DTYPES:
            return None
        return f"serves {FLOAT_DTYPES + INT_DTYPES}, not {call.dtype}"

    @classmethod
    def entry_for(cls, call: ElementwiseCall) -> Entry:
        dtype = torch.float32 if call.dtype in INT_DTYPES else call.dtype
        return call, lambda: cls(call.n_total, dtype)

    def __init__(self, n_total: int, dtype: torch.dtype) -> None:
        if _ASCEND_IMPORT_ERROR is not None:
            raise ImportError(
                "AscendReciprocalKernel requires tilelang.ascend, which failed to import: "
                f"{_ASCEND_IMPORT_ERROR}"
            ) from _ASCEND_IMPORT_ERROR
        super().__init__()
        self.n_total = n_total
        self.dtype = dtype
        self.kernel = _make_reciprocal_kernel(n_total, _dtype_str(dtype)) if n_total else None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Return ``1 / input`` elementwise; nothing is written in place.

        Args:
            input: ``self.n_total`` elements on an NPU device, contiguous in whatever shape the
                caller passed. Converted to ``self.dtype`` first when it arrives in a different
                dtype -- the integral-input redirect every call with an integer dtype takes
                (``entry_for`` already fixed ``self.dtype`` to ``torch.float32`` for those).

        Returns:
            A new contiguous tensor on the same device, shaped as ``input``, in ``self.dtype``.
        """
        if input.device.type != "npu":
            raise ValueError(
                f"{type(self).__name__} is an Ascend NPU kernel; got {input.device}. "
                "Another target's backend serves other devices."
            )
        if input.dtype != self.dtype:
            input = input.to(self.dtype)
        if not self.n_total and not input.numel():
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

    register_kernel_type("ReciprocalFwdOp", "ascend", AscendReciprocalKernel)
except ImportError:
    pass
