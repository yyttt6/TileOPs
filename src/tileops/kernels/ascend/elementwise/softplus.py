"""Ascend Softplus with persistent tiles and bounded arbitrary-size tails."""

import ctypes
import functools
import math
from typing import ClassVar, Optional

import torch

from tileops.kernels.elementwise.call_spec import SoftplusCall, SoftplusFwdInterface
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

__all__ = ["AscendSoftplusKernel"]

# SIMT UB budget per vector core, in bytes. Reused verbatim from AscendSiluKernel (silu.py),
# which discovered it via a real on-device compile failure on float32's dynamic-shared-memory
# limit -- this op has the identical per-element UB footprint shape (one input tile + one output
# tile, no extra scratch: the scaled/exp/log intermediates are per-thread registers, not UB
# allocations), so the same budget applies unchanged.
_UB_BUDGET_BYTES = 216 * 1024
# The one open round-1 parameter (schedule doc §3.3): built and measured as both 2 and 3 under
# the official measure_tileops_ascend.py protocol; see round_1_result.md §4 for the comparison.
_NUM_STAGES = 2
# Conservative fallback vector-core count if get_num_vector_cores() raises or returns a falsy
# value. 64 is this SKU's own measured-optimal value (Relu/Gelu/Silu/Exp/Log1p round 1/2), not an
# arbitrary guess, so a failed query still degrades to the right answer for this device.
_FALLBACK_VECTOR_CORES = 64
# Hard Ascend SIMT ceiling (dev_process.md §7.5): __launch_bounds__ above 2048 fails to compile.
_MAX_THREADS = 2048


def _floor_pow2(n: int) -> int:
    """Largest power of two ``<= n`` (``n >= 1``)."""
    p = 1
    while p * 2 <= n:
        p *= 2
    return p


@functools.lru_cache(maxsize=1)
def _calibrated_num_blocks(fallback: int = _FALLBACK_VECTOR_CORES) -> int:
    """Persistent-grid block count for the large, bandwidth-bound tiled path.

    Live-queries this SKU's real vector-core count (``get_num_vector_cores``, device 0) and
    floors it to the nearest power of two, matching ``AscendExpKernel``/``AscendLog1pKernel``'s
    ``_calibrated_num_blocks`` exactly -- five-plus independent pilots already swept this under
    the official (no-cache-flush) M4 protocol and converged on 64 as both the live-queried floor
    *and* the measured optimum on this SKU (72 real vector cores, ``Ascend950DT_9592``). This
    op's body is a cheap composite of two transcendentals, structurally close to those precedents,
    so round 1 expects the same answer rather than re-sweeping it from scratch (schedule doc
    §3.1).

    Deliberately **not** evaluated at module import time: the query touches the real device, so
    merely importing this module must stay a no-NPU-touch no-op (``dev_process.md`` §4).
    ``lru_cache`` makes it a one-time query per process either way.
    """
    try:
        queried = get_num_vector_cores(device_id=0)
    except Exception:
        queried = None
    if not queried:
        return fallback
    return _floor_pow2(int(queried))


def _tile_elems_for_dtype(dtype_str: str) -> int:
    """The largest power-of-two element count keeping the tiled path's UB use in budget.

    2 buffers (x tile, y tile) x ``_NUM_STAGES`` x ``itemsize`` x ``tile_elems`` must fit
    ``_UB_BUDGET_BYTES``. Resolves to 16384 for fp16/bf16 (itemsize 2) and 8192 for float32
    (itemsize 4) at both ``num_stages`` candidates this round considers (schedule doc §3.2's
    table confirms the result is invariant to that choice here) -- the same numbers
    ``ExpFwdOp``/``Log1pFwdOp`` converged on, reused rather than re-derived.
    """
    itemsize = getattr(torch, dtype_str).itemsize
    max_elems = _UB_BUDGET_BYTES // (2 * _NUM_STAGES * itemsize)
    return _floor_pow2(max_elems)


# Round 3: per-case SIMT scheduling selected by the real AutoTuner search over
# simt_threads x coalesced_width for the tight 8M bf16 case only. The table is
# consulted only at beta==1.0/threshold==20.0; every other call (other shapes,
# fp32, non-default beta/threshold) keeps the round-1 default layout 2048/0.
_TUNED_LAYOUTS = {
    (8388608, "bfloat16"): (1024, 2),
}


def _opmath_scalar(value: float) -> float:
    """Match the CPU operator's float scalar conversion, including finite overflow."""
    value = float(value)
    if math.isfinite(value) and abs(value) > 3.4028234663852886e38:
        raise RuntimeError("value cannot be converted to type float without overflow")
    return ctypes.c_float(value).value


def _layout_for_call(
    n_total: int, dtype_str: str, beta: float, threshold: float
) -> tuple[int, int]:
    """``(simt_threads, coalesced_width)`` for one full production call."""
    if beta == 1.0 and threshold == 20.0:
        tuned = _TUNED_LAYOUTS.get((n_total, dtype_str))
        if tuned is not None:
            return tuned
    return (_MAX_THREADS, 0)


if T is not None:

    @T.macro
    def _softplus_body(
        x_ub, y_ub, n: int, threads: int, beta: float, threshold: float,
        coalesced_width: int = 0,
    ) -> None:
        """``y_ub[:n] = softplus(x_ub[:n]; beta, threshold)``, computed in fp32.

        Mirrors the CUDA reference's ``_make_softplus_kernel`` body (``kernels/elementwise/
        activations.py``) exactly, minus its CUDA-only ``log_for_output_precision`` dispatch
        (schedule doc §2, §2.1): ``scaled = beta * x`` and the full ``log(1 + exp(scaled)) /
        beta`` composite are computed unconditionally for every element (binding ``softplus`` as
        a statement, not inside the guard -- the CUDA kernel's own comment records this as a
        measured 1.2% float16 win), then selected against ``x`` via ``T.if_then_else``. ``beta``
        and ``threshold`` are plain Python floats closed over at trace time (the caller's
        ``functools.lru_cache`` key), not tensor operands -- each distinct ``(beta, threshold)``
        pair traces and compiles its own kernel.

        A ``@T.macro``, not a plain Python helper (code-style.md: "sub-routines use
        ``@T.macro``, never nested ``prim_func``") -- TileLang's eager builder only rewrites
        ``@T.macro``/``@T.prim_func`` bodies, so an undecorated helper's
        ``for i in T.Parallel(n):`` fails with ``TypeError: 'ForFrame' object is not iterable``
        (dev_process.md §7.5).
        """
        with T.SimtVF(threads=threads):
            # coalesced_width is only a scheduling hint: 0 maps to None (kw
            # omitted by T.Parallel), 1/2 are passed as the upper bound. The
            # math below is byte-for-byte the round-1 SIMT body.
            for i in T.Parallel(
                n,
                coalesced_width=(
                    None if coalesced_width == 0 else T.int32(coalesced_width)
                ),
            ):
                one = T.Cast("float32", 1.0)
                t = T.Cast("float32", threshold)
                wide = T.Cast("float32", x_ub[i])
                if beta == 0.0:
                    # CPU Softplus: finite inputs divide log(2) by signed zero;
                    # infinite inputs form NaN in beta*x before the threshold test.
                    if threshold < 0.0:
                        finite = wide
                    else:
                        finite = T.cast(math.copysign(float("inf"), beta), "float32")
                    result = T.if_then_else(
                        T.abs(wide) == T.cast(float("inf"), "float32"),
                        T.cast(float("nan"), "float32"), finite,
                    )
                    y_ub[i] = T.cast(T.if_then_else(T.isnan(wide), wide, result), dtype=y_ub.dtype)
                else:
                    if beta == 1.0:
                        # beta is a trace-time constant in the cache key; when it is
                        # exactly 1.0 (the manifest default and every mlp-hidden* M4
                        # workload) ``scaled = wide * 1.0`` and ``sp / 1.0`` are exact
                        # identities, so the multiply and the per-element float32
                        # division are dropped. msprof_detail shows this kernel is
                        # vector-pipe bound (aiv_vec ~0.89-0.95, mte2/mte3 ~0.45/0.19),
                        # so removing the division -- the costliest non-transcendental
                        # vector op -- reduces the actual bottleneck, not just the
                        # instruction count (dev_process.md step-3 guidance; measured
                        # in resume_20261019/workers/SoftplusFwdOp).
                        scaled = wide
                        e = T.exp(scaled)
                        sp = T.log(one + e)
                    else:
                        b = T.Cast("float32", beta)
                        scaled = wide * b
                        e = T.exp(scaled)
                        sp = T.log(one + e) / b
                    y_ub[i] = T.if_then_else(scaled > t, x_ub[i], T.Cast(x_ub.dtype, sp))

    @functools.lru_cache(maxsize=64)
    def _build_tiled(
        n_total: int,
        dtype_str: str,
        num_blocks: int,
        tile_elems: int,
        num_stages: int,
        beta_key: str,
        threshold_key: str,
        simt_threads: int = _MAX_THREADS,
        coalesced_width: int = 0,
    ):
        """Persistent-block, pipelined main loop over an exact multiple of ``tile_elems``.

        Requires ``n_total`` to divide ``num_blocks * tile_elems`` exactly; the caller
        (``_build_softplus_kernel``) only reaches this builder for a shape that does (schedule
        doc §3.4: every manifest case and M3 fixture this op is ever asked to run divides evenly
        for both dtype classes), so no tail handling is implemented here.
        """
        beta, threshold = float.fromhex(beta_key), float.fromhex(threshold_key)
        block_span = num_blocks * tile_elems
        assert n_total % block_span == 0, (
            f"_build_tiled requires n_total ({n_total}) to divide num_blocks*tile_elems "
            f"({block_span}) exactly; the caller routes other shapes elsewhere"
        )
        iterations = n_total // block_span

        @T.prim_func
        def main(X: T.Tensor((n_total,), dtype_str), Y: T.Tensor((n_total,), dtype_str)):
            with T.Kernel(num_blocks) as bx:
                x_ub = T.alloc_shared((tile_elems,), dtype_str)
                y_ub = T.alloc_shared((tile_elems,), dtype_str)
                T.annotate_buffer_versions({x_ub: num_stages, y_ub: num_stages})
                for step in T.Pipelined(iterations, num_stages=num_stages):
                    begin = (step * num_blocks + bx) * tile_elems
                    T.copy(X[begin : begin + tile_elems], x_ub)
                    _softplus_body(
                        x_ub,
                        y_ub,
                        tile_elems,
                        threads=simt_threads,
                        beta=beta,
                        threshold=threshold,
                        coalesced_width=coalesced_width,
                    )
                    T.copy(y_ub, Y[begin : begin + tile_elems])

        return tilelang.compile(main, target="ascend", out_idx=-1)

    @functools.lru_cache(maxsize=128)
    def _build_with_tail(n_total: int, dtype_str: str, beta_key: str, threshold_key: str):
        """Bounded UB tail; only valid elements cross the global-memory boundary."""
        beta, threshold = float.fromhex(beta_key), float.fromhex(threshold_key)
        tile = 8192
        stages = _NUM_STAGES
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
                            T.copy(X[begin:begin + tile], x)
                            _softplus_body(x, y, tile, min(tile, 2048), beta, threshold)
                            T.copy(y, Y[begin:begin + tile])
                if tail:
                    if bx == 0:
                        xt = T.alloc_shared((padded_tail,), dtype_str)
                        yt = T.alloc_shared((padded_tail,), dtype_str)
                        T.copy(X[full * tile:n_total], xt[:tail])
                        _softplus_body(xt, yt, padded_tail, min(padded_tail, 2048), beta, threshold)
                        T.copy(yt[:tail], Y[full * tile:n_total])
        return tilelang.compile(main, target="ascend", out_idx=-1)

    def _build_softplus_kernel(n_total: int, dtype_str: str, beta: float, threshold: float):
        """Use persistent tiles when divisible, otherwise the bounded tail kernel."""
        beta, threshold = _opmath_scalar(beta), _opmath_scalar(threshold)
        beta_key, threshold_key = beta.hex(), threshold.hex()
        simt_threads, coalesced_width = _layout_for_call(
            n_total, dtype_str, beta, threshold
        )
        tile_elems = _tile_elems_for_dtype(dtype_str)
        num_blocks = _calibrated_num_blocks()
        block_span = num_blocks * tile_elems
        if n_total % block_span == 0:
            return _build_tiled(
                n_total, dtype_str, num_blocks, tile_elems, _NUM_STAGES,
                beta_key, threshold_key, simt_threads, coalesced_width,
            )
        return _build_with_tail(n_total, dtype_str, beta_key, threshold_key)


class AscendSoftplusKernel(Kernel, SoftplusFwdInterface):
    """Ascend NPU softplus: ``y = log(1 + exp(beta*x)) / beta`` if ``beta*x <= threshold`` else
    ``x``, computed in fp32 and cast back to the storage dtype at the boundary.

    See ``docs/ascend950/dev/SoftplusFwdOp/round_1_schedule.md`` for the tiling design and
    ``round_1_result.md`` for the measurement trail. ``beta``/``threshold`` are op-construction
    compile-time constants baked into the trace (``SoftplusCall``'s dataclass fields), not tensor
    inputs or runtime scalars -- each distinct ``(beta, threshold)`` pair produces a separately
    ``functools.lru_cache``-memoized compiled kernel.

    Uses plain ``T.log`` (fp32-promoted) for every dtype, never ``T.log1p`` -- the natural
    softplus formula already is the cheap ``log(1+exp(x))`` composite that ``Log1pFwdOp``'s round
    1 had to discover by elimination (``T.log1p``'s compiled ``log1pf`` measured 2.3-2.8x more
    expensive than this composite on this toolchain); there is no path in this design that
    evaluates genuine ``log1p`` at all. ``T.__log`` (the CUDA-only fast-math dunder the reference
    CUDA kernel's ``log_for_output_precision`` helper gates fp16/bf16 through) does not exist in
    the Ascend dialect and is never referenced here.

    ``devices`` is NPU only; ``supported_archs`` is left ``None`` -- unlike the CUDA
    ``MultiInputElementwiseKernel``/``UnaryKernel`` family this class does not inherit, there is
    no Ascend SoC-version gate here to replace the CUDA SM-version one (``Kernel._check_arch``
    skips the check entirely when ``supported_archs is None``).

    No ``refusal()`` override: the manifest admits exactly ``{float16, bfloat16, float32}`` for
    this op's ``T`` (no integer dtype in the union) and ``SoftplusFwdOp.kernel_types = {
    "softplus": SoftplusFwdKernel}`` has no integer-dtype sibling kernel to route around, so the
    base class's permissive default (serves every call) is already correct and complete.

    Tolerance (inherited from the shared elementwise test tolerance table, not re-derived here):
    fp16 rtol/atol 1e-3, bf16 1.6e-2, fp32 1e-5 (``SoftplusFwdOp`` is not in any of
    ``ElementwiseWorkload.verification``'s special-cased op-name lists, so it gets the default
    ``Exact()``).
    """

    devices: ClassVar[frozenset[str]] = frozenset({"npu"})
    supported_archs = None

    @classmethod
    def entry_for(cls, call: SoftplusCall) -> Entry:
        return call, lambda: cls(call.n_total, call.dtype, call.beta, call.threshold)

    def __init__(
        self, n_total: int, dtype: torch.dtype, beta: float = 1.0, threshold: float = 20.0
    ) -> None:
        if _ASCEND_IMPORT_ERROR is not None:
            raise ImportError(
                "AscendSoftplusKernel requires tilelang.ascend, which failed to import: "
                f"{_ASCEND_IMPORT_ERROR}"
            ) from _ASCEND_IMPORT_ERROR
        super().__init__()
        self.n_total = n_total
        self.dtype = dtype
        self.beta = _opmath_scalar(beta)
        self.threshold = _opmath_scalar(threshold)
        self.kernel = (_build_softplus_kernel(n_total, self.dtype_str, beta, threshold)
                       if n_total else None)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Apply softplus elementwise; nothing is written in place.

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
            raise ValueError("input dtype differs from the compiled specialization")
        flat_input = input.contiguous().reshape(-1)
        if flat_input.numel() != self.n_total:
            raise ValueError(
                f"{type(self).__name__} was built for n_total={self.n_total}, got "
                f"{flat_input.numel()} elements"
            )
        if not self.n_total:
            return torch.empty_like(input)
        flat_output = self.kernel(flat_input)
        return flat_output.reshape(input.shape)


try:
    from tileops.backend import register_kernel_type

    register_kernel_type("SoftplusFwdOp", "ascend", AscendSoftplusKernel)
except ImportError:
    pass
