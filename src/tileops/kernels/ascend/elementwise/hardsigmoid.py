"""Ascend 950 implementations for HardsigmoidFwdOp."""

import functools

import tilelang
import tilelang.ascend.language as T
import torch
from tilelang.carver.arch.driver import get_num_vector_cores

import tileops.backend
from tileops.kernels.elementwise.call_spec import ElementwiseCall, UnaryElementwiseFwdInterface
from tileops.kernels.kernel_base import Entry, Kernel

_FALLBACK_VECTOR_CORES = 72
_MAX_THREADS = 2048
_MANIFEST_NUM_BLOCKS = {7680: 24, 30720: 32}
_ONESHOT_MAX_CHUNK = 16384
_BOUNDED_TILE = 8192
_BOUNDED_NUM_STAGES = 2


@functools.lru_cache(maxsize=1)
def _calibrated_vector_cores() -> int:
    """This SKU's real vector-core count, live-queried once per process."""
    try:
        queried = get_num_vector_cores(device_id=0)
    except Exception:
        queried = None
    if not queried:
        return _FALLBACK_VECTOR_CORES
    return int(queried)


def _choose_fallback_num_blocks(n_total: int, vector_cores: int) -> int:
    """The largest divisor of ``n_total`` that does not exceed ``vector_cores``."""
    upper = min(n_total, vector_cores)
    for candidate in range(upper, 0, -1):
        if n_total % candidate == 0:
            return candidate
    return 1


def _num_blocks_for(n_total: int) -> int:
    manifest_choice = _MANIFEST_NUM_BLOCKS.get(n_total)
    if manifest_choice is not None:
        return manifest_choice
    return _choose_fallback_num_blocks(n_total, _calibrated_vector_cores())


def _dtype_str(dtype: torch.dtype) -> str:
    return str(dtype).split(".")[-1]


@T.macro
def _hardsigmoid_body(x_ub, y_ub, n: int, threads: int) -> None:
    """``y_ub[:n] = clamp(x_ub[:n] + 3, 0, 6) / 6``, NaN-propagating, in fp32 compute."""
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
    assert n_total % num_blocks == 0, (
        f"_make_hardsigmoid_oneshot requires num_blocks ({num_blocks}) to divide n_total ({n_total}) exactly; _num_blocks_for should only ever return a divisor"
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
    """Bounded-UB multi-block grid for a per-block chunk too large for one shot."""
    assert n_total % num_blocks == 0, (
        f"_make_hardsigmoid_bounded requires num_blocks ({num_blocks}) to divide n_total ({n_total}) exactly; _num_blocks_for should only ever return a divisor"
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
    num_blocks = _num_blocks_for(n_total)
    chunk = n_total // num_blocks
    if chunk <= _ONESHOT_MAX_CHUNK:
        return _make_hardsigmoid_oneshot(n_total, dtype, num_blocks)
    return _make_hardsigmoid_bounded(n_total, dtype, num_blocks, _BOUNDED_TILE, _BOUNDED_NUM_STAGES)


class AscendHardsigmoidKernel(Kernel, UnaryElementwiseFwdInterface):
    """Hardsigmoid on Ascend NPUs: ``y = clamp(x + 3, 0, 6) / 6``, NaN-propagating."""

    devices = frozenset({"npu"})

    @classmethod
    def entry_for(cls, call: ElementwiseCall) -> Entry:
        return (call, lambda: cls(call.n_total, call.dtype))

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
