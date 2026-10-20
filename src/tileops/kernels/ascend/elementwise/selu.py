"""Ascend 950 implementations for SeluFwdOp."""

import functools
from typing import ClassVar

import tilelang
import tilelang.ascend.language as T
import torch
from tilelang.carver.arch.driver import get_num_vector_cores

from tileops.backend import register_kernel_type
from tileops.kernels.elementwise.call_spec import ElementwiseCall, UnaryElementwiseFwdInterface
from tileops.kernels.kernel_base import Entry, Kernel

__all__ = ["AscendSeluKernel"]
_SELU_ALPHA = 1.6732632423543772
_SELU_SCALE = 1.0507009873554805
_UB_BUDGET_BYTES = 216 * 1024
_NUM_STAGES = 3
_FALLBACK_VECTOR_CORES = 72
_MAX_THREADS = 2048
_TILE_ELEMS_16BIT = 16384
_MANIFEST_TILING: dict[tuple[int, str], tuple[int, int, int]] = {}


def _num_vector_cores() -> int:
    """Best-effort AIV (vector) core count of device 0."""
    try:
        queried = get_num_vector_cores(device_id=0)
    except Exception:
        queried = None
    if not queried:
        return _FALLBACK_VECTOR_CORES
    return int(queried)


def _tile_elems_for_dtype(dtype_str: str) -> int:
    """Element count of one UB tile for the persistent path, by dtype."""
    if getattr(torch, dtype_str).itemsize >= 4:
        return 8192
    return _TILE_ELEMS_16BIT


def _choose_num_blocks(num_full_tiles: int, vector_cores: int) -> int:
    """Largest divisor of ``num_full_tiles`` not exceeding ``vector_cores``."""
    cap = max(1, min(num_full_tiles, vector_cores))
    for candidate in range(cap, 0, -1):
        if num_full_tiles % candidate == 0:
            return candidate
    return 1


@T.macro
def _selu_body(x_ub, y_ub, n: int, threads: int) -> None:
    """``y_ub[:n] = selu(x_ub[:n])``, compiler-vectorized over ``threads`` lanes."""
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
    """One block, one untiled GM->UB->compute->UB->GM pass over ``extent`` elements."""

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
    """Persistent-block, pipelined main loop over an exact multiple of ``tile_elems``."""
    block_span = num_blocks * tile_elems
    assert main_n % block_span == 0, (
        f"_build_tiled requires main_n ({main_n}) to divide num_blocks*tile_elems ({block_span}) exactly; the caller passes only the full-tile prefix"
    )
    assert num_stages >= 1, f"num_stages must be positive, got {num_stages}"
    assert 2 * tile_elems * num_stages * getattr(torch, dtype_str).itemsize <= _UB_BUDGET_BYTES, (
        f"UB budget exceeded: 2*{tile_elems}*{num_stages}*{getattr(torch, dtype_str).itemsize} > {_UB_BUDGET_BYTES}"
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
    """Dispatches one flat ``(n_total,)`` NPU tensor to one or two compiled Ascend programs."""

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
    """Build (memoized by ``(n_total, dtype_str)``) the Ascend SELU runner for ``n_total``."""
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
    """SELU on Ascend NPUs: ``y = scale * (x if x >= 0 else alpha * (exp(x) - 1))``."""

    devices: ClassVar[frozenset[str]] = frozenset({"npu"})
    supported_archs = None

    @classmethod
    def entry_for(cls, call: ElementwiseCall) -> Entry:
        return (call, lambda: cls(call.n_total, call.dtype))

    def __init__(self, n_total: int, dtype: torch.dtype) -> None:
        if n_total < 0:
            raise ValueError(f"n_total must be non-negative, got {n_total}")
        super().__init__()
        self.n_total = n_total
        self.dtype = dtype
        self.kernel = _build_selu_kernel(n_total, self.dtype_str) if n_total else None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Apply SELU elementwise; nothing is written in place."""
        if input.device.type != "npu":
            raise ValueError(
                f"{type(self).__name__} is an Ascend NPU kernel; got {input.device}. Another target's backend serves other devices."
            )
        flat_input = input.reshape(-1)
        if flat_input.numel() != self.n_total:
            raise ValueError(
                f"{type(self).__name__} was built for n_total={self.n_total}, got {flat_input.numel()} elements"
            )
        if self.n_total == 0:
            return input.clone()
        flat_output = self.kernel(flat_input)
        return flat_output.reshape(input.shape)


register_kernel_type("SeluFwdOp", "ascend", AscendSeluKernel)
