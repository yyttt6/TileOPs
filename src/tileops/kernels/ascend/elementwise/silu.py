"""Ascend 950 implementations for SiluFwdOp."""

import functools
from typing import ClassVar

import tilelang
import tilelang.ascend.language as T
import torch

from tileops.backend import register_kernel_type
from tileops.kernels.elementwise.call_spec import ElementwiseCall, UnaryElementwiseFwdInterface
from tileops.kernels.kernel_base import Entry, Kernel

__all__ = ["AscendSiluKernel"]
_UB_BUDGET_BYTES = 216 * 1024
_NUM_STAGES = 3
_FALLBACK_VECTOR_CORES = 48
_MAX_THREADS = 2048


def _silu_op(x):
    """``x / (1 + exp(-x))``, computed in fp32; mirrors ``SiluFwdKernel.op_func``'s algebra."""
    one = T.cast(1.0, "float32")
    wide = T.cast(x, "float32")
    return wide / (one + T.exp(-wide))


def _num_vector_cores() -> int:
    """Best-effort AIV (vector) core count of the current NPU device."""
    try:
        prop = torch.npu.get_device_properties(torch.npu.current_device())
        cube_cores = getattr(prop, "cube_core_num", None)
        if cube_cores:
            return int(cube_cores) * 2
    except Exception:
        pass
    return _FALLBACK_VECTOR_CORES


def _tile_elems_for_dtype(dtype_str: str) -> int:
    """The largest power-of-two element count keeping the multi-tile path's UB use in budget."""
    itemsize = getattr(torch, dtype_str).itemsize
    max_elems = _UB_BUDGET_BYTES // (2 * _NUM_STAGES * itemsize)
    tile = 1
    while tile * 2 <= max_elems:
        tile *= 2
    return tile


def _choose_num_blocks(num_full_tiles: int, vector_cores: int) -> int:
    """The largest divisor of ``num_full_tiles`` that does not exceed ``vector_cores``."""
    cap = max(1, min(num_full_tiles, vector_cores))
    for candidate in range(cap, 0, -1):
        if num_full_tiles % candidate == 0:
            return candidate
    return 1


@functools.lru_cache(maxsize=None)
def _build_single_block(extent: int, dtype_str: str):
    """One block, one untiled GM->UB->compute->UB->GM pass over ``extent`` elements."""

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
    divides evenly by ``num_cores``."""
    assert extent % num_cores == 0, (
        f"_build_decode_multicore requires extent={extent} to divide evenly by num_cores={num_cores}"
    )
    block = extent // num_cores
    threads = min(block, _MAX_THREADS)

    @T.prim_func
    def main(X: T.Tensor((extent,), dtype_str), Y: T.Tensor((extent,), dtype_str)):
        with T.Kernel(num_cores) as core_id:
            x_tile = T.alloc_shared((block,), dtype_str)
            y_tile = T.alloc_shared((block,), dtype_str)
            begin = core_id * block
            T.copy(X[begin : begin + block], x_tile)
            with T.SimtVF(threads=threads):
                for i in T.Parallel(block):
                    y_tile[i] = _silu_op(x_tile[i])
            T.copy(y_tile, Y[begin : begin + block])

    return tilelang.compile(main, target="ascend", out_idx=-1)


@functools.lru_cache(maxsize=None)
def _build_tiled(main_n: int, dtype_str: str, num_blocks: int, tile_elems: int):
    """Persistent-block, pipelined main loop over exact multiples of ``tile_elems``."""

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
                with T.SimtVF(threads=min(tile_elems, _MAX_THREADS)):
                    for i in T.Parallel(tile_elems):
                        y_tile[i] = _silu_op(x_tile[i])
                T.copy(y_tile, Y[begin : begin + tile_elems])

    return tilelang.compile(main, target="ascend", out_idx=-1)


class _SiluRunner:
    """Dispatches one flat ``(n_total,)`` NPU tensor to one or two compiled Ascend programs."""

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
    """Build (or reuse, by ``(n_total, dtype_str)``) the Ascend SiLU program."""
    if n_total <= 0:
        raise ValueError(f"n_total must be positive, got {n_total}")
    tile_elems = _tile_elems_for_dtype(dtype_str)
    if n_total <= tile_elems:
        num_cores = _choose_num_blocks(n_total, _num_vector_cores())
        if num_cores > 1:
            return _SiluRunner(
                _build_decode_multicore(n_total, dtype_str, num_cores), None, n_total
            )
        return _SiluRunner(_build_single_block(n_total, dtype_str), None, n_total)
    num_full_tiles = n_total // tile_elems
    tail = n_total - num_full_tiles * tile_elems
    main_n = num_full_tiles * tile_elems
    num_blocks = _choose_num_blocks(num_full_tiles, _num_vector_cores())
    main_kernel = _build_tiled(main_n, dtype_str, num_blocks, tile_elems)
    tail_kernel = _build_single_block(tail, dtype_str) if tail else None
    return _SiluRunner(main_kernel, tail_kernel, main_n)


class AscendSiluKernel(Kernel, UnaryElementwiseFwdInterface):
    """Ascend NPU SiLU: ``x / (1 + exp(-x))`` in fp32, mirroring ``SiluFwdKernel.op_func``."""

    devices: ClassVar[frozenset[str]] = frozenset({"npu"})
    supported_archs = None

    @classmethod
    def entry_for(cls, call: ElementwiseCall) -> Entry:
        return (call, lambda: cls(call.n_total, call.dtype))

    def __init__(self, n_total: int, dtype: torch.dtype) -> None:
        super().__init__()
        self.n_total = n_total
        self.dtype = dtype
        self.kernel = _build_silu_kernel(n_total, self.dtype_str)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Apply SiLU elementwise; nothing is written in place."""
        if input.device.type != "npu":
            raise ValueError(
                f"{type(self).__name__} is an Ascend NPU kernel; got {input.device}. Another target's backend serves other devices."
            )
        flat_input = input.reshape(-1)
        if flat_input.numel() != self.n_total:
            raise ValueError(
                f"{type(self).__name__} was built for n_total={self.n_total}, got {flat_input.numel()} elements"
            )
        flat_output = self.kernel(flat_input)
        return flat_output.reshape(input.shape)


register_kernel_type("SiluFwdOp", "ascend", AscendSiluKernel)
