"""Ascend 950 implementations for EluFwdOp."""

import functools
import math
from typing import ClassVar

import tilelang
import tilelang.ascend.language as T
import torch
from tilelang.carver.arch.driver import get_num_vector_cores

from tileops.backend import register_kernel_type
from tileops.kernels.elementwise.call_spec import EluCall, EluFwdInterface
from tileops.kernels.kernel_base import Entry, Kernel

__all__ = ["AscendEluKernel"]
_UB_BUDGET_BYTES = 216 * 1024
_NUM_STAGES = 3
_GRADED_SCHEDULE = {
    (8388608, "float16"): (64, 16384, 2, 1024, 1),
    (8388608, "bfloat16"): (64, 16384, 2, 1024, 1),
    (16777216, "float16"): (64, 16384, 2, 1024, 1),
    (16777216, "bfloat16"): (64, 16384, 2, 1024, 1),
}
_ALPHA1_KEY = float(1.0).hex()
_FALLBACK_VECTOR_CORES = 64
_MAX_THREADS = 2048


def _floor_pow2(n: int) -> int:
    """Largest power of two ``<= n`` (``n >= 1``)."""
    p = 1
    while p * 2 <= n:
        p *= 2
    return p


@functools.lru_cache(maxsize=1)
def _calibrated_num_blocks(fallback: int = _FALLBACK_VECTOR_CORES) -> int:
    """Persistent-grid block count for the large, bandwidth-bound tiled path."""
    try:
        queried = get_num_vector_cores(device_id=0)
    except Exception:
        queried = None
    if not queried:
        return fallback
    return _floor_pow2(int(queried))


_TILE_ELEMS_16BIT = 16384
_TILE_ELEMS_FP32 = 8192


def _tile_elems_for_dtype(dtype_str: str) -> int:
    """The per-dtype UB tile element count (``_TILE_ELEMS_16BIT`` / ``_TILE_ELEMS_FP32``)."""
    itemsize = getattr(torch, dtype_str).itemsize
    tile = _TILE_ELEMS_16BIT if itemsize == 2 else _TILE_ELEMS_FP32
    tile = _floor_pow2(tile)
    assert 2 * _NUM_STAGES * itemsize * tile <= _UB_BUDGET_BYTES, (
        f"tile_elems={tile} for itemsize={itemsize} at num_stages={_NUM_STAGES} needs {2 * _NUM_STAGES * itemsize * tile} B > UB budget {_UB_BUDGET_BYTES} B"
    )
    return tile


def _choose_num_blocks(num_full_tiles: int, cap: int) -> int:
    """Largest divisor of ``num_full_tiles`` not exceeding ``cap`` (always >= 1)."""
    hi = max(1, min(num_full_tiles, cap))
    for candidate in range(hi, 0, -1):
        if num_full_tiles % candidate == 0:
            return candidate
    return 1


def _dtype_str(dtype: torch.dtype) -> str:
    return str(dtype).split(".")[-1]


class _EluRunner:
    """Dispatch one flat ``(n_total,)`` NPU tensor to one or two compiled Ascend programs."""

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


@T.macro
def _elu_body(
    x_ub, y_ub, n: int, threads: int, alpha: float, zero_sign_bits: int, coalesced_width: int = 0
) -> None:
    """``y_ub[:n] = elu(x_ub[:n], alpha)``, compiler-vectorized over ``threads`` lanes."""
    with T.SimtVF(threads=threads):
        for i in T.Parallel(
            n, coalesced_width=None if coalesced_width == 0 else T.int32(coalesced_width)
        ):
            zero = T.cast(0, "float32")
            one = T.cast(1.0, "float32")
            a = T.cast(alpha, "float32")
            wide = T.cast(x_ub[i], "float32")
            if alpha > 0.0:
                base = T.exp(wide) - one
                base = T.if_then_else(wide < zero, T.if_then_else(base < zero, base, wide), base)
                neg = T.cast(a * base, y_ub.dtype)
                y_ub[i] = T.if_then_else(wide >= zero, x_ub[i], neg)
            elif alpha == 0.0:
                sign = T.reinterpret(wide, "uint32") & T.uint32(2147483648) ^ T.uint32(
                    zero_sign_bits
                )
                signed_zero = T.reinterpret(sign, "float32")
                y_ub[i] = T.if_then_else(
                    T.isnan(wide),
                    x_ub[i],
                    T.if_then_else(wide > zero, x_ub[i], T.cast(signed_zero, y_ub.dtype)),
                )
            else:
                base = T.if_then_else(wide == zero, wide, T.exp(wide) - one)
                base = T.if_then_else(wide < zero, T.if_then_else(base < zero, base, wide), base)
                neg = T.cast(a * base, y_ub.dtype)
                y_ub[i] = T.if_then_else(wide > zero, x_ub[i], neg)


@functools.lru_cache(maxsize=64)
def _build_single_block(n_total: int, dtype_str: str, alpha_key: str):
    """One block, one untiled GM->UB->compute->UB->GM pass over ``n_total`` elements."""
    alpha = float.fromhex(alpha_key)
    zero_sign_bits = 2147483648 if alpha == 0.0 and math.copysign(1.0, alpha) < 0 else 0

    @T.prim_func
    def main(X: T.Tensor((n_total,), dtype_str), Y: T.Tensor((n_total,), dtype_str)):
        with T.Kernel(1):
            x_ub = T.alloc_shared((n_total,), dtype_str)
            y_ub = T.alloc_shared((n_total,), dtype_str)
            T.copy(X, x_ub)
            _elu_body(
                x_ub,
                y_ub,
                n_total,
                threads=min(n_total, _MAX_THREADS),
                alpha=alpha,
                zero_sign_bits=zero_sign_bits,
            )
            T.copy(y_ub, Y)

    return tilelang.compile(main, target="ascend", out_idx=-1)


@functools.lru_cache(maxsize=64)
def _build_tiled(
    n_total: int,
    dtype_str: str,
    num_blocks: int,
    tile_elems: int,
    num_stages: int,
    alpha_key: str,
    simt_threads: int = _MAX_THREADS,
    coalesced_width: int = 0,
):
    """Persistent-block, pipelined main loop over an exact multiple of ``tile_elems``."""
    alpha = float.fromhex(alpha_key)
    zero_sign_bits = 2147483648 if alpha == 0.0 and math.copysign(1.0, alpha) < 0 else 0
    block_span = num_blocks * tile_elems
    assert n_total % block_span == 0, (
        f"_build_tiled requires n_total ({n_total}) to divide num_blocks*tile_elems ({block_span}) exactly; the caller routes other shapes elsewhere"
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
                _elu_body(
                    x_ub,
                    y_ub,
                    tile_elems,
                    threads=simt_threads,
                    alpha=alpha,
                    zero_sign_bits=zero_sign_bits,
                    coalesced_width=coalesced_width,
                )
                T.copy(y_ub, Y[begin : begin + tile_elems])

    return tilelang.compile(main, target="ascend", out_idx=-1)


def _build_elu_kernel(n_total: int, dtype_str: str, alpha: float) -> _EluRunner:
    """Build (or reuse, via the sub-builders' ``lru_cache``) the Ascend ELU program(s) for
    any positive ``n_total``, returned as an ``_EluRunner``."""
    if n_total <= 0:
        raise ValueError(f"n_total must be positive, got {n_total}")
    alpha_key = float(alpha).hex()
    tile_elems = _tile_elems_for_dtype(dtype_str)
    if n_total < tile_elems:
        return _EluRunner(_build_single_block(n_total, dtype_str, alpha_key), None, n_total)
    num_blocks = _calibrated_num_blocks()
    block_span = num_blocks * tile_elems
    if n_total % block_span == 0:
        schedule = _GRADED_SCHEDULE.get((n_total, dtype_str)) if alpha_key == _ALPHA1_KEY else None
        if schedule is not None:
            gen_blocks, gen_tile, gen_stages = schedule[:3]
            layout = tuple(schedule[3:])
            itemsize = 2 if dtype_str in ("float16", "bfloat16") else 4
            assert n_total % (gen_blocks * gen_tile) == 0
            assert 2 * gen_stages * gen_tile * itemsize <= _UB_BUDGET_BYTES
            main = _build_tiled(
                n_total, dtype_str, gen_blocks, gen_tile, gen_stages, alpha_key, *layout
            )
            return _EluRunner(main, None, n_total)
        main = _build_tiled(n_total, dtype_str, num_blocks, tile_elems, _NUM_STAGES, alpha_key)
        return _EluRunner(main, None, n_total)
    num_full_tiles = n_total // tile_elems
    tail = n_total - num_full_tiles * tile_elems
    main_n = num_full_tiles * tile_elems
    main_kernel = None
    if num_full_tiles:
        gen_blocks = _choose_num_blocks(num_full_tiles, num_blocks)
        main_kernel = _build_tiled(
            main_n, dtype_str, gen_blocks, tile_elems, _NUM_STAGES, alpha_key
        )
    tail_kernel = _build_single_block(tail, dtype_str, alpha_key) if tail else None
    return _EluRunner(main_kernel, tail_kernel, main_n)


class AscendEluKernel(Kernel, EluFwdInterface):
    """ELU on Ascend NPUs: ``y = x if x > 0 else alpha * (exp(x) - 1)``."""

    devices: ClassVar[frozenset[str]] = frozenset({"npu"})
    supported_archs = None

    @classmethod
    def entry_for(cls, call: EluCall) -> Entry:
        return (call, lambda: cls(call.n_total, call.dtype, call.alpha))

    def __init__(self, n_total: int, dtype: torch.dtype, alpha: float = 1.0) -> None:
        super().__init__()
        self.n_total = n_total
        self.dtype = dtype
        self.alpha = float(alpha)
        if n_total < 0:
            raise ValueError(f"n_total must be non-negative, got {n_total}")
        self.kernel = (
            _build_elu_kernel(n_total, self.dtype_str, self.alpha) if n_total > 0 else None
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Apply ELU elementwise; nothing is written in place."""
        if input.device.type != "npu":
            raise ValueError(
                f"{type(self).__name__} is an Ascend NPU kernel; got {input.device}. Another target's backend serves other devices."
            )
        if input.numel() != self.n_total:
            raise ValueError(
                f"{type(self).__name__} was built for n_total={self.n_total}, got {input.numel()} elements"
            )
        if self.n_total == 0:
            return torch.empty_like(input)
        flat_input = input.reshape(-1)
        flat_output = self.kernel(flat_input)
        return flat_output.reshape(input.shape)


register_kernel_type("EluFwdOp", "ascend", AscendEluKernel)
