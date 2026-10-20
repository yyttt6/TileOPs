"""Ascend 950 implementations for HardswishFwdOp."""

import functools
from typing import ClassVar

import tilelang
import tilelang.ascend.language as T
import torch
from tilelang.carver.arch.driver import get_num_vector_cores

from tileops.backend import register_kernel_type
from tileops.kernels.elementwise.call_spec import ElementwiseCall, UnaryElementwiseFwdInterface
from tileops.kernels.kernel_base import Entry, Kernel

__all__ = ["AscendHardswishKernel"]
_UB_BUDGET_BYTES = 216 * 1024
_TILE_FP32 = 8192
_TILE_16BIT = 16384
_NUM_STAGES = 2
_TUNED_SCHEDULE = {
    (9633792, "float16"): (64, 12288, 4),
    (9633792, "bfloat16"): (64, 12288, 4),
    (6021120, "float16"): (70, 16384, 3),
    (6021120, "bfloat16"): (70, 16384, 3),
}
_TUNED_UNROLL = {
    (9633792, "float16"): 4,
    (9633792, "bfloat16"): 4,
    (6021120, "float16"): 2,
    (6021120, "bfloat16"): 1,
}


def _tile_for_dtype(dtype_str: str) -> int:
    """Main-loop tile for ``dtype_str``: wide (16384) for 16-bit, 8192 for float32 (UB budget)."""
    return _TILE_FP32 if dtype_str == "float32" else _TILE_16BIT


_FALLBACK_VECTOR_CORES = 72
_MAX_THREADS = 2048


@functools.lru_cache(maxsize=1)
def _num_vector_cores() -> int:
    """Best-effort AIV (vector) core count of the current NPU device."""
    try:
        queried = get_num_vector_cores(device_id=0)
    except Exception:
        queried = None
    if not queried:
        return _FALLBACK_VECTOR_CORES
    return int(queried)


def _num_blocks_for_n_total(n_total: int, vector_cores: int) -> int:
    """The largest divisor of ``n_total`` itself that does not exceed ``vector_cores``."""
    cap = max(1, min(n_total, vector_cores))
    for candidate in range(cap, 0, -1):
        if n_total % candidate == 0:
            return candidate
    return 1


@T.macro
def _hardswish_body_native(x_ub, y_ub, n: int, threads: int) -> None:
    """``y_ub[:n] = x_ub[:n] * clamp(x_ub[:n] + 3, 0, 6) / 6``, native storage dtype."""
    with T.SimtVF(threads=threads):
        for i in T.Parallel(n):
            dt = x_ub.dtype
            three = T.cast(3.0, dt)
            six = T.cast(6.0, dt)
            zero = T.cast(0.0, dt)
            one_sixth = T.cast(1.0 / 6.0, dt)
            neg_three = T.cast(-3.0, dt)
            neg_zero = T.cast(-0.0, dt)
            clamped = T.min(T.max(x_ub[i] + three, zero), six)
            normal = x_ub[i] * clamped * one_sixth
            y_ub[i] = T.if_then_else(x_ub[i] < neg_three, neg_zero, normal)


@T.macro
def _hardswish_body_promoted(x_ub, y_ub, n: int, threads: int) -> None:
    """``y_ub[:n] = x_ub[:n] * clamp(x_ub[:n] + 3, 0, 6) / 6``, fp32-promoted."""
    with T.SimtVF(threads=threads):
        for i in T.Parallel(n):
            wide = T.cast(x_ub[i], "float32")
            three = T.cast(3.0, "float32")
            six = T.cast(6.0, "float32")
            zero = T.cast(0.0, "float32")
            one_sixth = T.cast(1.0 / 6.0, "float32")
            neg_three = T.cast(-3.0, "float32")
            neg_zero = T.cast(-0.0, "float32")
            clamped = T.min(T.max(wide + three, zero), six)
            normal = wide * clamped * one_sixth
            selected = T.if_then_else(wide < neg_three, neg_zero, normal)
            y_ub[i] = T.cast(selected, y_ub.dtype)


@T.macro
def _hardswish_body_vf(x_ub, y_ub, n: int, dtype_str: str, vf_unroll: int = 1) -> None:
    """Explicit-SIMD Hardswish over one full 64-lane-aligned tile, fp32 arithmetic."""
    with T.SimdVF():
        three = T.cast(3.0, "float32")
        six = T.cast(6.0, "float32")
        zero = T.cast(0.0, "float32")
        one_sixth = T.cast(1.0 / 6.0, "float32")
        neg_three = T.cast(-3.0, "float32")
        neg_zero = T.cast(-0.0, "float32")
        zero_v = T.simd.vdup(zero, "float32")
        six_v = T.simd.vdup(six, "float32")
        neg_zero_v = T.simd.vdup(neg_zero, "float32")
        for g in T.serial(n // 64 // vf_unroll):
            for s_unroll in T.unroll(vf_unroll, explicit=True):
                base = (g * vf_unroll + s_unroll) * 64
                if dtype_str == "float32":
                    wide = T.simd.vld(x_ub[base])
                else:
                    wide = T.simd.vcvt(T.simd.vld(x_ub[base], dist="UNPK_B16"), "float32", part=0)
                xp3 = T.simd.vadds(wide, three)
                low = T.simd.vsel(zero_v, xp3, T.simd.vcmps(xp3, zero, op="lt"))
                clamped = T.simd.vsel(six_v, low, T.simd.vcmps(low, six, op="gt"))
                normal = T.simd.vmuls(T.simd.vmul(wide, clamped), one_sixth)
                result = T.simd.vsel(neg_zero_v, normal, T.simd.vcmps(wide, neg_three, op="lt"))
                if dtype_str == "float32":
                    T.simd.vsts(y_ub[base], result)
                else:
                    T.simd.vsts(y_ub[base], T.simd.vcvt(result, dtype_str), dist="PK_B32")


@functools.lru_cache(maxsize=128)
def _make_hardswish_kernel(
    n_total: int, dtype_str: str, num_blocks: int, tile: int, num_stages: int, vf_unroll: int = 1
):
    body = _hardswish_body_native if dtype_str == "float32" else _hardswish_body_promoted
    use_vf_main = dtype_str != "float32" and tile % 64 == 0
    if vf_unroll not in (1, 2, 4):
        raise ValueError(f"vf_unroll must be 1/2/4, got {vf_unroll}")
    if use_vf_main and tile % (64 * vf_unroll) != 0:
        raise ValueError(f"tile {tile} not divisible by 64*vf_unroll={64 * vf_unroll}")
    per_block = n_total // num_blocks
    full_tiles, remainder = divmod(per_block, tile)

    @T.prim_func
    def main(X: T.Tensor((n_total,), dtype_str), Y: T.Tensor((n_total,), dtype_str)):
        with T.Kernel(num_blocks) as bx:
            base = bx * per_block
            if full_tiles:
                x_ub = T.alloc_shared((tile,), dtype_str)
                y_ub = T.alloc_shared((tile,), dtype_str)
                T.annotate_buffer_versions({x_ub: num_stages, y_ub: num_stages})
                for step in T.Pipelined(full_tiles, num_stages=num_stages):
                    begin = base + step * tile
                    T.copy(X[begin : begin + tile], x_ub)
                    if use_vf_main:
                        _hardswish_body_vf(x_ub, y_ub, tile, dtype_str, vf_unroll)
                    else:
                        body(x_ub, y_ub, tile, threads=min(_MAX_THREADS, tile))
                    T.copy(y_ub, Y[begin : begin + tile])
            if remainder:
                xt_ub = T.alloc_shared((remainder,), dtype_str)
                yt_ub = T.alloc_shared((remainder,), dtype_str)
                begin = base + full_tiles * tile
                T.copy(X[begin : begin + remainder], xt_ub)
                body(xt_ub, yt_ub, remainder, threads=min(_MAX_THREADS, remainder))
                T.copy(yt_ub, Y[begin : begin + remainder])

    return tilelang.compile(main, target="ascend", out_idx=-1)


def _build_hardswish_kernel(n_total: int, dtype_str: str):
    tuned = _TUNED_SCHEDULE.get((n_total, dtype_str))
    if tuned is not None:
        num_blocks, tile, num_stages = tuned
    else:
        num_blocks = _num_blocks_for_n_total(n_total, _num_vector_cores())
        tile = _tile_for_dtype(dtype_str)
        num_stages = _NUM_STAGES
    vf_unroll = _TUNED_UNROLL.get((n_total, dtype_str), 1)
    return _make_hardswish_kernel(n_total, dtype_str, num_blocks, tile, num_stages, vf_unroll)


class AscendHardswishKernel(Kernel, UnaryElementwiseFwdInterface):
    """Compute ``y = x * clamp(x + 3, 0, 6) / 6`` in fp32 to avoid fp16 overflow."""

    devices: ClassVar[frozenset[str]] = frozenset({"npu"})
    supported_archs = None

    @classmethod
    def entry_for(cls, call: ElementwiseCall) -> Entry:
        return (call, lambda: cls(call.n_total, call.dtype))

    def __init__(self, n_total: int, dtype: torch.dtype) -> None:
        super().__init__()
        self.n_total = n_total
        self.dtype = dtype
        self.kernel = _build_hardswish_kernel(n_total, self.dtype_str)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Apply Hardswish elementwise; nothing is written in place."""
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


register_kernel_type("HardswishFwdOp", "ascend", AscendHardswishKernel)
