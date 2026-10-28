"""Dispatch the Ascend 950 16-bit GEMM schedules to ``GemmFwdOp``."""

import functools
from typing import Optional

import torch
from tilelang.carver.arch.driver import get_num_cube_cores

import tileops.backend
from tileops.kernels.ascend.gemm.persistent import SCHEDULES, compile_schedule
from tileops.kernels.gemm import GemmCall, GemmFwdInterface
from tileops.kernels.kernel_base import Kernel

__all__ = ["AscendGemmKernel"]


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).split(".")[-1]


@functools.lru_cache(maxsize=32)
def _make_gemm_kernel(m: int, n: int, k: int, dtype: str, device_index: int):
    """Compile the schedule serving ``(m, n, k)`` and ``dtype`` once per call shape."""
    name = torch.npu.get_device_name(device_index)
    if "Ascend950" not in name:
        raise RuntimeError(f"These GEMM schedules require Ascend 950, got {name}")
    # Every schedule keeps one persistent block per Cube core of the measured device.
    cube_cores = get_num_cube_cores(device_id=device_index or 0)
    if cube_cores != 32:
        raise RuntimeError(f"These GEMM schedules are measured on 32 Cube cores, got {cube_cores}")
    return compile_schedule(m, n, k, dtype)


class AscendGemmKernel(Kernel, GemmFwdInterface):
    """Shape-specialized 16-bit GEMM schedules for the Ascend 950 matrix unit.

    Each schedule serves exactly one manifest case, named by ``(M, N, K)``, dtype
    and layout in :mod:`tileops.kernels.ascend.gemm.persistent`. All but
    ``square-1k`` store ``b`` as ``(N, K)``; ``square-1k`` stores it as ``(K, N)``.
    The contraction accumulates in float32 and writes the result in the operand
    dtype.
    """

    devices = frozenset({"npu"})

    @classmethod
    def refusal(cls, call: GemmCall) -> Optional[str]:
        if call.dtype not in (torch.bfloat16, torch.float16):
            return "requires float16 or bfloat16 operands"
        if call.trans_a:
            return "requires trans_a=False"
        dtype = _dtype_name(call.dtype)
        schedule = SCHEDULES.get((call.m, call.n, call.k, dtype))
        if schedule is None:
            return f"no schedule for M={call.m}, N={call.n}, K={call.k}, dtype={dtype}"
        if call.trans_b != schedule[1]:
            return (
                f"the M={call.m}, N={call.n}, K={call.k}, dtype={dtype} schedule "
                f"requires trans_b={schedule[1]}"
            )
        return super().refusal(call)

    def __init__(self, call: GemmCall):
        device_index = call.device.index
        super().__init__(device_index=device_index)
        self._compiled_fn = _make_gemm_kernel(
            call.m, call.n, call.k, _dtype_name(call.dtype), device_index
        )

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self._compiled_fn(a, b)


tileops.backend.register_kernel_type("GemmFwdOp", "ascend", AscendGemmKernel)
