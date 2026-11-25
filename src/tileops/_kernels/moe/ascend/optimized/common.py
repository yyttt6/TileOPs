"""Tensor validation and per-device compilation for staged Ascend MoE kernels."""

from __future__ import annotations

import math

import torch

from tileops._kernels.buffer_utils import tensors_overlap

__all__ = ["AscendMoEKernel"]


def check_tensor(tensor, name, shape, dtype, device=None):
    """Validate the static tensor ABI without reading device values."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if tuple(tensor.shape) != tuple(shape) or tensor.dtype != dtype:
        raise ValueError(f"{name} must have shape {shape} and dtype {dtype}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if device is not None and tensor.device != device:
        raise ValueError(f"{name} must be on {device}")


def check_dtype(dtype):
    if dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("Ascend MoE operands must be float16 or bfloat16")


def check_scale(scale):
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("routed_scaling_factor must be finite and positive")


def check_activation(activation):
    if activation not in (None, "silu_and_mul", "gelu_and_mul", "swigluoai"):
        raise ValueError(f"unknown gated activation {activation!r}")


def check_out(out, shape, dtype, device, inputs):
    if out is None:
        return
    check_tensor(out, "out", shape, dtype, device)
    if any(t is not None and tensors_overlap(out, t) for t in inputs):
        raise ValueError("out must not overlap an input")


class AscendMoEKernel:
    """Compile an explicit-output TileLang program once per NPU device."""

    def __init__(self, num_cores):
        if num_cores <= 0:
            raise ValueError("num_cores must be positive")
        self.config = {"num_cores": num_cores}
        self._compiled = {}

    def compile(self, device=None):
        if device is None or device.type not in ("npu", "privateuseone"):
            raise ValueError("an explicit NPU device is required")
        if device not in self._compiled:
            import tilelang

            # Explicit outputs preserve the public out-buffer ABI. No CUDA Kernel base,
            # CUDA tuning/SM queries or default target are involved.
            with torch.npu.device(device):
                self._compiled[device] = tilelang.compile(
                    self.make_program(), target="ascend", out_idx=None
                )
        return self._compiled[device]
