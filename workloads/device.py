"""Process-wide device selection shared by workloads, tests, and benchmarks."""

from __future__ import annotations

import os

import torch


def _default_device() -> str:
    requested = os.environ.get("TILEOPS_DEVICE")
    if requested:
        return requested
    npu = getattr(torch, "npu", None)
    if npu is not None and callable(getattr(npu, "is_available", None)):
        try:
            if npu.is_available():
                return "npu"
        except Exception:
            pass
    return "cuda"


DEVICE = _default_device()

__all__ = ["DEVICE"]
