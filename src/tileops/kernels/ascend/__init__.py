"""Ascend 950 kernels, registered through the backend entry point."""

from tileops.kernels.ascend import attention, elementwise, gemm, reduction

__all__ = ["attention", "elementwise", "gemm", "reduction"]
