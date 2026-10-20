"""Ascend 950 kernels, registered through the backend entry point."""

from tileops.kernels.ascend import elementwise, gemm, reduction

__all__ = ["elementwise", "gemm", "reduction"]
