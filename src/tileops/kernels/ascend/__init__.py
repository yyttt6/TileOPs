"""Ascend 950 kernels, registered through the backend entry point."""

from tileops.kernels.ascend import elementwise, fft, gemm, reduction

__all__ = ["elementwise", "fft", "gemm", "reduction"]
