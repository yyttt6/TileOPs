"""Ascend log with shape-independent numerical semantics and tiled tails."""

import functools

import torch

from tileops.kernels.ascend.elementwise._logroots import LogRootsKernel, T
from tileops.kernels.elementwise.call_spec import UnaryElementwiseFwdInterface

__all__ = ["AscendLogKernel"]


if T is not None:
    import tilelang

    @functools.lru_cache(maxsize=8)
    def _lookup_table(device="npu"):
        # All nonnegative patterns, followed by negative zero and negative-subnormal NaNs.
        # Every bit pattern above negative zero has a NaN logarithm.
        # No runtime input values participate in this constant construction.
        values = (
            torch.arange(32896, dtype=torch.int32, device="cpu")
            .to(torch.int16)
            .view(torch.bfloat16)
        )
        return torch.log(values.double()).to(torch.bfloat16).to(device)

    @T.macro
    def _lookup_body(a, c, lookup, width: int):
        with T.SimdVF():
            mask = T.simd.pset(16)
            for i in range(width // 128):
                value = T.simd.vld(a[i * 128], dist="NORM_B16")
                bits = T.reinterpret(value, "uint16x128")
                index = T.simd.vmins(bits, T.uint16(32769), mask)
                result = T.simd.vgather2(lookup[0], index, mask)
                T.simd.vsts(c[i * 128], result, mask, dist="NORM_B16")

    @functools.lru_cache(maxsize=128)
    def _build_lookup(n, tile=8192, blocks=64, stages=3):
        if n <= 0 or tile <= 0 or tile % 128 or blocks <= 0 or stages <= 0:
            raise ValueError("Expected positive size/grid and a tile divisible by 128")
        full, tail = divmod(n, tile)
        grid = min(blocks, full) if full else 1
        tail_width = (tail + 127) // 128 * 128
        max_stages = ((248 * 1024 - 32896 * 2) // 4 - tail_width) // tile
        if full:
            if max_stages < 1:
                raise ValueError("Lookup, tile and tail buffers exceed available UB capacity")
            stages = min(stages, max_stages)

        @T.prim_func
        def main(
            A: T.Tensor((n,), "bfloat16"),
            Lookup: T.Tensor((32896,), "bfloat16"),
            C: T.Tensor((n,), "bfloat16"),
        ):
            with T.Kernel(grid) as bx:
                lookup = T.alloc_shared((32896,), "bfloat16")
                T.copy(Lookup, lookup)
                if full:
                    a = T.alloc_shared((tile,), "bfloat16")
                    c = T.alloc_shared((tile,), "bfloat16")
                    T.annotate_buffer_versions({a: stages, c: stages})
                    for step in T.Pipelined((full + grid - 1) // grid, num_stages=stages):
                        block = step * grid + bx
                        if block < full:
                            begin = block * tile
                            T.copy(A[begin : begin + tile], a)
                            _lookup_body(a, c, lookup, tile)
                            T.copy(c, C[begin : begin + tile], l2_cache_ctrl=0)
                if tail:  # noqa: SIM102 - static guard excludes zero-size TIR buffers
                    if bx == 0:
                        at = T.alloc_shared((tail_width,), "bfloat16")
                        ct = T.alloc_shared((tail_width,), "bfloat16")
                        T.copy(A[full * tile : n], at[:tail])
                        _lookup_body(at, ct, lookup, tail_width)
                        T.copy(ct[:tail], C[full * tile : n], l2_cache_ctrl=0)

        return tilelang.compile(main, target="ascend", out_idx=-1)


class AscendLogKernel(LogRootsKernel, UnaryElementwiseFwdInterface):
    kind = "log"
    configs = {
        (16777216, "float16"): (8192, 64, 3),
        "float16": (8192, 64, 3),
        (16777216, "bfloat16"): (8192, 64, 3),
        "bfloat16": (8192, 64, 3),
        (16777216, "float32"): (4096, 64, 4),
        "float32": (8192, 64, 3),
        (268435456, "float16"): (8192, 64, 3),
        (268435456, "bfloat16"): (8192, 64, 4),
    }

    def __init__(self, n_total, dtype):
        if dtype != torch.bfloat16:
            super().__init__(n_total, dtype)
            return
        super().__init__(0, dtype)
        self.n_total = n_total
        if n_total:
            config = self.configs.get((n_total, "bfloat16"), self.configs["bfloat16"])
            self.compiled = _build_lookup(n_total, *config)
            self.lookup = _lookup_table(torch.device("npu", torch.npu.current_device()))

    def forward(self, input):
        if self.dtype != torch.bfloat16 or not self.n_total:
            return super().forward(input)
        if input.device.type != "npu" or input.dtype != self.dtype or input.numel() != self.n_total:
            raise ValueError("Input must match the NPU kernel dtype and element count")
        return self.compiled(input.reshape(-1), self.lookup).reshape(input.shape)


if T is not None:
    from tileops.backend import register_kernel_type

    register_kernel_type("LogFwdOp", "ascend", AscendLogKernel)
