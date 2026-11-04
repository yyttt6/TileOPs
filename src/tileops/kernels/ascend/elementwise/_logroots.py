"""Persistent SIMD kernels for logarithms and square roots on Ascend.

All shapes use the same value law, including subnormals and domain boundaries.
The final partial tile has padded on-chip storage but copies only valid elements.
"""

import functools

import torch

from tileops.kernels.kernel_base import Kernel

try:
    import tilelang
    import tilelang.ascend.language as T
except ImportError:
    T = None


if T is not None:

    @T.macro
    def _body(a, c, width: int, dtype: str, kind: str):
        with T.SimdVF():
            mask = T.simd.pset(32)
            for i in range(width // 64):
                x = (
                    T.simd.vld(a[i * 64])
                    if dtype == "float32"
                    else T.simd.vcvt(T.simd.vld(a[i * 64], dist="UNPK_B16"), "float32", part=0)
                )
                if kind == "log":
                    if dtype == "float16":
                        y = T.simd.vln(x, mask, precision="ftz_true")
                    else:
                        y = T.simd.vln(x, mask, precision="ftz_false")
                elif kind == "sqrt" or kind == "rsqrt":
                    if dtype == "float16":
                        # Every finite nonzero half becomes a normal FP32 value.
                        root = T.simd.vsqrt(x, mask, precision="ftz_true")
                    else:
                        # A subnormal is mantissa * 2**-149. Converting its
                        # integer mantissa avoids FTZ before sqrt; sqrt(2**-149)
                        # restores scale. Negative subnormals stay in sqrt's
                        # NaN domain, and a separate bit test preserves -0.
                        bits = T.reinterpret(x, "int32x64")
                        abs_bits = T.simd.vand(
                            bits, T.simd.vdup(T.int32(0x7FFFFFFF), "int32"), mask
                        )
                        subnormal = T.simd.vcmps(abs_bits, T.int32(0x00800000), mask, op="lt")
                        significand = T.simd.vcvt(bits, "float32")
                        normalized = T.simd.vsel(significand, x, subnormal)
                        raw_root = T.simd.vsqrt(normalized, mask, precision="ftz_true")
                        scaled_root = T.simd.vmuls(raw_root, 3.743392130574644e-23, mask)
                        corrected_root = T.simd.vsel(scaled_root, raw_root, subnormal)
                        zero = T.simd.vcmps(abs_bits, T.int32(0), mask, op="eq")
                        root = T.simd.vsel(x, corrected_root, zero)
                    if kind == "sqrt":
                        y = root
                    else:
                        one = T.simd.vdup(1.0, "float32")
                        y = T.simd.vdiv(one, root, mask, precision="ftz_true")
                else:
                    u = T.simd.vadds(x, 1.0, mask)
                    raw = T.simd.vln(u, mask, precision="ftz_true")
                    if dtype == "float32":
                        # The cubic remainder at |x| <= 1/64 is below 1.6e-8.
                        # This avoids cancellation near zero without a division.
                        p = T.simd.vmuls(x, 0.3333333333333333, mask)
                        q = T.simd.vadds(p, -0.5, mask)
                        r = T.simd.vmul(x, q, mask)
                        t = T.simd.vadds(r, 1.0, mask)
                        polynomial = T.simd.vmul(x, t, mask)
                        magnitude = T.simd.vabs(x, mask)
                        small = T.simd.vcmps(magnitude, 0.015625, mask, op="le")
                        corrected = T.simd.vsel(polynomial, raw, small)
                        tiny = T.simd.vcmps(magnitude, 5.960464477539063e-8, mask, op="le")
                        y = T.simd.vsel(x, corrected, tiny)
                    else:
                        # The quadratic correction is below the low-dtype
                        # rounding scale here. Exhaustive bit-pattern checks
                        # validate both the direct path and hardware ln path.
                        magnitude = T.simd.vabs(x, mask)
                        threshold = 0.00048828125 if dtype == "float16" else 0.00390625
                        small = T.simd.vcmps(magnitude, threshold, mask, op="le")
                        y = T.simd.vsel(x, raw, small)
                if dtype == "float32":
                    T.simd.vsts(c[i * 64], y, mask)
                else:
                    narrow = T.simd.vcvt(y, dtype)
                    T.simd.vsts(c[i * 64], narrow, mask, dist="PK_B32")

    @functools.lru_cache(maxsize=512)
    def build(n, dtype, kind, tile=8192, blocks=64, stages=2):
        if kind not in ("log", "log1p", "sqrt", "rsqrt"):
            raise ValueError(f"Unknown logarithm/root operation: {kind}")
        if n <= 0 or tile <= 0 or tile % 64 or blocks <= 0 or stages <= 0:
            raise ValueError("Expected a positive size/grid and a tile divisible by 64")
        full, tail = divmod(n, tile)
        grid = min(blocks, full) if full else 1
        tail_width = (tail + 63) // 64 * 64
        if full and tail:
            # Tail storage coexists with the versioned pipeline buffers. The
            # 8 KiB headroom keeps the total within the 256 KiB UB budget.
            element_bytes = 4 if dtype == "float32" else 2
            max_stages = ((248 * 1024) // (2 * element_bytes) - tail_width) // tile
            if max_stages < 1:
                raise ValueError("Tile and tail buffers exceed available UB capacity")
            stages = min(stages, max_stages)

        @T.prim_func
        def main(A: T.Tensor((n,), dtype), C: T.Tensor((n,), dtype)):
            with T.Kernel(grid) as bx:
                if full:
                    a = T.alloc_shared((tile,), dtype)
                    c = T.alloc_shared((tile,), dtype)
                    T.annotate_buffer_versions({a: stages, c: stages})
                    for step in T.Pipelined((full + grid - 1) // grid, num_stages=stages):
                        block = step * grid + bx
                        if block < full:
                            begin = block * tile
                            T.copy(A[begin : begin + tile], a)
                            _body(a, c, tile, dtype, kind)
                            T.copy(c, C[begin : begin + tile], l2_cache_ctrl=0)
                if tail:  # noqa: SIM102 - static guard excludes zero-size TIR buffers
                    if bx == 0:
                        at = T.alloc_shared((tail_width,), dtype)
                        ct = T.alloc_shared((tail_width,), dtype)
                        T.copy(A[full * tile : n], at[:tail])
                        _body(at, ct, tail_width, dtype, kind)
                        T.copy(ct[:tail], C[full * tile : n], l2_cache_ctrl=0)

        return tilelang.compile(main, target="ascend", out_idx=-1)


class LogRootsKernel(Kernel):
    devices = frozenset({"npu"})
    supported_archs = None
    kind = ""
    configs = {}

    @classmethod
    def entry_for(cls, call):
        return call, lambda: cls(call.n_total, call.dtype)

    def __init__(self, n_total, dtype):
        super().__init__()
        self.n_total = n_total
        self.dtype = dtype
        if T is None:
            raise ImportError("tilelang.ascend is required")
        if dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise TypeError(f"Unsupported dtype: {dtype}")
        dtype_name = str(dtype).removeprefix("torch.")
        tile, blocks, stages = self.configs.get(
            (n_total, dtype_name), self.configs.get(dtype_name, (8192, 64, 2))
        )
        self.compiled = (
            build(n_total, dtype_name, self.kind, tile, blocks, stages) if n_total else None
        )

    def forward(self, input):
        if input.device.type != "npu" or input.dtype != self.dtype or input.numel() != self.n_total:
            raise ValueError("Input must match the NPU kernel dtype and element count")
        if not self.n_total:
            return torch.empty_like(input)
        return self.compiled(input.reshape(-1)).reshape(input.shape)
