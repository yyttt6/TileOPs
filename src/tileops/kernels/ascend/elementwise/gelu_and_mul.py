"""TileLang-Ascend kernel for ``GeluAndMulFwdOp``: y = gelu(gate) * value, exact erf GELU.

New, additive file: it does not edit any existing kernel/op module. It self-registers
with ``tileops.backend.register_kernel_type`` at import time (see
``docs/ascend950/dev_process.md`` section 1), so importing this module is what makes
``AscendGeluAndMulKernel`` participate in ``GeluAndMulFwdOp`` dispatch for npu tensors;
nothing else needs to change.

The input ``x`` is a single ``(M, 2*N)`` row-major tensor; ``gate = x[:, :N]`` and
``value = x[:, N:]``, and the output is ``gelu(gate) * value`` with shape ``(M, N)``
(``FusedGatedFwdInterface``, ``tileops/kernels/elementwise/call_spec.py``).

The erf used on fp16/bf16 is the same clamped degree-10 polynomial the CUDA reference
kernel uses (``tileops/kernels/elementwise/_erf.py``'s ``gelu()``), ported as math only
(the constants and formula, not the code): that file imports ``tilelang.language``, the
CUDA dialect, which this module must not use. Ascend's SIMD MicroAPI exposes no
``verf``/``vtanh`` instruction, so avoiding a transcendental call entirely is also a
correctness requirement here, not just a performance choice.
"""

import functools
from typing import ClassVar

import torch
from tileops.kernels.elementwise._erf import gelu as _exact_gelu

try:
    import tilelang
    import tilelang.ascend.language as T

    from tileops.backend import register_kernel_type
    from tileops.kernels.constants import INV_SQRT2
    from tileops.kernels.elementwise.call_spec import FusedGatedCall, FusedGatedFwdInterface
    from tileops.kernels.kernel_base import Entry, Kernel

    _ASCEND_IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - exercised only without an Ascend env
    _ASCEND_IMPORT_ERROR = exc

__all__ = ["AscendGeluAndMulKernel"] if _ASCEND_IMPORT_ERROR is None else []


if _ASCEND_IMPORT_ERROR is None:
    # erf(x) = clip(t * P(w), -1, 1), t = clamp(x, -_ERF_CLAMP, _ERF_CLAMP),
    # w = 1 - (t / _ERF_CLAMP)**2, highest degree first. Ported verbatim (values only) from
    # tileops/kernels/elementwise/_erf.py: worst case over the real line is 1.7e-5, an order
    # below half a float16 ulp at 1.0. Ascend behavior of this exact polynomial has not been
    # measured before this round; M3 is what confirms it still holds under Ascend's FMA/
    # rounding.
    _ERF_CLAMP = 3.6
    _ERF_POLY_COEFFS = (
        8.971932411193848,
        -33.1397590637207,
        53.60123062133789,
        -48.02798080444336,
        26.02545738220215,
        -8.488715171813965,
        1.7499406337738037,
        -0.09359221160411835,
        0.11330155283212662,
        0.1387363225221634,
        0.2777777910232544,
    )

    _TORCH_DTYPE_TO_TL = {
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
        torch.float32: "float32",
    }

    # Ascend 950 AIV core count this round's tiling assumes (same value used by the other
    # batch-1 Ascend examples: example_rmsnorm.py, example_per_token_cast_to_fp8.py). Not
    # independently queried from the device; if measurement shows a different effective
    # core count this constant is the first thing to revisit.
    _NUM_AIV_CORES = 64

    def _largest_divisor_leq(n: int, upper_bound: int) -> int:
        """The largest divisor of *n* that is <= *upper_bound* (upper_bound clamped to >= 1)."""
        upper_bound = max(1, min(upper_bound, n))
        for candidate in range(upper_bound, 0, -1):
            if n % candidate == 0:
                return candidate
        return 1  # unreachable: 1 always divides n

    def _pick_block_n(m: int, n: int) -> int:
        """The column tile width.

        When rows alone give at least one tile per core (``m >= _NUM_AIV_CORES``, the
        prefill-shaped case), tile wide for DMA efficiency but cap at 8192 elements so
        double-buffered gate/value/output UB stays a comfortable fraction of the per-core
        budget. When rows do not (``m < _NUM_AIV_CORES``, the decode-shaped case, including
        ``m == 1``), parallelism can only come from splitting N, so target one tile per
        core instead.
        """
        if n <= 0:
            raise ValueError(f"N must be positive, got {n}")
        if m >= _NUM_AIV_CORES:
            return _largest_divisor_leq(n, 8192)
        return _largest_divisor_leq(n, max(1, n // _NUM_AIV_CORES))

    def _gelu_and_mul_program(m: int, n: int, dtype: str):
        """Build the ``(m, 2*n) -> (m, n)`` GeluAndMul prim_func for one (m, n, dtype)."""
        block_n = _pick_block_n(m, n)
        col_tiles = n // block_n
        total_tiles = m * col_tiles
        # Only worth double-buffering when a core actually revisits the pipeline (more than
        # one tile per core); a single pass per core (e.g. the decode case, where
        # total_tiles == _NUM_AIV_CORES) has nothing to overlap with.
        num_stages = 2 if total_tiles // _NUM_AIV_CORES >= 2 else 1
        threads = min(1024, block_n)

        @T.prim_func
        def main(X: T.Tensor((m, 2 * n), dtype), Y: T.Tensor((m, n), dtype)):
            with T.Kernel(_NUM_AIV_CORES) as core_id:
                for tile in T.Persistent([total_tiles], _NUM_AIV_CORES, core_id, num_stages=num_stages):
                    row = tile // col_tiles
                    col = (tile % col_tiles) * block_n
                    gate_ub = T.alloc_shared((block_n,), dtype)
                    value_ub = T.alloc_shared((block_n,), dtype)
                    out_ub = T.alloc_shared((block_n,), dtype)
                    T.annotate_buffer_versions(
                        {gate_ub: num_stages, value_ub: num_stages, out_ub: num_stages}
                    )
                    T.copy(X[row, col : col + block_n], gate_ub)
                    T.copy(X[row, n + col : n + col + block_n], value_ub)
                    _tail_math(gate_ub, value_ub, out_ub, block_n, dtype)
                    T.copy(out_ub, Y[row, col : col + block_n])

        return main

    @T.macro
    def _tail_math(gate, value, output, width: int, dtype: str):
        with T.SimtVF(threads=min(1024, width)):
            for i in T.Parallel(width):
                activated = _exact_gelu(gate[i], dtype)
                output[i] = T.cast(activated, dtype) * value[i]

    def _gelu_and_mul_tail_program(m: int, n: int, dtype: str):
        # Keep exact GM transfer lengths; only UB math includes padded lanes.
        tile, stages = 8192, 2
        itemsize = 4 if dtype == "float32" else 2
        while True:
            full, tail = divmod(n, tile)
            padded = (tail + 127) // 128 * 128
            if (3 * stages * tile * bool(full) + 3 * padded) * itemsize <= 248 * 1024:
                break
            tile //= 2
        columns = full + bool(tail)
        total = m * columns
        grid = min(64, total)

        @T.prim_func
        def main(X: T.Tensor((m, 2 * n), dtype), Y: T.Tensor((m, n), dtype)):
            with T.Kernel(grid) as bx:
                if full:
                    gate = T.alloc_shared((tile,), dtype)
                    value = T.alloc_shared((tile,), dtype)
                    output = T.alloc_shared((tile,), dtype)
                    T.annotate_buffer_versions({gate: stages, value: stages, output: stages})
                    for step in T.Pipelined((m * full + grid - 1) // grid, num_stages=stages):
                        block = step * grid + bx
                        if block < m * full:
                            row = block // full
                            col = block % full * tile
                            T.copy(X[row, col:col + tile], gate)
                            T.copy(X[row, n + col:n + col + tile], value)
                            _tail_math(gate, value, output, tile, dtype)
                            T.copy(output, Y[row, col:col + tile], l2_cache_ctrl=0)
                if tail:
                    gt = T.alloc_shared((padded,), dtype)
                    vt = T.alloc_shared((padded,), dtype)
                    ot = T.alloc_shared((padded,), dtype)
                    for step in T.serial((m + grid - 1) // grid):
                        row = step * grid + bx
                        if row < m:
                            T.copy(X[row, full * tile:n], gt[:tail])
                            T.copy(X[row, n + full * tile:2 * n], vt[:tail])
                            _tail_math(gt, vt, ot, padded, dtype)
                            T.copy(ot[:tail], Y[row, full * tile:n], l2_cache_ctrl=0)
        return main

    @functools.lru_cache(maxsize=None)
    def _build_gelu_and_mul_kernel(m: int, n: int, dtype_str: str):
        program = (_gelu_and_mul_tail_program(m, n, dtype_str)
                   if n % 128 or _pick_block_n(m, n) < 64
                   else _gelu_and_mul_program(m, n, dtype_str))
        return tilelang.compile(program, target="ascend", out_idx=-1)

    class AscendGeluAndMulKernel(Kernel, FusedGatedFwdInterface):
        """TileLang-Ascend GeluAndMul: ``y = gelu(gate) * value`` (exact erf GELU).

        On float16 and bfloat16 the error function is evaluated with the same clamped
        degree-10 polynomial as the in-tree CUDA kernel
        (``tileops/kernels/elementwise/_erf.py``), promoted to float32 and cast back to
        the storage dtype at the boundary; its worst case over the real line is 1.7e-5.
        Float32 uses the shared erf path; 16-bit storage uses its polynomial.
        """

        devices: ClassVar[frozenset[str]] = frozenset({"npu"})

        @classmethod
        def entry_for(cls, call: FusedGatedCall) -> Entry:
            return call, lambda: cls(call.m, call.n, call.dtype)

        def __init__(self, m: int, n: int, dtype: torch.dtype):
            super().__init__()
            if dtype not in _TORCH_DTYPE_TO_TL:
                raise ValueError(f"AscendGeluAndMulKernel does not support dtype {dtype!r}")
            self.m = m
            self.n = n
            self.dtype = dtype
            self._dtype_str = _TORCH_DTYPE_TO_TL[dtype]
            self.kernel = _build_gelu_and_mul_kernel(m, n, self._dtype_str) if m and n else None

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Return ``gelu(x[:, :n]) * x[:, n:]``; nothing is written in place."""
            if x.device.type != "npu" or x.dtype != self.dtype or tuple(x.shape) != (self.m, 2 * self.n):
                raise ValueError("GeluAndMul input differs from the compiled NPU signature")
            if not self.m or not self.n:
                return torch.empty((self.m, self.n), dtype=self.dtype, device=x.device)
            if not x.is_contiguous():
                x = x.contiguous()
            return self.kernel(x)

    register_kernel_type("GeluAndMulFwdOp", "ascend", AscendGeluAndMulKernel)
