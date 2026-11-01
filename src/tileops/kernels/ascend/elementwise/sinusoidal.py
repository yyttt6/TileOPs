"""Ascend sinusoidal positional encoding with reusable column divisors."""
import functools
from typing import ClassVar
import torch
from tileops.kernels.elementwise.call_spec import SinusoidalFwdInterface
from tileops.kernels.kernel_base import Kernel
try:
    import tilelang
    import tilelang.ascend.language as T
except ImportError:
    T = None

if T is not None:
    @T.macro
    def _angles(divisor, angles, rows: int, cols: int, row0):
        with T.SimdVF():
            mask = T.simd.pset(32)
            for r in range(rows):
                pos_int = T.simd.vdup(T.cast(row0 + r, 'int32'), 'int32')
                pos = T.simd.vcvt(pos_int, 'float32')
                for c in range(cols // 64):
                    den = T.simd.vld(divisor[c * 64])
                    angle = T.simd.vdiv(pos, den, mask, precision='exact')
                    T.simd.vsts(angles[r, c * 64], angle, mask)

    @T.macro
    def _encode(angles, output, rows: int, cols: int, dtype: str):
        with T.SimtVF(threads=min(2048, rows * cols)):
            for chunk in T.serial((rows * cols + min(2048, rows * cols) - 1) // min(2048, rows * cols)):
                for lane in T.Parallel(min(2048, rows * cols)):
                    pair = chunk * min(2048, rows * cols) + lane
                    if pair < rows * cols:
                        r = pair // cols
                        c = pair % cols
                        angle = angles[r, c]
                        output[r, 2 * c] = T.cast(T.sin(angle), dtype)
                        output[r, 2 * c + 1] = T.cast(T.cos(angle), dtype)

    @functools.lru_cache(maxsize=128)
    def build(seq_len, d_model, dtype):
        half = d_model // 2
        rows, cols, stages = min(16, seq_len), min(256, (half + 63) // 64 * 64), 2
        row_tiles, col_tiles = (seq_len + rows - 1) // rows, (half + cols - 1) // cols
        total = row_tiles * col_tiles
        grid = min(64, total)
        @T.prim_func
        def main(F: T.Tensor(((half + 255) // 256 * 256,), 'float32'), Y: T.Tensor((seq_len, d_model), dtype)):
            with T.Kernel(grid) as bx:
                divisor = T.alloc_shared((cols,), 'float32')
                angles = T.alloc_shared((rows, cols), 'float32')
                output = T.alloc_shared((rows, 2 * cols), dtype)
                T.annotate_buffer_versions({divisor: stages, angles: stages, output: stages})
                for step in T.Pipelined((total + grid - 1) // grid, num_stages=stages):
                    block = step * grid + bx
                    if block < total:
                        row0 = (block // col_tiles) * rows
                        col0 = (block % col_tiles) * cols
                        valid_rows = T.min(rows, seq_len - row0)
                        valid_cols = T.min(cols, half - col0)
                        T.copy(F[col0:col0 + cols], divisor)
                        _angles(divisor, angles, rows, cols, row0)
                        _encode(angles, output, rows, cols, dtype)
                        T.copy(output[:valid_rows, :2 * valid_cols], Y[row0:row0 + valid_rows, 2 * col0:2 * (col0 + valid_cols)], l2_cache_ctrl=0)
        return tilelang.compile(main, target='ascend', out_idx=[1])

class AscendSinusoidalKernel(Kernel, SinusoidalFwdInterface):
    devices: ClassVar[frozenset[str]] = frozenset({'npu'})
    @classmethod
    def entry_for(cls, call):
        return call, lambda: cls(call.seq_len, call.d_model, call.dtype, call.device)
    def __init__(self, seq_len, d_model, dtype, device=None):
        super().__init__()
        if seq_len <= 0 or d_model <= 0 or d_model % 2:
            raise ValueError('Sinusoidal requires positive length and positive even model dimension')
        if dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError(f'Unsupported sinusoidal dtype: {dtype}')
        self.seq_len, self.d_model, self.dtype = seq_len, d_model, dtype
        self.device = torch.device(device) if device is not None else torch.device('npu', torch.npu.current_device())
        # These O(D) constants depend only on the construction-time dimension.
        # CPU float32 pow follows the reference's rounding before angle division.
        half = d_model // 2
        frequencies = torch.ones(((half + 255) // 256 * 256,), dtype=torch.float32, device='cpu')
        frequencies[:half] = torch.pow(10000., torch.arange(0, d_model, 2, dtype=torch.float32, device='cpu') / d_model)
        self.frequencies = frequencies.to(self.device)
        self.compiled = build(seq_len, d_model, str(dtype).removeprefix('torch.'))
    def forward(self):
        with torch.npu.device(self.device):
            return self.compiled(self.frequencies)

if T is not None:
    from tileops.backend import register_kernel_type
    register_kernel_type('SinusoidalFwdOp', 'ascend', AscendSinusoidalKernel)
