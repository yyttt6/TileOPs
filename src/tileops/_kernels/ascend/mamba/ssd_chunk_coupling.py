"""Causal chunk coupling on Ascend: FP32 contraction before masking/casting."""
import functools
import torch
import torch.nn.functional as F
import tilelang
import tilelang.ascend.language as T
from tileops._kernels.kernel_base import Kernel
from tileops._kernels.mamba.call_spec import SSDChunkCouplingFwdInterface
from tileops.backend._registry import register_kernel_type


@functools.lru_cache(maxsize=128)
def build(groups, q, n, dtype):
    # Packing pads both dimensions, so Cube loads never cross a chunk boundary.
    width = (q + 127) // 128 * 128
    depth = (n + 127) // 128 * 128
    tiles = width // 128
    blocks = min(32, groups * tiles * tiles)

    @T.prim_func
    def product(C: T.Tensor((groups, width, depth), dtype),
                B: T.Tensor((groups, width, depth), dtype),
                O: T.Tensor((groups, width, width), 'float32')):
        with T.Kernel(blocks) as bx:
            a = T.alloc_l1((128, 128), dtype)
            b = T.alloc_l1((128, 128), dtype)
            acc = T.alloc_l0c((128, 128), 'float32')
            T.annotate_buffer_versions({a: 2, b: 2})
            for step in T.serial(T.ceildiv(groups * tiles * tiles, blocks)):
                tile = step * blocks + bx
                if tile < groups * tiles * tiles:
                    group = tile // (tiles * tiles)
                    row = tile // tiles % tiles * 128
                    col = tile % tiles * 128
                    for ki in T.Pipelined(depth // 128, num_stages=2):
                        T.copy(C[group, row:row+128, ki*128:ki*128+128], a)
                        T.copy(B[group, col:col+128, ki*128:ki*128+128], b)
                        T.gemm(a, b, acc, transpose_B=True, clear_accum=ki == 0)
                    T.copy(acc, O[group, row:row+128, col:col+128])

    row_tile = 16
    row_tiles = (q + row_tile - 1) // row_tile
    col_tile = min(1024, width)
    col_tiles = (width + col_tile - 1) // col_tile
    tasks = groups * row_tiles * col_tiles
    vector_blocks = min(64, tasks)
    @T.prim_func
    def mask_cast(X: T.Tensor((groups, width, width), 'float32'),
                  Y: T.Tensor((groups, q, width), dtype)):
        with T.Kernel(vector_blocks) as bx:
            buf = T.alloc_shared((row_tile, col_tile), 'float32')
            result = T.alloc_shared((row_tile, col_tile), dtype)
            for step in T.serial(T.ceildiv(tasks, vector_blocks)):
                task = step * vector_blocks + bx
                if task < tasks:
                    g = task // (row_tiles * col_tiles)
                    first = task // col_tiles % row_tiles * row_tile
                    first_col = task % col_tiles * col_tile
                    columns = T.min(col_tile, width-first_col)
                    count = T.min(row_tile, q - first)
                    T.copy(X[g, first:first+count, first_col:first_col+columns], buf[:count, :columns])
                    with T.SimtVF(threads=256):
                        for i, j in T.Parallel(row_tile, col_tile):
                            value = T.alloc_var('float32')
                            value = T.float32(0)
                            if i < count and j < columns:
                                value = buf[i, j] * T.if_then_else(first_col+j <= first+i, T.float32(1), T.float32(0))
                            result[i, j] = T.cast(value, dtype)
                    T.copy(result[:count, :columns], Y[g, first:first+count, first_col:first_col+columns])
    return (tilelang.compile(product, target='ascend', out_idx=-1),
            tilelang.compile(mask_cast, target='ascend', out_idx=-1))


class AscendSSDChunkCouplingKernel(Kernel, SSDChunkCouplingFwdInterface):
    devices = frozenset({'npu'})

    @classmethod
    def entry_for(cls, call):
        return call, lambda: cls(call)

    def __init__(self, call):
        super().__init__()
        self.call = call
        if call.chunk_len <= 0 or call.seq_len % call.chunk_len:
            raise ValueError('chunk_size must be positive and divide sequence length')
        self.chunks = call.seq_len // call.chunk_len
        self.compiled, self.mask_compiled = build(call.batch*self.chunks*call.n_groups,
                                                call.chunk_len, call.d_state,
                                                str(call.dtype).split('.')[-1])

    def forward(self, C, B):
        c = self.call
        q, n = c.chunk_len, c.d_state
        width, depth = (q+127)//128*128, (n+127)//128*128
        def pack(x):
            x = x.reshape(c.batch, self.chunks, q, c.n_groups, n)
            x = x.permute(0, 1, 3, 2, 4).contiguous()
            if width != q or depth != n:
                x = F.pad(x, (0, depth-n, 0, width-q))
            return x.reshape(c.batch*self.chunks*c.n_groups, width, depth)
        raw = self.compiled(pack(C), pack(B))
        out = self.mask_compiled(raw)
        return out[:, :, :q].contiguous().reshape(c.batch, self.chunks, c.n_groups, q, q)


register_kernel_type('SSDChunkCouplingFwdOp', 'ascend', AscendSSDChunkCouplingKernel)
