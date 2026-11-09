"""Chunk-local FP32 scan, with independent storage conversion."""
import functools
from typing import ClassVar

import torch
import tilelang
import tilelang.ascend.language as T

from tileops._kernels.kernel_base import Kernel
from tileops._kernels.mamba.call_spec import SSDChunkCumsumFwdInterface
from tileops._kernels.ascend.reduction.cumsum import _scan
from tileops.backend._registry import register_kernel_type


@T.macro
def _preprocess(raw, stored, decay, bias_values, heads, width, dtype: str, bias: bool, softplus: bool, lower: float, upper: float):
    with T.SimdVF():
        mask = T.simd.pset(32)
        for r in T.serial(heads):
            av = T.simd.vld(decay[r], dist='BRC_B32')
            bv = T.simd.vld(bias_values[r], dist='BRC_B32')
            for part in T.serial(width // 64):
                value = T.simd.vld(raw[r, part*64])
                if bias:
                    value = T.simd.vadd(value, bv, mask)
                original = value
                if softplus:
                    e = T.simd.vexp(T.simd.vmins(value, T.float32(20), mask), mask, precision='ftz_false')
                    normal = T.simd.vln(T.simd.vadds(e, T.float32(1), mask), mask, precision='ftz_false')
                    poly = T.simd.vadds(T.simd.vmuls(e, T.float32(-0.25), mask), T.float32(1.0/3.0), mask)
                    poly = T.simd.vadds(T.simd.vmul(e, poly, mask), T.float32(-0.5), mask)
                    poly = T.simd.vmul(e, T.simd.vadds(T.simd.vmul(e, poly, mask), T.float32(1), mask), mask)
                    result = T.simd.vsel(poly, normal, T.simd.vcmps(value, T.float32(-4), mask, op='lt'))
                    value = T.simd.vsel(value, result, T.simd.vcmps(value, T.float32(20), mask, op='gt'))
                value = T.simd.vmins(T.simd.vmaxs(value, T.float32(lower), mask), T.float32(upper), mask)
                value = T.simd.vsel(original, value, T.simd.vcmp(original, original, mask, op='ne'))
                if dtype == 'float32':
                    T.simd.vsts(stored[r, part*64], value, mask, extent=64)
                else:
                    T.simd.vsts(stored[r, part*64], T.simd.vcvt(value, dtype, sat=False), mask, dist='PK_B32', extent=64)
                T.simd.vsts(raw[r, part*64], T.simd.vmul(value, av, mask), mask, extent=64)
        T.simd.mem_bar('VST_VLD')


@functools.lru_cache(maxsize=128)
def build(b, s, h, q, dtype, bias, softplus, lower, upper):
    chunks = s // q
    rows = b * h * chunks
    heads = min(16, h)
    head_tiles = (h + heads - 1) // heads
    groups = b * head_tiles * chunks
    pitch = (q + 63) // 64 * 64
    width = min(1024, pitch)
    blocks = min(64, groups)
    dma_input = h % 8 == 0 and h % heads == 0

    @T.prim_func
    def main(DT: T.Tensor((b, s, h), 'float32'),
             A: T.Tensor((h,), 'float32'), Bias: T.Tensor((h,), 'float32'),
             Out: T.Tensor((rows, pitch), dtype),
             Prefix: T.Tensor((rows, pitch), 'float32')):
        with T.Kernel(blocks) as bx:
            product = T.alloc_shared((heads, width + 8), 'float32')
            input_tile = T.alloc_shared((width, heads), 'float32')
            stored = T.alloc_shared((heads, width), dtype)
            state = T.alloc_shared((heads, 8), 'float32')
            decay = T.alloc_shared((max(8, heads),), 'float32')
            bias_values = T.alloc_shared((max(8, heads),), 'float32')
            for step in T.serial(T.ceildiv(groups, blocks)):
                group = step * blocks + bx
                if group < groups:
                    batch = group // (head_tiles * chunks)
                    first_head = group // chunks % head_tiles * heads
                    chunk = group % chunks
                    T.fill(state, 0)
                    for part in T.serial(T.ceildiv(q, width)):
                        start = part * width
                        count = T.min(width, q - start)
                        physical = T.min(width, pitch - start)
                        if dma_input:
                            T.copy(DT[batch, chunk*q+start:chunk*q+start+count, first_head:first_head+heads], input_tile[:count, :])
                        with T.SimtVF(threads=256):
                            for local_head in T.Parallel(heads):
                                head = first_head + local_head
                                av = T.alloc_var('float32')
                                bv = T.alloc_var('float32')
                                av = T.float32(0)
                                bv = T.float32(0)
                                if head < h:
                                    av = A[head]
                                    if bias:
                                        bv = Bias[head]
                                decay[local_head] = av
                                bias_values[local_head] = bv
                            for j, local_head in T.Parallel(width, heads):
                                head = first_head + local_head
                                value = T.alloc_var('float32')
                                value = T.float32(0)
                                if head < h and j < count:
                                    if dma_input:
                                        value = input_tile[j, local_head]
                                    else:
                                        value = DT[batch, chunk*q + start+j, head]
                                product[local_head, j] = value
                        _preprocess(product, stored, decay, bias_values, heads, width, dtype, bias, softplus, lower, upper)
                        _scan(product, product, state, heads, count, 'float32')
                        for local_head in T.serial(heads):
                            if first_head + local_head < h:
                                row = (batch*h + first_head + local_head)*chunks + chunk
                                T.copy(stored[local_head, :physical], Out[row, start:start+physical])
                                T.copy(product[local_head, :physical], Prefix[row, start:start+physical])
    return tilelang.compile(main, target='ascend', out_idx=[3, 4])


class AscendSSDChunkCumsumKernel(Kernel, SSDChunkCumsumFwdInterface):
    devices: ClassVar[frozenset[str]] = frozenset({'npu'})

    @classmethod
    def entry_for(cls, call):
        return call, lambda: cls(call)

    def __init__(self, call):
        super().__init__()
        self.call = call
        self.compiled = build(call.batch, call.seq_len, call.n_heads, call.chunk_len,
                              str(call.out_dtype).removeprefix('torch.'), call.has_dt_bias,
                              call.dt_softplus, call.dt_min, call.dt_max)

    def forward(self, dt, A, dt_bias=None):
        c = self.call
        bias = A if dt_bias is None else dt_bias.contiguous()
        outputs = self.compiled(dt.contiguous(), A.contiguous(), bias)
        shape = (c.batch, c.n_heads, c.seq_len // c.chunk_len, c.chunk_len)
        return tuple(x[:, :c.chunk_len].contiguous().reshape(shape) for x in outputs)


register_kernel_type('SSDChunkCumsumFwdOp', 'ascend', AscendSSDChunkCumsumKernel)
