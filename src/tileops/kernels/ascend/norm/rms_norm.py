"""Ascend RMSNorm with bounded UB storage and FP32 accumulation."""
import functools
from typing import ClassVar
import torch
from tileops.backend import register_kernel_type
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.norm.call_spec import RMSNormFwdInterface
import tilelang
import tilelang.ascend.language as T


@functools.lru_cache(maxsize=128)
def build_batched(m,n,dtype,eps,weighted,rows=8,threads=32,max_blocks=64):
    width=tilelang.next_power_of_2(n)
    pitch=max(16,(n+15)//16*16)
    groups=(m+rows-1)//rows
    blocks=min(max_blocks,groups)
    assert n<=1024 and rows*pitch*8+pitch*2<192*1024
    @T.prim_func
    def main(X:T.Tensor((m,n),dtype),W:T.Tensor((n,),dtype),Y:T.Tensor((m,n),dtype)):
        with T.Kernel(blocks) as bx:
            x=T.alloc_shared((rows,pitch),dtype)
            y=T.alloc_shared((rows,pitch),dtype)
            w=T.alloc_shared((pitch,),dtype)
            T.annotate_buffer_versions({x:2,y:2})
            if weighted:T.copy(W[:n],w[:n])
            for step in T.Pipelined((groups+blocks-1)//blocks,num_stages=2):
                row=(step*blocks+bx)*rows
                if row<m:
                    valid_rows=T.min(rows,m-row)
                    T.copy(X[row:row+valid_rows,:n],x[:valid_rows,:n])
                    with T.SimtVF(threads=min(threads,width)):
                        xf=T.alloc_fragment((rows,width),'float32')
                        total=T.alloc_reducer((rows,),'float32',op='sum')
                        T.reducer_init(total)
                        for r,i in T.Parallel(rows,width):
                            xf[r,i]=T.if_then_else(r<valid_rows and i<n,T.cast(x[r,i],'float32'),T.float32(0))
                            T.reducer_update(total[r],xf[r,i]*xf[r,i])
                        reduced=T.alloc_fragment((rows,),'float32')
                        T.finalize_reducer(total,reduced)
                        for r in T.Parallel(rows):reduced[r]=T.rsqrt(reduced[r]/n+eps)
                        for r,i in T.Parallel(rows,width):
                            if r<valid_rows and i<n:
                                if weighted:y[r,i]=xf[r,i]*reduced[r]*T.cast(w[i],'float32')
                                else:y[r,i]=xf[r,i]*reduced[r]
                    T.copy(y[:valid_rows,:n],Y[row:row+valid_rows,:n])
    return tilelang.compile(main,target='ascend',out_idx=[2])


@functools.lru_cache(maxsize=256)
def _build_fallback(m, n, dtype, eps, weighted):
    if 0 < n <= 1024:
        rows = min(32, max(1, 8192 // max(16, (n + 15) // 16 * 16)))
        return build_batched(m, n, dtype, eps, weighted, rows, 128, 64)
    width = min(4096, tilelang.next_power_of_2(n))
    pitch = max(16, min(width, (n + 15) // 16 * 16))
    blocks = min(64, m)

    @T.prim_func
    def main(X: T.Tensor((m, n), dtype), W: T.Tensor((n,), dtype), Y: T.Tensor((m, n), dtype)):
        with T.Kernel(blocks) as bx:
            x = T.alloc_shared((pitch,), dtype)
            w = T.alloc_shared((pitch,), dtype)
            y = T.alloc_shared((pitch,), dtype)
            accum = T.alloc_shared((16,), 'float32')
            for step in T.serial((m + blocks - 1) // blocks):
                row = step * blocks + bx
                if row < m:
                    T.clear(accum)
                    for chunk in T.serial((n + width - 1) // width):
                        begin = chunk * width
                        valid = T.min(width, n - begin)
                        T.copy(X[row, begin:begin+valid], x[:valid])
                        with T.SimtVF(threads=min(256, width)):
                            total = T.alloc_reducer((1,), 'float32', op='sum')
                            T.reducer_init(total)
                            for i in T.Parallel(width):
                                if i < valid:
                                    v = T.cast(x[i], 'float32')
                                    T.reducer_update(total[0], v * v)
                            reduced = T.alloc_fragment((1,), 'float32')
                            T.finalize_reducer(total, reduced)
                            accum[0] = accum[0] + reduced[0]
                    with T.SimtVF(threads=32):
                        accum[0] = T.rsqrt(accum[0] / n + eps)
                    for chunk in T.serial((n + width - 1) // width):
                        begin = chunk * width
                        valid = T.min(width, n - begin)
                        T.copy(X[row, begin:begin+valid], x[:valid])
                        if weighted:
                            T.copy(W[begin:begin+valid], w[:valid])
                        with T.SimtVF(threads=min(256, width)):
                            for i in T.Parallel(width):
                                if i < valid:
                                    if weighted:
                                        y[i] = T.cast(x[i], 'float32') * accum[0] * T.cast(w[i], 'float32')
                                    else:
                                        y[i] = T.cast(x[i], 'float32') * accum[0]
                        T.copy(y[:valid], Y[row, begin:begin+valid])
    return tilelang.compile(main, target='ascend', out_idx=[2])


@functools.lru_cache(maxsize=256)
def build(m, n, dtype, eps, weighted):
    from tileops.kernels.ascend.norm import _rms_dispatch
    return _rms_dispatch.select_build(_build_fallback, m, n, dtype, eps, weighted)


class AscendRMSNormKernel(Kernel, RMSNormFwdInterface):
    devices: ClassVar[frozenset[str]] = frozenset({'npu'})

    @classmethod
    def entry_for(cls, call):
        return call, lambda: cls(call)

    def __init__(self, call):
        super().__init__()
        self.call = call
        if call.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise TypeError('Unsupported RMSNorm dtype')
        if call.n <= 0:
            raise ValueError('RMSNorm width must be positive')

    def forward(self, x, weight=None):
        n = self.call.n
        if x.numel() == 0:
            return torch.empty_like(x)
        if x.numel() % n or x.dtype != self.call.dtype:
            raise ValueError('Invalid RMSNorm input signature')
        if weight is not None and (weight.numel() != n or weight.dtype != x.dtype):
            raise ValueError('Invalid RMSNorm weight signature')
        rows = x.reshape(-1, n)
        # An unused input view avoids an allocation for the unweighted signature.
        w = rows[0] if weight is None else weight.reshape(n)
        kernel = build(rows.shape[0], n, str(x.dtype).split('.')[-1], self.call.eps, weight is not None)
        return kernel(rows, w).reshape(x.shape)


register_kernel_type('RMSNormFwdOp', 'ascend', AscendRMSNormKernel)
