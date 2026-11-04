"""Ascend InstanceNorm input statistics with dtype-specific CPU-reference rounding."""

# Registration is optional when the Ascend toolchain is unavailable.
try:
    import tilelang.ascend.language as T
except ImportError:
    pass
else:
    import functools,struct
    from typing import ClassVar
    import torch
    from tileops.backend import register_kernel_type
    from tileops.kernels.kernel_base import Kernel
    from tileops.kernels.norm.call_spec import GroupNormFwdInterface
    import tilelang


    @functools.lru_cache(maxsize=256)
    def build(batch, channels, spatial, groups, dtype, eps, weighted, biased):
        zero_inverse=1/(eps**0.5)
        m=batch*groups
        cpg=channels//groups
        n=cpg*spatial
        raw_n=struct.unpack('I',struct.pack('f',float(n)))[0]
        rounded_n=(raw_n+0x7fff+((raw_n>>16)&1))&0xffff0000
        bf_n=struct.unpack('f',struct.pack('I',rounded_n))[0]
        width = min(4096, tilelang.next_power_of_2(n))
        pitch = max(16, min(width, (n + 15) // 16 * 16))
        blocks = min(64, m)

        @T.prim_func
        def main(X: T.Tensor((m, n), dtype), W: T.Tensor((channels,), dtype), B: T.Tensor((channels,), dtype), Y: T.Tensor((m, n), dtype)):
            with T.Kernel(blocks) as bx:
                x = T.alloc_shared((pitch,), dtype)
                w = T.alloc_shared((16,), dtype)
                b = T.alloc_shared((16,), dtype)
                y = T.alloc_shared((pitch,), dtype)
                products = T.alloc_shared((pitch,), 'float32')
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
                                        T.reducer_update(total[0], v)
                                reduced = T.alloc_fragment((1,), 'float32')
                                T.finalize_reducer(total, reduced)
                                accum[0] = accum[0] + reduced[0]
                        with T.SimtVF(threads=32):
                            accum[0] = accum[0] / n
                        for chunk in T.serial((n + width - 1) // width):
                            begin = chunk * width
                            valid = T.min(width, n - begin)
                            if n > width:
                                T.copy(X[row, begin:begin+valid], x[:valid])
                            with T.SimtVF(threads=min(256, width)):
                                variance = T.alloc_reducer((1,), 'float32', op='sum')
                                T.reducer_init(variance)
                                for i in T.Parallel(width):
                                    if i < valid:
                                        centered = T.cast(x[i], 'float32') - accum[0]
                                        T.reducer_update(variance[0], centered * centered)
                                reduced_var = T.alloc_fragment((1,), 'float32')
                                T.finalize_reducer(variance, reduced_var)
                                accum[1] = accum[1] + reduced_var[0]
                        with T.SimtVF(threads=32):
                            if dtype=='bfloat16':
                                sum_bf=T.cast(T.cast(accum[1],dtype),'float32')
                                variance_bf=T.cast(T.cast(sum_bf/T.float32(bf_n),dtype),'float32')
                                accum[1]=T.if_then_else(variance_bf==0,T.float32(zero_inverse),T.rsqrt(variance_bf+eps))
                            else:
                                accum[1] = T.if_then_else(accum[1]==0,T.float32(zero_inverse),T.rsqrt(accum[1] / n + eps))
                        with T.SimtVF(threads=32):
                            accum[0]=T.cast(T.cast(accum[0],dtype),'float32')
                            accum[1]=T.cast(T.cast(accum[1],dtype),'float32')
                        for channel in T.serial(cpg):
                            for chunk in T.serial((spatial + width - 1) // width):
                                begin = channel * spatial + chunk * width
                                valid = T.min(width, spatial - chunk * width)
                                if n > width or cpg != 1:
                                    T.copy(X[row, begin:begin+valid], x[:valid])
                                if weighted:
                                    T.copy(W[(row % groups)*cpg+channel:(row % groups)*cpg+channel+1], w[:1])
                                if biased:
                                    T.copy(B[(row % groups)*cpg+channel:(row % groups)*cpg+channel+1], b[:1])
                                with T.SimtVF(threads=min(256, width)):
                                    for i in T.Parallel(width):
                                        if i < valid:
                                            weight_value=T.cast(w[0],'float32') if weighted else T.float32(1)
                                            bias_value=T.cast(b[0],'float32') if biased else T.float32(0)
                                            alpha=accum[1]*weight_value
                                            beta=T.call_extern('float32','fmaf',-accum[0],alpha,bias_value)
                                            products[i]=T.cast(x[i],'float32')*alpha
                                            if i==0:
                                                accum[2]=beta
                                                if dtype!='float32' and spatial%16!=0:accum[3]=alpha
                                with T.SimtVF(threads=min(256,width)):
                                    for i in T.Parallel(width):
                                        if i<valid:
                                            if dtype!='float32' and spatial%16!=0 and chunk*width+i>=spatial//16*16:y[i]=T.call_extern('float32','fmaf',T.cast(x[i],'float32'),accum[3],accum[2])
                                            else:y[i]=products[i]+accum[2]
                                T.copy(y[:valid], Y[row, begin:begin+valid])
        return tilelang.compile(main, target='ascend', out_idx=[3],pass_configs={'tl.device_compile_flags':['-ffp-contract=off']})


