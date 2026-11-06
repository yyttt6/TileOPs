"""InstanceNorm training kernels: normalize, round per-sample updates, then average."""

# Registration is optional when the Ascend toolchain is unavailable.
try:
    import tilelang.ascend.language as T
except ImportError:
    pass
else:
    import struct
    import functools
    from typing import ClassVar
    import torch
    from tileops.backend import register_kernel_type
    from tileops.kernels.kernel_base import Kernel
    import tilelang


    def bf16_constant(value):
        bits=struct.unpack('<I',struct.pack('<f',float(value)))[0]
        bits=(bits+0x7fff+((bits>>16)&1))&0xffff0000
        return struct.unpack('<f',struct.pack('<I',bits))[0]

    @functools.lru_cache(maxsize=256)
    def build_train(batch, channels, spatial, dtype, eps, momentum, weighted, biased):
        bf_momentum=bf16_constant(momentum)
        bf_previous=bf16_constant(1.0-bf_momentum)
        bf_divisor=bf16_constant(spatial-1)
        groups=channels
        m=batch*groups
        cpg=channels//groups
        n=cpg*spatial
        width = min(4096, tilelang.next_power_of_2(n))
        pitch = max(16, min(width, (n + 15) // 16 * 16))
        blocks = min(64, m)

        @T.prim_func
        def main(X: T.Tensor((m, n), dtype), W: T.Tensor((channels,), dtype), B: T.Tensor((channels,), dtype), RM: T.Tensor((channels,), 'float32'), RV: T.Tensor((channels,), 'float32'), Y: T.Tensor((m, n), dtype), U: T.Tensor((m, 2), 'float32')):
            with T.Kernel(blocks) as bx:
                x = T.alloc_shared((pitch,), dtype)
                w = T.alloc_shared((16,), dtype)
                b = T.alloc_shared((16,), dtype)
                y = T.alloc_shared((pitch,), dtype)
                accum = T.alloc_shared((16,), 'float32')
                old_mean=T.alloc_shared((16,), 'float32')
                old_var=T.alloc_shared((16,), 'float32')
                update=T.alloc_shared((16,), 'float32')
                lanes=T.alloc_shared((32,), 'float32')
                tail=T.alloc_shared((16,), 'float32')
                for step in T.serial((m + blocks - 1) // blocks):
                    row = step * blocks + bx
                    if row < m:
                        T.clear(accum)
                        T.clear(lanes);T.clear(tail)
                        for chunk in T.serial((n + width - 1) // width):
                            begin=chunk*width;valid=T.min(width,n-begin)
                            T.copy(X[row,begin:begin+valid],x[:valid])
                            with T.SimtVF(threads=32):
                                for i in T.Parallel(32):
                                    for j in T.serial(valid//32):
                                        lanes[i]=lanes[i]+T.cast(x[j*32+i%16],'float32')
                                        lanes[i]=lanes[i]+T.cast(x[j*32+i%16+16],'float32')
                                for j in T.serial(valid%32):
                                    tail[0]=tail[0]+T.cast(x[(valid//32)*32+j],'float32')
                        with T.SimtVF(threads=32):
                            accum[0]=lanes[0]
                            for i in T.serial(1,16):accum[0]=accum[0]+lanes[i]
                            accum[0]=(accum[0]+tail[0])/n
                        T.clear(lanes);T.clear(tail)
                        for chunk in T.serial((n + width - 1) // width):
                            begin=chunk*width;valid=T.min(width,n-begin)
                            T.copy(X[row,begin:begin+valid],x[:valid])
                            with T.SimtVF(threads=32):
                                for i in T.Parallel(32):
                                    for j in T.serial(valid//32):
                                        a=T.cast(x[j*32+i%16],'float32')-accum[0]
                                        bvalue=T.cast(x[j*32+i%16+16],'float32')-accum[0]
                                        lanes[i]=lanes[i]+a*a
                                        lanes[i]=lanes[i]+bvalue*bvalue
                                for j in T.serial(valid%32):
                                    value=T.cast(x[(valid//32)*32+j],'float32')-accum[0]
                                    tail[0]=tail[0]+value*value
                        with T.SimtVF(threads=32):
                            accum[1]=lanes[0]
                            for i in T.serial(1,16):accum[1]=accum[1]+lanes[i]
                            accum[1]=accum[1]+tail[0]
                        T.copy(RM[row % channels:row % channels+1],old_mean[:1])
                        T.copy(RV[row % channels:row % channels+1],old_var[:1])
                        with T.SimtVF(threads=32):
                            old_m=T.cast(T.cast(old_mean[0],dtype),'float32')
                            old_v=T.cast(T.cast(old_var[0],dtype),'float32')
                            if dtype=='bfloat16':
                                mom=T.float32(bf_momentum)
                                previous=T.float32(bf_previous)
                                mean=T.cast(T.cast(accum[0],dtype),'float32')
                                variance_sum=T.cast(T.cast(accum[1],dtype),'float32')
                                divisor=T.float32(bf_divisor)
                                unbiased=T.cast(T.cast(variance_sum/divisor,dtype),'float32')
                                mean_new=T.cast(T.cast(mom*mean,dtype),'float32')
                                mean_old=T.cast(T.cast(previous*old_m,dtype),'float32')
                                var_old=T.cast(T.cast(previous*old_v,dtype),'float32')
                                update[0]=T.cast(T.cast(mean_new+mean_old,dtype),'float32')
                                update[1]=T.cast(T.cast(mom*unbiased+var_old,dtype),'float32')
                            else:
                                update[0]=T.cast(T.cast(momentum*accum[0]+(1-momentum)*old_m,dtype),'float32')
                                update[1]=T.cast(T.cast(momentum*(accum[1]/(n-1))+(1-momentum)*old_v,dtype),'float32')
                            accum[1] = T.rsqrt(accum[1] / n + eps)
                        T.copy(update[:2],U[row,:2])
                        for channel in T.serial(cpg):
                            for chunk in T.serial((spatial + width - 1) // width):
                                begin = channel * spatial + chunk * width
                                valid = T.min(width, spatial - chunk * width)
                                T.copy(X[row, begin:begin+valid], x[:valid])
                                if weighted:
                                    T.copy(W[(row % groups)*cpg+channel:(row % groups)*cpg+channel+1], w[:1])
                                if biased:
                                    T.copy(B[(row % groups)*cpg+channel:(row % groups)*cpg+channel+1], b[:1])
                                with T.SimtVF(threads=min(256, width)):
                                    for i in T.Parallel(width):
                                        if i < valid:
                                            value = (T.cast(x[i], 'float32') - accum[0]) * accum[1]
                                            if weighted:
                                                if biased:
                                                    y[i] = value * T.cast(w[0], 'float32') + T.cast(b[0], 'float32')
                                                else:
                                                    y[i] = value * T.cast(w[0], 'float32')
                                            else:
                                                if biased:
                                                    y[i] = value + T.cast(b[0], 'float32')
                                                else:
                                                    y[i] = value
                                T.copy(y[:valid], Y[row, begin:begin+valid])
        return tilelang.compile(main, target='ascend', out_idx=[5,6])


    @functools.lru_cache(maxsize=128)
    def build_update(batch,channels,dtype):
        blocks=min(64,channels)
        @T.prim_func
        def main(U:T.Tensor((batch*channels,2),'float32'),RM:T.Tensor((channels,),'float32'),RV:T.Tensor((channels,),'float32')):
            with T.Kernel(blocks) as bx:
                sample=T.alloc_shared((16,),'float32')
                accum=T.alloc_shared((16,),'float32')
                for step in T.serial((channels+blocks-1)//blocks):
                    channel=step*blocks+bx
                    if channel<channels:
                        T.clear(accum)
                        for n in T.serial(batch):
                            T.copy(U[n*channels+channel,:2],sample[:2])
                            with T.SimtVF(threads=32):
                                accum[0]=accum[0]+sample[0]
                                accum[8]=accum[8]+sample[1]
                        with T.SimtVF(threads=32):
                            accum[0]=T.cast(T.cast(accum[0]*(1.0/batch),dtype),'float32')
                            accum[8]=T.cast(T.cast(accum[8]*(1.0/batch),dtype),'float32')
                        T.copy(accum[:1],RM[channel:channel+1])
                        T.copy(accum[8:9],RV[channel:channel+1])
        return tilelang.compile(main,target='ascend',out_idx=[])
