"""Ascend GroupNorm with bounded storage and per-channel affine parameters."""

# Registration is optional when the Ascend toolchain is unavailable.
try:
    import tilelang.ascend.language as T
except ImportError:
    pass
else:
    import functools,math,struct
    from typing import ClassVar
    import torch
    from tileops.backend import register_kernel_type
    from tileops.kernels.kernel_base import Kernel
    from tileops.kernels.norm.call_spec import GroupNormFwdInterface
    import tilelang


    @functools.lru_cache(maxsize=256)
    def build_lowprecision(batch, channels, spatial, groups, dtype, eps, weighted, biased):
        zero_variance_inverse=1.0/math.sqrt(eps)
        m=batch*groups
        cpg=channels//groups
        n=cpg*spatial
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
                            accum[1] = T.if_then_else(accum[1]==0,T.float32(zero_variance_inverse),T.rsqrt(accum[1] / n + eps))
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
                                            if weighted or biased:
                                                scale=accum[1]
                                                if weighted:scale=scale*T.cast(w[0],'float32')
                                                offset=-scale*accum[0]
                                                if biased:offset=offset+T.cast(b[0],'float32')
                                                y[i]=T.call_extern('float32','fmaf',scale,T.cast(x[i],'float32'),offset)
                                            else:
                                                y[i]=(T.cast(x[i],'float32')-accum[0])*accum[1]
                                T.copy(y[:valid], Y[row, begin:begin+valid])
        return tilelang.compile(main, target='ascend', out_idx=[3])

    @functools.lru_cache(maxsize=256)
    def build(batch,channels,spatial,groups,dtype,eps,weighted,biased):
        n = channels // groups * spatial
        pitch = max(16, (n + 15) // 16 * 16)
        if (dtype in ('float16', 'bfloat16') and 0 < n <= 32768
                and 8 * pitch + 4 * channels < 208 * 1024):
            from ._group_lowprecision_simd import build as build_simd
            return build_simd(batch, channels, spatial, groups, dtype, eps, weighted, biased)
        if dtype == 'float32':
            from ._group_fp32_resident import supported, build_small
            if supported(channels, spatial, groups):
                return build_small(batch, channels, spatial, groups, dtype, eps, weighted, biased)
        if dtype!='float32':return build_lowprecision(batch,channels,spatial,groups,dtype,eps,weighted,biased)
        m=batch*groups;cpg=channels//groups;n=cpg*spatial;blocks=min(64,m)
        nv=n//8;chunks=(nv+15)//16;depth=max(1,(max(1,chunks)-1).bit_length())
        zero_inverse=1/math.sqrt(eps)
        eps_hi=struct.unpack("f",struct.pack("f",eps))[0]
        eps_lo=eps-eps_hi
        @T.prim_func
        def main(X:T.Tensor((m,n),'float32'),W:T.Tensor((channels,),'float32'),B:T.Tensor((channels,),'float32'),Y:T.Tensor((m,n),'float32')):
            with T.Kernel(blocks) as bx:
                x=T.alloc_shared((128,),'float32');y=T.alloc_shared((128,),'float32')
                w=T.alloc_shared((16,),'float32');b=T.alloc_shared((16,),'float32')
                sm=T.alloc_shared((depth,32),'float32');sv=T.alloc_shared((depth,32),'float32');sc=T.alloc_shared((depth,),'int32')
                lm=T.alloc_shared((32,),'float32');lv=T.alloc_shared((32,),'float32');acc=T.alloc_shared((16,),'float32')
                for step in T.serial((m+blocks-1)//blocks):
                    row=step*blocks+bx
                    if row<m:
                        T.clear(sm);T.clear(sv);T.clear(sc);T.clear(acc)
                        for chunk in T.serial(chunks):
                            vectors=T.min(16,nv-chunk*16)
                            T.copy(X[row,chunk*128:chunk*128+vectors*8],x[:vectors*8])
                            T.clear(lm);T.clear(lv)
                            with T.SimtVF(threads=32):
                                for lane in T.Parallel(32):
                                    for j in T.serial(vectors):
                                        value=x[j*8+lane%8]
                                        delta=value-lm[lane]
                                        reciprocal=T.float32(1)/T.cast(j+1,'float32')
                                        rem=T.call_extern('float32','fmaf',-reciprocal,T.cast(j+1,'float32'),T.float32(1))
                                        c=reciprocal+rem/T.cast(j+1,'float32')
                                        lm[lane]=T.call_extern('float32','fmaf',c,delta,lm[lane])
                                        lv[lane]=T.call_extern('float32','fmaf',delta,value-lm[lane],lv[lane])
                                    count=sc[0]+vectors
                                    ratio=T.cast(vectors,'float32')/T.cast(count,'float32')
                                    delta=lm[lane]-sm[0,lane]
                                    cd=ratio*delta
                                    md=delta*T.cast(sc[0],'float32')
                                    sm[0,lane]=sm[0,lane]+cd
                                    sv[0,lane]=T.call_extern('float32','fmaf',md,cd,sv[0,lane]+lv[lane])
                                sc[0]=sc[0]+vectors
                            for level in T.serial(1,depth):
                                if (chunk+1)%(1<<level)==0:
                                    with T.SimtVF(threads=32):
                                        for lane in T.Parallel(32):
                                            count=sc[level]+sc[level-1]
                                            ratio=T.cast(sc[level-1],'float32')/T.cast(count,'float32')
                                            delta=sm[level-1,lane]-sm[level,lane]
                                            cd=ratio*delta;md=delta*T.cast(sc[level],'float32')
                                            sm[level,lane]=sm[level,lane]+cd
                                            sv[level,lane]=T.call_extern('float32','fmaf',md,cd,sv[level,lane]+sv[level-1,lane])
                                            sm[level-1,lane]=0;sv[level-1,lane]=0
                                        sc[level]=sc[level]+sc[level-1];sc[level-1]=0
                        for level in T.serial(1,depth):
                            with T.SimtVF(threads=32):
                                for lane in T.Parallel(32):
                                    count=sc[0]+sc[level]
                                    ratio=T.if_then_else(count==0,T.float32(0),T.cast(sc[level],'float32')/T.cast(count,'float32'))
                                    delta=sm[level,lane]-sm[0,lane]
                                    cd=ratio*delta;md=delta*T.cast(sc[0],'float32')
                                    sm[0,lane]=sm[0,lane]+cd
                                    sv[0,lane]=T.call_extern('float32','fmaf',md,cd,sv[0,lane]+sv[level,lane])
                                sc[0]=sc[0]+sc[level]
                        if n%8:
                            T.copy(X[row,nv*8:n],x[:n%8])
                            with T.SimtVF(threads=32):
                                for j in T.serial(n%8):
                                    delta=x[j]-acc[0]
                                    acc[2]=acc[2]+1
                                    acc[0]=acc[0]+delta/acc[2]
                                    acc[1]=T.call_extern('float32','fmaf',delta,x[j]-acc[0],acc[1])
                        with T.SimtVF(threads=32):
                            for lane in T.serial(8):
                                count=acc[2]+nv
                                ratio=T.if_then_else(count==0,T.float32(0),T.float32(nv)/count)
                                delta=sm[0,lane]-acc[0]
                                acc[0]=T.call_extern('float32','fmaf',ratio,delta,acc[0])
                                term=(delta*delta)*ratio
                                merged=T.call_extern('float32','fmaf',term,acc[2],sv[0,lane])
                                acc[1]=acc[1]+merged;acc[2]=count
                            estimate=acc[1]/n
                            remainder=T.call_extern('float32','fmaf',-estimate,T.float32(n),acc[1])
                            variance=estimate+remainder/n
                            argument=variance+T.float32(eps_hi)
                            displaced=argument-variance
                            low=(variance-(argument-displaced))+(T.float32(eps_hi)-displaced)+T.float32(eps_lo)
                            inverse=T.rsqrt(argument)
                            square=inverse*inverse
                            square_error=T.call_extern('float32','fmaf',inverse,inverse,-square)
                            error=T.call_extern('float32','fmaf',-argument,square,T.float32(1))-argument*square_error-low*square
                            corrected=inverse+(T.float32(0.5)*inverse)*error
                            acc[1]=T.if_then_else(variance==0,T.float32(zero_inverse),corrected)
                        for channel in T.serial(cpg):
                            if weighted:T.copy(W[(row%groups)*cpg+channel:(row%groups)*cpg+channel+1],w[:1])
                            if biased:T.copy(B[(row%groups)*cpg+channel:(row%groups)*cpg+channel+1],b[:1])
                            for chunk in T.serial((spatial+127)//128):
                                begin=channel*spatial+chunk*128;valid=T.min(128,spatial-chunk*128)
                                T.copy(X[row,begin:begin+valid],x[:valid])
                                with T.SimtVF(threads=128):
                                    for i in T.Parallel(128):
                                        if i<valid:
                                            if weighted or biased:
                                                scale=acc[1]
                                                if weighted:scale=scale*w[0]
                                                bias=T.float32(0)
                                                if biased:bias=b[0]
                                                offset=T.call_extern('float32','fmaf',-scale,acc[0],bias)
                                                y[i]=T.call_extern('float32','fmaf',scale,x[i],offset)
                                            else:y[i]=(x[i]-acc[0])*acc[1]
                                T.copy(y[:valid],Y[row,begin:begin+valid])
        return tilelang.compile(main,target='ascend',out_idx=[3])

    class AscendGroupNormKernel(Kernel, GroupNormFwdInterface):
        devices: ClassVar[frozenset[str]] = frozenset({'npu'})
        @classmethod
        def entry_for(cls,call):return call,lambda:cls(call)
        def __init__(self,call):
            super().__init__();self.call=call
            if call.num_groups<=0 or call.c%call.num_groups:raise ValueError('Groups must divide channels')
            if call.dtype not in (torch.float16,torch.bfloat16,torch.float32):raise TypeError('Unsupported dtype')
        def forward(self,x,weight=None,bias=None):
            c=self.call
            if x.numel()==0:return torch.empty_like(x)
            if x.shape[1]!=c.c or x.dtype!=c.dtype:raise ValueError('Invalid GroupNorm input')
            for param in (weight,bias):
                if param is not None and (param.numel()!=c.c or param.dtype!=c.dtype):raise ValueError('Invalid affine parameter')
            rows=x.reshape(x.shape[0]*c.num_groups,(c.c//c.num_groups)*c.spatial)
            unused=x.reshape(-1)[:c.c]
            w=unused if weight is None else weight.reshape(c.c)
            b=unused if bias is None else bias.reshape(c.c)
            self.compiled=build(x.shape[0],c.c,c.spatial,c.num_groups,str(c.dtype).split('.')[-1],c.eps,weight is not None,bias is not None)
            return self.compiled(rows,w,b).reshape(x.shape)
    register_kernel_type('GroupNormFwdOp','ascend',AscendGroupNormKernel)
