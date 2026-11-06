"""Resident FP32 GroupNorm preserving CPU Welford order with private merge state."""
import functools,math,struct
import tilelang
import tilelang.ascend.language as T
def supported(channels,spatial,groups):
    n=channels//groups*spatial
    pitch=max(128,(n+127)//128*128)+64
    capacity=1 << (max(1,(n//8+15)//16)-1).bit_length()
    weight_pitch=max(16,(channels+15)//16*16)
    return 128<=n<=16384 and spatial%16==0 and 16*pitch+64*capacity+8*weight_pitch+320<208*1024

@functools.lru_cache(maxsize=256)
def build_small(batch,channels,spatial,groups,dtype,eps,weighted,biased):
    assert dtype=='float32'
    m=batch*groups;cpg=channels//groups;n=cpg*spatial;blocks=min(64,m)
    assert supported(channels,spatial,groups)
    pitch=max(128,(n+127)//128*128)+64
    nv=n//8;chunks=(nv+15)//16;depth=max(1,(max(1,chunks)-1).bit_length())
    chunk_capacity=tilelang.next_power_of_2(max(1,chunks))
    output_width=pitch
    weight_pitch=max(16,(channels+15)//16*16)
    zero_inverse=1/math.sqrt(eps)
    eps_hi=struct.unpack("f",struct.pack("f",eps))[0]
    eps_lo=eps-eps_hi
    @T.prim_func
    def main(X:T.Tensor((m,n),'float32'),W:T.Tensor((channels,),'float32'),B:T.Tensor((channels,),'float32'),Y:T.Tensor((m,n),'float32')):
        with T.Kernel(blocks) as bx:
            x=T.alloc_shared((pitch,),'float32');y=T.alloc_shared((output_width,),'float32')
            w=T.alloc_shared((weight_pitch,),'float32');b=T.alloc_shared((weight_pitch,),'float32')
            lm=T.alloc_shared((32,),'float32');lv=T.alloc_shared((32,),'float32')
            cm=T.alloc_shared((chunk_capacity,8),'float32');cv=T.alloc_shared((chunk_capacity,8),'float32')
            acc=T.alloc_shared((16,),'float32')
            T.annotate_buffer_versions({x:2,y:2})
            if weighted:T.copy(W[:channels],w[:channels])
            if biased:T.copy(B[:channels],b[:channels])
            for step in T.Pipelined((m+blocks-1)//blocks,num_stages=2):
                row=step*blocks+bx
                if row<m:
                    T.copy(X[row,:n],x[:n])
                    with T.SimtVF(threads=512):
                        means=T.alloc_fragment((chunk_capacity,8),'float32')
                        variances=T.alloc_fragment((chunk_capacity,8),'float32')
                        for chunk,lane in T.Parallel(chunk_capacity,8):
                            means[chunk,lane]=0
                            variances[chunk,lane]=0
                            for j in T.unroll(16):
                                if chunk*16+j<nv:
                                    value=x[chunk*128+j*8+lane]
                                    delta=value-means[chunk,lane]
                                    reciprocal=T.float32(1)/T.cast(j+1,'float32')
                                    rem=T.call_extern('float32','fmaf',-reciprocal,T.cast(j+1,'float32'),T.float32(1))
                                    c=reciprocal+rem/T.cast(j+1,'float32')
                                    means[chunk,lane]=T.call_extern('float32','fmaf',c,delta,means[chunk,lane])
                                    variances[chunk,lane]=T.call_extern('float32','fmaf',delta,value-means[chunk,lane],variances[chunk,lane])
                        for chunk,lane in T.Parallel(chunk_capacity,8):
                            cm[chunk,lane]=means[chunk,lane]
                            cv[chunk,lane]=variances[chunk,lane]
                        for level in T.unroll(1,depth+1):
                            next_mean=T.alloc_fragment((max(1,chunk_capacity//2),8),'float32')
                            next_variance=T.alloc_fragment((max(1,chunk_capacity//2),8),'float32')
                            for group,lane in T.Parallel(max(1,chunk_capacity//2),8):
                                left=T.min(group*(1<<level),chunk_capacity-1)
                                next_mean[group,lane]=cm[left,lane]
                                next_variance[group,lane]=cv[left,lane]
                                if group<chunks//(1<<level):
                                    right=left+(1<<(level-1))
                                    count_left=T.min((1<<(level-1))*16,nv-left*16)
                                    count_right=T.min((1<<(level-1))*16,nv-right*16)
                                    ratio=T.cast(count_right,'float32')/T.cast(T.max(1,count_left+count_right),'float32')
                                    delta=cm[right,lane]-cm[left,lane]
                                    cd=ratio*delta
                                    md=delta*T.cast(count_left,'float32')
                                    next_mean[group,lane]=cm[left,lane]+cd
                                    next_variance[group,lane]=T.call_extern('float32','fmaf',md,cd,cv[left,lane]+cv[right,lane])
                            for group,lane in T.Parallel(max(1,chunk_capacity//2),8):
                                if group<T.max(1,chunk_capacity//(1<<level)):
                                    left=group*(1<<level)
                                    cm[left,lane]=next_mean[group,lane]
                                    cv[left,lane]=next_variance[group,lane]
                    with T.SimtVF(threads=32):
                        pm=T.alloc_fragment((32,),'float32')
                        pv=T.alloc_fragment((32,),'float32')
                        pc=T.alloc_fragment((32,),'int32')
                        nm=T.alloc_fragment((32,),'float32')
                        vv=T.alloc_fragment((32,),'float32')
                        nc=T.alloc_fragment((32,),'int32')
                        for lane in T.Parallel(32):
                            pm[lane]=0
                            pv[lane]=0
                            pc[lane]=0
                            for level in T.unroll(depth):
                                if chunks==(1<<depth) and level==depth-1:
                                    nm[lane]=cm[0,lane%8]
                                    vv[lane]=cv[0,lane%8]
                                    nc[lane]=nv
                                elif (chunks & (1<<level))!=0:
                                    start=(chunks//(1<<(level+1)))*(1<<(level+1))
                                    nm[lane]=cm[start,lane%8]
                                    vv[lane]=cv[start,lane%8]
                                    nc[lane]=T.min((1<<level)*16,nv-start*16)
                                else:
                                    nm[lane]=T.float32(0)
                                    vv[lane]=T.float32(0)
                                    nc[lane]=0
                                if level==0:
                                    pm[lane]=nm[lane]
                                    pv[lane]=vv[lane]
                                    pc[lane]=nc[lane]
                                else:
                                    count=pc[lane]+nc[lane]
                                    ratio=T.if_then_else(count==0,T.float32(0),T.cast(nc[lane],'float32')/T.cast(count,'float32'))
                                    delta=nm[lane]-pm[lane]
                                    cd=ratio*delta
                                    md=delta*T.cast(pc[lane],'float32')
                                    pm[lane]=pm[lane]+cd
                                    pv[lane]=T.call_extern('float32','fmaf',md,cd,pv[lane]+vv[lane])
                                    pc[lane]=count
                            lm[lane]=pm[lane]
                            lv[lane]=pv[lane]
                        am=T.alloc_fragment((1,),'float32')
                        av=T.alloc_fragment((1,),'float32')
                        ac=T.alloc_fragment((1,),'float32')
                        for one in T.Parallel(1):
                            am[one]=0
                            av[one]=0
                            ac[one]=0
                            if n % 8:
                                for j in T.serial(n % 8):
                                    delta = x[nv * 8 + j] - am[one]
                                    ac[one] = ac[one] + 1
                                    am[one] = am[one] + delta / ac[one]
                                    av[one] = T.call_extern('float32', 'fmaf', delta, x[nv * 8 + j] - am[one], av[one])
                            for lane in T.unroll(8):
                                count = ac[one] + nv
                                ratio = T.if_then_else(count == 0, T.float32(0), T.float32(nv) / count)
                                delta = lm[lane] - am[one]
                                am[one] = T.call_extern('float32', 'fmaf', ratio, delta, am[one])
                                term = delta * delta * ratio
                                merged = T.call_extern('float32', 'fmaf', term, ac[one], lv[lane])
                                av[one] = av[one] + merged
                                ac[one] = count
                            estimate = av[one] / n
                            remainder = T.call_extern('float32', 'fmaf', -estimate, T.float32(n), av[one])
                            variance = estimate + remainder / n
                            argument = variance + T.float32(eps_hi)
                            displaced = argument - variance
                            low = variance - (argument - displaced) + (T.float32(eps_hi) - displaced) + T.float32(eps_lo)
                            inverse = T.rsqrt(argument)
                            square = inverse * inverse
                            square_error = T.call_extern('float32', 'fmaf', inverse, inverse, -square)
                            error = T.call_extern('float32', 'fmaf', -argument, square, T.float32(1)) - argument * square_error - low * square
                            corrected = inverse + T.float32(0.5) * inverse * error
                            av[one] = T.if_then_else(variance == 0, T.float32(zero_inverse), corrected)
                            acc[0]=am[one]
                            acc[1]=av[one]
                    with T.SimdVF():
                        mask=T.simd.pset(32)
                        mean=T.simd.vld(acc[0],dist='BRC_B32');inverse=T.simd.vld(acc[1],dist='BRC_B32')
                        offset=T.simd.alloc_local(1,'float32');value=T.simd.alloc_local(1,'float32')
                        for channel in range(cpg):
                            scale=inverse
                            if weighted:scale=T.simd.vmul(scale,T.simd.vld(w[(row%groups)*cpg+channel],dist='BRC_B32'),mask)
                            bias=T.simd.vdup(T.float32(0),'float32')
                            if biased:bias=T.simd.vld(b[(row%groups)*cpg+channel],dist='BRC_B32')
                            if weighted or biased:
                                offset[0]=T.simd.vneg(scale,mask)
                                T.simd.vmadd(offset[0],mean,bias,mask)
                            for i in range((spatial+63)//64):
                                tail=T.simd.update_mask(T.min(64,spatial-i*64),32)
                                value[0]=T.simd.vld(x[channel*spatial+i*64])
                                if weighted or biased:T.simd.vmadd(value[0],scale,offset[0],tail)
                                else:value[0]=T.simd.vmul(T.simd.vsub(value[0],mean,tail),inverse,tail)
                                T.simd.vsts(y[channel*spatial+i*64],value[0],tail)
                    T.copy(y[:n],Y[row,:n])
    return tilelang.compile(main,target='ascend',out_idx=[3])

