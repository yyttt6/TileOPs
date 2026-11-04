"""Candidate: resident full row, preserving original cascade and rounding order."""
import functools
import tilelang
import tilelang.ascend.language as T
@functools.lru_cache(maxsize=128)
def build(m,n,dtype,quant_dtype,eps,cap):
    assert n%32==0 and n<=8192
    width=n;blocks=min(64,m)
    groups=n//32
    power=max(4,(max(1,groups)-1).bit_length()//4)
    step=1<<power
    width=max(width,step*32)
    maximum=127.0 if quant_dtype=='int8' else 448.0
    minimum=-128.0 if quant_dtype=='int8' else -448.0
    @T.prim_func
    def main(X:T.Tensor((m,n),dtype),R:T.Tensor((m,n),dtype),W:T.Tensor((n,),dtype),Q:T.Tensor((m,n),quant_dtype),S:T.Tensor((m,1),'float32'),O:T.Tensor((m,n),dtype)):
        with T.Kernel(blocks) as bx:
            x=T.alloc_shared((width,),dtype);r=T.alloc_shared((width,),dtype);w=T.alloc_shared((width,),dtype)
            q=T.alloc_shared((width,),quant_dtype);o=T.alloc_shared((width,),dtype)
            stats=T.alloc_shared((16,),'float32')
            scale_out=T.alloc_shared((16,),'float32')
            squared=T.alloc_shared((width,),"float32")
            weighted_row=T.alloc_shared((width,),dtype)
            cascade=T.alloc_shared((4,32),"float32")
            partial=T.alloc_shared((32,),"float32")
            T.copy(W[:n],w[:n])
            for row_step in T.serial((m+blocks-1)//blocks):
                row=row_step*blocks+bx
                if row<m:
                    T.clear(stats)
                    T.clear(cascade)
                    T.copy(X[row,:n],x[:n]);T.copy(R[row,:n],r[:n])
                    with T.SimtVF(threads=256):
                        for i in T.Parallel(n):
                            value=T.cast(x[i],'float32')+T.cast(r[i],'float32')
                            squared[i]=value*value
                    with T.SimtVF(threads=32):
                        state0=T.alloc_fragment((32,),'float32')
                        state1=T.alloc_fragment((32,),'float32')
                        state2=T.alloc_fragment((32,),'float32')
                        state3=T.alloc_fragment((32,),'float32')
                        for lane in T.Parallel(32):
                            state0[lane]=0
                            state1[lane]=0
                            state2[lane]=0
                            state3[lane]=0
                            for chunk in T.serial((groups+step-1)//step):
                                count=T.min(step,groups-chunk*step)
                                start=chunk*step*32
                                for j in T.serial(count):state0[lane]=state0[lane]+squared[start+j*32+lane]
                                if count==step:
                                    if ((chunk+1)*step)%(1<<(power*1))==0:
                                        state1[lane]=state1[lane]+state0[lane]
                                        state0[lane]=0
                                    if ((chunk+1)*step)%(1<<(power*2))==0:
                                        state2[lane]=state2[lane]+state1[lane]
                                        state1[lane]=0
                                    if ((chunk+1)*step)%(1<<(power*3))==0:
                                        state3[lane]=state3[lane]+state2[lane]
                                        state2[lane]=0
                        for lane in T.Parallel(32):
                            state0[lane]=state0[lane]+state1[lane]
                            state0[lane]=state0[lane]+state2[lane]
                            state0[lane]=state0[lane]+state3[lane]
                        for lane in T.Parallel(32):cascade[0,lane]=state0[lane]
                    if n%32:
                        T.copy(X[row,groups*32:n],x[:n%32]);T.copy(R[row,groups*32:n],r[:n%32])
                        with T.SimtVF(threads=32):
                            for i in T.Parallel(32):
                                if i<n%32:
                                    value=T.cast(x[i],'float32')+T.cast(r[i],'float32')
                                    squared[i]=value*value
                        with T.SimtVF(threads=32):
                            for lane in T.Parallel(32):
                                for j in T.serial((n%32)//8):
                                    cascade[0,lane]=cascade[0,lane]+T.if_then_else(lane<8,squared[j*8+lane%8],T.float32(0))
                    with T.SimtVF(threads=32):
                        for lane in T.Parallel(32):
                            partial[lane]=cascade[0,lane%8]
                            for ilp in T.serial(1,4):partial[lane]=partial[lane]+cascade[0,ilp*8+lane%8]
                        for j in T.serial(n%8):stats[0]=stats[0]+squared[((n%32)//8)*8+j]
                        for lane in T.serial(8):stats[0]=stats[0]+partial[lane]
                        mean=stats[0]/n
                        mean_remainder=T.call_extern('float32','fmaf',-mean,T.float32(n),stats[0])
                        stats[3]=(mean+mean_remainder/n)+T.float32(eps)
                        root_estimate=T.sqrt(stats[3])
                        root_remainder=T.call_extern('float32','fmaf',-root_estimate,root_estimate,stats[3])
                        root=root_estimate+root_remainder/(T.float32(2)*root_estimate)
                        stats[4]=root
                        inverse=T.float32(1)/root
                        remainder=T.call_extern('float32','fmaf',-inverse,root,T.float32(1))
                        stats[0]=inverse+remainder/root
                    for chunk in T.serial((n+width-1)//width):
                        start=chunk*width;valid=T.min(width,n-start)
                        with T.SimtVF(threads=256):
                            peak=T.alloc_reducer((1,),'float32',op='max');T.reducer_init(peak)
                            for i in T.Parallel(width):
                                if i<valid:
                                    value=T.cast(x[i],'float32')+T.cast(r[i],'float32')
                                    normalized=T.cast(value*stats[0],dtype)
                                    o[i]=value
                                    weighted=T.cast(T.cast(normalized,'float32')*T.cast(w[i],'float32'),dtype)
                                    weighted_row[i]=weighted
                                    T.reducer_update(peak[0],T.abs(T.cast(weighted,'float32')))
                            result=T.alloc_fragment((1,),'float32');T.finalize_reducer(peak,result)
                            stats[1]=T.max(stats[1],result[0])
                    with T.SimtVF(threads=32):
                        if cap is not None:stats[1]=T.min(stats[1],T.float32(cap))
                        estimate=stats[1]/maximum
                        remainder=T.call_extern('float32','fmaf',-estimate,T.float32(maximum),stats[1])
                        stats[2]=estimate+remainder*T.float32(1.0/maximum)
                        stats[2]=T.max(stats[2],T.float32(1.0/(448.0*512.0) if quant_dtype=='float8_e4m3fn' else 1.1920928955078125e-7))
                        scale_out[0]=stats[2]
                    T.copy(scale_out[:1],S[row,:1])
                    with T.SimtVF(threads=1024):
                        for chunk in T.serial((n+1023)//1024):
                            start=chunk*1024;valid=T.min(1024,n-start)
                            for i in T.Parallel(1024):
                                if i<valid:
                                    weighted=weighted_row[start+i]
                                    denominator=T.if_then_else(stats[2]==0,T.float32(1),stats[2])
                                    quotient=T.cast(weighted,'float32')/denominator
                                    remainder=T.call_extern('float32','fmaf',-quotient,denominator,T.cast(weighted,'float32'))
                                    divided=quotient+remainder/denominator
                                    if quant_dtype=='int8':divided=T.nearbyint(divided)
                                    converted=T.cast(T.min(T.max(divided,T.float32(minimum)),T.float32(maximum)),quant_dtype)
                                    if quant_dtype=='float8_e4m3fn':
                                        signed_zero=T.reinterpret(T.cast((T.reinterpret(T.cast(weighted,'float32'),'uint32') >> 24) & 128,'uint8'),'float8_e4m3fn')
                                        q[start+i]=T.if_then_else(T.cast(weighted,'float32')==0,signed_zero,converted)
                                    else:
                                        q[start+i]=converted
                    T.copy(q[:n],Q[row,:n]);T.copy(o[:n],O[row,:n])
    return tilelang.compile(main,target='ascend',out_idx=[3,4,5])
