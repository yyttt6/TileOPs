"""INT8 candidate: SIMD row reductions, explicit dtype rounding, SIMT final packing."""
import functools
import tilelang
import tilelang.ascend.language as T
@functools.lru_cache(maxsize=128)
def build(m,n,dtype,quant_dtype,eps,cap):
    assert quant_dtype=='float8_e4m3fn' and n%128==0 and n<=8192
    rows=4 if n<=5120 and m%4==0 else (2 if m%2==0 else 1)
    groups=m//rows;blocks=min(64,groups)
    @T.prim_func
    def main(X:T.Tensor((m,n),dtype),R:T.Tensor((m,n),dtype),W:T.Tensor((n,),dtype),Q:T.Tensor((m,n),quant_dtype),S:T.Tensor((m,1),'float32'),O:T.Tensor((m,n),dtype)):
        with T.Kernel(blocks) as bx:
            x=T.alloc_shared((rows,n),dtype);r=T.alloc_shared((rows,n),dtype);w=T.alloc_shared((n,),dtype)
            sx=T.view(x,(rows,n//2),dtype='float32');sr=T.view(r,(rows,n//2),dtype='float32')
            if n==5120 and rows==4:
                q=T.view(x,(rows,n*2),dtype=quant_dtype)
            else:
                q=T.alloc_shared((rows,n),quant_dtype)
            sc=T.alloc_shared((rows,1),'float32')
            if n==5120 and rows==4:
                T.annotate_buffer_versions({x:2,r:2,sc:2})
            else:
                T.annotate_buffer_versions({x:2,r:2,q:2,sc:2})
            T.copy(W[:n],w[:n])
            for step in T.Pipelined((groups+blocks-1)//blocks,num_stages=2):
                row=(step*blocks+bx)*rows
                if row<m:
                    T.copy(X[row:row+rows,:n],x[:rows,:n]);T.copy(R[row:row+rows,:n],r[:rows,:n])
                    with T.SimdVF():
                        mask=T.simd.pset(32);narrow=T.simd.pset(16);byte_mask=T.simd.pset(8)
                        zero16=T.simd.vdup(T.int16(0),'int16',narrow)
                        signless=T.simd.vdup(T.int16(32767),'int16',narrow)
                        lowbyte=T.simd.vdup(T.uint16(255),'uint16',narrow)
                        zero=T.simd.vdup(T.float32(0),'float32',mask);one=T.simd.vdup(T.float32(1),'float32',mask)
                        for j in range(rows):
                            total=T.simd.alloc_local(1,'float32');total[0]=zero
                            for i in range(n//128):
                                px=T.simd.vld(x[j,i*128]);pr=T.simd.vld(r[j,i*128])
                                a=T.simd.vadd(T.simd.vcvt(px,'float32',mask=narrow,part=0),T.simd.vcvt(pr,'float32',mask=narrow,part=0),mask)
                                b=T.simd.vadd(T.simd.vcvt(px,'float32',mask=narrow,part=1),T.simd.vcvt(pr,'float32',mask=narrow,part=1),mask)
                                T.simd.vsts(sx[j,i*64],a,mask)
                                T.simd.vsts(sr[j,i*64],b,mask)
                                total[0]=T.simd.vadd(total[0],T.simd.vadd(T.simd.vmul(a,a,mask),T.simd.vmul(b,b,mask),mask),mask)
                            T.simd.mem_bar("VST_VLD")
                            mean=T.simd.vmuls(T.simd.vdupv(T.simd.vcadd(total[0],mask),mask),T.float32(1.0/n),mask)
                            inverse=T.simd.vdiv(one,T.simd.vsqrt(T.simd.vadds(mean,T.float32(eps),mask),mask),mask,precision='ftz_true')
                            peak=T.simd.alloc_local(1,'int16');peak[0]=zero16
                            for i in range(n//128):
                                pw=T.simd.vld(w[i*128])
                                a=T.simd.vld(sx[j,i*64]);b=T.simd.vld(sr[j,i*64])
                                lo=T.simd.vcvt(a,dtype,mask=narrow,part=0,sat=False);hi=T.simd.vcvt(b,dtype,mask=narrow,part=1,sat=False)
                                residual_packed=T.reinterpret(T.simd.vor(T.reinterpret(lo,'uint16x128'),T.reinterpret(hi,'uint16x128'),narrow),dtype+'x128')
                                T.simd.vsts(r[j,i*128],residual_packed,narrow)
                                na=T.simd.vcvt(T.simd.vmul(a,inverse,mask),dtype,mask=narrow,part=0,sat=False)
                                nb=T.simd.vcvt(T.simd.vmul(b,inverse,mask),dtype,mask=narrow,part=1,sat=False)
                                normalized_packed=T.reinterpret(T.simd.vor(T.reinterpret(na,'uint16x128'),T.reinterpret(nb,'uint16x128'),narrow),dtype+'x128')
                                packed=T.simd.vmul(normalized_packed,pw,narrow)
                                T.simd.vsts(x[j,i*128],packed,narrow)
                                absolute_bits=T.simd.vand(T.reinterpret(packed,'int16x128'),signless,narrow)
                                absolute_bits=T.simd.vsel(absolute_bits,zero16,T.simd.vcmps(absolute_bits,T.int16(32640 if dtype=='bfloat16' else 31744),narrow,op='le'))
                                peak[0]=T.simd.vmax(peak[0],absolute_bits,narrow)
                            T.simd.mem_bar("VST_VLD")
                            peak_bits=T.simd.vcmax(peak[0],narrow)
                            maximum=T.simd.vdupv(T.simd.vcvt(T.reinterpret(peak_bits,dtype+'x128'),'float32',mask=narrow,part=0),mask)
                            if cap is not None:maximum=T.simd.vmins(maximum,T.float32(cap),mask)
                            scale=T.simd.vmaxs(T.simd.vmuls(maximum,T.float32(1.0/448),mask),T.float32(1.0/(448*512)),mask)
                            T.simd.vsts(sc[j,0],scale,mask,dist='ONEPT_B32')
                            invscale=T.simd.vdiv(one,scale,mask,precision='ftz_true')
                            for i in range(n//128):
                                packed=T.simd.vld(x[j,i*128])
                                even=T.simd.vcvt(packed,'float32',mask=narrow,part=0);odd=T.simd.vcvt(packed,'float32',mask=narrow,part=1)
                                qa=T.simd.vmul(even,invscale,mask);qb=T.simd.vmul(odd,invscale,mask)
                                if cap is not None:
                                    qa=T.simd.vsel(qa,T.simd.vdup(T.float32(448),'float32',mask),T.simd.vcmp(qa,qa,mask,op='eq'))
                                    qb=T.simd.vsel(qb,T.simd.vdup(T.float32(448),'float32',mask),T.simd.vcmp(qb,qb,mask,op='eq'))
                                if cap is not None:
                                    qa=T.simd.vmins(T.simd.vmaxs(qa,T.float32(-448),mask),T.float32(448),mask)
                                    qb=T.simd.vmins(T.simd.vmaxs(qb,T.float32(-448),mask),T.float32(448),mask)
                                f0=T.simd.vcvt(qa,'float8_e4m3fn',mask=byte_mask,round='ROUND_R',sat=False,part=0)
                                f1=T.simd.vcvt(qb,'float8_e4m3fn',mask=byte_mask,round='ROUND_R',sat=False,part=2)
                                joined=T.simd.vor(T.reinterpret(f0,'uint8x256'),T.reinterpret(f1,'uint8x256'),byte_mask)
                                if cap is None:
                                    bits16=T.reinterpret(joined,'uint16x128')
                                    absolute=T.simd.vand(bits16,T.simd.vdup(T.uint16(127),'uint16',narrow),narrow)
                                    repaired=T.simd.vsel(T.simd.vdup(T.uint16(126),'uint16',narrow),bits16,T.simd.vcmps(absolute,T.uint16(127),narrow,op='eq'))
                                    joined=T.reinterpret(repaired,'uint8x256')
                                T.simd.vsts(q[j,i*128],T.reinterpret(joined,'float8_e4m3fnx256'),byte_mask,dist='PK_B16')
                    T.copy(q[:rows,:n],Q[row:row+rows,:n]);T.copy(r[:rows,:n],O[row:row+rows,:n]);T.copy(sc[:rows,:1],S[row:row+rows,:1])
    return tilelang.compile(main,target='ascend',out_idx=[3,4,5],pass_configs={'tirx.disable_vectorize':True})
