"""Pure CPU PrimFunc factories: resident short channels, SIMD streamed aligned long channels."""
import tilelang
import tilelang.ascend.language as T

def resident(batch,channels,spatial,dtype):
    length=batch*spatial;width=tilelang.next_power_of_2(length);pitch=max(16,length);blocks=min(64,channels)
    @T.prim_func
    def main(D:T.Tensor((batch,channels,spatial),dtype),X:T.Tensor((batch,channels,spatial),dtype),W:T.Tensor((channels,),'float32'),M:T.Tensor((channels,),'float32'),R:T.Tensor((channels,),'float32'),DX:T.Tensor((batch,channels,spatial),dtype),DW:T.Tensor((channels,),'float32'),DB:T.Tensor((channels,),'float32')):
        with T.Kernel(blocks) as bx:
            x=T.alloc_shared((pitch,),dtype);dy=T.alloc_shared((pitch,),dtype);dx=T.alloc_shared((pitch,),dtype)
            w=T.alloc_shared((channels,),'float32');mean=T.alloc_shared((channels,),'float32');rstd=T.alloc_shared((channels,),'float32')
            T.copy(W[:channels],w[:channels]);T.copy(M[:channels],mean[:channels]);T.copy(R[:channels],rstd[:channels])
            for step in T.serial((channels+blocks-1)//blocks):
                c=step*blocks+bx
                if c<channels:
                    if spatial==1:
                        with T.SimtVF(threads=128):
                            for n in T.Parallel(batch):
                                x[n]=X[n,c,0];dy[n]=D[n,c,0]
                    else:
                        for n in T.serial(batch):
                            T.copy(X[n,c,:spatial],x[n*spatial:(n+1)*spatial]);T.copy(D[n,c,:spatial],dy[n*spatial:(n+1)*spatial])
                    with T.SimtVF(threads=256):
                        db=T.alloc_reducer((1,),'float32',op='sum');dw=T.alloc_reducer((1,),'float32',op='sum');T.reducer_init(db);T.reducer_init(dw)
                        for i in T.Parallel(width):
                            if i<length:
                                g=T.cast(dy[i],'float32');z=(T.cast(x[i],'float32')-mean[c])*rstd[c];T.reducer_update(db[0],g);T.reducer_update(dw[0],g*z)
                        sb=T.alloc_fragment((1,),'float32');sw=T.alloc_fragment((1,),'float32');T.finalize_reducer(db,sb);T.finalize_reducer(dw,sw)
                        for j in T.Parallel(1):DB[c]=sb[0];DW[c]=sw[0]
                        values=T.alloc_fragment((width,),'float32')
                        for i in T.Parallel(width):
                            if i<length:
                                z=(T.cast(x[i],'float32')-mean[c])*rstd[c];values[i]=w[c]*rstd[c]/length*(length*T.cast(dy[i],'float32')-sb[0]-z*sw[0])
                        for i in T.Parallel(width):
                            if i<length:dx[i]=values[i]
                    if spatial==1:
                        with T.SimtVF(threads=128):
                            for n in T.Parallel(batch):DX[n,c,0]=dx[n]
                    else:
                        for n in T.serial(batch):T.copy(dx[n*spatial:(n+1)*spatial],DX[n,c,:spatial])
    return main

def stream(batch,channels,spatial,dtype):
    width=8192;chunks=spatial//width;blocks=min(64,channels);length=batch*spatial
    assert dtype in ('float16','bfloat16') and spatial%width==0
    @T.prim_func
    def main(D:T.Tensor((batch,channels,spatial),dtype),X:T.Tensor((batch,channels,spatial),dtype),W:T.Tensor((channels,),'float32'),M:T.Tensor((channels,),'float32'),R:T.Tensor((channels,),'float32'),DX:T.Tensor((batch,channels,spatial),dtype),DW:T.Tensor((channels,),'float32'),DB:T.Tensor((channels,),'float32')):
        with T.Kernel(blocks) as bx:
            x=T.alloc_shared((width,),dtype);dy=T.alloc_shared((width,),dtype);dx=T.alloc_shared((width,),dtype)
            w=T.alloc_shared((channels,),'float32');mean=T.alloc_shared((channels,),'float32');rstd=T.alloc_shared((channels,),'float32');sums=T.alloc_shared((2,64),'float32')
            T.annotate_buffer_versions({x:2,dy:2,dx:2});T.copy(W[:channels],w[:channels]);T.copy(M[:channels],mean[:channels]);T.copy(R[:channels],rstd[:channels])
            for step in T.serial((channels+blocks-1)//blocks):
                c=step*blocks+bx
                if c<channels:
                    T.clear(sums)
                    for tile in T.Pipelined(batch*chunks,num_stages=2):
                        n=tile//chunks;start=tile%chunks*width
                        T.copy(X[n,c,start:start+width],x[:width]);T.copy(D[n,c,start:start+width],dy[:width])
                        with T.SimdVF():
                            mask=T.simd.pset(32);m=T.simd.vld(mean[c],dist='BRC_B32');r=T.simd.vld(rstd[c],dist='BRC_B32');sb=T.simd.alloc_local(1,'float32');sw=T.simd.alloc_local(1,'float32');sb[0]=T.simd.vld(sums[0,0]);sw[0]=T.simd.vld(sums[1,0])
                            for i in range(width//128):
                                px=T.simd.vld(x[i*128]);pg=T.simd.vld(dy[i*128]);a=T.simd.vcvt(pg,'float32',part=0);b=T.simd.vcvt(pg,'float32',part=1)
                                za=T.simd.vmul(T.simd.vsub(T.simd.vcvt(px,'float32',part=0),m,mask),r,mask);zb=T.simd.vmul(T.simd.vsub(T.simd.vcvt(px,'float32',part=1),m,mask),r,mask)
                                sb[0]=T.simd.vadd(sb[0],T.simd.vadd(a,b,mask),mask);sw[0]=T.simd.vadd(sw[0],T.simd.vadd(T.simd.vmul(a,za,mask),T.simd.vmul(b,zb,mask),mask),mask)
                            T.simd.vsts(sums[0,0],sb[0],mask);T.simd.vsts(sums[1,0],sw[0],mask)
                    with T.SimdVF():
                        mask=T.simd.pset(32);sb=T.simd.vdupv(T.simd.vcadd(T.simd.vld(sums[0,0]),mask),mask);sw=T.simd.vdupv(T.simd.vcadd(T.simd.vld(sums[1,0]),mask),mask);T.simd.vsts(sums[0,0],sb,mask);T.simd.vsts(sums[1,0],sw,mask)
                    T.copy(sums[0,:1],DB[c:c+1]);T.copy(sums[1,:1],DW[c:c+1])
                    for tile in T.Pipelined(batch*chunks,num_stages=2):
                        n=tile//chunks;start=tile%chunks*width
                        T.copy(X[n,c,start:start+width],x[:width]);T.copy(D[n,c,start:start+width],dy[:width])
                        with T.SimdVF():
                            mask=T.simd.pset(32);narrow=T.simd.pset(16);m=T.simd.vld(mean[c],dist='BRC_B32');r=T.simd.vld(rstd[c],dist='BRC_B32');wgt=T.simd.vld(w[c],dist='BRC_B32');factor=T.simd.vmuls(T.simd.vmul(wgt,r,mask),T.float32(1.0/length),mask);sb=T.simd.vld(sums[0,0]);sw=T.simd.vld(sums[1,0])
                            for i in range(width//128):
                                px=T.simd.vld(x[i*128]);pg=T.simd.vld(dy[i*128]);a=T.simd.vcvt(pg,'float32',part=0);b=T.simd.vcvt(pg,'float32',part=1)
                                za=T.simd.vmul(T.simd.vsub(T.simd.vcvt(px,'float32',part=0),m,mask),r,mask);zb=T.simd.vmul(T.simd.vsub(T.simd.vcvt(px,'float32',part=1),m,mask),r,mask)
                                va=T.simd.vmul(factor,T.simd.vsub(T.simd.vsub(T.simd.vmuls(a,T.float32(length),mask),sb,mask),T.simd.vmul(za,sw,mask),mask),mask);vb=T.simd.vmul(factor,T.simd.vsub(T.simd.vsub(T.simd.vmuls(b,T.float32(length),mask),sb,mask),T.simd.vmul(zb,sw,mask),mask),mask)
                                lo=T.simd.vcvt(va,dtype,part=0,sat=False);hi=T.simd.vcvt(vb,dtype,part=1,sat=False);bits=T.simd.vor(T.reinterpret(lo,'uint16x128'),T.reinterpret(hi,'uint16x128'),narrow);T.simd.vsts(dx[i*128],T.reinterpret(bits,dtype+'x128'),narrow)
                        T.copy(dx[:width],DX[n,c,start:start+width])
    return main

def fc(batch,channels,spatial,dtype):
    rows=16;groups=channels//rows;blocks=min(64,groups);width=tilelang.next_power_of_2(batch)
    assert spatial==1 and channels%rows==0
    @T.prim_func
    def main(D:T.Tensor((batch,channels,spatial),dtype),X:T.Tensor((batch,channels,spatial),dtype),W:T.Tensor((channels,),'float32'),M:T.Tensor((channels,),'float32'),R:T.Tensor((channels,),'float32'),DX:T.Tensor((batch,channels,spatial),dtype),DW:T.Tensor((channels,),'float32'),DB:T.Tensor((channels,),'float32')):
        with T.Kernel(blocks) as bx:
            x=T.alloc_shared((batch,rows),dtype);dy=T.alloc_shared((batch,rows),dtype);dx=T.alloc_shared((batch,rows),dtype)
            w=T.alloc_shared((channels,),'float32');mean=T.alloc_shared((channels,),'float32');rstd=T.alloc_shared((channels,),'float32');sums=T.alloc_shared((2,rows),'float32')
            T.copy(W[:channels],w[:channels]);T.copy(M[:channels],mean[:channels]);T.copy(R[:channels],rstd[:channels])
            for step in T.serial((groups+blocks-1)//blocks):
                first=(step*blocks+bx)*rows
                if first<channels:
                    T.copy(X[:batch,first:first+rows,0],x[:batch,:rows]);T.copy(D[:batch,first:first+rows,0],dy[:batch,:rows])
                    with T.SimtVF(threads=256):
                        db=T.alloc_reducer((rows,),'float32',op='sum');dw=T.alloc_reducer((rows,),'float32',op='sum');T.reducer_init(db);T.reducer_init(dw)
                        for r,n in T.Parallel(rows,width):
                            if n<batch:
                                g=T.cast(dy[n,r],'float32');z=(T.cast(x[n,r],'float32')-mean[first+r])*rstd[first+r];T.reducer_update(db[r],g);T.reducer_update(dw[r],g*z)
                        sb=T.alloc_fragment((rows,),'float32');sw=T.alloc_fragment((rows,),'float32');T.finalize_reducer(db,sb);T.finalize_reducer(dw,sw)
                        for r in T.Parallel(rows):sums[0,r]=sb[r];sums[1,r]=sw[r]
                    T.copy(sums[0,:rows],DB[first:first+rows]);T.copy(sums[1,:rows],DW[first:first+rows])
                    with T.SimtVF(threads=256):
                        for n,r in T.Parallel(batch,rows):
                            z=(T.cast(x[n,r],'float32')-mean[first+r])*rstd[first+r];dx[n,r]=w[first+r]*rstd[first+r]/batch*(batch*T.cast(dy[n,r],'float32')-sums[0,r]-z*sums[1,r])
                    T.copy(dx[:batch,:rows],DX[:batch,first:first+rows,0])
    return main

def simd_short(batch,channels,spatial,dtype):
    length=batch*spatial;rows=4 if length<=4096 and channels%4==0 else 1
    pitch=(spatial+63)//64*64;groups=channels//rows;blocks=min(64,groups)
    assert dtype=='float32'
    @T.prim_func
    def main(D:T.Tensor((batch,channels,spatial),dtype),X:T.Tensor((batch,channels,spatial),dtype),W:T.Tensor((channels,),'float32'),M:T.Tensor((channels,),'float32'),R:T.Tensor((channels,),'float32'),DX:T.Tensor((batch,channels,spatial),dtype),DW:T.Tensor((channels,),'float32'),DB:T.Tensor((channels,),'float32')):
        with T.Kernel(blocks) as bx:
            x=T.alloc_shared((batch,rows,pitch),dtype);dy=T.alloc_shared((batch,rows,pitch),dtype);dx=T.alloc_shared((batch,rows,pitch),dtype)
            w=T.alloc_shared((channels,),'float32');mean=T.alloc_shared((channels,),'float32');rstd=T.alloc_shared((channels,),'float32');sums=T.alloc_shared((rows,2,64),'float32')
            T.copy(W[:channels],w[:channels]);T.copy(M[:channels],mean[:channels]);T.copy(R[:channels],rstd[:channels])
            for step in T.serial((groups+blocks-1)//blocks):
                first=(step*blocks+bx)*rows
                if first<channels:
                    for n in T.serial(batch):
                        T.copy(X[n,first:first+rows,:spatial],x[n,:rows,:spatial]);T.copy(D[n,first:first+rows,:spatial],dy[n,:rows,:spatial])
                    with T.SimdVF():
                        mask=T.simd.pset(32);zero=T.simd.vdup(T.float32(0),'float32')
                        for r in range(rows):
                            m=T.simd.vld(mean[first+r],dist='BRC_B32');inv=T.simd.vld(rstd[first+r],dist='BRC_B32');weight=T.simd.vld(w[first+r],dist='BRC_B32');sb=T.simd.alloc_local(1,'float32');sw=T.simd.alloc_local(1,'float32');sb[0]=T.simd.vdup(T.float32(0),'float32');sw[0]=T.simd.vdup(T.float32(0),'float32')
                            for n in range(batch):
                                for i in range(pitch//64):
                                    tail=T.simd.update_mask(T.min(64,spatial-i*64),32);g=T.simd.vsel(T.simd.vld(dy[n,r,i*64]),zero,tail);v=T.simd.vsel(T.simd.vld(x[n,r,i*64]),zero,tail);z=T.simd.vmul(T.simd.vsub(v,m,mask),inv,mask)
                                    sb[0]=T.simd.vadd(sb[0],g,mask);sw[0]=T.simd.vadd(sw[0],T.simd.vmul(g,z,mask),mask)
                            db=T.simd.vdupv(T.simd.vcadd(sb[0],mask),mask);dw=T.simd.vdupv(T.simd.vcadd(sw[0],mask),mask);T.simd.vsts(sums[r,0,0],db,mask);T.simd.vsts(sums[r,1,0],dw,mask);factor=T.simd.vmuls(T.simd.vmul(weight,inv,mask),T.float32(1.0/length),mask)
                            for n in range(batch):
                                for i in range(pitch//64):
                                    tail=T.simd.update_mask(T.min(64,spatial-i*64),32);z=T.simd.vmul(T.simd.vsub(T.simd.vld(x[n,r,i*64]),m,mask),inv,mask);v=T.simd.vmul(factor,T.simd.vsub(T.simd.vsub(T.simd.vmuls(T.simd.vld(dy[n,r,i*64]),T.float32(length),mask),db,mask),T.simd.vmul(z,dw,mask),mask),mask);T.simd.vsts(dx[n,r,i*64],v,tail)
                    with T.SimtVF(threads=32):
                        for r in T.Parallel(rows):DB[first+r]=sums[r,0,0];DW[first+r]=sums[r,1,0]
                    for n in T.serial(batch):T.copy(dx[n,:rows,:spatial],DX[n,first:first+rows,:spatial])
    return main

def simd_short_lowp(batch,channels,spatial,dtype):
    length=batch*spatial;rows=4;pitch=(spatial+127)//128*128;groups=channels//rows;blocks=min(64,groups)
    assert dtype in ('float16','bfloat16') and channels%4==0 and spatial%16==0
    @T.prim_func
    def main(D:T.Tensor((batch,channels,spatial),dtype),X:T.Tensor((batch,channels,spatial),dtype),W:T.Tensor((channels,),'float32'),M:T.Tensor((channels,),'float32'),R:T.Tensor((channels,),'float32'),DX:T.Tensor((batch,channels,spatial),dtype),DW:T.Tensor((channels,),'float32'),DB:T.Tensor((channels,),'float32')):
        with T.Kernel(blocks) as bx:
            x=T.alloc_shared((batch,rows,pitch),dtype);dy=T.alloc_shared((batch,rows,pitch),dtype);dx=T.alloc_shared((batch,rows,pitch),dtype)
            w=T.alloc_shared((channels,),'float32');mean=T.alloc_shared((channels,),'float32');rstd=T.alloc_shared((channels,),'float32');sums=T.alloc_shared((rows,2,64),'float32')
            T.copy(W[:channels],w[:channels]);T.copy(M[:channels],mean[:channels]);T.copy(R[:channels],rstd[:channels])
            for step in T.serial((groups+blocks-1)//blocks):
                first=(step*blocks+bx)*rows
                if first<channels:
                    for n in T.serial(batch):
                        T.copy(X[n,first:first+rows,:spatial],x[n,:rows,:spatial]);T.copy(D[n,first:first+rows,:spatial],dy[n,:rows,:spatial])
                    with T.SimdVF():
                        mask=T.simd.pset(32);narrow=T.simd.pset(16);zero=T.simd.vdup(T.float32(0),'float32')
                        for r in range(rows):
                            m=T.simd.vld(mean[first+r],dist='BRC_B32');inv=T.simd.vld(rstd[first+r],dist='BRC_B32');weight=T.simd.vld(w[first+r],dist='BRC_B32');sb=T.simd.alloc_local(1,'float32');sw=T.simd.alloc_local(1,'float32');sb[0]=T.simd.vdup(T.float32(0),'float32');sw[0]=T.simd.vdup(T.float32(0),'float32')
                            for n in range(batch):
                                for i in range(pitch//128):
                                    tail=T.simd.update_mask(T.min(64,(spatial-i*128)//2),32);px=T.simd.vld(x[n,r,i*128]);pg=T.simd.vld(dy[n,r,i*128])
                                    a=T.simd.vsel(T.simd.vcvt(pg,'float32',part=0),zero,tail);b=T.simd.vsel(T.simd.vcvt(pg,'float32',part=1),zero,tail);va=T.simd.vsel(T.simd.vcvt(px,'float32',part=0),zero,tail);vb=T.simd.vsel(T.simd.vcvt(px,'float32',part=1),zero,tail)
                                    za=T.simd.vmul(T.simd.vsub(va,m,mask),inv,mask);zb=T.simd.vmul(T.simd.vsub(vb,m,mask),inv,mask);sb[0]=T.simd.vadd(sb[0],T.simd.vadd(a,b,mask),mask);sw[0]=T.simd.vadd(sw[0],T.simd.vadd(T.simd.vmul(a,za,mask),T.simd.vmul(b,zb,mask),mask),mask)
                            db=T.simd.vdupv(T.simd.vcadd(sb[0],mask),mask);dw=T.simd.vdupv(T.simd.vcadd(sw[0],mask),mask);T.simd.vsts(sums[r,0,0],db,mask);T.simd.vsts(sums[r,1,0],dw,mask);factor=T.simd.vmuls(T.simd.vmul(weight,inv,mask),T.float32(1.0/length),mask)
                            for n in range(batch):
                                for i in range(pitch//128):
                                    tail=T.simd.update_mask(T.min(128,spatial-i*128),16);px=T.simd.vld(x[n,r,i*128]);pg=T.simd.vld(dy[n,r,i*128]);a=T.simd.vcvt(pg,'float32',part=0);b=T.simd.vcvt(pg,'float32',part=1)
                                    za=T.simd.vmul(T.simd.vsub(T.simd.vcvt(px,'float32',part=0),m,mask),inv,mask);zb=T.simd.vmul(T.simd.vsub(T.simd.vcvt(px,'float32',part=1),m,mask),inv,mask)
                                    va=T.simd.vmul(factor,T.simd.vsub(T.simd.vsub(T.simd.vmuls(a,T.float32(length),mask),db,mask),T.simd.vmul(za,dw,mask),mask),mask);vb=T.simd.vmul(factor,T.simd.vsub(T.simd.vsub(T.simd.vmuls(b,T.float32(length),mask),db,mask),T.simd.vmul(zb,dw,mask),mask),mask)
                                    lo=T.simd.vcvt(va,dtype,part=0,sat=False);hi=T.simd.vcvt(vb,dtype,part=1,sat=False);bits=T.simd.vor(T.reinterpret(lo,'uint16x128'),T.reinterpret(hi,'uint16x128'),narrow);T.simd.vsts(dx[n,r,i*128],T.reinterpret(bits,dtype+'x128'),tail)
                    with T.SimtVF(threads=32):
                        for r in T.Parallel(rows):DB[first+r]=sums[r,0,0];DW[first+r]=sums[r,1,0]
                    for n in T.serial(batch):T.copy(dx[n,:rows,:spatial],DX[n,first:first+rows,:spatial])
    return main


def route(batch,channels,spatial,dtype):
    if dtype not in ('float16','bfloat16','float32') or min(batch,channels,spatial)<=0:return None
    size=4 if dtype=='float32' else 2
    length=batch*spatial
    budget=208*1024
    if spatial==1 and batch&(batch-1)==0 and channels%16==0 and 1<length<=8192 and 3*batch*16*size+12*channels+128<budget:return 'fc'
    rows=4 if length<=4096 and channels%4==0 else 1
    pitch=(spatial+63)//64*64
    if dtype=='float32' and 1<length<=8192 and spatial%16==0 and 3*batch*rows*pitch*size+12*channels+512*rows<budget:return 'simd_short'
    pitch=(spatial+127)//128*128
    if dtype in ('float16','bfloat16') and channels>=256 and channels%4==0 and 1<length<=4096 and spatial%16==0 and 3*batch*4*pitch*size+12*channels+2048<budget:return 'simd_short_lowp'
    if 1<length<=8192 and ((spatial==1 and batch&(batch-1)==0) or spatial%16==0) and 3*max(16,length)*size+12*channels<budget:return 'resident'
    if dtype in ('float16','bfloat16') and spatial%8192==0 and length>8192 and 3*8192*2*size+12*channels+512<budget:return 'stream'
    return None

import functools
@functools.lru_cache(maxsize=128)
def build(batch,channels,spatial,dtype):
    selected=route(batch,channels,spatial,dtype)
    assert selected is not None
    main=globals()[selected](batch,channels,spatial,dtype)
    return tilelang.compile(main,target='ascend',out_idx=[5,6,7])
