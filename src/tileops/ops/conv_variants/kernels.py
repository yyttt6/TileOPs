"""T307 original TileLang transforms and convolution compositions.

Forward arithmetic reuses a byte-identical snapshot of this workspace's
TileOPs/src/tileops/kernels/{convolution,gemm,common}.py (see provenance.json).
No third-party snippets are introduced here. The copied GEMM retains its
Tile-AI attribution in its original module docstring.
Torch is used only for allocation and layout views/copies, never convolution,
unfold, matmul, autograd, bias arithmetic or ReLU in the implementation.
"""
from functools import lru_cache
import math
import torch
import tilelang
import tilelang.language as T
from .donor_convolution import build_conv2d_kernel
from .gemm import build_gemm_kernel


def pair(value):
    return tuple(value) if isinstance(value, (tuple,list)) else (value,value)


#: Output lanes staged in UB per chunk by the contiguous unfold (32 KiB at fp16).
_UNFOLD_TILE_MAX = 16384
#: Launch geometry, unchanged from the scalar `transform` it replaces.
_UNFOLD_BLOCKS = 48


def _unfold_plan(shape, params, dtype):
    """Geometry for the contiguous unfold, or ``None`` to refuse.

    ADMISSION.  The scalar `transform('unfold')` computes one address per output
    element and issues one scalar GM load for it.  R345 measured what that costs
    on ``Im2colOp/highres-3x3-s1/fp16``: ``aiv_scalar_ratio`` 0.989,
    ``aiv_scalar_time`` 58,830 us of a 59,557 us kernel, and
    ``aiv_mte2_time`` = **0.002 us** -- the vector LOAD pipe never runs at all.

    The fix is to make the staging a ``T.copy``, which needs the lanes of a
    chunk to be CONTIGUOUS in the input.  Write ``p = oy*ow + ox`` for the flat
    lane inside one ``(b, c, ky, kx)`` output plane.  If the horizontal stride is
    1 AND ``ow == w``, then

        flat_in(p) = bc*h*w + p + delta,
        delta = (ky*dh - ph)*w + (kx*dw - pw)

    with ``delta`` a per-plane scalar: the whole plane is one contiguous run.
    Lanes whose shifted coordinate leaves its axis are still READ (the address is
    inside the tensor, just in the wrong row) and are zeroed afterwards, which is
    correct because im2col defines those lanes as zero.  This is the same
    argument `_vector_window_plan` makes for the convolution kernel in
    ``tileops/kernels/convolution.py``; only the admission is different, because
    unfold has no ``oh == h`` requirement (nothing accumulates across rows).

    Deliberately NOT admitted, and why:
      * horizontal stride > 1 -- the lanes are then strided in the input, not
        contiguous, and ``T.copy`` cannot express that.
      * ``ow != w`` -- ``flat_in`` is then not ``p + const`` and the argument
        collapses.
      * a plane too short to hold one 32-byte-aligned row group.  Those are the
        7x7 workloads, whose whole unfold is 49 lanes per plane.
    """
    n, ci, h, w = shape
    kh, kw, sh, sw, ph, pw, dh, dw = params
    oh = (h + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    ow = (w + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    if sw != 1 or ow != w:
        return None
    itemsize = 4 if dtype == 'float32' else 2
    plane = oh * ow
    # A chunk is a whole number of output ROWS, so the row-band zeroing below
    # indexes it without a division, and its extent is a multiple of 32 bytes,
    # so the DMA is not split.  Both are what `_vector_window_plan`'s
    # `group_rows` search buys the convolution kernel.
    unit_rows = 1
    while (unit_rows * ow * itemsize) % 32:
        unit_rows += 1
    unit = unit_rows * ow
    if unit > plane or unit > _UNFOLD_TILE_MAX:
        return None
    tile = (min(_UNFOLD_TILE_MAX, plane) // unit) * unit
    return dict(oh=oh, ow=ow, plane=plane, tile=tile, unit=unit,
                rows=tile // ow, unit_groups=tile // unit,
                chunks=-(-plane // tile), planes=n * ci * kh * kw,
                numel_x=n * ci * h * w, total=n * ci * kh * kw * plane)


@lru_cache(None)
def _unfold_contiguous(shape, params, dtype):
    """im2col whose window staging is ONE contiguous ``T.copy``, not a gather.

    Admission and the ``p + delta`` argument are documented on `_unfold_plan`.
    Per chunk this kernel issues 1 vector load and 1 vector store where the
    scalar kernel it replaces issued ``tile`` scalar GM loads.
    """
    n, ci, h, w = shape
    kh, kw, sh, sw, ph, pw, dh, dw = params
    plan = _unfold_plan(shape, params, dtype)
    ow = plan['ow']; plane = plan['plane']; tile = plan['tile']
    unit = plan['unit']; rows = plan['rows']; unit_groups = plan['unit_groups']
    chunks = plan['chunks']; planes = plan['planes']
    numel_x = plan['numel_x']; total = plan['total']
    last_start = plane - tile
    items = planes * chunks
    blocks = min(_UNFOLD_BLOCKS, -(-items // 2))
    repeats = -(-items // (blocks * 2))

    @tilelang.jit(out_idx=[-1], pass_configs={tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
        tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True}, compile_flags=['-O3','-DENABLE_BF16'])
    def factory():
        @T.prim_func
        def main(x: T.Tensor((numel_x,), dtype), y: T.Tensor((total,), dtype)):
            with T.Kernel(blocks, is_npu=True) as (bid, vid):
                buf = T.alloc_ub((tile,), dtype)
                with T.Scope('V'):
                    for repeat in T.serial(repeats):
                        item = (repeat * blocks + bid) * 2 + vid
                        if item < items:
                            chunk = item % chunks
                            pl = item // chunks
                            kx = pl % kw
                            ky = pl // kw % kh
                            bc = pl // (kw * kh)
                            # The last chunk of a plane is CLAMPED, not
                            # truncated: it recomputes the lanes the previous
                            # chunk already produced and stores the same values
                            # over them.  A store is idempotent, so this removes
                            # the need for `tile` to divide `plane` and keeps
                            # every copy extent a compile-time constant.
                            start = T.if_then_else(chunk * tile < last_start,
                                                   chunk * tile, last_start)
                            cshift = kx * dw - pw
                            rshift = ky * dh - ph
                            src = bc * (h * w) + start + rshift * w + cshift
                            T.tile.fill(buf, 0.0)
                            if src >= 0 and src + tile <= numel_x:
                                T.copy(x[src:src + tile], buf)
                            else:
                                # Boundary chunk: part of its source range is
                                # outside the tensor.  Guard each 32-byte-aligned
                                # sub-block and fall back to per-lane moves only
                                # for the ones that really are out.  Those lanes
                                # are all zeroed below anyway; this branch exists
                                # so the kernel never dereferences outside x.
                                for grp in T.serial(unit_groups):
                                    gsrc = src + grp * unit
                                    if gsrc >= 0 and gsrc + unit <= numel_x:
                                        T.copy(x[gsrc:gsrc + unit],
                                               buf[grp * unit:grp * unit + unit])
                                    else:
                                        for lane in T.serial(unit):
                                            if gsrc + lane >= 0 and gsrc + lane < numel_x:
                                                buf[grp * unit + lane] = x[gsrc + lane]
                            lo = T.if_then_else(cshift < 0, -cshift, 0)
                            hi = T.if_then_else(cshift > 0, cshift, 0)
                            for r in T.serial(rows):
                                iy = start // ow + r + rshift
                                if iy < 0 or iy >= h or lo >= ow or hi >= ow:
                                    for col in T.serial(ow):
                                        buf[r * ow + col] = 0
                                else:
                                    for col in T.serial(lo):
                                        buf[r * ow + col] = 0
                                    for col in T.serial(hi):
                                        buf[r * ow + ow - 1 - col] = 0
                            T.copy(buf, y[pl * plane + start:pl * plane + start + tile])
        return main
    return factory()


@lru_cache(None)
def transform(mode, shape, params, dtype):
    """Copy-only transforms: unfold, zero insertion, weight adjoint, causal pad.

    Flattened output index defines every source index; invalid locations are
    zero. No scatter or atomics: each output is written once, including tails.
    """
    if mode == 'unfold':
        n, ci, h, w = shape
        kh,kw,sh,sw,ph,pw,dh,dw = params
        oh,ow = (h+2*ph-dh*(kh-1)-1)//sh+1,(w+2*pw-dw*(kw-1)-1)//sw+1
        total=n*ci*kh*kw*oh*ow
        # Contiguous staging when the geometry admits it; the scalar body below
        # stays as the fallback for the shapes `_unfold_plan` refuses.
        if _unfold_plan(shape, params, dtype) is not None:
            return _unfold_contiguous(shape, params, dtype)
    elif mode == 'insert':
        n,ci,h,w=shape
        sh,sw,oph,opw=params
        oh,ow=(h-1)*sh+1+oph,(w-1)*sw+1+opw
        total=n*ci*oh*ow
    elif mode == 'adjoint':
        co,cig,kh,kw=shape
        groups,=params
        cog=co//groups
        total=math.prod(shape)
    elif mode == 'pad1d':
        n,ci,length=shape
        left,=params
        total=n*ci*(length+left)
    else:
        raise ValueError(mode)
    size=math.prod(shape)
    tile=256
    blocks=min(48, (total+2*tile-1)//(2*tile))
    repeats=(total+blocks*2*tile-1)//(blocks*2*tile)

    @tilelang.jit(out_idx=[-1], pass_configs={tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
        tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True}, compile_flags=['-O3','-DENABLE_BF16'])
    def factory():
        @T.prim_func
        def main(x:T.Tensor((size,),dtype), y:T.Tensor((total,),dtype)):
            with T.Kernel(blocks,is_npu=True) as (bid,vid):
                buf=T.alloc_ub((tile,),dtype)
                with T.Scope('V'):
                    for repeat in T.serial(repeats):
                        base=(repeat*blocks+bid)*2*tile+vid*tile
                        for lane in T.Parallel(tile):
                            idx=base+lane
                            buf[lane]=0
                            if idx<total:
                                if mode=='unfold':
                                    ox=idx%ow
                                    oy=idx//ow%oh
                                    kx=idx//(oh*ow)%kw
                                    ky=idx//(oh*ow*kw)%kh
                                    c=idx//(oh*ow*kw*kh)%ci
                                    b=idx//(oh*ow*kw*kh*ci)
                                    ix=ox*sw-pw+kx*dw
                                    iy=oy*sh-ph+ky*dh
                                    if ix>=0 and ix<w and iy>=0 and iy<h:
                                        buf[lane]=x[((b*ci+c)*h+iy)*w+ix]
                                elif mode=='insert':
                                    ox=idx%ow
                                    oy=idx//ow%oh
                                    bc=idx//(oh*ow)
                                    if ox%sw==0 and oy%sh==0 and ox//sw<w and oy//sh<h:
                                        buf[lane]=x[(bc*h+oy//sh)*w+ox//sw]
                                elif mode=='adjoint':
                                    kx=idx%kw
                                    ky=idx//kw%kh
                                    oc=idx//(kw*kh)%cog
                                    ic=idx//(kw*kh*cog)%cig
                                    g=idx//(kw*kh*cog*cig)
                                    buf[lane]=x[(((g*cog+oc)*cig+ic)*kh+kh-1-ky)*kw+kw-1-kx]
                                else:
                                    t=idx%(length+left)-left
                                    bc=idx//(length+left)
                                    if t>=0 and t<length:
                                        buf[lane]=x[bc*length+t]
                        for lane in T.Parallel(tile):
                            if base+lane<total:
                                y[base+lane]=buf[lane]
        return main
    return factory()


@lru_cache(None)
def relu_kernel(total,dtype):
    tile=256
    blocks=min(48,(total+511)//512)
    repeats=(total+blocks*512-1)//(blocks*512)
    @tilelang.jit(out_idx=[-1], pass_configs={tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC:True,
        tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING:True},compile_flags=['-O3','-DENABLE_BF16'])
    def factory():
        @T.prim_func
        def main(x:T.Tensor((total,),dtype),y:T.Tensor((total,),dtype)):
            with T.Kernel(blocks,is_npu=True) as (bid,vid):
                buf=T.alloc_ub((tile,),dtype)
                wide=T.alloc_ub((tile,),'float32')
                with T.Scope('V'):
                    for r in T.serial(repeats):
                        base=(r*blocks+bid)*512+vid*256
                        for j in T.Parallel(tile):
                            buf[j]=0
                            if base+j<total:
                                buf[j]=x[base+j]
                        T.tile.cast(wide,buf,'CAST_NONE',tile)
                        T.tile.max(wide,wide,0.0)
                        T.tile.cast(buf,wide,'CAST_RINT',tile)
                        for j in T.Parallel(tile):
                            if base+j<total:
                                y[base+j]=buf[j]
        return main
    return factory()


def forward(x, weight, bias, w):
    kernel=build_conv2d_kernel(tuple(x.shape),tuple(weight.shape),x.dtype,
        stride=pair(w.get('stride',1)),padding=pair(w.get('padding',0)),
        dilation=pair(w.get('dilation',1)),groups=w.get('groups',1),has_bias=bias is not None)
    return kernel(x,weight,bias)


def causal(x,weight,bias,w):
    n,ci,length=x.shape
    left=w.get('dilation',1)*(w['kW']-1)
    padded=transform('pad1d',tuple(x.shape),(left,),str(x.dtype).split('.')[-1])(x.reshape(-1))
    params=dict(stride=(1,w.get('stride',1)),padding=0,dilation=(1,w.get('dilation',1)),groups=w.get('groups',1))
    y=forward(padded.reshape(n,ci,1,length+left),weight.unsqueeze(2),bias,params)
    return y.squeeze(2)


def adjoint(grad_output,weight,bias,w):
    n,ci,h,width=w['input_shape']
    co=w['C_out']
    kh,kw=w['kH'],w['kW']
    sh,sw=pair(w.get('stride',1)); ph,pw=pair(w.get('padding',0)); dh,dw=pair(w.get('dilation',1))
    groups=w.get('groups',1)
    oh,ow=grad_output.shape[-2:]
    oph=h-((oh-1)*sh-2*ph+dh*(kh-1)+1)
    opw=width-((ow-1)*sw-2*pw+dw*(kw-1)+1)
    assert 0<=oph<sh and 0<=opw<sw
    dtype=str(weight.dtype).split('.')[-1]
    if sh==1 and sw==1 and oph==0 and opw==0:
        # Zero insertion with unit stride and no output padding is the IDENTITY:
        # every output lane copies x[bc, oy, ox] with oy==oy//1 and ox==ox//1, and
        # the output extents equal the input extents.  Running it anyway costs a
        # whole kernel launch plus a full read-and-write of grad_output through
        # the scalar-gather `transform` -- R345 measured that path at 195 ns of
        # AIV scalar-pipe busy per element (aiv_mte2_time = 0.002 us, i.e. the
        # vector load pipe never runs).  Skipping it is bit-exact by definition.
        z=grad_output
    else:
        z=transform('insert',tuple(grad_output.shape),(sh,sw,oph,opw),dtype)(grad_output.reshape(-1))
    z=z.reshape(n,co,(oh-1)*sh+1+oph,(ow-1)*sw+1+opw)
    wt=transform('adjoint',tuple(weight.shape),(groups,),dtype)(weight.reshape(-1))
    wt=wt.reshape(ci,co//groups,kh,kw)
    # These donors all have nonnegative adjoint padding. No chosen padding.
    assert dh*(kh-1)-ph>=0 and dw*(kw-1)-pw>=0
    return forward(z,wt,bias,dict(stride=1,padding=(dh*(kh-1)-ph,dw*(kw-1)-pw),dilation=(dh,dw),groups=groups))


def unfold(x,w):
    kh,kw=w['kH'],w['kW']
    s,p,d=pair(w.get('stride',1)),pair(w.get('padding',0)),pair(w.get('dilation',1))
    output=transform('unfold',tuple(x.shape),(kh,kw,*s,*p,*d),str(x.dtype).split('.')[-1])(x.reshape(-1))
    return output.reshape(x.shape[0],x.shape[1]*kh*kw,-1)


def wgrad(x,grad_output,w):
    groups=w.get('groups',1)
    n,ci,_,_=x.shape
    co,kh,kw=w['C_out'],w['kH'],w['kW']
    k=ci//groups*kh*kw
    # Move N beside spatial position, so one GEMM accumulates across all N.
    cols=unfold(x,w).reshape(n,groups,k,-1).permute(1,2,0,3).contiguous().reshape(groups,k,-1)
    grad=grad_output.reshape(n,groups,co//groups,-1).permute(1,2,0,3).contiguous().reshape(groups,co//groups,-1)
    kernel=build_gemm_kernel(tuple(grad[0].shape),tuple(cols[0].shape),x.dtype,trans_a=False,trans_b=True)
    return torch.stack([kernel(grad[g],cols[g]) for g in range(groups)]).reshape(co,ci//groups,kh,kw)
