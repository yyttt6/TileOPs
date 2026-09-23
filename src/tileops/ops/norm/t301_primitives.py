"""Fixed-geometry TileLang primitives used only by the T301 normalization ops.

Derived from the vector-scope, aligned-DMA and reduction design in
TileOPs/src/tileops/kernels/{elementwise_unary,normalization_spatial}.py.
Rewritten as small composable fp32 kernels; no example imports or autotuning.
See the adjacent NOTICE.T301. Layout copies and allocations use torch; all
arithmetic, reductions and output conversions below execute in TileLang.
"""
from functools import lru_cache
import math
import os

import torch
import tilelang
import tilelang.language as T

_TILE = 1024


def _t362_arm() -> str:
    """R362's control-arm switch.  "base" = the tree as T362 found it."""
    return os.environ.get("TILEOPS_T362_ARM", "r362")
_CONFIG = {tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
           tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
           tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True}


def _name(dtype):
    return str(dtype).removeprefix('torch.')


@lru_cache(maxsize=512)
def _map(size, adtype, bdtype, outdtype, kind, scalar):
    blocks = min(24, max(1, math.ceil(size / (2 * _TILE))))
    repeats = math.ceil(size / (blocks * 2 * _TILE))
    @tilelang.jit(out_idx=[2], pass_configs=_CONFIG)
    def build():
        @T.prim_func
        def main(A: T.Tensor([size], adtype), B: T.Tensor([size], bdtype),
                 O: T.Tensor([size], outdtype)):
            with T.Kernel(blocks, is_npu=True) as (bid, vid):
                ar = T.alloc_ub((_TILE,), adtype)
                br = T.alloc_ub((_TILE,), bdtype)
                a = T.alloc_ub((_TILE,), 'float32')
                b = T.alloc_ub((_TILE,), 'float32')
                out = T.alloc_ub((_TILE,), outdtype)
                with T.Scope('V'):
                    for repeat in T.serial(repeats):
                        T.barrier_all()
                        offset = ((repeat * blocks + bid) * 2 + vid) * _TILE
                        if offset < size:
                            T.copy(A[offset], ar)
                            if adtype == 'float32':
                                T.copy(ar, a)
                            else:
                                T.tile.cast(a, ar, 'CAST_NONE', _TILE)
                            if kind == 'add' or kind == 'sub' or kind == 'mul' or kind == 'div':
                                T.copy(B[offset], br)
                                if bdtype == 'float32':
                                    T.copy(br, b)
                                else:
                                    T.tile.cast(b, br, 'CAST_NONE', _TILE)
                            if kind == 'add':
                                T.tile.add(a, a, b)
                            elif kind == 'sub':
                                T.tile.sub(a, a, b)
                            elif kind == 'mul':
                                T.tile.mul(a, a, b)
                            elif kind == 'div':
                                T.tile.div(a, a, b)
                            elif kind == 'scale':
                                T.tile.mul(a, a, scalar)
                            elif kind == 'offset':
                                T.tile.add(a, a, scalar)
                            elif kind == 'sqrt':
                                T.tile.sqrt(a, a)
                            elif kind == 'abs':
                                T.tile.abs(a, a)
                            elif kind == 'floor':
                                T.tile.max(a, a, scalar)
                            if outdtype == 'float32':
                                T.copy(a, out)
                            else:
                                T.tile.cast(out, a, 'CAST_RINT', _TILE)
                            T.copy(out, O[offset])
        return main
    return build()


def _pad_flat(x):
    """Zero-extend a flat view to a whole number of ``_TILE`` lanes.

    ⚠️ R362: this used ``torch.nn.functional.pad``.  For the DECODE shapes every
    per-row statistic is a ONE-element tensor, so every arithmetic step on it
    pays a pad from 1 lane to 1024 -- and CANN charges 14.7 us for that pad
    (``aclnnConstantPadNd`` splits into MemSet 6.24 us + PadV3 8.43 us,
    measured R362 `probe/base/p1_RMSNormQuantFwdOp.stdout`, reps=20).
    `llama-8b-decode` paid SEVEN of them: 102.1 us out of a 137.8 us operator,
    against 31.5 us of actual arithmetic in our own 17 kernels.

    ``torch.zeros`` + a prefix copy writes the SAME BYTES (the tail is zero
    either way) using ``FillScalar`` + ``InplaceCopy``, ~1.3 us each.
    """
    flat = x.contiguous().view(-1)
    tail = (-flat.numel()) % _TILE
    if not tail:
        return flat
    if _t362_arm() == "base":
        return torch.nn.functional.pad(flat, (0, tail))
    out = torch.zeros(flat.numel() + tail, dtype=flat.dtype, device=flat.device)
    out[: flat.numel()] = flat
    return out


def calc(kind, a, b=None, scalar=0.0, dtype=torch.float32):
    if b is not None:
        a, b = torch.broadcast_tensors(a, b)
    shape = a.shape
    aa = _pad_flat(a)
    bb = aa if b is None else _pad_flat(b)
    return _map(aa.numel(), _name(aa.dtype), _name(bb.dtype), _name(dtype),
                kind, float(scalar))(aa, bb)[:math.prod(shape)].reshape(shape)


@lru_cache(maxsize=256)
def _reduce(rows, width, kind):
    blocks = min(24, max(1, math.ceil(rows / 2)))
    repeats = math.ceil(rows / (blocks * 2))
    @tilelang.jit(out_idx=[1], pass_configs=_CONFIG)
    def build():
        @T.prim_func
        def main(A: T.Tensor([rows, width], 'float32'), O: T.Tensor([rows, 8], 'float32')):
            with T.Kernel(blocks, is_npu=True) as (bid, vid):
                a = T.alloc_ub((1, _TILE), 'float32')
                part = T.alloc_ub((1,), 'float32')
                acc = T.alloc_ub((1,), 'float32')
                packed = T.alloc_ub((1, 8), 'float32')
                with T.Scope('V'):
                    for repeat in T.serial(repeats):
                        T.barrier_all()
                        row = (repeat * blocks + bid) * 2 + vid
                        if row < rows:
                            T.tile.fill(acc, 0.0)
                            for chunk in T.serial(width // _TILE):
                                T.copy(A[row, chunk * _TILE], a)
                                if kind == 'sum':
                                    T.reduce_sum(a, part, dim=-1)
                                    T.tile.add(acc, acc, part)
                                else:
                                    T.reduce_max(a, part, dim=-1)
                                    T.tile.max(acc, acc, part)
                            T.tile.broadcast(packed, acc)
                            T.copy(packed[0, :], O[row, :])
        return main
    return build()


def reduce_last(x, kind='sum'):
    """Sum or max(abs(x)) callers; max's neutral element is zero."""
    shape = x.shape
    width = shape[-1]
    rows = x.numel() // width
    if _t362_arm() != "base" and x.dtype == torch.float32 and x.is_contiguous():
        # R362: `calc('cast', ...)` on an fp32 input compiles to `_map` with
        # kind='cast', whose body takes NO arithmetic branch -- it is
        # GM -> UB -> UB -> GM, a pure copy of the whole tensor.  Measured at
        # 66.19 us and 66.07 us for the two `reduce_last` calls of
        # llama-8b-prefill (32 MiB each way), i.e. 132 us of a 1519.6 us
        # operator spent copying a tensor onto itself.  Bit-exact by
        # construction: the branch it replaces moves bits, it does not round.
        xx = x.reshape(rows, width)
    else:
        xx = calc('cast', x).reshape(rows, width)
    tail = (-width) % _TILE
    if tail:
        xx = torch.nn.functional.pad(xx, (0, tail))
    return _reduce(rows, width + tail, kind)(xx)[:, 0].contiguous().reshape(*shape[:-1], 1)


def reduce_axes(x, axes):
    axes = tuple(sorted(a % x.ndim for a in axes))
    keep = tuple(a for a in range(x.ndim) if a not in axes)
    permuted = x.permute(keep + axes).contiguous()
    kept_shape = tuple(x.shape[a] for a in keep)
    width = math.prod(x.shape[a] for a in axes)
    return reduce_last(permuted.reshape(-1, width)).reshape(kept_shape)


@lru_cache(maxsize=128)
def _encode_e4m3(size):
    """Encode finite, saturated E4M3FN with exact binary scaling and RNE.

    Each exponent bin is selected explicitly: no approximate log/exp and no
    hardware FP8 cast. Uint8 is storage, not an int8 quantization format.
    """
    blocks=min(24,max(1,math.ceil(size/(2*_TILE))))
    repeats=math.ceil(size/(blocks*2*_TILE))
    # ⚠️ MUST be a Python bool: inside a prim_func body the TVMScript parser
    # turns a non-bool `if` into a TIR predicate and walks BOTH branches.
    binary_bins=_t362_arm()!="base"
    @tilelang.jit(out_idx=[1],pass_configs=_CONFIG)
    def build():
        @T.prim_func
        def main(A:T.Tensor([size],'float32'),O:T.Tensor([size],'uint8')):
            with T.Kernel(blocks,is_npu=True) as (bid,vid):
                x=T.alloc_ub((_TILE,),'float32')
                mag=T.alloc_ub((_TILE,),'float32')
                scaled=T.alloc_ub((_TILE,),'float32')
                code=T.alloc_ub((_TILE,),'float32')
                candidate=T.alloc_ub((_TILE,),'float32')
                ints=T.alloc_ub((_TILE,),'int32')
                threshold=T.alloc_ub((_TILE,),'float32')
                factor=T.alloc_ub((_TILE,),'float32')
                lower=T.alloc_ub((_TILE,),'float32')
                offset_code=T.alloc_ub((_TILE,),'float32')
                mask=T.alloc_ub((_TILE//8,),'uint8')
                output=T.alloc_ub((_TILE,),'uint8')
                code_half=T.alloc_ub((_TILE,),'float16')
                with T.Scope('V'):
                    for repeat in T.serial(repeats):
                        T.barrier_all()
                        offset=((repeat*blocks+bid)*2+vid)*_TILE
                        if offset<size:
                            T.copy(A[offset],x)
                            T.tile.abs(mag,x)
                            T.tile.min(mag,mag,448.0)
                            # Subnormals have a fixed quantum 2**-9.
                            T.tile.mul(scaled,mag,512.0)
                            T.tile.cast(ints,scaled,'CAST_RINT',_TILE)
                            T.tile.cast(code,ints,'CAST_NONE',_TILE)
                            T.tile.fill(factor,512.0)
                            T.tile.fill(lower,0.015625)
                            T.tile.fill(offset_code,0.0)
                            if binary_bins:
                                # R362.  The loop below it replaces walks the 15
                                # exponent bins e = 0..14 (lower_e = 2**(e-6))
                                # IN ORDER and keeps the last one whose
                                # predicate `mag >= lower_e` holds: 9 vector ops
                                # x 15 = 135 per tile.  Measured 725.8 us of the
                                # 1519.6 us llama-8b-prefill operator, 82.5% of
                                # it aiv_vec_time (R362
                                # probe/base/p1_RMSNormQuantFwdOp.stdout).
                                #
                                # The predicate is MONOTONE in e, so the largest
                                # satisfying e can be found by binary search --
                                # 4 steps of 7 ops plus a 4-op epilogue, 32 ops.
                                #
                                # Bit-exact, three reasons:
                                #  * it selects the same bin E (monotone
                                #    predicate, standard largest-true search);
                                #  * `factor` = 512 * prod(2**-step) and `lower`
                                #    = 2**-6 * prod(2**step) over the taken
                                #    steps are exact powers of two in fp32, and
                                #    `offset_code` = sum(8*step) is an exact
                                #    small integer, so all three end at exactly
                                #    the linear scan's bin-E values;
                                #  * the search can only overshoot to e = 15,
                                #    whose threshold is 2**9 = 512, and `mag` is
                                #    clamped to 448 one line above -- so e = 15
                                #    is never taken and the range stays 0..14.
                                # `scaled` / `candidate` / `threshold` are dead
                                # here and double as the three scratch tiles.
                                # ⚠️ The four steps are written out rather than
                                # looped: the TVMScript parser rejects a Python
                                # tuple in a `for` inside a prim_func body
                                # ("Expect the for loop to be one of the
                                # following: range, T.serial, ..."), measured
                                # R362 probe/c-smoke.log.
                                T.tile.mul(scaled,lower,256.0)
                                T.tile.compare(mask,mag,scaled,'GE')
                                T.tile.select(lower,mask,scaled,lower,'VSEL_TENSOR_TENSOR_MODE')
                                T.tile.mul(candidate,factor,0.00390625)
                                T.tile.select(factor,mask,candidate,factor,'VSEL_TENSOR_TENSOR_MODE')
                                T.tile.add(threshold,offset_code,64.0)
                                T.tile.select(offset_code,mask,threshold,offset_code,'VSEL_TENSOR_TENSOR_MODE')
                                T.tile.mul(scaled,lower,16.0)
                                T.tile.compare(mask,mag,scaled,'GE')
                                T.tile.select(lower,mask,scaled,lower,'VSEL_TENSOR_TENSOR_MODE')
                                T.tile.mul(candidate,factor,0.0625)
                                T.tile.select(factor,mask,candidate,factor,'VSEL_TENSOR_TENSOR_MODE')
                                T.tile.add(threshold,offset_code,32.0)
                                T.tile.select(offset_code,mask,threshold,offset_code,'VSEL_TENSOR_TENSOR_MODE')
                                T.tile.mul(scaled,lower,4.0)
                                T.tile.compare(mask,mag,scaled,'GE')
                                T.tile.select(lower,mask,scaled,lower,'VSEL_TENSOR_TENSOR_MODE')
                                T.tile.mul(candidate,factor,0.25)
                                T.tile.select(factor,mask,candidate,factor,'VSEL_TENSOR_TENSOR_MODE')
                                T.tile.add(threshold,offset_code,16.0)
                                T.tile.select(offset_code,mask,threshold,offset_code,'VSEL_TENSOR_TENSOR_MODE')
                                T.tile.mul(scaled,lower,2.0)
                                T.tile.compare(mask,mag,scaled,'GE')
                                T.tile.select(lower,mask,scaled,lower,'VSEL_TENSOR_TENSOR_MODE')
                                T.tile.mul(candidate,factor,0.5)
                                T.tile.select(factor,mask,candidate,factor,'VSEL_TENSOR_TENSOR_MODE')
                                T.tile.add(threshold,offset_code,8.0)
                                T.tile.select(offset_code,mask,threshold,offset_code,'VSEL_TENSOR_TENSOR_MODE')
                                T.tile.mul(scaled,mag,factor)
                                T.tile.cast(ints,scaled,'CAST_RINT',_TILE)
                                T.tile.cast(code,ints,'CAST_NONE',_TILE)
                                T.tile.add(code,code,offset_code)
                            else:
                                for exponent in T.serial(15):
                                    T.tile.mul(scaled,mag,factor)
                                    T.tile.cast(ints,scaled,'CAST_RINT',_TILE)
                                    T.tile.cast(candidate,ints,'CAST_NONE',_TILE)
                                    T.tile.add(candidate,candidate,offset_code)
                                    T.tile.compare(mask,mag,lower,'GE')
                                    T.tile.select(code,mask,candidate,code,'VSEL_TENSOR_TENSOR_MODE')
                                    T.tile.mul(factor,factor,0.5)
                                    T.tile.mul(lower,lower,2.0)
                                    T.tile.add(offset_code,offset_code,8.0)
                            T.tile.fill(threshold,0.0)
                            T.tile.compare(mask,x,threshold,'LT')
                            T.tile.add(candidate,code,128.0)
                            T.tile.select(code,mask,candidate,code,'VSEL_TENSOR_TENSOR_MODE')
                            # Codes 0..254 are exactly representable in fp16.
                            T.tile.cast(code_half,code,'CAST_RINT',_TILE)
                            T.tile.cast(output,code_half,'CAST_RINT',_TILE)
                            T.copy(output,O[offset])
        return main
    return build()


def quant_e4m3(y):
    maximum=reduce_last(calc('abs',y),'max')
    scale=calc('scale',calc('floor',maximum,scalar=1e-12),scalar=1.0/448.0)
    normalized=calc('div',y,scale)
    flat=_pad_flat(normalized)
    payload=_encode_e4m3(flat.numel())(flat)[:y.numel()].reshape(y.shape)
    return payload.view(torch.float8_e4m3fn),scale.squeeze(-1)
