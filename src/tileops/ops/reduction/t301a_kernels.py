"""Fixed first implementations for T301 A, colocated inside authorized ops/.

Derived from Tile-AI in-tree kernels/reduction.py (row/UB layout),
kernels/elementwise_binary.py (vector cast), kernels/elementwise_predicate.py
(compare/where), and tilelang-ascend/examples/unsorted_segment_sum/
unsorted_segment_sum.py (fp32 atomic scatter). The merge sort is a newly written
standard bottom-up merge, inspired by examples/sort/example_merge_sort.py's
sorted-run representation. Examples were read as text, never imported.
No autotuning, candidate search or performance-dependent dispatch is used.
See TileOPs/NOTICE. Registration uses the normal backend API, not harness hooks.
"""
from functools import lru_cache
from math import prod
import torch
import tilelang
import tilelang.language as T
from tileops.backend import register_kernel_builder

_PC = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}
_REGISTERED = False


def _dtype(dtype):
    return str(dtype).removeprefix('torch.')


def _axis(shape, dim):
    if isinstance(dim, bool) or not isinstance(dim, int) or not -len(shape) <= dim < len(shape):
        raise ValueError('dim must name an input axis')
    if any(s <= 0 for s in shape):
        raise ValueError('empty dimensions are not supported')
    return dim % len(shape)


@lru_cache(None)
def _affine(total, source, dest, scale, zero):
    width = 1024
    chunks = (total + width - 1) // width
    blocks = min(24, (chunks + 1) // 2)
    padded = chunks * width
    @tilelang.jit(out_idx=[1], pass_configs=_PC)
    def factory():
        @T.prim_func
        def main(X: T.Tensor((total,), source), Y: T.Tensor((padded,), dest)):
            with T.Kernel(blocks, is_npu=True) as (cid, vid):
                a = T.alloc_ub((width,), source)
                h = T.alloc_ub((width,), 'float16')
                f = T.alloc_ub((width,), 'float32')
                b = T.alloc_ub((width,), dest)
                with T.Scope('V'):
                    for step in T.serial((chunks + 2 * blocks - 1) // (2 * blocks)):
                        chunk = cid * 2 + vid + step * blocks * 2
                        if chunk < chunks:
                            base = chunk * width
                            if base + width <= total:
                                T.copy(X[base:base + width], a)
                            else:
                                if source in ('int8', 'uint8'):
                                    for j in T.serial(width):
                                        a[j] = 0
                                else:
                                    T.tile.fill(a, 0)
                                for j in T.serial(width):
                                    if base + j < total:
                                        a[j] = X[base + j]
                            if source == 'float32':
                                T.copy(a, f)
                            elif source in ('int8', 'uint8'):
                                T.tile.cast(h, a, 'CAST_NONE', width)
                                T.tile.cast(f, h, 'CAST_NONE', width)
                            else:
                                T.tile.cast(f, a, 'CAST_NONE', width)
                            if zero != 0:
                                T.tile.add(f, f, T.float32(-zero))
                            if scale != 1.0:
                                T.tile.mul(f, f, T.float32(scale))
                            if dest == 'float32':
                                T.copy(f, b)
                            else:
                                T.tile.cast(b, f, 'CAST_RINT', width)
                            T.copy(b, Y[base:base + width])
        return main
    return factory()


@lru_cache(None)
def _cummax(m, n):
    width = 256
    pad = ((n + width - 1) // width) * width
    blocks = min(24, (m + 1) // 2)
    @tilelang.jit(out_idx=[1, 2], pass_configs=_PC)
    def factory():
        @T.prim_func
        def main(X: T.Tensor((m, pad), 'float32'), V: T.Tensor((m, pad), 'float32'), I: T.Tensor((m, pad), 'int32')):
            with T.Kernel(blocks, is_npu=True) as (cid, vid):
                a = T.alloc_ub((width,), 'float32')
                b = T.alloc_ub((width,), 'float32')
                ids = T.alloc_ub((width,), 'int32')
                best = T.alloc_var('float32')
                index = T.alloc_var('int32')
                with T.Scope('V'):
                    for rr in T.serial((m + 2 * blocks - 1) // (2 * blocks)):
                        row = cid * 2 + vid + rr * 2 * blocks
                        if row < m:
                            best = -T.infinity('float32')
                            index = 0
                            for c in T.serial(pad // width):
                                T.copy(X[row, c * width:c * width + width], a)
                                for j in T.serial(width):
                                    if a[j] >= best or a[j] != a[j]:
                                        best = a[j]
                                        index = c * width + j
                                    b[j] = best
                                    ids[j] = index
                                T.copy(b, V[row, c * width:c * width + width])
                                T.copy(ids, I[row, c * width:c * width + width])
        return main
    return factory()


@lru_cache(None)
def _merge(m, padded, n, run, descending):
    """Merge adjacent runs; bounded UB storage and deterministic tie handling."""
    assert run >= 8
    width = min(run, 256)
    pairs_per_row = padded // (2 * run)
    jobs = m * pairs_per_row
    blocks = min(24, (jobs + 1) // 2)
    @tilelang.jit(out_idx=[2, 3], pass_configs=_PC)
    def factory():
        @T.prim_func
        def main(X: T.Tensor((m, padded), 'float32'), IX: T.Tensor((m, padded), 'int32'),
                 Y: T.Tensor((m, padded), 'float32'), IY: T.Tensor((m, padded), 'int32')):
            with T.Kernel(blocks, is_npu=True) as (cid, vid):
                lv = T.alloc_ub((width,), 'float32')
                rv = T.alloc_ub((width,), 'float32')
                li = T.alloc_ub((width,), 'int32')
                ri = T.alloc_ub((width,), 'int32')
                ov = T.alloc_ub((width,), 'float32')
                oi = T.alloc_ub((width,), 'int32')
                lp = T.alloc_var('int32')
                rp = T.alloc_var('int32')
                loaded_l = T.alloc_var('int32')
                loaded_r = T.alloc_var('int32')
                take_l = T.alloc_var('int32')
                with T.Scope('V'):
                    for repeat in T.serial((jobs + 2 * blocks - 1) // (2 * blocks)):
                        job = cid * 2 + vid + repeat * 2 * blocks
                        if job < jobs:
                            row = job // pairs_per_row
                            base = (job % pairs_per_row) * 2 * run
                            lp = 0
                            rp = 0
                            loaded_l = -1
                            loaded_r = -1
                            for out_chunk in T.serial(2 * run // width):
                                for j in T.serial(width):
                                    if lp < run and lp // width != loaded_l:
                                        loaded_l = lp // width
                                        if width >= 8:
                                            T.copy(X[row, base + loaded_l * width:base + (loaded_l + 1) * width], lv)
                                            T.copy(IX[row, base + loaded_l * width:base + (loaded_l + 1) * width], li)
                                        else:
                                            for k in T.serial(width):
                                                lv[k] = X[row, base + lp + k]
                                                li[k] = IX[row, base + lp + k]
                                    if rp < run and rp // width != loaded_r:
                                        loaded_r = rp // width
                                        if width >= 8:
                                            T.copy(X[row, base + run + loaded_r * width:base + run + (loaded_r + 1) * width], rv)
                                            T.copy(IX[row, base + run + loaded_r * width:base + run + (loaded_r + 1) * width], ri)
                                        else:
                                            for k in T.serial(width):
                                                rv[k] = X[row, base + run + rp + k]
                                                ri[k] = IX[row, base + run + rp + k]
                                    take_l = 0
                                    if lp < run:
                                        if rp >= run:
                                            take_l = 1
                                        elif ri[rp % width] >= n:
                                            take_l = 1
                                        elif li[lp % width] < n:
                                            if descending:
                                                if lv[lp % width] != lv[lp % width]:
                                                    take_l = 1
                                                elif rv[rp % width] == rv[rp % width] and lv[lp % width] >= rv[rp % width]:
                                                    take_l = 1
                                            else:
                                                if rv[rp % width] != rv[rp % width]:
                                                    take_l = 1
                                                elif lv[lp % width] == lv[lp % width] and lv[lp % width] <= rv[rp % width]:
                                                    take_l = 1
                                    if take_l == 1:
                                        ov[j] = lv[lp % width]
                                        oi[j] = li[lp % width]
                                        lp = lp + 1
                                    else:
                                        ov[j] = rv[rp % width]
                                        oi[j] = ri[rp % width]
                                        rp = rp + 1
                                if width >= 8:
                                    T.copy(ov, Y[row, base + out_chunk * width:base + (out_chunk + 1) * width])
                                    T.copy(oi, IY[row, base + out_chunk * width:base + (out_chunk + 1) * width])
                                else:
                                    for k in T.serial(width):
                                        Y[row, base + out_chunk * width + k] = ov[k]
                                        IY[row, base + out_chunk * width + k] = oi[k]
        return main
    return factory()


@lru_cache(None)
def _segment(rows, cols, segments):
    width = 256
    padded = ((cols + width - 1) // width) * width
    jobs = rows * (padded // width)
    blocks = min(24, (jobs + 1) // 2)
    @tilelang.jit(pass_configs=_PC)
    def factory():
        @T.prim_func
        def main(X: T.Tensor((rows, padded), 'float32'), IDs: T.Tensor((rows,), 'int32'), Y: T.Tensor((segments, padded), 'float32')):
            with T.Kernel(blocks, is_npu=True) as (cid, vid):
                a = T.alloc_ub((width,), 'float32')
                with T.Scope('V'):
                    for repeat in T.serial((jobs + 2 * blocks - 1) // (2 * blocks)):
                        job = cid * 2 + vid + repeat * blocks * 2
                        if job < jobs:
                            row = job // (padded // width)
                            col = job % (padded // width) * width
                            seg = IDs[row]
                            if seg >= 0 and seg < segments:
                                T.copy(X[row, col:col + width], a)
                                T.tile.atomic_add(Y[seg, col], a)
        return main
    return factory()


def _padded_rows(x, axis, grain):
    view = x.movedim(axis, -1)
    n = view.shape[-1]
    m = x.numel() // n
    padded = ((n + grain - 1) // grain) * grain
    a = view.contiguous().reshape(m, n).float()
    if padded != n:
        a = torch.nn.functional.pad(a, (0, padded - n))
    return a, view.shape, padded


def build_cummax(x, *, dim=-1):
    axis = _axis(x.shape, dim)
    n = x.shape[axis]
    m = prod(x.shape) // n
    compiled = _cummax(m, n)
    def launch(a):
        padded, shape, _ = _padded_rows(a, axis, 256)
        values, indices = compiled(padded)
        return (values[:, :n].to(a.dtype).reshape(shape).movedim(-1, axis),
                indices[:, :n].long().reshape(shape).movedim(-1, axis))
    return launch


def build_sort(x, *, dim=-1, descending=False):
    axis = _axis(x.shape, dim)
    n = x.shape[axis]
    padded = max(8, 1 << (n - 1).bit_length())
    m = prod(x.shape) // n
    seed = _sort_seed(m, padded, n, descending)
    stages = [_merge(m, padded, n, 1 << k, descending) for k in range(3, (padded - 1).bit_length())]
    def launch(a):
        values, shape, _ = _padded_rows(a, axis, padded)
        values, indices = seed(values)
        for stage in stages:
            values, indices = stage(values, indices)
        return (values[:, :n].to(a.dtype).reshape(shape).movedim(-1, axis),
                indices[:, :n].long().reshape(shape).movedim(-1, axis))
    return launch


def build_median(x, *, dim=-1, keepdim=False):
    axis = _axis(x.shape, dim)
    sort = build_sort(x, dim=dim)
    middle = (x.shape[axis] - 1) // 2
    def launch(a):
        values, indices = sort(a)
        # torch.median propagates a NaN if any input element is NaN. In the
        # ascending order the last value supplies such a NaN and a valid index.
        v, i = values.select(axis, middle), indices.select(axis, middle)
        last = values.select(axis, x.shape[axis] - 1)
        nan = torch.isnan(last)
        v = torch.where(nan, last, v)
        i = torch.where(nan, indices.select(axis, x.shape[axis] - 1), i)
        return (v.unsqueeze(axis), i.unsqueeze(axis)) if keepdim else (v, i)
    return launch


def build_welford(x, *, dim=None, correction=1, keepdim=False):
    from tileops.kernels.reduction import build_reduction_kernel
    kernel = build_reduction_kernel(x.shape, x.dtype, dim, keepdim,
        op_kind='var_mean', op_name='MeanVarWelfordFwdOp', correction=correction)
    def launch(a):
        var, mean = kernel(a)
        return mean, var
    return launch


def build_segment(x, segment_ids, *, num_segments=16):
    rows, cols = x.shape[0], prod(x.shape[1:])
    if tuple(segment_ids.shape) != (rows,):
        raise ValueError('one int32 segment ID is required per input row')
    padded = ((cols + 255) // 256) * 256
    compiled = _segment(rows, cols, num_segments)
    def launch(a, ids):
        flat = a.contiguous().reshape(rows, cols).float()
        if padded != cols:
            flat = torch.nn.functional.pad(flat, (0, padded - cols))
        result = torch.zeros((num_segments, padded), dtype=torch.float32, device=a.device)
        compiled(flat, ids.contiguous(), result)
        return result[:, :cols].to(a.dtype).reshape((num_segments,) + tuple(a.shape[1:]))
    return launch


def build_masked(x, mask, *, dim=None, keepdim=False):
    from tileops.kernels.elementwise_predicate import build_where_kernel
    from tileops.kernels.reduction import build_reduction_kernel
    where = build_where_kernel(x.shape, x.shape, (), x.dtype)
    reduce = build_reduction_kernel(x.shape, x.dtype, dim, keepdim,
        op_kind='sum', op_name='MaskedReduceSumFwdOp')
    def launch(a, mask):
        zero = torch.zeros((), dtype=a.dtype, device=a.device)
        selected = where(mask, a, zero)
        return reduce(selected)
    return launch


def build_cast(input, *, out_dtype='float32'):
    compiled = _affine(prod(input.shape), _dtype(input.dtype), out_dtype, 1.0, 0)
    return lambda a: compiled(a.contiguous().reshape(-1))[:a.numel()].reshape(a.shape)


def build_compare(input, other, *, predicate='eq'):
    from tileops.kernels.elementwise_predicate import build_predicate_binary
    if predicate == 'ne':
        # Ascend vector NE is ordered: invert EQ to preserve NaN != x.
        # This is the count_nonzero reference's complement-of-EQ idea.
        from tileops.kernels.elementwise_unary import build_unary_kernel
        equal = build_predicate_binary(input.shape, other.shape, input.dtype,
            op_kind='EQ', op_name='CompareFwdOp')
        shape = tuple(torch.broadcast_shapes(input.shape, other.shape))
        invert = build_unary_kernel(shape, torch.bool, op_kind='logical_not',
            supported_dtypes=(torch.bool,), output_dtype=torch.bool, op_name='CompareFwdOp')
        return lambda a, b: invert(equal(a, b))
    return build_predicate_binary(input.shape, other.shape, input.dtype,
        op_kind=predicate.upper(), op_name='CompareFwdOp')


def build_dequant(input_tensor, *, scale=0.0078125, zero_point=0):
    compiled = _affine(prod(input_tensor.shape), _dtype(input_tensor.dtype), 'float32', scale, zero_point)
    return lambda a: compiled(a.contiguous().reshape(-1))[:a.numel()].reshape(a.shape)


def register_builders():
    global _REGISTERED
    if _REGISTERED:
        return
    for name, build in (
        ('CummaxFwdOp', build_cummax), ('SortFwdOp', build_sort),
        ('MedianFwdOp', build_median), ('MeanVarWelfordFwdOp', build_welford),
        ('SegmentSumFwdOp', build_segment), ('MaskedReduceSumFwdOp', build_masked),
        ('CastFwdOp', build_cast), ('CompareFwdOp', build_compare),
        ('DequantizeFwdOp', build_dequant),
    ):
        register_kernel_builder(op=name, target='ascend', build_kernel=build)
    _REGISTERED = True


@lru_cache(None)
def _sort_seed(m, padded, n, descending):
    """Sort aligned eight-element seeds; scalar writes stay entirely in UB.

    Scalar GM writeback is not reliable in this lowering (see in-tree
    kernels/indexed_reduce.py), so even the initial run owns a whole 32B line.
    """
    jobs = m * (padded // 8)
    blocks = min(24, (jobs + 1) // 2)
    @tilelang.jit(out_idx=[1, 2], pass_configs=_PC)
    def factory():
        @T.prim_func
        def main(X: T.Tensor((m, padded), 'float32'), Y: T.Tensor((m, padded), 'float32'), I: T.Tensor((m, padded), 'int32')):
            with T.Kernel(blocks, is_npu=True) as (cid, vid):
                val = T.alloc_ub((8,), 'float32')
                idx = T.alloc_ub((8,), 'int32')
                swap = T.alloc_var('int32')
                tv = T.alloc_var('float32')
                ti = T.alloc_var('int32')
                with T.Scope('V'):
                    for repeat in T.serial((jobs + blocks * 2 - 1) // (blocks * 2)):
                        job = cid * 2 + vid + repeat * blocks * 2
                        if job < jobs:
                            row = job // (padded // 8)
                            col = job % (padded // 8) * 8
                            T.copy(X[row, col:col + 8], val)
                            for j in T.serial(8):
                                idx[j] = col + j
                            for rep in T.serial(8):
                                for j in T.serial(7):
                                    swap = 0
                                    if idx[j + 1] < n:
                                        if idx[j] >= n:
                                            swap = 1
                                        elif descending:
                                            if val[j] == val[j] and val[j + 1] != val[j + 1]:
                                                swap = 1
                                            elif val[j] == val[j] and val[j + 1] == val[j + 1] and val[j] < val[j + 1]:
                                                swap = 1
                                        else:
                                            if val[j] != val[j] and val[j + 1] == val[j + 1]:
                                                swap = 1
                                            elif val[j] == val[j] and val[j + 1] == val[j + 1] and val[j] > val[j + 1]:
                                                swap = 1
                                    if swap == 1:
                                        tv = val[j]
                                        ti = idx[j]
                                        val[j] = val[j + 1]
                                        idx[j] = idx[j + 1]
                                        val[j + 1] = tv
                                        idx[j + 1] = ti
                            T.copy(val, Y[row, col:col + 8])
                            T.copy(idx, I[row, col:col + 8])
        return main
    return factory()
