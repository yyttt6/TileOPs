"""T301 A operators, with fixed initial kernels and no autotuning.

Derived from the dispatch/shape handling of ops/reduction/reduce.py and
ops/elementwise/comparison.py (Tile-AI); implementations are in t301a_kernels.py.
See TileOPs/NOTICE. Workload provenance is recorded verbatim in t301a.yaml.
"""
import torch
from tileops.ops.op_base import Op
from tileops.manifest.shape_rules import reduced_shape


class _InitialOp(Op):
    def _init(self, target, tune):
        if tune:
            raise ValueError('T301 initial implementations do not support tuning')
        self.target = target
        self.tune = False
        from .t301a_kernels import register_builders
        register_builders()

    def _run(self, names, tensors):
        self._validate_dtypes(*tensors)
        if any(t.device != tensors[0].device for t in tensors):
            raise ValueError('all inputs must be on one device')
        if tensors[0].device.type != 'npu':
            raise ValueError('T301 A kernels require Ascend NPU')
        self.dtype = tensors[0].dtype
        for name, t in zip(names, tensors):
            setattr(self, name + '_shape', tuple(t.shape))
        return self.get_or_build_kernel('main', tensors)(*tensors)


class CummaxFwdOp(_InitialOp):
    """Inclusive cumulative maximum and int64 indices; ties take the last index.

    Derived from the row tiling in tileops/kernels/reduction.py, not torch.cummax.
    """
    def __init__(self, dim=-1, *, target=None, tune=False):
        self.dim = dim
        self._init(target, tune)
        self.dispatch_kernel()

    def _infer_output_shapes(self, x_shape):
        return {'values': tuple(x_shape), 'indices': tuple(x_shape)}

    def forward(self, x):
        return self._run(('x',), (x,))


class SortFwdOp(_InitialOp):
    """Stable axis sort returning values and int64 source indices.

    Uses a standard bottom-up merge sort, with Ascend UB tiling informed by
    tilelang-ascend/examples/sort/example_merge_sort.py (read only).
    """
    def __init__(self, dim=-1, descending=False, *, target=None, tune=False):
        self.dim, self.descending = dim, bool(descending)
        self._init(target, tune)
        self.dispatch_kernel()

    def _infer_output_shapes(self, x_shape):
        return {'values': tuple(x_shape), 'indices': tuple(x_shape)}

    def forward(self, x):
        return self._run(('x',), (x,))


class MedianFwdOp(_InitialOp):
    """Lower median along dim, with an index selecting that value.

    Sort-based implementation derived from SortFwdOp in this module. For ties,
    any source index selecting the median is valid (as in torch.median).
    """
    def __init__(self, dim=-1, keepdim=False, *, target=None, tune=False):
        self.dim, self.keepdim = dim, bool(keepdim)
        self._init(target, tune)
        self.dispatch_kernel()

    def _infer_output_shapes(self, x_shape):
        shape = reduced_shape(x_shape, self.dim, self.keepdim)
        return {'values': shape, 'indices': shape}

    def forward(self, x):
        return self._run(('x',), (x,))


class MeanVarWelfordFwdOp(_InitialOp):
    """Return (mean, variance) with Welford accumulation, correction default 1.

    Reuses tileops/kernels/reduction.py's existing Welford kernel; unlike
    VarMeanFwdOp, this explicit API returns mean first. No copied kernel body.
    """
    def __init__(self, dim=None, keepdim=False, correction=1, *, target=None, tune=False):
        if not isinstance(correction, int) or correction < 0:
            raise ValueError('correction must be a nonnegative integer')
        self.dim, self.keepdim, self.correction = dim, bool(keepdim), correction
        self._init(target, tune)
        self.dispatch_kernel()

    def _infer_output_shapes(self, x_shape):
        shape = reduced_shape(x_shape, self.dim, self.keepdim)
        return {'mean': shape, 'var': shape}

    def forward(self, x):
        return self._run(('x',), (x,))


class SegmentSumFwdOp(_InitialOp):
    """Unsorted segment sum over axis 0; negative/out-of-range IDs are ignored.

    IDs have shape [x.shape[0]]; output is [num_segments, *x.shape[1:]].
    Derived from the atomic accumulation idea in tilelang-ascend/examples/
    unsorted_segment_sum/unsorted_segment_sum.py; accumulation uses float32.
    """
    def __init__(self, num_segments=16, *, target=None, tune=False):
        if not isinstance(num_segments, int) or num_segments <= 0:
            raise ValueError('num_segments must be positive')
        self.num_segments = num_segments
        self._init(target, tune)
        self.dispatch_kernel()

    def _infer_output_shapes(self, x_shape, segment_ids_shape):
        if not x_shape or tuple(segment_ids_shape) != (x_shape[0],):
            raise ValueError('segment_ids must have one ID per input row')
        return {'output': (self.num_segments,) + tuple(x_shape[1:])}

    def forward(self, x, segment_ids):
        self._infer_output_shapes(tuple(x.shape), tuple(segment_ids.shape))
        return self._run(('x', 'segment_ids'), (x, segment_ids))


class MaskedReduceSumFwdOp(_InitialOp):
    """Sum selected elements in float32; masked NaNs contribute exactly zero.

    Reuses the in-tree Where and Sum kernels. The mask has exactly x.shape;
    reduction supports the SumFwdOp dim/keepdim contract.
    """
    def __init__(self, dim=None, keepdim=False, *, target=None, tune=False):
        self.dim, self.keepdim = dim, bool(keepdim)
        self._init(target, tune)
        self.dispatch_kernel()

    def _infer_output_shapes(self, x_shape, mask_shape):
        if tuple(x_shape) != tuple(mask_shape):
            raise ValueError('mask must have exactly x.shape')
        return {'output': reduced_shape(x_shape, self.dim, self.keepdim)}

    def forward(self, x, mask):
        self._infer_output_shapes(tuple(x.shape), tuple(mask.shape))
        return self._run(('x', 'mask'), (x, mask))


class CastFwdOp(_InitialOp):
    """Floating cast among float16, bfloat16 and float32; default float32.

    Derived from tileops/kernels/elementwise_binary.py's DMA/cast idiom.
    """
    def __init__(self, out_dtype='float32', *, target=None, tune=False):
        out_dtype = str(out_dtype).removeprefix('torch.')
        if out_dtype not in ('float16', 'bfloat16', 'float32'):
            raise ValueError('out_dtype must be float16, bfloat16 or float32')
        self.out_dtype = out_dtype
        self._init(target, tune)
        self.dispatch_kernel()

    def _infer_output_shapes(self, input_shape):
        return {'output': tuple(input_shape)}

    def forward(self, input):
        return self._run(('input',), (input,))


class CompareFwdOp(_InitialOp):
    """Broadcast comparison, predicate eq/ne/lt/le/gt/ge, bool output.

    Reuses tileops/kernels/elementwise_predicate.py. NE inverts EQ through
    elementwise_unary.py to retain unordered NaN semantics.
    """
    def __init__(self, predicate='eq', *, target=None, tune=False):
        if predicate not in ('eq', 'ne', 'lt', 'le', 'gt', 'ge'):
            raise ValueError('unsupported comparison predicate')
        self.predicate = predicate
        self._init(target, tune)
        self.dispatch_kernel()

    def _infer_output_shapes(self, input_shape, other_shape):
        return {'output': tuple(torch.broadcast_shapes(input_shape, other_shape))}

    def forward(self, input, other):
        return self._run(('input', 'other'), (input, other))


class DequantizeFwdOp(_InitialOp):
    """Affine dequantization: float32(input_tensor - zero_point) * scale.

    Accepts integer codes or codes stored in floating tensors; this is affine
    dequantization, not FP8 bit decoding. Scale and zero point are per-tensor.
    Derived from tileops/kernels/elementwise_binary.py's float32 arithmetic.
    """
    def __init__(self, scale=0.0078125, zero_point=0, *, target=None, tune=False):
        self.scale, self.zero_point = float(scale), int(zero_point)
        self._init(target, tune)
        self.dispatch_kernel()

    def _infer_output_shapes(self, input_tensor_shape):
        return {'output': tuple(input_tensor_shape)}

    def forward(self, input_tensor):
        return self._run(('input_tensor',), (input_tensor,))
