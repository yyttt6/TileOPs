"""T301 B normalization operators and their Ascend composition builders.

Saved-stat interfaces follow /home/dyq/op-list-150.pdf, entries 88, 90,
96, 98, 100-102 and 104-108. Arithmetic uses the rewritten TileLang
primitives in t301_primitives.py; design provenance is in NOTICE.T301.
The kernel implementation lives under ops to respect T301's file boundary.
No tuning or timing is performed by this module.
"""
import math
import torch

from ..op_base import Op
from tileops.kernels._registry import register


def _p():
    from . import t301_primitives
    return t301_primitives


def _norm(x, eps, centered=False):
    p = _p()
    xf = p.calc('cast', x)
    if centered:
        mean = p.calc('scale', p.reduce_last(xf), scalar=1.0 / x.shape[-1])
        xf = p.calc('sub', xf, mean)
    square = p.calc('mul', xf, xf)
    variance = p.calc('scale', p.reduce_last(square), scalar=1.0 / x.shape[-1])
    denom = p.calc('sqrt', p.calc('offset', variance, scalar=eps))
    inv = p.calc('div', torch.ones_like(denom), denom)
    return p.calc('mul', xf, inv), inv


def _row_bwd(x, grad_out, weight, mean, rstd, normalized_shape, centered):
    p = _p()
    n = math.prod(normalized_shape)
    xx = p.calc('cast', x).reshape(-1, n)
    dy = p.calc('cast', grad_out).reshape(-1, n)
    inv = rstd.reshape(-1, 1)
    z = p.calc('sub', xx, mean.reshape(-1, 1)) if centered else xx
    z = p.calc('mul', z, inv)
    weighted_dy = p.calc('mul', dy, weight.reshape(1, n))
    cross = p.calc('scale', p.reduce_last(p.calc('mul', weighted_dy, z)), scalar=1.0/n)
    dx = p.calc('sub', weighted_dy, p.calc('mul', z, cross))
    if centered:
        dx = p.calc('sub', dx, p.calc('scale', p.reduce_last(weighted_dy), scalar=1.0/n))
    dx = p.calc('cast', p.calc('mul', dx, inv), dtype=x.dtype).reshape(x.shape)
    dw = p.calc('cast', p.reduce_axes(p.calc('mul', dy, z), (0,)), dtype=weight.dtype).reshape(weight.shape)
    if not centered:
        return dx, dw
    db = p.calc('cast', p.reduce_axes(dy, (0,)), dtype=weight.dtype).reshape(weight.shape)
    return dx, dw, db


def _spatial_bwd(x, grad_out, weight, mean, rstd, groups, inference=False):
    p = _p()
    batch, channels = x.shape[:2]
    spatial = math.prod(x.shape[2:])
    width = channels // groups * spatial
    xx = p.calc('cast', x).reshape(batch * groups, width)
    dy = p.calc('cast', grad_out)
    channel_view = (1, channels) + (1,) * (x.ndim - 2)
    weighted = dy if weight is None else p.calc('mul', dy, weight.reshape(channel_view))
    weighted = weighted.reshape(batch * groups, width)
    inv = rstd.reshape(batch * groups, 1)
    z = p.calc('mul', p.calc('sub', xx, mean.reshape(batch * groups, 1)), inv)
    dx = weighted
    if not inference:
        cross = p.calc('scale', p.reduce_last(p.calc('mul', weighted, z)), scalar=1.0/width)
        dx = p.calc('sub', dx, p.calc('mul', z, cross))
        dx = p.calc('sub', dx, p.calc('scale', p.reduce_last(weighted), scalar=1.0/width))
    dx = p.calc('cast', p.calc('mul', dx, inv), dtype=x.dtype).reshape(x.shape)
    axes = (0,) + tuple(range(2, x.ndim))
    dw = p.calc('cast', p.reduce_axes(p.calc('mul', dy, z.reshape(x.shape)), axes), dtype=x.dtype)
    db = p.calc('cast', p.reduce_axes(dy, axes), dtype=x.dtype)
    return dx, dw, db


def _run_layer_bwd(x, grad_out, weight, mean, rstd, *, normalized_shape):
    return _row_bwd(x, grad_out, weight, mean, rstd, normalized_shape, True)


def _run_rms_bwd(x, grad_out, weight, rstd, *, normalized_shape):
    return _row_bwd(x, grad_out, weight, None, rstd, normalized_shape, False)


def _run_group_bwd(x, grad_out, weight, mean, rstd, bias, *, num_groups):
    return _spatial_bwd(x, grad_out, weight, mean, rstd, num_groups)


def _run_instance_bwd(x, grad_out, weight, mean, rstd, bias, running_mean, running_var, *, eps):
    if running_mean is not None:
        p = _p()
        mean = running_mean.expand(x.shape[0], -1)
        denom = p.calc('sqrt', p.calc('offset', running_var, scalar=eps))
        rstd = p.calc('div', torch.ones_like(denom), denom).expand(x.shape[0], -1)
    return _spatial_bwd(x, grad_out, weight, mean, rstd, x.shape[1], running_mean is not None)


def _run_weight(x, g, *, normalized_shape, dim):
    p = _p()
    axis = dim % x.ndim
    order = (axis,) + tuple(a for a in range(x.ndim) if a != axis)
    xx = p.calc('cast', x).permute(order).contiguous().reshape(x.shape[axis], -1)
    norm = p.calc('sqrt', p.reduce_last(p.calc('mul', xx, xx)))
    output = p.calc('div', p.calc('mul', xx, g.reshape(-1, 1)), norm)
    output = output.reshape(tuple(x.shape[a] for a in order)).permute(tuple(order.index(a) for a in range(x.ndim)))
    return p.calc('cast', output, dtype=x.dtype), norm.reshape(-1)


def _run_gemma(x, weight, *, normalized_shape, eps):
    p = _p()
    shape = x.shape
    z, _ = _norm(x.reshape(-1, math.prod(normalized_shape)), eps)
    affine = p.calc('offset', weight, scalar=1.0).reshape(1, -1)
    return p.calc('cast', p.calc('mul', z, affine), dtype=x.dtype).reshape(shape)


def _run_qk(x, k, q_weight, k_weight, *, normalized_shape, eps):
    p = _p()
    n = math.prod(normalized_shape)
    qz, _ = _norm(x.reshape(-1, n), eps)
    kz, _ = _norm(k.reshape(-1, n), eps)
    return (p.calc('cast', p.calc('mul', qz, q_weight.reshape(1, n)), dtype=x.dtype).reshape(x.shape),
            p.calc('cast', p.calc('mul', kz, k_weight.reshape(1, n)), dtype=k.dtype).reshape(k.shape))


def _run_group_rms(x, weight, *, normalized_shape, eps):
    p = _p()
    z, inv = _norm(x, eps)
    return p.calc('cast', p.calc('mul', z, weight), dtype=x.dtype), inv.squeeze(-1)


def _run_batch(x, weight, bias, running_mean, running_var, *, eps):
    p = _p()
    view = (1, x.shape[1]) + (1,) * (x.ndim - 2)
    denom = p.calc('sqrt', p.calc('offset', running_var, scalar=eps))
    z = p.calc('div', p.calc('sub', x, running_mean.reshape(view)), denom.reshape(view))
    return p.calc('cast', p.calc('add', p.calc('mul', z, weight.reshape(view)), bias.reshape(view)), dtype=x.dtype)


def _run_spectral(x, u, v, *, normalized_shape, n_iter, eps):
    p = _p()
    w = p.calc('cast', x)
    ut, vt = u, v
    def unit(vec):
        length = p.calc('sqrt', p.reduce_last(p.calc('mul', vec, vec)))
        return p.calc('div', vec, p.calc('floor', length, scalar=eps))
    for _ in range(n_iter):
        vt = unit(p.reduce_last(p.calc('mul', w.t(), ut.reshape(1, -1))).reshape(-1))
        ut = unit(p.reduce_last(p.calc('mul', w, vt.reshape(1, -1))).reshape(-1))
    wv = p.reduce_last(p.calc('mul', w, vt.reshape(1, -1))).reshape(-1)
    sigma = p.reduce_last(p.calc('mul', ut, wv)).reshape(())
    return sigma, ut.clone(), vt.clone()


def _run_quant(x, weight, bias=None, *, normalized_shape, eps, centered):
    p = _p()
    shape = x.shape
    z, _ = _norm(x.reshape(-1, math.prod(normalized_shape)), eps, centered)
    y = p.calc('mul', z, weight.reshape(1, -1))
    if bias is not None:
        y = p.calc('add', y, bias.reshape(1, -1))
    # Encode into bytes, then return a metadata-only FP8 dtype view.
    q, scale = p.quant_e4m3(y)
    return q.reshape(shape), scale.reshape(shape[:-len(normalized_shape)])


class _T301Norm(Op):
    """Common validation/dispatch for this addition; no global dispatch changes."""
    def __init__(self, *, target='ascend', tune=False):
        if tune:
            raise ValueError('T301 operators have fixed geometry; tuning is not implemented')
        self.target = target
        self.tune = False
        self.dispatch_kernel()

    def _validate_dtypes(self, *inputs):
        tensors = [t for t in inputs if isinstance(t, torch.Tensor)]
        if any(t.device.type != 'npu' for t in tensors):
            raise ValueError('T301 normalization kernels require NPU tensors')
        if len({t.device for t in tensors}) != 1:
            raise ValueError('all inputs must be on the same NPU')
        if any(t.dtype not in (torch.float16, torch.bfloat16, torch.float32) for t in tensors):
            raise ValueError('expected float16, bfloat16, or float32')
        if any(t.numel() == 0 for t in tensors):
            raise ValueError('empty inputs are not supported')

    def _infer_output_shapes(self, x_shape, **kwargs):
        return {'output': tuple(x_shape)}

    def eval_roofline(self):
        if not hasattr(self, '_roof'):
            raise RuntimeError('eval_roofline requires a prior call')
        return self._roof

    def _run(self, *inputs):
        _T301Norm._validate_dtypes(self, *inputs)
        self._validate_dtypes(**dict(zip(self._manifest_input_names, inputs)))
        self.dtype = inputs[0].dtype
        for name, value in zip(self._manifest_input_names, inputs):
            setattr(self, name+'_shape', tuple(value.shape) if value is not None else None)
        self._check(inputs)
        result = self.get_or_build_kernel('t301_norm', inputs)(*inputs)
        outputs = result if isinstance(result, tuple) else (result,)
        # Analytic compulsory IO, not intermediate-traffic accounting.
        total_io = sum(t.numel()*t.element_size() for t in (*inputs,*outputs) if t is not None)
        self._roof = (self._flops(inputs), total_io)
        return result

    def _flops(self, inputs):
        return self.flops_per_element * inputs[0].numel()

    def _check(self, inputs):
        x = inputs[0]
        ns = getattr(self, 'normalized_shape', None)
        if ns is not None and (not ns or min(ns) <= 0 or tuple(x.shape[-len(ns):]) != ns):
            raise ValueError('x trailing dimensions must equal normalized_shape')


class _Rows(_T301Norm):
    def __init__(self, normalized_shape, eps=1e-6, *, target='ascend', tune=False):
        self.normalized_shape = tuple(normalized_shape)
        self.eps = float(eps)
        if self.eps <= 0:
            raise ValueError('eps must be positive')
        super().__init__(target=target, tune=tune)


class LayerNormBwdOp(_Rows):
    """Saved-stat LayerNorm backward; dx, dweight, dbias use input dtype."""
    flops_per_element = 12
    _manifest_input_names = ('x','grad_out','weight','mean','rstd')
    def __init__(self, normalized_shape, *, target='ascend', tune=False):
        super().__init__(normalized_shape, target=target, tune=tune)
    def forward(self, x, grad_out, weight, mean, rstd):
        return self._run(x, grad_out, weight, mean, rstd)
    def _infer_output_shapes(self, x_shape, grad_out_shape, weight_shape, mean_shape, rstd_shape):
        return {'grad_x':x_shape, 'grad_weight':weight_shape, 'grad_bias':weight_shape}
    def _check(self, inputs):
        super()._check(inputs)
        x, dy, w, mean, inv = inputs
        _check_row_bwd(x, dy, w, inv, self.normalized_shape)
        _check_stat(mean, x.shape[:-len(self.normalized_shape)], 'mean')


class RMSNormBwdOp(_Rows):
    """Saved-rstd RMSNorm backward with fp32 accumulation."""
    flops_per_element = 9
    _manifest_input_names = ('x','grad_out','weight','rstd')
    def __init__(self, normalized_shape, *, target='ascend', tune=False):
        super().__init__(normalized_shape, target=target, tune=tune)
    def forward(self, x, grad_out, weight, rstd):
        return self._run(x, grad_out, weight, rstd)
    def _infer_output_shapes(self, x_shape, grad_out_shape, weight_shape, rstd_shape):
        return {'grad_x':x_shape, 'grad_weight':weight_shape}
    def _check(self, inputs):
        super()._check(inputs)
        _check_row_bwd(*inputs, self.normalized_shape)


class GroupNormBwdOp(_T301Norm):
    """Saved-stat group backward; absent weight means implicit unit affine."""
    flops_per_element = 12
    _manifest_input_names = ('x','grad_out','weight','mean','rstd','bias')
    def __init__(self, num_groups, *, target='ascend', tune=False):
        self.num_groups = int(num_groups)
        super().__init__(target=target,tune=tune)
    def forward(self, x, grad_out, weight, mean, rstd, bias=None):
        return self._run(x,grad_out,weight,mean,rstd,bias)
    def _infer_output_shapes(self, x_shape, grad_out_shape, weight_shape, mean_shape, rstd_shape, bias_shape=None):
        return {'grad_x':x_shape,'grad_weight':(x_shape[1],),'grad_bias':(x_shape[1],)}
    def _check(self, inputs):
        x,dy,w,mean,inv,bias = inputs
        _check_spatial(x,dy,w,mean,inv,self.num_groups)
        if bias is not None:
            _check_param(bias,(x.shape[1],),x.dtype,'bias')


class InstanceNormBwdOp(_T301Norm):
    """Instance backward; running stats select fixed-stat inference derivative.

    Saved mean/rstd are used on the input-stat branch. Optional running stats
    mirror the complete existing InstanceNormFwdOp workload, without dropping
    its inference cases. Neither saved nor running statistics are mutated.
    """
    flops_per_element = 12
    _manifest_input_names = ('x','grad_out','weight','mean','rstd','bias','running_mean','running_var')
    def __init__(self, eps=1e-5, *, target='ascend', tune=False):
        self.eps=float(eps)
        super().__init__(target=target,tune=tune)
    def forward(self,x,grad_out,weight,mean,rstd,bias=None,running_mean=None,running_var=None):
        return self._run(x,grad_out,weight,mean,rstd,bias,running_mean,running_var)
    def _infer_output_shapes(self,x_shape,grad_out_shape,weight_shape,mean_shape,rstd_shape,bias_shape=None,running_mean_shape=None,running_var_shape=None):
        return {'grad_x':x_shape,'grad_weight':(x_shape[1],),'grad_bias':(x_shape[1],)}
    def _check(self,inputs):
        x,dy,w,mean,inv,bias,rm,rv=inputs
        _check_spatial(x,dy,w,mean,inv,x.shape[1])
        if (rm is None)!=(rv is None):
            raise ValueError('running statistics must be supplied together')
        for t in (rm,rv):
            if t is not None:
                _check_stat(t,(x.shape[1],),'running statistics')
        if bias is not None:
            _check_param(bias,(x.shape[1],),x.dtype,'bias')


class WeightNormFwdOp(_Rows):
    """Normalize over all dimensions except dim; return weight and fp32 norm."""
    flops_per_element=4
    _manifest_input_names=('x','g')
    def __init__(self,normalized_shape,dim=0,*,target='ascend',tune=False):
        self.dim=int(dim)
        super().__init__(normalized_shape,target=target,tune=tune)
    def forward(self,x,g):
        return self._run(x,g)
    def _infer_output_shapes(self,x_shape,g_shape):
        return {'output':x_shape,'norm':g_shape}
    def _check(self,inputs):
        super()._check(inputs)
        x,g=inputs
        if not -x.ndim<=self.dim<x.ndim:
            raise ValueError('dim out of range')
        _check_param(g,(x.shape[self.dim],),x.dtype,'g')


class GemmaRMSNormFwdOp(_Rows):
    """Gemma RMSNorm: normalized x times fp32 (1 + weight); NOTICE.T301."""
    flops_per_element=5
    _manifest_input_names=('x','weight')
    def forward(self,x,weight):
        return self._run(x,weight)
    def _infer_output_shapes(self, x_shape, weight_shape):
        return {'output': x_shape}
    def _check(self,inputs):
        super()._check(inputs)
        _check_param(inputs[1],self.normalized_shape,inputs[0].dtype,'weight')


class QKNormFwdOp(_Rows):
    """Independent per-head RMS norms; Q/K leading shapes may differ."""
    flops_per_element=4
    _manifest_input_names=('x','k','q_weight','k_weight')
    def forward(self,x,k,q_weight,k_weight):
        return self._run(x,k,q_weight,k_weight)
    def _infer_output_shapes(self,x_shape,k_shape,q_weight_shape,k_weight_shape):
        return {'q_out':x_shape,'k_out':k_shape}
    def _flops(self,inputs):
        return 4*(inputs[0].numel()+inputs[1].numel())
    def _check(self,inputs):
        super()._check(inputs)
        x,k,qw,kw=inputs
        if tuple(k.shape[-len(self.normalized_shape):])!=self.normalized_shape or k.dtype!=x.dtype:
            raise ValueError('K normalized shape and dtype must match Q')
        for t in (qw,kw):
            _check_param(t,self.normalized_shape,x.dtype,'weight')


class GroupRMSNormFwdOp(_Rows):
    """x[..., G, D], weight[G, D], independent D reductions plus saved rstd.

    The inherited RMS workload's 2-D x is the valid no-leading-batch case.
    """
    flops_per_element=4
    _manifest_input_names=('x','weight')
    def forward(self,x,weight):
        return self._run(x,weight)
    def _infer_output_shapes(self,x_shape,weight_shape):
        return {'output':x_shape,'rstd':x_shape[:-1]}
    def _check(self,inputs):
        super()._check(inputs)
        x,w=inputs
        if x.ndim<2 or len(self.normalized_shape)!=1:
            raise ValueError('expected [..., G, D] and normalized_shape=(D,)')
        _check_param(w,tuple(x.shape[-2:]),x.dtype,'weight')


class BatchNormInferenceFwdOp(_T301Norm):
    """Fixed running-stat normalization; statistics are never updated."""
    flops_per_element=4
    _manifest_input_names=('x','weight','bias','running_mean','running_var')
    def __init__(self,eps=1e-5,*,target='ascend',tune=False):
        self.eps=float(eps)
        super().__init__(target=target,tune=tune)
    def forward(self,x,weight,bias,running_mean,running_var):
        return self._run(x,weight,bias,running_mean,running_var)
    def _infer_output_shapes(self, x_shape, weight_shape, bias_shape, running_mean_shape, running_var_shape):
        return {'output': x_shape}
    def _check(self,inputs):
        x,*params=inputs
        if x.ndim<2:
            raise ValueError('expected [N,C,*spatial]')
        for t in params:
            _check_stat(t,(x.shape[1],),'channel parameter')


class SpectralNormPowerIterFwdOp(_Rows):
    """v <- normalize(W.T u), u <- normalize(W v), sigma <- u.T W v."""
    _manifest_input_names=('x','u','v')
    def __init__(self,normalized_shape,n_iter=1,eps=1e-12,*,target='ascend',tune=False):
        if not isinstance(n_iter,int) or n_iter<0:
            raise ValueError('n_iter must be a nonnegative integer')
        self.n_iter=n_iter
        super().__init__(normalized_shape,eps,target=target,tune=tune)
    def forward(self,x,u,v):
        return self._run(x,u,v)
    def _infer_output_shapes(self,x_shape,u_shape,v_shape):
        return {'sigma':(),'u_out':u_shape,'v_out':v_shape}
    def _flops(self,inputs):
        m,n=inputs[0].shape
        return (4*self.n_iter+2)*m*n + (4*self.n_iter+2)*(m+n)
    def _check(self,inputs):
        super()._check(inputs)
        x,u,v=inputs
        if x.ndim!=2:
            raise ValueError('expected matrix x')
        _check_stat(u,(x.shape[0],),'u')
        _check_stat(v,(x.shape[1],),'v')


class LayerNormQuantFwdOp(_Rows):
    """LayerNorm followed by per-row symmetric E4M3FN quantization.

    q has torch.float8_e4m3fn dtype; its payload is encoded by TileLang.
    scale=max(max(abs(y)),1e-12)/448; decode(q)*scale reconstructs y.
    """
    flops_per_element=8
    _manifest_input_names=('x','weight','bias')
    def __init__(self,normalized_shape,eps=1e-5,*,target='ascend',tune=False):
        super().__init__(normalized_shape,eps,target=target,tune=tune)
    def forward(self,x,weight,bias=None):
        return self._run(x,weight,bias)
    def _infer_output_shapes(self,x_shape,weight_shape,bias_shape=None):
        return {'q':x_shape,'scale':x_shape[:-len(self.normalized_shape)]}
    def _check(self,inputs):
        super()._check(inputs)
        for t in inputs[1:]:
            if t is not None:
                _check_param(t,self.normalized_shape,inputs[0].dtype,'affine')


class RMSNormQuantFwdOp(_Rows):
    """RMSNorm + per-row torch.float8_e4m3fn output and fp32 scale."""
    flops_per_element=7
    _manifest_input_names=('x','weight')
    def forward(self,x,weight):
        return self._run(x,weight)
    def _infer_output_shapes(self,x_shape,weight_shape):
        return {'q':x_shape,'scale':x_shape[:-len(self.normalized_shape)]}
    def _check(self,inputs):
        super()._check(inputs)
        _check_param(inputs[1],self.normalized_shape,inputs[0].dtype,'weight')


def _check_param(t, shape, dtype, name):
    if t is None or tuple(t.shape)!=tuple(shape) or t.dtype!=dtype:
        raise ValueError(f'{name} must have shape {tuple(shape)} and dtype {dtype}')


def _check_stat(t,shape,name):
    _check_param(t,shape,torch.float32,name)


def _check_row_bwd(x,dy,w,inv,ns):
    _check_param(dy,x.shape,x.dtype,'grad_out')
    _check_param(w,ns,x.dtype,'weight')
    _check_stat(inv,x.shape[:-len(ns)],'rstd')


def _check_spatial(x,dy,w,mean,inv,groups):
    if x.ndim<2 or groups<=0 or x.shape[1]%groups:
        raise ValueError('positive num_groups must divide C')
    _check_param(dy,x.shape,x.dtype,'grad_out')
    if w is not None:
        _check_param(w,(x.shape[1],),x.dtype,'weight')
    for t in (mean,inv):
        _check_stat(t,(x.shape[0],groups),'saved statistic')


# Registration is local to these 12 operators. Imports compile no kernels.
def _builder(fn, **fixed):
    def build(*specs, **params):
        def launch(*tensors):
            return fn(*tensors, **params, **fixed)
        return launch
    return build


for _cls, _fn, _fixed in (
    (LayerNormBwdOp,_run_layer_bwd,{}), (RMSNormBwdOp,_run_rms_bwd,{}),
    (GroupNormBwdOp,_run_group_bwd,{}), (InstanceNormBwdOp,_run_instance_bwd,{}),
    (WeightNormFwdOp,_run_weight,{}), (GemmaRMSNormFwdOp,_run_gemma,{}),
    (QKNormFwdOp,_run_qk,{}), (GroupRMSNormFwdOp,_run_group_rms,{}),
    (BatchNormInferenceFwdOp,_run_batch,{}), (SpectralNormPowerIterFwdOp,_run_spectral,{}),
    (LayerNormQuantFwdOp,_run_quant,{'centered':True}),
    (RMSNormQuantFwdOp,_run_quant,{'centered':False}),
):
    register(_cls.__name__)(_builder(_fn,**_fixed))

__all__ = ['LayerNormBwdOp','RMSNormBwdOp','GroupNormBwdOp','InstanceNormBwdOp',
           'WeightNormFwdOp','LayerNormQuantFwdOp','RMSNormQuantFwdOp','QKNormFwdOp',
           'GemmaRMSNormFwdOp','GroupRMSNormFwdOp','BatchNormInferenceFwdOp',
           'SpectralNormPowerIterFwdOp']
