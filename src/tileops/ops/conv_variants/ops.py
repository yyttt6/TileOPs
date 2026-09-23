"""Standalone T307 Op classes: explicit donor geometry, no global registry."""
import copy
from . import kernels

class VariantOp:
    name = None
    def __init__(self,workload):
        self.workload=copy.deepcopy(workload)
        w=self.workload
        g=w.get('groups',1); ci=w['input_shape'][1]
        assert ci%g==0 and w['C_out']%g==0
        if self.name=='pointwise_conv2d': assert w['kH']==w['kW']==1
        if self.name=='depthwise_conv2d': assert g==ci
        if self.name=='grouped_conv2d': assert 1<g<ci
        if self.name=='dilated_conv2d': assert 'dilation' in w and any(v!=1 for v in kernels.pair(w['dilation']))
        if self.name=='conv2d_bias_relu': assert 'bias_shape' in w
        if self.name=='conv1d_causal':
            assert len(w['input_shape'])==3
            assert 'padding' not in w, 'Causal donor padding semantics require PM decision'
    def __call__(self,*inputs):
        self._elem_bytes = inputs[0].element_size()
        if any(not t.is_contiguous() for t in inputs if t is not None):
            raise ValueError('T307 requires contiguous inputs')
        w=self.workload
        if self.name=='im2col': return kernels.unfold(inputs[0],w)
        if self.name=='conv2d_wgrad': return kernels.wgrad(*inputs,w)
        if self.name=='conv2d_dgrad': return kernels.adjoint(*inputs,None,w)
        if self.name=='conv2d_transpose': return kernels.adjoint(*inputs,w)
        if self.name=='conv1d_causal': return kernels.causal(*inputs,w)
        y=kernels.forward(*inputs,w)
        if self.name=='conv2d_bias_relu':
            return kernels.relu_kernel(y.numel(),str(y.dtype).split('.')[-1])(y.reshape(-1)).reshape(y.shape)
        return y

    def eval_roofline(self):
        """Evaluate the unchanged T307 manifest formula in donor coordinates."""
        from tileops.manifest import load_manifest
        w = self.workload
        n, ci, *space = w['input_shape']
        env = dict(N=n, C_in=ci, C_out=w['C_out'],
                   C_in_g=ci//w.get('groups',1), kW=w['kW'],
                   elem_bytes=self._elem_bytes,
                   bias=('present' if 'bias_shape' in w else None))
        if self.name == 'conv1d_causal':
            env.update(L_in=space[0], L_out=(space[0]-1)//w.get('stride',1)+1)
        else:
            s,p,d = (kernels.pair(w.get(key,default)) for key,default in
                     (('stride',1),('padding',0),('dilation',1)))
            out = [(v+2*pa-di*(k-1)-1)//st+1 for v,pa,di,k,st in
                   zip(space,p,d,(w['kH'],w['kW']),s)]
            env.update(H=space[0],W=space[1],out_H=out[0],out_W=out[1],kH=w['kH'])
        formula = load_manifest()[type(self).__name__]['roofline']
        for key,expr in formula.get('vars',{}).items():
            env[key] = eval(str(expr), {'__builtins__': {}}, env)
        return tuple(int(eval(str(formula[key]), {'__builtins__': {}}, env))
                     for key in ('flops','bytes'))

class PointwiseConv2dOp(VariantOp): name='pointwise_conv2d'
class DepthwiseConv2dOp(VariantOp): name='depthwise_conv2d'
class GroupedConv2dOp(VariantOp): name='grouped_conv2d'
class DilatedConv2dOp(VariantOp): name='dilated_conv2d'
class Conv2dBiasReluOp(VariantOp): name='conv2d_bias_relu'
class Conv1dCausalOp(VariantOp): name='conv1d_causal'
class Conv2dTransposeOp(VariantOp): name='conv2d_transpose'
class Conv2dDgradOp(VariantOp): name='conv2d_dgrad'
class Conv2dWgradOp(VariantOp): name='conv2d_wgrad'
class Im2colOp(VariantOp): name='im2col'

OPS={cls.name:cls for cls in (PointwiseConv2dOp,DepthwiseConv2dOp,GroupedConv2dOp,DilatedConv2dOp,
    Conv2dBiasReluOp,Conv1dCausalOp,Conv2dTransposeOp,Conv2dDgradOp,Conv2dWgradOp,Im2colOp)}
