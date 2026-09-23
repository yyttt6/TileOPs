"""T307 adapters and independent torch references; no timings or registration.

Input distribution and tolerances fixed before execution: randn * 0.25,
fp16 atol=rtol=1e-3 and bf16 atol=rtol=1.6e-2, matching the current workspace
tileops-ascend-harness/adapters/gemm_convolution.py correctness policy.
"""
import hashlib
import torch
import torch.nn.functional as F
from .ops import OPS
from .kernels import pair

TOLERANCE={'float16':dict(atol=1e-3,rtol=1e-3),'bfloat16':dict(atol=1.6e-2,rtol=1.6e-2)}

def reference_bias_relu(input,weight,bias,**params):
    """Manifest reference for the complete bias-plus-ReLU operation."""
    return F.relu(F.conv2d(input,weight,bias,**params))

def reference_causal(input,weight,bias=None,stride=1,dilation=1,groups=1):
    """Manifest reference: causal left padding and donor stride."""
    left=dilation*(weight.shape[-1]-1)
    return F.conv1d(F.pad(input,(left,0)),weight,bias,stride,0,dilation,groups)

def run_case(name,w,dtype_name,device):
    seed=int.from_bytes(hashlib.sha256((name+'/'+w['label']+'/'+dtype_name).encode()).digest()[:4],'little')
    generator=torch.Generator(device='cpu').manual_seed(seed)
    dtype=getattr(torch,dtype_name)
    def rand(shape):
        return (torch.randn(shape,generator=generator,dtype=dtype)*0.25).to(device)
    n,ci,*space=w['input_shape']; co=w['C_out']; groups=w.get('groups',1)
    kernel=(w['kW'],) if name=='conv1d_causal' else (w['kH'],w['kW'])
    weight_shape=(co,ci//groups,*kernel)
    params={k:tuple(v) if isinstance(v,list) else v for k,v in w.items() if k in ('stride','padding','dilation','groups')}
    op=OPS[name](w)
    if name in ('conv2d_transpose','conv2d_dgrad','conv2d_wgrad'):
        s,p,d=pair(w.get('stride',1)),pair(w.get('padding',0)),pair(w.get('dilation',1))
        out=[(v+2*pa-di*(k-1)-1)//st+1 for v,pa,di,k,st in zip(space,p,d,kernel,s)]
        dy=rand((n,co,*out))
        if name=='conv2d_wgrad':
            x=rand(w['input_shape']); inputs=(x,dy)
            got=op(*inputs)
            ref=torch.nn.grad.conv2d_weight(x,weight_shape,dy,**params)
        else:
            weight=rand(weight_shape)
            if name=='conv2d_dgrad':
                inputs=(dy,weight); got=op(*inputs)
                ref=torch.nn.grad.conv2d_input(tuple(w['input_shape']),weight,dy,**params)
            else:
                bias=rand((ci,)) if 'bias_shape' in w else None
                inputs=(dy,weight,bias); got=op(*inputs)
                output_padding=tuple(v-((o-1)*st-2*pa+di*(k-1)+1) for v,o,st,pa,di,k in zip(space,out,s,p,d,kernel))
                ref=F.conv_transpose2d(dy,weight,bias,output_padding=output_padding,**params)
    elif name=='im2col':
        x=rand(w['input_shape']); inputs=(x,); got=op(x)
        params.pop('groups',None)
        ref=F.unfold(x,kernel,**params)
    else:
        x=rand(w['input_shape']); weight=rand(weight_shape)
        bias=rand(w['bias_shape']) if 'bias_shape' in w else None
        inputs=(x,weight,bias); got=op(*inputs)
        if name=='conv1d_causal':
            left=w.get('dilation',1)*(w['kW']-1)
            ref=F.conv1d(F.pad(x,(left,0)),weight,bias,**params)
        else:
            ref=F.conv2d(x,weight,bias,**params)
            if name=='conv2d_bias_relu': ref=F.relu(ref)
    torch.npu.synchronize()
    got_cpu,ref_cpu=got.detach().cpu(),ref.detach().cpu()
    shape_ok=got_cpu.shape==ref_cpu.shape
    dtype_ok=got_cpu.dtype==ref_cpu.dtype==dtype
    finite=bool(torch.isfinite(got_cpu).all() and torch.isfinite(ref_cpu).all())
    tol=dict(atol=0.0,rtol=0.0) if name=='im2col' else TOLERANCE[dtype_name]
    passed=shape_ok and dtype_ok and finite and torch.allclose(got_cpu,ref_cpu,**tol)
    delta=(got_cpu.float()-ref_cpu.float()).abs() if shape_ok else None
    return dict(op=name,label=w['label'],dtype=dtype_name,seed=seed,passed=bool(passed),
        input_shapes=[list(t.shape) if t is not None else None for t in inputs],output_shape=list(got.shape),
        reference_shape=list(ref.shape),tolerance=tol,finite=finite,dtype_ok=dtype_ok,
        max_abs=float(delta.max()) if delta is not None else None,
        mismatched_elements=int((delta>(tol['atol']+tol['rtol']*ref_cpu.float().abs())).sum()) if delta is not None else None,
        elements=got.numel())

def pointwise_conv2d(w,dtype,device): return run_case('pointwise_conv2d',w,dtype,device)
def depthwise_conv2d(w,dtype,device): return run_case('depthwise_conv2d',w,dtype,device)
def grouped_conv2d(w,dtype,device): return run_case('grouped_conv2d',w,dtype,device)
def dilated_conv2d(w,dtype,device): return run_case('dilated_conv2d',w,dtype,device)
def conv2d_bias_relu(w,dtype,device): return run_case('conv2d_bias_relu',w,dtype,device)
def conv1d_causal(w,dtype,device): return run_case('conv1d_causal',w,dtype,device)
def conv2d_transpose(w,dtype,device): return run_case('conv2d_transpose',w,dtype,device)
def conv2d_dgrad(w,dtype,device): return run_case('conv2d_dgrad',w,dtype,device)
def conv2d_wgrad(w,dtype,device): return run_case('conv2d_wgrad',w,dtype,device)
def im2col(w,dtype,device): return run_case('im2col',w,dtype,device)
