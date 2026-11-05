"""CPU oracle for the existing TileOPs CUDA Div semantics.

Source: kernels/elementwise/arithmetic.py DivTruncFwdKernel and
FloorDivideFwdKernel._floor_divide. CPU low-width torch.div floor is not
shape-invariant on this installation; NPU torch.div trunc uses another order.
No tolerance is changed. Inputs must be CPU tensors.
"""
import torch
import math

def div_reference(a,b,mode=None):
 assert a.device.type==b.device.type=='cpu'
 dtype=a.dtype;x,y=a.float(),b.float();q=x/y
 if mode is None:return q.to(dtype)
 if mode=='trunc':return q.to(dtype).float().trunc().to(dtype)
 assert mode=='floor'
 # Python math.fmod uses scalar libm. This CPU build's vector fmod
 # returns NaN for finite operands with extreme ratios.
 xx,yy=torch.broadcast_tensors(x,y)
 def remainder(u,v):
  if math.isinf(u) or v==0:return float('nan')
  return math.fmod(u,v)
 rem=torch.tensor([remainder(u,v) for u,v in zip(xx.flatten().tolist(),yy.flatten().tolist())],dtype=torch.float32).reshape(xx.shape)
 div=(x-rem)/y
 div=torch.where((rem!=0)&((y<0)!=(rem<0)),div-1,div)
 floored=div.floor().to(dtype).float()
 rounded=torch.where(div-floored>.5,(floored+1).to(dtype).float(),floored)
 signed_zero=torch.copysign(torch.zeros_like(q),q)
 slow=torch.where(y==0,q,torch.where(div!=0,rounded,signed_zero))
 # Original CUDA fast tier proves mathematical floor below these limits.
 # FP64 division of two FP32 operands is sufficient to resolve the integer
 # boundary in this range, independently of the kernel's FP32 residual.
 limit={torch.float16:2**13,torch.bfloat16:2**16,torch.float32:2**22}[dtype]
 fast=torch.copysign((x.double()/y.double()).floor().float(),q)
 return torch.where((q.abs()<limit)&(y.abs()<float('inf')),fast,slow).to(dtype)
