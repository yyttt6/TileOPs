"""Adaptive LayerNorm candidate dispatcher; no production registration change."""
from . import _ada_resident as resident
from . import _ada_fp32_simd as simd

def select_build(original,m,n,dtype,eps,gated):
    if dtype=='float32' and n%64==0 and 0<n and 10*n*4<208*1024:
        return simd.build(m,n,dtype,eps,gated)
    if 0<n<=4096:return resident.build(m,n,dtype,eps,gated)
    return original(m,n,dtype,eps,gated)
