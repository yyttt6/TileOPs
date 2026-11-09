"""Two-level affine prefixes for long dense sequences."""
import functools
import tilelang
import tilelang.ascend.language as T
from .deltanet_scan import build_scan


def gather(B,N,H,K,V,dtype,G):
    NG=(N+G-1)//G;J=B*NG*H;grid=min(64,J)
    @T.prim_func
    def main(M:T.Tensor((B*N*H,K,K),dtype),U:T.Tensor((B*N*H,K,V),dtype),
             MG:T.Tensor((J,K,K),dtype),UG:T.Tensor((J,K,V),dtype)):
        with T.Kernel(grid) as bx:
            m=T.alloc_shared((K,K),dtype);u=T.alloc_shared((K,V),dtype)
            for step in T.serial((J+grid-1)//grid):
                j=step*grid+bx
                if j<J:
                    b=j//(NG*H);g=j//H%NG
                    r=(b*N+T.min((g+1)*G,N)-1)*H+j%H
                    T.copy(M[r,:,:],m);T.copy(U[r,:,:],u)
                    T.copy(m,MG[j,:,:]);T.copy(u,UG[j,:,:])
    return tilelang.compile(main,target='ascend',out_idx=[2,3])


def apply_groups(B,N,H,K,V,dtype,G,initial):
    NG=(N+G-1)//G;J=B*N*H;JG=B*NG*H;grid=min(32,J);R=K//2
    @T.prim_func
    def main(M:T.Tensor((J,K,K),dtype),U:T.Tensor((J,K,V),dtype),
             MG:T.Tensor((JG,K,K),dtype),UG:T.Tensor((JG,K,V),dtype),
             MO:T.Tensor((J,K,K),dtype),UF:T.Tensor((J,K,V),'float32')):
        with T.Kernel(grid) as bx:
            a=T.alloc_l1((K,K),dtype);b=T.alloc_l1((K,K),dtype);v=T.alloc_l1((V,K),dtype)
            aa=T.alloc_l0a((K,K),dtype);bb=T.alloc_l0b((K,K),dtype);vv=T.alloc_l0b((V,K),dtype)
            mm=T.alloc_l0c((K,K),'float32');uu=T.alloc_l0c((K,V),'float32')
            mp=T.alloc_shared((R,K),'float32');up=T.alloc_shared((R,V),'float32')
            oldm=T.alloc_shared((R,K),dtype);oldu=T.alloc_shared((R,V),dtype)
            newm=T.alloc_shared((R,K),dtype)
            for step in T.serial((J+grid-1)//grid):
                j=step*grid+bx
                if j<J:
                    local_group=(j//H%N)//G;group=j//(N*H)*NG+local_group
                    r=(group-T.min(local_group,1))*H+j%H
                    T.copy(M[j,:,:],a);T.copy(a,aa)
                    if initial:
                        T.copy(MG[r,:,:],b,transpose=True);T.copy(b,bb)
                        T.gemm(aa,bb,mm,transpose_B=True,clear_accum=True)
                        T.dual_copy(mm,mp);T.dual_copy(M[j,:,:],oldm)
                        with T.SimtVF(threads=256):
                            for i,k in T.Parallel(R,K):newm[i,k]=mp[i,k] if local_group>0 else T.cast(oldm[i,k],'float32')
                        T.dual_copy(newm,MO[j,:,:])
                    T.copy(UG[r,:,:],v,transpose=True);T.copy(v,vv)
                    T.gemm(aa,vv,uu,transpose_B=True,clear_accum=True)
                    T.dual_copy(uu,up);T.dual_copy(U[j,:,:],oldu)
                    with T.SimtVF(threads=256):
                        for i,k in T.Parallel(R,V):up[i,k]=T.cast(oldu[i,k],'float32')+(up[i,k] if local_group>0 else T.float32(0))
                    T.dual_copy(up,UF[j,:,:])
    return tilelang.compile(main,target='ascend',out_idx=[4,5])

class GroupScan:
    def __init__(self,B,N,H,K,V,dtype,initial,G=4):
        assert N>=G
        self.G=G;self.NG=(N+G-1)//G;self.initial=initial
        self.local=build_scan(B,N,H,K,V,dtype,save_full=False,group_size=G)
        self.gather=gather(B,N,H,K,V,dtype,G)
        self.groups=build_scan(B,self.NG,H,K,V,dtype,save_full=False)
        self.apply=apply_groups(B,N,H,K,V,dtype,G,initial)
    def __call__(self,m,u):
        stride=1
        while stride<self.G:
            m,u,_=self.local(m,u,stride);stride*=2
        mg,ug=self.gather(m,u);stride=1
        while stride<self.NG:
            mg,ug,_=self.groups(mg,ug,stride);stride*=2
        mo,full=self.apply(m,u,mg,ug)
        return mo if self.initial else m,full
    def get_kernel_source(self):
        return '\n'.join(k.get_kernel_source() for k in (self.local,self.gather,self.groups,self.apply))
build_group_scan=functools.lru_cache(maxsize=64)(GroupScan)
