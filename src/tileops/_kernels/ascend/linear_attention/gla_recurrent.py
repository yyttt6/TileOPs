"""Single-step GLA: FP32 state update and readout, with tiled value columns."""

import functools
from typing import ClassVar

from tileops._kernels.kernel_base import Kernel
from tileops._kernels.linear_attention.call_spec import GLADecodeFwdInterface

try:
    import tilelang
    import tilelang.ascend.language as T
except ImportError:
    T = None

if T is not None:

    @functools.lru_cache(maxsize=128)
    def build(batch, heads, dk, dv, dtype, scale):
        columns = 16 if batch * heads < 16 else 32
        keys = tilelang.next_power_of_2(dk)
        kp = (dk + 15) // 16 * 16
        tiles = (dv + columns - 1) // columns
        jobs = batch * heads * tiles
        grid = min(64, jobs)
        scale = dk**-0.5 if scale <= 0 else scale

        @T.prim_func
        def main(
            Q: T.Tensor((batch * heads, dk), dtype),
            K: T.Tensor((batch * heads, dk), dtype),
            V: T.Tensor((batch * heads, dv), dtype),
            G: T.Tensor((batch * heads, dk), dtype),
            State: T.Tensor((batch * heads, dk, dv), dtype),
            O: T.Tensor((batch * heads, dv), dtype),
            New: T.Tensor((batch * heads, dk, dv), dtype),
        ):
            with T.Kernel(grid) as bx:
                q = T.alloc_shared((kp,), dtype)
                k = T.alloc_shared((kp,), dtype)
                g = T.alloc_shared((kp,), dtype)
                v = T.alloc_shared((columns,), dtype)
                state = T.alloc_shared((kp, columns), dtype)
                updated = T.alloc_shared((kp, columns), dtype)
                out = T.alloc_shared((columns,), dtype)
                for step in T.Pipelined((jobs + grid - 1) // grid, num_stages=2):
                    job = step * grid + bx
                    if job < jobs:
                        head = job // tiles
                        col = job % tiles * columns
                        valid = T.min(columns, dv - col)
                        T.copy(Q[head, :], q[:dk])
                        T.copy(K[head, :], k[:dk])
                        T.copy(G[head, :], g[:dk])
                        T.copy(V[head, col : col + valid], v[:valid])
                        T.copy(State[head, :, col : col + valid], state[:dk, :valid])
                        with T.SimtVF(threads=min(1024, keys * columns)):
                            sums = T.alloc_reducer((columns,), "float32", op="sum")
                            T.reducer_init(sums)
                            for i, j in T.Parallel(keys, columns):
                                if i < dk and j < valid:
                                    value = T.exp(T.cast(g[i], "float32")) * T.cast(
                                        state[i, j], "float32"
                                    ) + T.cast(k[i], "float32") * T.cast(v[j], "float32")
                                    updated[i, j] = value
                                    T.reducer_update(sums[j], T.cast(q[i], "float32") * value)
                            result = T.alloc_fragment((columns,), "float32")
                            T.finalize_reducer(sums, result)
                            for j in T.Parallel(columns):
                                out[j] = result[j] * scale
                        T.copy(updated[:dk, :valid], New[head, :, col : col + valid])
                        T.copy(out[:valid], O[head, col : col + valid])

        return tilelang.compile(main, target="ascend", out_idx=[5, 6])


class AscendGLARecurrentKernel(Kernel, GLADecodeFwdInterface):
    devices: ClassVar[frozenset[str]] = frozenset({"npu"})

    @classmethod
    def entry_for(cls, call):
        return call, lambda: cls(call)

    def __init__(self, call):
        super().__init__()
        self.call = call
        self.compiled = build(
            call.batch,
            call.heads,
            call.dim_k,
            call.dim_v,
            str(call.dtype).removeprefix("torch."),
            call.scale,
        )

    def forward(self, q, k, v, gk, state):
        c = self.call
        bh = c.batch * c.heads
        o, new = self.compiled(
            q.contiguous().view(bh, c.dim_k),
            k.contiguous().view(bh, c.dim_k),
            v.contiguous().view(bh, c.dim_v),
            gk.contiguous().view(bh, c.dim_k),
            state.contiguous().view(bh, c.dim_k, c.dim_v),
        )
        return o.view(c.batch, c.heads, c.dim_v), new.view_as(state)


if T is not None:
    from tileops.backend._registry import register_kernel_type

    register_kernel_type("GLARecurrentFwdOp", "ascend", AscendGLARecurrentKernel)
