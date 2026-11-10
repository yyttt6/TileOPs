"""Memory-bounded GDN scheduling for dense and packed inference calls."""

import torch

from tileops._kernels.ascend.linear_attention.gdn import recurrent
from tileops._kernels.ascend.linear_attention.gdn_chunked_paired import build as build_chunked


class Pipeline:
    fill_cores = True
    preferred_vt = 128
    batch_multiplier = 2

    def __init__(self, call, offsets):
        self.call = call
        self.offsets = offsets
        self.used = {}

    def dense(self, q, k, v, g, beta, state, A_log, dt_bias):
        c = self.call
        B, S = q.shape[:2]
        limit = 4096 if c.value_heads >= 32 else 8192
        if S >= 128 and c.dim_k in (64, 128) and c.dim_v in (64, 128):
            from tileops._kernels.ascend.linear_attention.gdn_pipeline_vector64 import (
                Pipeline as OutputPipeline,
            )

            return OutputPipeline.dense(self, q, k, v, g, beta, state, A_log, dt_bias)
        if S == 0:
            return v.clone(), state if state is not None else torch.zeros(
                (B, c.value_heads, c.dim_k, c.dim_v), device=q.device, dtype=torch.float32
            )
        batch_size = max(1, limit // min(S, limit))
        batch_outputs = []
        batch_states = []
        for first in range(0, B, batch_size):
            end = min(B, first + batch_size)
            carry = state[first:end].contiguous() if state is not None else None
            outputs = []
            for start in range(0, S, limit):
                stop = min(S, start + limit)
                length = stop - start
                args = (
                    end - first,
                    length,
                    c.heads,
                    c.value_heads,
                    c.dim_k,
                    c.dim_v,
                    str(c.dtype).removeprefix("torch."),
                    c.scale,
                    carry is not None,
                    c.l2norm,
                    c.gate_in_kernel,
                    c.beta_sigmoid,
                    c.allow_neg_eigval,
                )
                if length >= 128 and c.dim_k in (64, 128) and c.dim_v in (64, 128):
                    kernel = build_chunked(*args)
                else:
                    kernel = recurrent(*args, False, ())
                self.used[args] = kernel
                h0 = (
                    carry
                    if carry is not None
                    else torch.empty(
                        (end - first, c.value_heads, c.dim_k, c.dim_v),
                        device=q.device,
                        dtype=torch.float32,
                    )
                )
                xs = tuple(x[first:end, start:stop].contiguous() for x in (q, k, v, g, beta))
                out, carry = kernel(*xs, h0, A_log, dt_bias)
                outputs.append(out)
            batch_outputs.append(torch.cat(outputs, dim=1) if len(outputs) > 1 else outputs[0])
            batch_states.append(carry)
        return (
            torch.cat(batch_outputs, dim=0) if len(batch_outputs) > 1 else batch_outputs[0],
            torch.cat(batch_states, dim=0) if len(batch_states) > 1 else batch_states[0],
        )

    def __call__(self, q, k, v, g, beta, state, A_log, dt_bias):
        c = self.call
        if state is not None and c.state_v_first:
            state = state.transpose(-1, -2).contiguous()
        if not self.offsets:
            out, final = self.dense(q, k, v, g, beta, state, A_log, dt_bias)
        else:
            lengths = [b - a for a, b in zip(self.offsets, self.offsets[1:], strict=False)]
            if lengths and min(lengths) > 0 and len(set(lengths)) == 1:
                n = len(lengths)
                length = lengths[0]
                xs = tuple(x.reshape(n, length, *x.shape[2:]) for x in (q, k, v, g, beta))
                out, final = self.dense(*xs, state, A_log, dt_bias)
                out = out.reshape_as(v)
            else:
                outputs = []
                states = []
                for i, (a, b) in enumerate(zip(self.offsets, self.offsets[1:], strict=False)):
                    xs = tuple(x[:, a:b].contiguous() for x in (q, k, v, g, beta))
                    h0 = state[i : i + 1] if state is not None else None
                    o, h = self.dense(*xs, h0, A_log, dt_bias)
                    outputs.append(o)
                    states.append(h)
                out = torch.cat(outputs, dim=1)
                final = torch.cat(states, dim=0)
        return out, final.transpose(-1, -2).contiguous() if c.state_v_first else final

    def get_kernel_source(self):
        return "\n".join(k.get_kernel_source() for k in self.used.values())
