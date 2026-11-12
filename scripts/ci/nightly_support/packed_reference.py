"""Independent token equations for Ascend packed GLA and ungated DeltaNet.

Pinned FLA's packed path is retained as historical comparison evidence, but is
not a numerical oracle here: it returns nonfinite GLA states and inaccurate
DeltaNet outputs on audited manifest inputs. Dense/decode keep the old oracle.
"""

import hashlib
from pathlib import Path

from reference_placement import ReferencePlan, cpu_tree


def packed_recurrence(
    op, q, k, v, gate, initial, cu_seqlens, cu_seqlens_cpu, *, scale=None, l2norm=False
):
    import torch

    if op not in {"GLAFwdOp", "DeltaNetFwdOp"}:
        raise ValueError(op)
    if q.device.type != "cpu" or q.shape[0] != 1:
        raise ValueError("Packed oracle requires copied CPU inputs with batch=1")
    bounds = cu_seqlens.tolist()
    if (
        bounds[0] != 0
        or bounds[-1] != q.shape[1]
        or any(b < a for a, b in zip(bounds[:-1], bounds[1:], strict=True))
    ):
        raise ValueError("Invalid packed prefix offsets")
    if cu_seqlens_cpu is not None and cu_seqlens_cpu.tolist() != bounds:
        raise ValueError("CPU and device prefix offsets disagree")
    if initial is not None and initial.shape[0] != len(bounds) - 1:
        raise ValueError("Initial state count differs from packed sequence count")
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    outputs, states = [], []
    for i, (start, end) in enumerate(zip(bounds[:-1], bounds[1:], strict=True)):
        qs, ks, vs = q[:, start:end], k[:, start:end], v[:, start:end].double()
        if op == "DeltaNetFwdOp" and l2norm:
            # Same epsilon and FP32-normalize/input-dtype materialization as the
            # declared FLA contract, followed by an FP64 recurrence.
            qs = (qs.float() * torch.rsqrt(qs.float().square().sum(-1, keepdim=True) + 1e-6)).to(
                q.dtype
            )
            ks = (ks.float() * torch.rsqrt(ks.float().square().sum(-1, keepdim=True) + 1e-6)).to(
                k.dtype
            )
        qs, ks = qs.double(), ks.double()
        state = (
            torch.zeros(1, q.shape[2], q.shape[3], v.shape[-1], dtype=torch.float64)
            if initial is None
            else initial[i : i + 1].double().clone()
        )
        sequence = []
        for t in range(end - start):
            kt, vt = ks[:, t], vs[:, t]
            if op == "GLAFwdOp":
                state = state * gate[:, start + t].double().exp().unsqueeze(-1) + kt.unsqueeze(
                    -1
                ) * vt.unsqueeze(-2)
            else:
                residual = gate[:, start + t].double().unsqueeze(-1) * (
                    vt - (kt.unsqueeze(-1) * state).sum(-2)
                )
                state = state + kt.unsqueeze(-1) * residual.unsqueeze(-2)
            sequence.append(((qs[:, t].unsqueeze(-1) * state).sum(-2) * scale).to(q.dtype))
        outputs.append(
            torch.stack(sequence, 1)
            if sequence
            else torch.empty(1, 0, q.shape[2], v.shape[-1], dtype=q.dtype)
        )
        states.append(state.float())
    return torch.cat(outputs, 1), torch.cat(states, 0)


class PackedCase:
    def __init__(self, op, original):
        self.original = original
        self.reference = lambda *inputs: packed_recurrence(
            op,
            *cpu_tree(inputs),
            scale=original.arguments["scale"],
            l2norm=original.arguments.get("use_qk_l2norm_in_kernel", False),
        )

    def __getattr__(self, name):
        return getattr(self.original, name)


def prepare_packed_reference(op, case):
    if (
        op not in {"GLAFwdOp", "DeltaNetFwdOp"}
        or case.inputs[0].device.type != "npu"
        or case.inputs[5] is None
    ):
        return None
    return ReferencePlan(
        PackedCase(op, case),
        cpu_tree,
        dict(
            reference_packed="independent-fp64-packed-token-equations-v1",
            reference_packed_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        ),
    )
