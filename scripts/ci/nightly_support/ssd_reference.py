"""Independent FP64 causal coupling reference, with one final output rounding."""

import hashlib
from pathlib import Path

from reference_placement import ReferencePlan, cpu_tree


def coupling(C, B, chunk_size):
    import torch

    batch, length, groups, state = C.shape
    if length % chunk_size:
        raise ValueError("chunk_size must divide sequence length")
    chunks = length // chunk_size
    shape = (batch, chunks, chunk_size, groups, state)
    # Native FP32 matmul can round the contraction to the wrong side of a
    # BF16 midpoint. Keep the mathematical contraction in FP64 on CPU, then
    # round only once to the declared output dtype; no candidate data is used.
    out = torch.einsum("bcqgn,bcsgn->bcgqs", C.reshape(shape).double(), B.reshape(shape).double())
    mask = torch.tril(torch.ones(chunk_size, chunk_size, dtype=torch.bool))
    return (out * mask).to(C.dtype)


class CouplingCase:
    def __init__(self, original):
        self.original = original
        self.reference = lambda *inputs: coupling(
            *cpu_tree(inputs), original.arguments["chunk_size"]
        )

    def __getattr__(self, name):
        return getattr(self.original, name)


def prepare_ssd_reference(op, case):
    if op != "SSDChunkCouplingFwdOp" or case.inputs[0].device.type != "npu":
        return None
    return ReferencePlan(
        CouplingCase(case),
        cpu_tree,
        dict(
            reference_coupling="cpu-fp64-causal-contraction-v1",
            reference_coupling_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        ),
    )
