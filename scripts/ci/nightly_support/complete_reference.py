"""Complete independent CPU oracles for chunk-training saved outputs.

The six DeltaNet outputs are defined by its public WY representation. This
module imports no candidate kernels and never obtains a reference from the op.
"""

import hashlib
from pathlib import Path

from reference_placement import ReferencePlan, cpu_tree


def delta_chunk_reference(q, k, v, beta, chunk_size):
    import torch

    dtype = q.dtype
    batch, heads, length, dim_k = q.shape
    dim_v = v.shape[-1]
    if length % chunk_size:
        raise ValueError("DeltaNet training requires complete chunks")
    # Solve (I + strict_lower(beta_i <k_i,k_j>)) A = I independently
    # of the candidate's inverse/scan kernels, in FP64 on CPU.
    state = torch.zeros(batch, heads, dim_k, dim_v, dtype=torch.float64)
    states = [state]
    outputs, inverses, weights, values = [], [], [], []
    eye = torch.eye(chunk_size, dtype=torch.float64)
    eye_batch = eye.expand(batch, heads, chunk_size, chunk_size)
    for start in range(0, length, chunk_size):
        sl = slice(start, start + chunk_size)
        qc, kc, vc = (x[:, :, sl].double() for x in (q, k, v))
        bc = beta[:, :, sl].double()
        system = eye + torch.tril(bc.unsqueeze(-1) * (kc @ kc.transpose(-1, -2)), -1)
        inverse = torch.linalg.solve_triangular(system, eye_batch, upper=False, unitriangular=True)
        w = inverse @ (bc.unsqueeze(-1) * kc)
        u = inverse @ (bc.unsqueeze(-1) * vc)
        residual = u - w @ state
        output = qc @ state + torch.tril(qc @ kc.transpose(-1, -2)) @ residual
        state = state + kc.transpose(-1, -2) @ residual
        outputs.append(output)
        states.append(state)
        inverses.append(inverse)
        weights.append(w)
        values.append(u)
    aw = torch.cat(inverses, dim=2).to(dtype)
    return (
        torch.cat(outputs, dim=2).to(dtype),
        torch.stack(states, dim=2).float(),
        aw,
        aw.clone(),
        torch.cat(weights, dim=2).to(dtype),
        torch.cat(values, dim=2).to(dtype),
    )


def full_chunk_verification(op, dtype):
    import torch

    from workloads.numerics import Custom, Exact, compare_outputs

    # Preserve the existing forward bounds; extend them to every saved output.
    tolerances = (
        {torch.float32: 1e-2, torch.float16: 5e-2, torch.bfloat16: 1e-1}
        if op == "GLAChunkFwdOp"
        else {torch.float32: 1e-3, torch.float16: 2e-2, torch.bfloat16: 5e-2}
    )
    tol = tolerances[dtype]

    def validate(got, expected):
        compare_outputs(got, expected, Exact(atol=tol, rtol=tol))
        for actual, reference in zip(got, expected, strict=True):
            a, b = actual.double().flatten(), reference.double().flatten()
            if torch.count_nonzero(b):
                cosine = torch.dot(a, b) / (
                    torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
                )
                assert cosine > 0.99, "saved output lost its reference signal"

    return Custom(validate, "every saved output, original bounds and FP64 signal alignment")


def delta_chunk_verification(dtype):
    return full_chunk_verification("DeltaNetChunkFwdOp", dtype)


class CompleteChunkCase:
    def __init__(self, op, original):
        self.original = original
        if op == "GLAChunkFwdOp":
            # The declared workload already returns o/final/h. The benchmark's
            # legacy Partial(2) classification incorrectly hid that third output.
            self.reference = lambda *inputs: original.reference(*cpu_tree(inputs))
            self.verification = full_chunk_verification(op, original.inputs[0].dtype)
        elif op == "DeltaNetChunkFwdOp":
            self.reference = lambda *inputs: delta_chunk_reference(
                *cpu_tree(inputs), original.arguments["chunk_size"]
            )
            self.verification = delta_chunk_verification(original.inputs[0].dtype)
        else:
            raise ValueError(op)

    def __getattr__(self, name):
        return getattr(self.original, name)


def prepare_complete_reference(op, case):
    if op not in {"GLAChunkFwdOp", "DeltaNetChunkFwdOp"} or case.inputs[0].device.type != "npu":
        return None
    return ReferencePlan(
        CompleteChunkCase(op, case),
        cpu_tree,
        dict(
            reference_completion="chunk-training-full-output-cpu-v1",
            reference_completion_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        ),
    )
