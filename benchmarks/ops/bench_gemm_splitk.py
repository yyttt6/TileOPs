"""Manifest-driven benchmark for split-K dense GEMM (T267, op 113).

The workloads are ``GemmFwdOp``'s twelve, reused verbatim through
``load_workloads`` on this op's own manifest entry (which copies them).  Split-K
computes the same product as ``gemm``, so ``GemmWorkload`` is the workload
class unchanged and ``torch.matmul`` is the reference.

``op.split_k`` records what the kernel's planner decided for each shape.  Only
some of the twelve shapes are starved for blocks, so only some take the split
path at all; the rest fall through to the ordinary dense GEMM.  The parameter
is reported per case rather than assumed -- see docs/reports/R268.md.
"""

import pytest
import torch

from benchmarks.benchmark_base import ManifestBenchmark, workload_params
from tileops.manifest import load_workloads
from tileops.ops import GemmSplitKFwdOp
from workloads.gemm import GemmWorkload

_OP_NAME = "GemmSplitKFwdOp"


@pytest.mark.parametrize(
    *workload_params(load_workloads(_OP_NAME), "m, n, k, dtype, trans_a, trans_b")
)
def test_bench_gemm_splitk(m, n, k, dtype, trans_a, trans_b):
    workload = GemmWorkload(m, n, k, dtype, trans_a=trans_a, trans_b=trans_b)
    op = GemmSplitKFwdOp(trans_a=trans_a, trans_b=trans_b)
    inputs = workload.gen_inputs()
    tol = 1.6e-2 if dtype is torch.bfloat16 else 1e-2
    torch.testing.assert_close(
        op(*inputs), workload.ref_program(*inputs), rtol=tol, atol=tol
    )
    bench = ManifestBenchmark(_OP_NAME, op, workload)
    bench.run(inputs)
