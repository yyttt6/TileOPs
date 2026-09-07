"""Manifest-driven benchmark for the fused GEMM epilogue ops (T264, 110-112).

The workloads are ``GemmFwdOp``'s twelve, reused verbatim through
``load_workloads`` on each op's own manifest entry (which copies them).
"""

import pytest
import torch

from benchmarks.benchmark_base import ManifestBenchmark, workload_params
from tileops.manifest import load_workloads
from tileops.ops import GemmBiasFwdOp, GemmBiasGeluFwdOp, GemmBiasReluFwdOp
from workloads.gemm import (
    GemmBiasGeluWorkload,
    GemmBiasReluWorkload,
    GemmBiasWorkload,
)

_BIAS_OP_NAME = "GemmBiasFwdOp"
_RELU_OP_NAME = "GemmBiasReluFwdOp"
_GELU_OP_NAME = "GemmBiasGeluFwdOp"

_CASES = (
    (_BIAS_OP_NAME, GemmBiasFwdOp, GemmBiasWorkload),
    (_RELU_OP_NAME, GemmBiasReluFwdOp, GemmBiasReluWorkload),
    (_GELU_OP_NAME, GemmBiasGeluFwdOp, GemmBiasGeluWorkload),
)


def _prepare(op_cls, workload_cls, m, n, k, dtype, trans_a, trans_b):
    """Build op + workload and check the numbers before anything is timed."""
    workload = workload_cls(m, n, k, dtype, trans_a=trans_a, trans_b=trans_b)
    op = op_cls(trans_a=trans_a, trans_b=trans_b)
    inputs = workload.gen_inputs()
    tol = 1.6e-2 if dtype is torch.bfloat16 else 1e-2
    torch.testing.assert_close(
        op(*inputs), workload.ref_program(*inputs), rtol=tol, atol=tol
    )
    return op, workload, inputs


@pytest.mark.parametrize(
    *workload_params(load_workloads(_BIAS_OP_NAME), "m, n, k, dtype, trans_a, trans_b")
)
def test_bench_gemm_bias(m, n, k, dtype, trans_a, trans_b):
    op, workload, inputs = _prepare(
        GemmBiasFwdOp, GemmBiasWorkload, m, n, k, dtype, trans_a, trans_b)
    ManifestBenchmark("GemmBiasFwdOp", op, workload).run(lambda: op(*inputs))


@pytest.mark.parametrize(
    *workload_params(load_workloads(_RELU_OP_NAME), "m, n, k, dtype, trans_a, trans_b")
)
def test_bench_gemm_bias_relu(m, n, k, dtype, trans_a, trans_b):
    op, workload, inputs = _prepare(
        GemmBiasReluFwdOp, GemmBiasReluWorkload, m, n, k, dtype, trans_a, trans_b)
    ManifestBenchmark("GemmBiasReluFwdOp", op, workload).run(lambda: op(*inputs))


@pytest.mark.parametrize(
    *workload_params(load_workloads(_GELU_OP_NAME), "m, n, k, dtype, trans_a, trans_b")
)
def test_bench_gemm_bias_gelu(m, n, k, dtype, trans_a, trans_b):
    op, workload, inputs = _prepare(
        GemmBiasGeluFwdOp, GemmBiasGeluWorkload, m, n, k, dtype, trans_a, trans_b)
    ManifestBenchmark("GemmBiasGeluFwdOp", op, workload).run(lambda: op(*inputs))
