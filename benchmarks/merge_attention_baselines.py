"""Ascend attention-state baselines shared by pytest and the guarded profiler launcher.

The launcher accepts
``--implementation-factory benchmarks.merge_attention_baselines:ascend_implementations``.
"""

import warnings

from benchmarks import api as bench
from benchmarks.baselines import optional_baseline
from benchmarks.vllm_ascend_dcp import dcp_implementations

__all__ = ["ascend_implementations"]


def _attention_update_merge(merge):
    """Adapt the native flattened natural-log API, including layout copies in each call."""

    def run(v_a, s_a, v_b, s_b):
        dim = v_a.shape[-1]
        value, lse = merge(
            [s_a.contiguous().view(-1), s_b.contiguous().view(-1)],
            [v_a.contiguous().view(-1, dim), v_b.contiguous().view(-1, dim)],
            1,
        )
        return value.view_as(v_a), lse.view_as(s_a)

    return run


def ascend_implementations(
    op_name: str, case: bench.Case, *, vllm_ascend_source=None
) -> dict[str, bench.Implementation]:
    """Return eager and available native NPU comparisons for an attention-state merge.

    Every returned implementation uses the case's reference and numerical policy. Native
    execution errors propagate to the runner, including an unavailable underlying CANN op.
    """
    if op_name != "MergeAttentionStatesFwdOp":
        return {}

    implementations = {"torch-ref": bench.Implementation(case.reference)}
    dim = case.inputs[0].shape[-1]
    if not 8 <= dim <= 512 or dim % 8:
        warnings.warn(
            f"torch-npu-attention-update omitted: head dimension {dim} must be an "
            "8-element multiple in [8, 512]",
            stacklevel=2,
        )
    else:
        merge = optional_baseline("torch_npu", "torch-npu", "npu_attention_update")
        if merge is None:
            warnings.warn(
                "torch-npu-attention-update omitted: torch_npu.npu_attention_update is unavailable",
                stacklevel=2,
            )
        else:
            implementations["torch-npu-attention-update"] = bench.Implementation(
                _attention_update_merge(merge)
            )
    implementations.update(dcp_implementations(case.inputs, source_root=vllm_ascend_source))
    return implementations
