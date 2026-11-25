"""Case factories of the quantization family."""

from typing import Any

from benchmarks._cases import Entry, default_binder
from benchmarks.api import Implementation
from benchmarks.baselines import private_inputs
from workloads.quantization.fp8_quant import FP8QuantWorkload
from workloads.quantization.int8_dequant import (
    INT8DequantPerBlockWorkload,
    INT8DequantPerChannelWorkload,
    INT8DequantPerTensorWorkload,
)
from workloads.quantization.quantize import (
    FP4QuantWorkload,
    FP8QuantPerBlockWorkload,
    INT4QuantPerGroupWorkload,
    INT8QuantPerBlockWorkload,
    INT8QuantPerChannelWorkload,
    INT8QuantPerTensorWorkload,
    SmoothQuantWorkload,
)

ENTRIES = {
    "FP8QuantPerTokenGroupFwdOp": Entry(FP8QuantWorkload.from_call),
    "FP8QuantPerBlockFwdOp": Entry(FP8QuantPerBlockWorkload.from_call),
    "INT4QuantPerGroupFwdOp": Entry(INT4QuantPerGroupWorkload.from_call),
    "INT8DequantPerTensorFwdOp": Entry(INT8DequantPerTensorWorkload.from_call),
    "INT8DequantPerChannelFwdOp": Entry(INT8DequantPerChannelWorkload.from_call),
    "INT8DequantPerBlockFwdOp": Entry(INT8DequantPerBlockWorkload.from_call),
    "INT8QuantPerTensorFwdOp": Entry(INT8QuantPerTensorWorkload.from_call),
    "INT8QuantPerChannelFwdOp": Entry(INT8QuantPerChannelWorkload.from_call),
    "INT8QuantPerTokenGroupFwdOp": Entry(INT8QuantPerBlockWorkload.from_call),
    "SmoothQuantFwdOp": Entry(SmoothQuantWorkload.from_call),
}


def _quantize_binder(op: Any, case: Any) -> Implementation:
    """The op on private copies of what it writes: ``x`` where it fake-quantizes ``x`` in
    place, otherwise any ``out_q`` the call passes after ``x``."""
    positions = (0,) if case.arguments["inplace"] else range(1, len(case.inputs))
    return private_inputs(default_binder(op, case).run, case.inputs, *positions)


ENTRIES.update(
    {
        "FP4QuantPerGroupFwdOp": Entry(FP4QuantWorkload.from_call, binder=_quantize_binder),
    }
)
