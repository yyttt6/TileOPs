"""Place the existing GDN FP64 oracle/checks on CPU where Ascend lacks FP64 matmul.

The operator still executes on the NPU. No oracle formula, dtype, output set,
tolerance, negative control or probe is changed. Only long GDN sequences adapt.
"""

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROTOCOL = "gdn-existing-fp64-reference-on-cpu-v1"


def cpu_tree(value):
    import torch

    if isinstance(value, torch.Tensor):
        if value.requires_grad:
            raise ValueError("GDN inference CPU reference does not adapt autograd inputs")
        return value.detach().to("cpu", copy=True)
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"Unsupported reference value: {type(value)}")


def needs_cpu(op, inputs):
    if op != "GDNFwdOp" or inputs[0].device.type != "npu":
        return False
    if inputs[6] is None:
        return inputs[0].shape[1] > 1
    offsets = inputs[7] if inputs[7] is not None else inputs[6].cpu()
    bounds = offsets.tolist()
    return any(end - start > 1 for start, end in zip(bounds[:-1], bounds[1:], strict=True))


class ReferenceCase:
    def __init__(self, original):
        self.original = original
        reference = original.reference
        if reference is None:
            raise ValueError("GDN CPU placement requires its declared independent reference")

        def on_cpu(*inputs):
            return reference(*cpu_tree(inputs))

        self.reference = on_cpu

    def __getattr__(self, name):
        return getattr(self.original, name)


@dataclass
class ReferencePlan:
    case: Any
    comparison_transform: Any = None
    cache_identity: dict = field(default_factory=dict)


def prepare_reference(op, case):
    if not needs_cpu(op, case.inputs):
        return ReferencePlan(case)
    return ReferencePlan(
        ReferenceCase(case),
        cpu_tree,
        dict(
            reference_placement=PROTOCOL,
            reference_placement_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        ),
    )
