"""Verify existing KDA manifest workloads through the public Op on Ascend."""

import pytest
import torch

from tests.workload_test_base import TestBase
from tileops.linear_attention import KDAFwdOp
from tileops.manifest import load_adts, load_manifest, load_workloads
from tileops.manifest._plan import entry_plan
from tileops.manifest._workload import instantiate
from workloads.device import run_device, run_device_available
from workloads.linear_attention.kda import KDAFwdCall


class AscendKDATest(KDAFwdCall, TestBase):
    def ref_program(self, *inputs):
        # The existing golden and normalized-error check require FP64.
        cpu_inputs = tuple(x.cpu() if x is not None else None for x in inputs)
        return super().ref_program(*cpu_inputs)


def _cases():
    name = "KDAFwdOp"
    plan = entry_plan(name, load_manifest()[name], load_adts(), resolve=False)
    smoke_labels = {"kda-tp1-4x1024", "kda-tp1-decode-b1", "kda-bounded-packed"}
    smoke, full = [], []
    for row in load_workloads(name):
        for dtype in row["dtype_cases"]:
            call = instantiate(plan, row, dtype)
            if row["label"] in smoke_labels:
                smoke.append(pytest.param(call, id=call.case_id, marks=pytest.mark.smoke))
            else:
                full.append(pytest.param(call, id=call.case_id, marks=pytest.mark.full))
    return smoke + full


@pytest.mark.parametrize("call", _cases())
def test_public_ascend_kda(call):
    if torch.device(run_device()).type != "npu" or not run_device_available():
        pytest.skip("requires an Ascend NPU and --tileops-device=npu")

    workload = AscendKDATest(call)
    op = KDAFwdOp(**workload.arguments())

    def run_on_npu(*inputs):
        outputs = op(*inputs)
        assert all(x.device == inputs[0].device for x in outputs)
        return tuple(x.cpu() for x in outputs)

    workload.check(op, *workload.gen_inputs(), runs=run_on_npu)

    # Resolve the class after calling the Op so the test exercises plugin discovery.
    from tileops._kernels.ascend.linear_attention.kda import AscendKDAFwdKernel

    assert [type(kernel) for kernel in op.iter_kernels()] == [AscendKDAFwdKernel]
