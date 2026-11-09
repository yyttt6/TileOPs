"""Empty projections retain negative controls for both public decode outputs."""

import pytest
import torch

from workloads.numerics import Request, verify
from workloads.sequence_modeling.engram import EngramDecodeWorkload

pytestmark = pytest.mark.smoke


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("batch,d,k,dilation", [(1, 16, 1, 1), (17, 17, 3, 2)])
@pytest.mark.parametrize("zero_history", [False, True])
def test_empty_projection_verification_checks_both_outputs(
    dtype, batch, d, k, dilation, zero_history
):
    workload = EngramDecodeWorkload(batch, 0, d, 4, k, dilation, dtype)
    inputs = workload.gen_inputs()
    if zero_history:
        inputs[2].zero_()
    evidence = workload.verification(*inputs)
    assert len(evidence.controls) == 2
    result = verify(
        workload.ref_program,
        inputs,
        evidence=evidence,
        requests={"reference": Request(workload.ref_program, inputs)},
    )["reference"]
    assert result.checked_outputs == result.total_outputs == 2
    for index in range(2):

        def stale(*args, index=index):
            outputs = list(workload.ref_program(*args))
            outputs[index] = outputs[index] + 1
            return tuple(outputs)

        with pytest.raises(AssertionError):
            verify(
                workload.ref_program,
                inputs,
                evidence=evidence,
                requests={"stale": Request(stale, inputs)},
            )
