"""Shared-expert public-call correctness and stream ownership on Ascend."""

import pytest
import torch

from tests.workload_test_base import TestBase
from tileops.manifest import load_adts, load_manifest, load_workloads
from tileops.manifest.plan import entry_plan
from tileops.manifest.workload import instantiate
from tileops.moe import SharedExpertMLPFwdOp
from workloads.device import run_device, run_device_available
from workloads.moe import SharedExpertMLPWorkload
from workloads.numerics import compare_outputs
from workloads.workload_base import manifest_call


def _manifest_cases():
    name = "SharedExpertMLPFwdOp"
    plan = entry_plan(name, load_manifest()[name], load_adts(), resolve=False)
    return [
        instantiate(plan, row, dtype)
        for row in load_workloads(name)
        for dtype in row["dtype_cases"]
    ]


@pytest.fixture
def ascend_device():
    if torch.device(run_device()).type != "npu" or not run_device_available():
        pytest.skip("requires an Ascend NPU and --tileops-device=npu")


@pytest.mark.smoke
@pytest.mark.parametrize("case", _manifest_cases(), ids=lambda case: case.case_id)
def test_shared_expert_manifest(case, ascend_device):
    """All declared decode/prefill workloads use the existing numerical contract."""
    workload = SharedExpertMLPWorkload(case)
    TestBase.check(workload, SharedExpertMLPFwdOp(**case.arguments({})), *workload.gen_inputs())


@pytest.mark.smoke
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize(
    ("tokens", "hidden", "ffn"),
    [
        (0, 16, 16),
        (1, 16, 16),
        (17, 128, 96),
        (32, 144, 80),
        (32, 256, 4096),
        (32, 5120, 1536),
        (96, 4224, 2048),
        (129, 144, 80),
        (255, 4224, 2048),
        (384, 4224, 2048),
        (512, 256, 2048),
        (512, 4224, 2048),
        (512, 4224, 6144),
        (600, 4224, 2048),
        (1018, 4224, 2048),
        (4096, 4096, 256),
    ],
    ids=[
        "empty",
        "minimal",
        "token-tail",
        "fused-activation-transition",
        "fused-buffer-reuse",
        "small-down-ring",
        "fused-tail-entry",
        "prefill-and-reduction-tail",
        "fused-tail-single-row",
        "fallback-tail-boundary",
        "pipelined-prefill",
        "prefill-fusion-odd-k-tiles",
        "prefill-narrow-fusion-wave-tail",
        "fused-tail-remainder",
        "fused-tail-near-full",
        "prefill-down-ring-tail-grid",
    ],
)
def test_shared_expert_reuses_entries_on_a_nondefault_stream(
    dtype, tokens, hidden, ffn, ascend_device
):
    """Changed weights and shapes must be read on the caller's stream without input writes.

    Minimal widths cover aligned short DMA; seventeen tokens and a 129-token prefill
    cover partial Cube rows. Non-power-of-two widths cover smaller reduction tiles.
    Reusing one op at a second shape checks that a specialization does not capture it.
    A 512-token prefill covers wide Cube tiles and pipelined activation for both dtypes;
    its transition to 513 tokens covers the fused tail after an aligned count.
    The 32-to-33 transition covers the fused activation and its separate-stage fallback.
    A wider shared expert retains several output tiles per core, checking scratch reuse.
    The GLM shape covers the short down projection's two accumulator slots.
    Wide prefill covers an odd K tile count in the fused L0 ring; a short down
    contraction covers a partially filled task grid and its 4097-token fallback.
    A wider FFN covers the 192-column fusion plan, its larger M contraction,
    and the tail after that aligned plan.
    A 96-token call is the fused prefill's lower entry, 255 fills its last tile but
    one row, and 600/1018 cover a remainder and a near-full padded tail; 384 keeps
    the 128-row fallback where the padded 256-row tile would waste a whole wave.
    """
    op = SharedExpertMLPFwdOp()
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        retained = []
        for rows in (tokens, tokens, tokens + 1):
            call = manifest_call("SharedExpertMLPFwdOp", {"D": dtype}, T=rows, H=hidden, S=ffn)
            workload = SharedExpertMLPWorkload(call)
            args = workload.gen_inputs()
            snapshots = [x.clone() for x in args]
            expected = workload.ref_program(*args)
            actual = op(*args)
            # Consume the output before host synchronization; previous outputs stay live
            # across later launches, so scratch reuse cannot overwrite them.
            consumed = actual.clone()
            compare_outputs(consumed, expected, workload.verification(*args))
            retained.append((actual, expected, workload.verification(*args)))
            assert all(actual.data_ptr() != x.data_ptr() for x in args if x.numel())
            for current, before in zip(args, snapshots, strict=True):
                torch.testing.assert_close(current, before, rtol=0, atol=0)
        for actual, expected, policy in retained:
            compare_outputs(actual, expected, policy)
    stream.synchronize()


@pytest.mark.smoke
@pytest.mark.in_tree_kernels
def test_shared_expert_refuses_unaligned_cube_inputs(ascend_device):
    """An unsupported contraction or output stride fails with an alignment reason."""
    for hidden, ffn in ((17, 16), (16, 17)):
        call = manifest_call("SharedExpertMLPFwdOp", {"D": "float16"}, T=1, H=hidden, S=ffn)
        args = SharedExpertMLPWorkload(call).gen_inputs()
        with pytest.raises(ValueError, match="divisible by 16"):
            SharedExpertMLPFwdOp()(*args)
