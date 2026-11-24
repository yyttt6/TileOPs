"""The Ascend expert MLP must follow live masked counts through both projections."""

import pytest
import torch

from tileops.moe import MaskedLayoutSpec, MoEExpertMLPFwdOp
from workloads.device import run_device
from workloads.moe import ref_moe_grouped_gemm


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("activation", "dtype", "bias_gate", "bias_down"),
    [
        ("gelu_and_mul", torch.float16, True, False),
        ("silu_and_mul", torch.bfloat16, False, True),
        ("swigluoai", torch.float16, True, True),
    ],
)
def test_expert_mlp_masked_counts_and_biases(activation, dtype, bias_gate, bias_down):
    device = run_device()
    if torch.device(device).type != "npu":
        pytest.skip("Ascend masked expert MLP regression")
    groups, capacity, hidden, ffn = 3, 19, 64, 32
    layout = MaskedLayoutSpec(max_m=capacity)
    op = MoEExpertMLPFwdOp(layout, activation=activation)
    x = torch.randn(groups, capacity, hidden, dtype=dtype, device=device)
    gate = torch.randn(groups, 2 * ffn, hidden, dtype=dtype, device=device) * 0.1
    down = torch.randn(groups, hidden, ffn, dtype=dtype, device=device) * 0.1
    b_gate = (
        torch.randn(groups, 2 * ffn, dtype=dtype, device=device) if bias_gate else None
    )
    b_down = (
        torch.randn(groups, hidden, dtype=dtype, device=device) if bias_down else None
    )
    counts = torch.empty(groups, dtype=torch.int32, device=device)
    workspace = torch.full(
        (groups * capacity + 2, hidden), 137, dtype=dtype, device=device
    )
    out = workspace[1:-1].view_as(x)
    for sizes in ([0, 13, 19], [7, 0, 3]):
        meta = torch.tensor(sizes, dtype=torch.int32)
        counts.copy_(meta)
        out.fill_(137)
        result = op(x, gate, down, counts, b_gate_up=b_gate, b_down=b_down, out=out)
        assert result is out
        up = ref_moe_grouped_gemm(
            x.cpu(),
            gate.cpu(),
            meta,
            layout,
            activation=activation,
            bias=b_gate.cpu() if bias_gate else None,
        )
        expected = ref_moe_grouped_gemm(
            up, down.cpu(), meta, layout, bias=b_down.cpu() if bias_down else None
        )
        mask = torch.arange(capacity)[None, :] < meta[:, None]
        actual = result.cpu()
        torch.testing.assert_close(
            actual[mask].float(), expected[mask].float(), rtol=0.02, atol=0.1
        )
        guards = workspace.cpu()[[0, -1]]
        torch.testing.assert_close(guards, torch.full_like(guards, 137), rtol=0, atol=0)
