"""Packed recurrence boundary and normalization controls; no device execution."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "nightly_support"))
from packed_reference import packed_recurrence, prepare_packed_reference

pytestmark = pytest.mark.smoke


@pytest.mark.parametrize("op", ["GLAFwdOp", "DeltaNetFwdOp"])
def test_independent_sequences_and_empty_sequence(op):
    q = torch.ones(1, 3, 1, 2)
    k = torch.ones_like(q) * 0.25
    v = torch.ones(1, 3, 1, 3)
    gate = torch.zeros_like(q) if op == "GLAFwdOp" else torch.ones(1, 3, 1) * 0.5
    initial = torch.arange(18).reshape(3, 1, 2, 3).float() * 0.01
    offsets = torch.tensor([0, 1, 1, 3])
    result = packed_recurrence(op, q, k, v, gate, initial, offsets, offsets, scale=0.25)
    separate = []
    finals = []
    for i, (a, b) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
        local = torch.tensor([0, int(b - a)])
        o, h = packed_recurrence(
            op,
            q[:, a:b],
            k[:, a:b],
            v[:, a:b],
            gate[:, a:b],
            initial[i : i + 1],
            local,
            local,
            scale=0.25,
        )
        separate.append(o)
        finals.append(h)
    torch.testing.assert_close(result[0], torch.cat(separate, 1))
    torch.testing.assert_close(result[1], torch.cat(finals, 0))
    torch.testing.assert_close(result[1][1], initial[1])
    # Zero scale suppresses readout, never the state update.
    zeros, states = packed_recurrence(op, q, k, v, gate, initial, offsets, offsets, scale=0.0)
    assert torch.count_nonzero(zeros) == 0
    torch.testing.assert_close(states, result[1])
    with pytest.raises(ValueError, match="disagree"):
        packed_recurrence(op, q, k, v, gate, initial, offsets, torch.tensor([0, 1, 2, 3]))


def test_single_delta_step_has_declared_l2norm_rounding():
    q = torch.tensor([[[[0.13, 0.41]]]], dtype=torch.bfloat16)
    k = q.clone()
    v = torch.tensor([[[[0.3]]]], dtype=q.dtype)
    beta = torch.ones(1, 1, 1, dtype=q.dtype) * 0.5
    offsets = torch.tensor([0, 1])
    o, h = packed_recurrence(
        "DeltaNetFwdOp", q, k, v, beta, None, offsets, None, l2norm=True, scale=1.0
    )
    qq = (
        (q.float() * torch.rsqrt(q.float().square().sum(-1, keepdim=True) + 1e-6))
        .to(q.dtype)
        .double()
    )
    expected_h = qq[:, 0].unsqueeze(-1) * v[:, 0].double().unsqueeze(-2) * 0.5
    torch.testing.assert_close(h, expected_h.float())
    torch.testing.assert_close(o[:, 0], (qq[:, 0].unsqueeze(-1) * expected_h).sum(-2).to(q.dtype))


def test_adapter_does_not_change_dense_or_cpu_cases():
    cpu = SimpleNamespace(
        inputs=(torch.zeros(1, 2, 1, 2), None, None, None, None, torch.tensor([0, 2]))
    )
    assert prepare_packed_reference("GLAFwdOp", cpu) is None
    q = SimpleNamespace(device=SimpleNamespace(type="npu"))
    dense = SimpleNamespace(inputs=(q, None, None, None, None, None))
    assert prepare_packed_reference("GLAFwdOp", dense) is None
