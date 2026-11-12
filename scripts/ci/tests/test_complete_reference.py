"""Independent recurrence controls for full saved-output references, CPU only."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "nightly_support"))
from complete_reference import CompleteChunkCase, delta_chunk_reference, delta_chunk_verification
from reference_plan import prepare_reference

from workloads.numerics import compare_outputs

pytestmark = pytest.mark.smoke


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_delta_full_outputs_against_per_token_recurrence(dtype):
    torch.manual_seed(91)
    q, k, v = (torch.randn(1, 2, 12, d).to(dtype) * 0.1 for d in (4, 4, 3))
    beta = torch.rand(1, 2, 12).to(dtype) * 0.5
    got = delta_chunk_reference(q, k, v, beta, 4)
    state = torch.zeros(1, 2, 4, 3, dtype=torch.float64)
    expected, states = [], [state.clone()]
    for t in range(12):
        kt, vt = k[:, :, t].double(), v[:, :, t].double()
        residual = (vt - (kt.unsqueeze(-1) * state).sum(-2)) * beta[:, :, t].double().unsqueeze(-1)
        state = state + kt.unsqueeze(-1) * residual.unsqueeze(-2)
        expected.append((q[:, :, t].double().unsqueeze(-1) * state).sum(-2))
        if (t + 1) % 4 == 0:
            states.append(state.clone())
    torch.testing.assert_close(got[0], torch.stack(expected, 2).to(dtype))
    torch.testing.assert_close(got[1], torch.stack(states, 2).float())
    evidence = delta_chunk_verification(dtype)
    result = compare_outputs(got, got, evidence)
    assert result.checked_outputs == result.total_outputs == 6
    for index in range(6):
        corrupt = list(got)
        corrupt[index] = torch.zeros_like(got[index])
        with pytest.raises(AssertionError):
            compare_outputs(tuple(corrupt), got, evidence)
    # Both public inverse outputs satisfy their defining triangular system.
    for start in range(0, 12, 4):
        kc = k[:, :, start : start + 4].double()
        bc = beta[:, :, start : start + 4].double()
        system = torch.eye(4) + torch.tril(bc.unsqueeze(-1) * (kc @ kc.transpose(-1, -2)), -1)
        torch.testing.assert_close(
            system @ got[2][:, :, start : start + 4].double(),
            torch.eye(4).double().expand(1, 2, 4, 4),
            atol=0.002,
            rtol=0.002,
        )


def test_gla_third_output_is_checked():
    from workloads.linear_attention.gla import gla_fwd_chunked_torch

    torch.manual_seed(92)
    q, k, v = (torch.randn(1, 8, 2, 4) * 0.1 for _ in range(3))
    g = -torch.rand_like(q)
    original = SimpleNamespace(
        inputs=(q, k, v, g, None),
        reference=lambda *x: (lambda o, s: (o, s[:, -1], s))(
            *gla_fwd_chunked_torch(*x[:4], 4, initial_state=x[4], with_chunk_states=True)
        ),
    )
    case = CompleteChunkCase("GLAChunkFwdOp", original)
    expected = case.reference(*original.inputs)
    assert compare_outputs(expected, expected, case.verification).checked_outputs == 3
    corrupt = list(expected)
    corrupt[2] = torch.zeros_like(corrupt[2])
    with pytest.raises(AssertionError):
        compare_outputs(tuple(corrupt), expected, case.verification)


def test_unaffected_cpu_reference_has_unchanged_identity():
    case = SimpleNamespace(inputs=(torch.zeros(1, 1, 1, 1),))
    assert prepare_reference("GLAFwdOp", case).case is case
    assert prepare_reference("GLAFwdOp", case).cache_identity == {}


def test_scoped_adapter_does_not_change_gdn_cache_identity():
    from reference_placement import prepare_reference as existing

    q = SimpleNamespace(shape=(1, 4, 2, 4), device=SimpleNamespace(type="npu"))
    case = SimpleNamespace(
        inputs=(q, None, None, None, None, None, None, None), reference=lambda *x: x
    )
    before, after = existing("GDNFwdOp", case), prepare_reference("GDNFwdOp", case)
    assert before.cache_identity == after.cache_identity
    assert type(before.case) is type(after.case)


def test_complete_adapter_identity_is_explicit_and_npu_scoped():
    q = SimpleNamespace(device=SimpleNamespace(type="npu"), dtype=torch.float16)
    case = SimpleNamespace(inputs=(q,), arguments={"chunk_size": 4}, reference=lambda *x: x)
    plan = prepare_reference("DeltaNetChunkFwdOp", case)
    assert plan.case.original is case
    assert plan.cache_identity["reference_completion"] == "chunk-training-full-output-cpu-v1"
    assert len(plan.cache_identity["reference_completion_sha256"]) == 64
    assert prepare_reference("ReLUFwdOp", case).cache_identity == {}


def test_large_gla_signal_reduction_and_corruption():
    from complete_reference import full_chunk_verification

    torch.manual_seed(93)
    x = (torch.randn(2, 16384, 4, 64) * 0.001).half()
    y = x.clone()
    y.flatten()[::1024] += 0.00001
    evidence = full_chunk_verification("GLAChunkFwdOp", x.dtype)
    assert compare_outputs((y,), (x,), evidence).checked_outputs == 1
    with pytest.raises(AssertionError):
        compare_outputs((torch.zeros_like(x),), (x,), evidence)
