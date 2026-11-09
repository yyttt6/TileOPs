"""Focused Ascend linear attention inputs and independent CPU references."""

import torch

from workloads.device import run_device
from workloads.linear_attention.gdn import gdn_ref


def gla_direct_recurrence(q, k, v, gate, initial, lengths, scale):
    """Direct FP64 recurrence, independent of the chunked attention factors."""
    q, k, v, gate = (x.cpu().double() for x in (q, k, v, gate))
    output = torch.empty_like(v)
    states = []
    start = 0
    for sequence, length in enumerate(lengths):
        state = initial[sequence].cpu().double().clone()
        for token in range(start, start + length):
            state = gate[0, token].exp()[..., None] * state
            state = state + k[0, token, ..., None] * v[0, token, :, None, :]
            output[0, token] = (q[0, token, ..., None] * state).sum(-2) * scale
        states.append(state.float())
        start += length
    return (output, torch.stack(states))


def gla_diagonal_inputs(lengths, dtype):
    torch.manual_seed(7731)
    size = (1, sum(lengths), 2, 128)
    q, k, v = (torch.randn(size, device=run_device(), dtype=dtype) * 0.1 for _ in range(3))
    gate = -torch.rand(size, device=run_device(), dtype=dtype)
    initial = torch.randn(len(lengths), 2, 128, 128, device=run_device()) * 0.1
    inputs = (q, k, v, gate, initial)
    before = tuple((x.cpu().clone() for x in inputs))
    offsets_cpu = torch.tensor(
        [0, *torch.tensor(lengths, device="cpu").cumsum(0).tolist()],
        dtype=torch.int64,
        device="cpu",
    )
    offsets = offsets_cpu.to(run_device()) if len(lengths) > 1 else None
    scale = 0.23
    return (inputs, before, offsets, offsets_cpu, scale)


def gla_training_inputs(chunk, keys, values, dtype):
    torch.manual_seed(4219)
    length = 2 * chunk
    q, k, gate = (
        torch.randn(1, length, 2, keys, device=run_device(), dtype=dtype) * 0.1 for _ in range(3)
    )
    gate = -gate.abs()
    v = torch.randn(1, length, 2, values, device=run_device(), dtype=dtype) * 0.1
    initial = torch.randn(1, 2, keys, values, device=run_device(), dtype=torch.float32) * 0.1
    inputs = (q, k, v, gate, initial)
    saved = tuple((x.cpu().clone() for x in inputs))
    return (inputs, saved, length)


def gla_training_reference(saved, length, chunk, dtype):
    cq, ck, cv, cg, state = (x.double() for x in saved)
    boundaries, outputs = ([state.clone()], [])
    for token in range(length):
        state = cg[:, token].exp().unsqueeze(-1) * state
        state = state + ck[:, token].unsqueeze(-1) * cv[:, token].unsqueeze(-2)
        outputs.append((cq[:, token].unsqueeze(-1) * state).sum(-2) * 0.23)
        if (token + 1) % chunk == 0:
            boundaries.append(state.clone())
    return (torch.stack(outputs, 1).to(dtype), state.float(), torch.stack(boundaries, 1).float())


def gla_gradient_case(chunk, keys, values, dtype):
    torch.manual_seed(4220)
    length = 2 * chunk
    q, k, v, gate, do = (
        torch.randn(1, length, 2, width, device="cpu", dtype=dtype) * 0.1
        for width in (keys, keys, values, keys, values)
    )
    gate = -gate.abs()
    initial, dfinal = (
        torch.randn(1, 2, keys, values, device="cpu", dtype=torch.float32) * 0.1 for _ in range(2)
    )
    leaves = [x.double().detach().requires_grad_(True) for x in (q, k, v, gate, initial)]
    cq, ck, cv, cg, state = leaves
    states, outputs = ([state.clone()], [])
    for token in range(length):
        state = cg[:, token].exp().unsqueeze(-1) * state
        state = state + ck[:, token].unsqueeze(-1) * cv[:, token].unsqueeze(-2)
        outputs.append((cq[:, token].unsqueeze(-1) * state).sum(-2) * 0.23)
        if (token + 1) % chunk == 0:
            states.append(state.clone())
    loss = (torch.stack(outputs, 1) * do.double()).sum() + (state * dfinal.double()).sum()
    expected = tuple((x.float() for x in torch.autograd.grad(loss, leaves)))
    boundaries = torch.stack(states, 1).detach().float()
    before = (do, dfinal, q, k, v, gate, boundaries)
    inputs = tuple((x.to(run_device()) for x in before))
    return (inputs, before, expected)


def delta_normalized_case(dtype, length, initial):
    torch.manual_seed(4382)
    q, k, v = (
        torch.randn(1, length, 3, 128, device=run_device(), dtype=dtype) * 0.1 for _ in range(3)
    )
    beta = torch.rand(1, length, 3, device=run_device(), dtype=dtype) * 0.5
    state = torch.randn(1, 3, 128, 128, device=run_device()) * 0.01 if initial else None
    inputs = (q, k, v, beta, state)
    cpu = tuple((x.cpu() if x is not None else None for x in inputs))
    expected = gdn_ref(
        cpu[0], cpu[1], cpu[2], torch.zeros_like(cpu[3]), cpu[3], cpu[4], l2norm=True
    )
    return (inputs, cpu, expected)


def gdn_packed_rectangular_case(dtype, keys, values):
    torch.manual_seed(9471)
    length, heads, value_heads = (257, 2, 6)
    q, k = (
        torch.randn(1, length, heads, keys, device=run_device(), dtype=dtype) * 0.1
        for _ in range(2)
    )
    v = torch.randn(1, length, value_heads, values, device=run_device(), dtype=dtype) * 0.1
    gate, beta = (
        torch.randn(1, length, value_heads, device=run_device(), dtype=dtype) for _ in range(2)
    )
    offsets_cpu = torch.tensor([0, 0, 1, 32, 257], device="cpu", dtype=torch.int64)
    offsets = offsets_cpu.to(run_device())
    state = torch.randn(4, value_heads, values, keys, device=run_device()) * 0.01
    rates = torch.zeros(value_heads, device=run_device())
    bias = torch.full((value_heads,), -3.0, device=run_device())
    inputs = (q, k, v, gate, beta, state, offsets, offsets_cpu, rates, bias)
    saved = tuple((x.cpu().clone() for x in inputs))
    expected = gdn_ref(
        *saved,
        raw_gate=True,
        beta_sigmoid=True,
        allow_neg_eigval=True,
        l2norm=True,
        state_v_first=True,
    )
    return (inputs, saved, expected)
