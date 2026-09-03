import importlib.util
from pathlib import Path

import torch


_KERNEL_PATH = (
    Path(__file__).parents[1]
    / "src"
    / "tileops_ascend"
    / "kernels"
    / "attention_decode.py"
)
_SPEC = importlib.util.spec_from_file_location("_attention_decode_kernel", _KERNEL_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_KERNEL = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_KERNEL)
build_gqa_decode_kernel = _KERNEL.build_gqa_decode_kernel
build_mha_decode_kernel = _KERNEL.build_mha_decode_kernel
build_mla_decode_kernel = _KERNEL.build_mla_decode_kernel


ATOL = 5e-3
RTOL = 1e-5


def _reference(q, k, v, cache_seqlens):
    q_cpu = q.detach().cpu().float()[:, 0]
    k_cpu = k.detach().cpu().float()
    v_cpu = v.detach().cpu().float()
    lengths = cache_seqlens.detach().cpu().tolist()
    outputs = []
    for bid, length in enumerate(lengths):
        score = torch.einsum("hd,khd->hk", q_cpu[bid], k_cpu[bid, :length])
        probability = torch.softmax(score / q.shape[-1] ** 0.5, dim=-1)
        outputs.append(torch.einsum("hk,khd->hd", probability, v_cpu[bid, :length]))
    return torch.stack(outputs).unsqueeze(1)


def _run(batch, heads, max_seqlen, lengths, dtype):
    torch.manual_seed(73)
    q_cpu = torch.randn((batch, 1, heads, 128), dtype=dtype, device="cpu")
    k_cpu = torch.randn((batch, max_seqlen, heads, 128), dtype=dtype, device="cpu")
    v_cpu = torch.randn((batch, max_seqlen, heads, 128), dtype=dtype, device="cpu")
    lengths_cpu = torch.tensor(lengths, dtype=torch.int32)
    q, k, v = q_cpu.npu(), k_cpu.npu(), v_cpu.npu()
    cache_seqlens = lengths_cpu.npu()
    kernel = build_mha_decode_kernel(batch, heads, max_seqlen, 128, dtype)
    actual = kernel(q, k, v, cache_seqlens)
    torch.npu.synchronize()
    expected = _reference(q_cpu, k_cpu, v_cpu, lengths_cpu)
    actual_cpu = actual.detach().cpu().float()
    error = (actual_cpu - expected).abs()
    max_error = float(error.max())
    print(
        f"mha batch={batch} heads={heads} max_seqlen={max_seqlen} "
        f"lengths={lengths} dtype={dtype} max_error={max_error:.8f}"
    )
    assert torch.isfinite(actual_cpu).all()
    assert torch.allclose(actual_cpu, expected, atol=ATOL, rtol=RTOL), max_error
    return max_error


def test_mha_decode_varlen_fp16():
    assert _run(2, 4, 128, [37, 113], torch.float16) <= ATOL


def test_mha_decode_bfloat16():
    assert _run(1, 4, 128, [97], torch.bfloat16) <= ATOL


def test_mha_decode_long_cache_finite():
    assert _run(1, 1, 32768, [32731], torch.float16) <= ATOL


def test_mha_decode_grid_stride_head_task():
    assert _run(1, 26, 64, [53], torch.float16) <= ATOL


def test_gqa_decode_varlen():
    batch, heads, heads_kv, max_seqlen = 2, 8, 2, 128
    lengths_cpu = torch.tensor([41, 119], dtype=torch.int32)
    torch.manual_seed(74)
    q_cpu = torch.randn((batch, heads, 128), dtype=torch.float16)
    k_cpu = torch.randn((batch, max_seqlen, heads_kv, 128), dtype=torch.float16)
    v_cpu = torch.randn((batch, max_seqlen, heads_kv, 128), dtype=torch.float16)
    kernel = build_gqa_decode_kernel(
        batch, heads, heads_kv, max_seqlen, 128, torch.float16
    )
    actual = kernel(q_cpu.npu(), k_cpu.npu(), v_cpu.npu(), lengths_cpu.npu())
    torch.npu.synchronize()
    expected = []
    group_size = heads // heads_kv
    for bid, length in enumerate(lengths_cpu.tolist()):
        k_expanded = k_cpu[bid, :length].repeat_interleave(group_size, dim=1)
        v_expanded = v_cpu[bid, :length].repeat_interleave(group_size, dim=1)
        score = torch.einsum("hd,khd->hk", q_cpu[bid].float(), k_expanded.float())
        probability = torch.softmax(score / 128**0.5, dim=-1)
        expected.append(torch.einsum("hk,khd->hd", probability, v_expanded.float()))
    expected = torch.stack(expected)
    error = (actual.detach().cpu().float() - expected).abs().max()
    print(f"gqa varlen max_error={float(error):.8f}")
    assert torch.isfinite(actual).all()
    assert float(error) <= ATOL


def test_mla_decode_compressed_kv():
    batch, heads, heads_kv, max_seqlen = 1, 8, 1, 128
    length = 103
    torch.manual_seed(75)
    q = torch.randn((batch, heads, 128), dtype=torch.float16)
    q_pe = torch.randn((batch, heads, 64), dtype=torch.float16)
    k = torch.randn((batch, max_seqlen, heads_kv, 128), dtype=torch.float16)
    k_pe = torch.randn((batch, max_seqlen, heads_kv, 64), dtype=torch.float16)
    lengths = torch.tensor([length], dtype=torch.int32)
    kernel = build_mla_decode_kernel(
        batch, heads, heads_kv, max_seqlen, 128, 64, torch.float16
    )
    actual = kernel(q.npu(), q_pe.npu(), k.npu(), k_pe.npu(), lengths.npu())
    torch.npu.synchronize()
    k_expanded = k[0, :length].repeat_interleave(heads // heads_kv, dim=1)
    k_pe_expanded = k_pe[0, :length].repeat_interleave(heads // heads_kv, dim=1)
    score = torch.einsum("hd,khd->hk", q[0].float(), k_expanded.float())
    score += torch.einsum("hd,khd->hk", q_pe[0].float(), k_pe_expanded.float())
    probability = torch.softmax(score / (128 + 64) ** 0.5, dim=-1)
    expected = torch.einsum("hk,khd->hd", probability, k_expanded.float()).unsqueeze(0)
    error = (actual.detach().cpu().float() - expected).abs().max()
    print(f"mla compressed-kv max_error={float(error):.8f}")
    assert torch.isfinite(actual).all()
    assert float(error) <= ATOL
