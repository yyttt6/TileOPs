"""T114 correctness, dual-regime timing, and canonical coverage evidence."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
import torch_npu

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [
    str(ROOT / "TileOPs"),
    str(ROOT / "tileops-ascend-harness"),
    str(ROOT / "TileOPs/src"),
]
import harnesslib as common  # noqa: E402
from adapters.graph_replay import dual_case, paired_regimes, perf_fields, summary  # noqa: E402

from tileops.kernels.attention_decode import (  # noqa: E402
    build_gqa_decode_kernel,
    build_gqa_decode_paged_kernel,
    build_gqa_prefill_paged_kernel,
    build_mla_decode_kernel,
)
from tileops.perf.formulas import (  # noqa: E402
    deepseek_mla_decode_roofline,
    gqa_decode_paged_roofline,
    gqa_decode_roofline,
    gqa_prefill_paged_with_kv_cache_fwd_roofline,
)

ATOL = 5e-3
RTOL = 1e-5
OPS = (
    "GroupedQueryAttentionDecodeWithKVCacheFwdOp",
    "MultiHeadLatentAttentionDecodeWithKVCacheFwdOp",
    "GroupedQueryAttentionDecodePagedWithKVCacheFwdOp",
    "GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp",
)


def _command_output(command):
    return subprocess.run(
        command, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
    ).stdout.strip()


def _record(label, actual, reference, shape, dtype, **metadata):
    torch.npu.synchronize()
    delta = (actual.float() - reference.float()).abs()
    finite = bool(torch.isfinite(actual).all())
    passed = finite and bool(torch.allclose(actual, reference, atol=ATOL, rtol=RTOL))
    result = {
        "shape": shape,
        "dtype": dtype,
        "label": label,
        "max_abs_err": float(delta.max().cpu()),
        "max_rel_err": float(
            (delta / reference.float().abs().clamp_min(1e-12)).max().cpu()
        ),
        "tolerance": {"atol": ATOL, "rtol": RTOL},
        "passed": passed,
        "finite": finite,
        **metadata,
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    if not passed:
        raise AssertionError(f"{label}: max_abs_err={result['max_abs_err']}")
    return result


def _rejection(op_name, build):
    try:
        build()
    except TypeError as exc:
        message = str(exc)
        required = ("requires dtype=torch.float16", "received torch.bfloat16", "1.5625e-2", "atol=5e-3")
        passed = all(text in message for text in required)
        if not passed:
            raise AssertionError(f"incomplete D022 error: {message}")
        return {
            "shape": {},
            "dtype": "bfloat16",
            "label": "d022-explicit-rejection",
            "max_abs_err": 0.0,
            "max_rel_err": 0.0,
            "tolerance": {"atol": ATOL, "rtol": RTOL},
            "passed": True,
            "finite": True,
            "error": message,
            "op": op_name,
        }
    raise AssertionError(f"{op_name} unexpectedly accepted bfloat16")


def _gqa_reference(q, k, v):
    return F.scaled_dot_product_attention(
        q.unsqueeze(2), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3), enable_gqa=True
    ).squeeze(2)


def _mla_reference(q, q_pe, k, k_pe):
    q_cat = torch.cat((q.float(), q_pe.float()), dim=-1)
    k_cat = torch.cat((k.float(), k_pe.float()), dim=-1)
    head_map = torch.arange(q.shape[1], device=q.device) // (q.shape[1] // k.shape[2])
    k_cat = k_cat[:, :, head_map, :]
    values = k.float()[:, :, head_map, :]
    scores = torch.einsum("bhd,bnhd->bhn", q_cat, k_cat) * q_cat.shape[-1] ** -0.5
    return torch.einsum("bhn,bnhd->bhd", torch.softmax(scores, dim=-1), values).to(q.dtype)


def _paged_logical(cache, table, page_size, length):
    indices = []
    for pos in range(length):
        indices.append(int(table[pos // page_size]) * page_size + pos % page_size)
    return cache.index_select(0, torch.tensor(indices, device=cache.device, dtype=torch.long))


def _paged_reference(q, k, v, lengths, tables, page_size):
    outputs = []
    for bid, length in enumerate(lengths):
        logical_k = _paged_logical(k, tables[bid], page_size, length).unsqueeze(0)
        logical_v = _paged_logical(v, tables[bid], page_size, length).unsqueeze(0)
        outputs.append(_gqa_reference(q[bid : bid + 1], logical_k, logical_v)[0])
    return torch.stack(outputs)


def _pack_pages(logical, table, page_size, physical_tokens):
    pages = torch.zeros(
        physical_tokens, logical.shape[1], logical.shape[2], dtype=logical.dtype
    )
    for logical_page, physical_page in enumerate(table):
        start = logical_page * page_size
        if start >= logical.shape[0]:
            break
        count = min(page_size, logical.shape[0] - start)
        dst = int(physical_page) * page_size
        pages[dst : dst + count] = logical[start : start + count]
    return pages


def _prefill_reference(q, k_new, v_new, old_k, old_v, q_lens, old_lens):
    outputs = []
    start = 0
    heads, heads_kv = q.shape[1], k_new.shape[1]
    head_map = torch.arange(heads, device=q.device) // (heads // heads_kv)
    for q_len, old_len, old_k_b, old_v_b in zip(q_lens, old_lens, old_k, old_v, strict=True):
        q_b = q[start : start + q_len]
        k_b = torch.cat((old_k_b, k_new[start : start + q_len]), dim=0)[:, head_map]
        v_b = torch.cat((old_v_b, v_new[start : start + q_len]), dim=0)[:, head_map]
        scores = torch.einsum("qhd,khd->hqk", q_b.float(), k_b.float()) * q.shape[-1] ** -0.5
        q_pos = torch.arange(q_len, device=q.device).view(1, q_len, 1)
        k_pos = torch.arange(old_len + q_len, device=q.device).view(1, 1, -1)
        scores = scores.masked_fill(k_pos > old_len + q_pos, float("-inf"))
        outputs.append(
            torch.einsum("hqk,khd->qhd", torch.softmax(scores, dim=-1), v_b.float()).to(q.dtype)
        )
        start += q_len
    return torch.cat(outputs)


def _perf_case(label, shape, dtype_name, tested, baseline, flops, nbytes):
    profile = common.load_profile(common.PROFILE_PATH)
    sol_hw, bound_hw = common.sol_us(flops, nbytes, dtype_name, profile, "peak")
    sol_ach, bound_ach = common.sol_us(flops, nbytes, dtype_name, profile, "achieved")
    regimes, unavailable = paired_regimes(
        tested, baseline, torch, torch.npu.synchronize, atol=ATOL, rtol=RTOL, common=common
    )
    return dual_case(
        {
            "shape": shape,
            "dtype": dtype_name,
            "label": label,
            "flops": flops,
            "bytes": nbytes,
            "roofline_source": "TileOPs attention formula",
            "roofline_error": None,
            "sol_us_hw": sol_hw,
            "sol_bound_hw": bound_hw,
            "sol_us_achieved": sol_ach,
            "sol_bound_achieved": bound_ach,
        },
        regimes,
        unavailable,
        common=common,
        sol_hw=sol_hw,
        sol_achieved=sol_ach,
    )


def _gqa_cases():
    cases = []
    perf = None
    for n, heads, heads_kv, label in ((129, 4, 2, "nondiv-s129"), (32768, 1, 1, "long-cache-32k")):
        torch.manual_seed(114 + n)
        q = torch.randn(1, heads, 128, dtype=torch.float16).npu()
        k = torch.randn(1, n, heads_kv, 128, dtype=torch.float16).npu()
        v = torch.randn_like(k)
        lengths = torch.tensor([n], dtype=torch.int32).npu()
        kernel = build_gqa_decode_kernel(1, heads, heads_kv, n, 128, torch.float16)
        def tested():
            return kernel(q, k, v, lengths)

        def baseline():
            return _gqa_reference(q, k, v)
        actual, reference = tested(), baseline()
        shape = {"q": list(q.shape), "k": list(k.shape), "v": list(v.shape), "out": list(q.shape)}
        cases.append(_record(label, actual, reference, shape, "float16"))
        if n == 129:
            flops, nbytes = gqa_decode_roofline(
                batch=1, heads=heads, heads_kv=heads_kv, seqlen_kv=n, dim=128, dtype="float16"
            )
            perf = _perf_case(label, shape, "float16", tested, baseline, flops, nbytes)
    cases.append(
        _rejection(
            OPS[0], lambda: build_gqa_decode_kernel(1, 4, 2, 129, 128, torch.bfloat16)
        )
    )
    return cases, [perf]


def _mla_cases():
    cases = []
    perf = None
    for n, dtype, label in (
        (129, torch.float16, "nondiv-s129-fp16"),
        (193, torch.bfloat16, "nondiv-s193-bf16"),
        (32768, torch.bfloat16, "long-cache-32k-bf16"),
    ):
        torch.manual_seed(214 + n)
        q = torch.randn(1, 4 if n != 32768 else 1, 128, dtype=dtype).npu()
        q_pe = torch.randn(q.shape[0], q.shape[1], 64, dtype=dtype).npu()
        k = torch.randn(1, n, 1, 128, dtype=dtype).npu()
        k_pe = torch.randn(1, n, 1, 64, dtype=dtype).npu()
        lengths = torch.tensor([n], dtype=torch.int32).npu()
        kernel = build_mla_decode_kernel(1, q.shape[1], 1, n, 128, 64, dtype)
        def tested():
            return kernel(q, q_pe, k, k_pe, lengths)

        def baseline():
            return _mla_reference(q, q_pe, k, k_pe)
        actual, reference = tested(), baseline()
        shape = {"q": list(q.shape), "q_pe": list(q_pe.shape), "k": list(k.shape), "k_pe": list(k_pe.shape), "out": list(q.shape)}
        dtype_name = "bfloat16" if dtype == torch.bfloat16 else "float16"
        cases.append(_record(label, actual, reference, shape, dtype_name))
        if n == 129:
            flops, nbytes = deepseek_mla_decode_roofline(
                batch=1, heads=q.shape[1], heads_kv=1, seqlen_kv=n, dim=128, pe_dim=64, dtype=dtype_name
            )
            perf = _perf_case(label, shape, dtype_name, tested, baseline, flops, nbytes)
    return cases, [perf]


def _paged_cases():
    cases = []
    perf = None
    specs = (
        (1, 4, 2, 512, 128, [193], [[3, 2, 1, 0]], "reverse-tail"),
        (2, 4, 2, 1024, 256, [37, 701], [[3, 1, 0, 2], [1, 3, 2, 0]], "dynamic-varlen-tail"),
        (1, 1, 1, 32768, 256, [32731], [list(reversed(range(128)))], "long-cache-32k"),
    )
    for batch, heads, heads_kv, physical, page_size, lengths_cpu, tables_cpu, label in specs:
        torch.manual_seed(314 + physical)
        q = torch.randn(batch, heads, 128, dtype=torch.float16).npu()
        k = torch.randn(physical, heads_kv, 128, dtype=torch.float16).npu()
        v = torch.randn_like(k)
        lengths = torch.tensor(lengths_cpu, dtype=torch.int32).npu()
        tables = torch.tensor(tables_cpu, dtype=torch.int32).npu()
        kernel = build_gqa_decode_paged_kernel(
            batch, heads, heads_kv, physical, 128, page_size, torch.float16
        )
        def tested():
            return kernel(q, k, v, lengths, tables)

        logical_pairs = [
            (
                _paged_logical(k, tables_cpu[bid], page_size, length),
                _paged_logical(v, tables_cpu[bid], page_size, length),
            )
            for bid, length in enumerate(lengths_cpu)
        ]

        def baseline():
            return torch.stack(
                [
                    _gqa_reference(q[bid : bid + 1], logical_k.unsqueeze(0), logical_v.unsqueeze(0))[0]
                    for bid, (logical_k, logical_v) in enumerate(logical_pairs)
                ]
            )
        actual, reference = tested(), baseline()
        shape = {"q": list(q.shape), "k": list(k.shape), "v": list(v.shape), "out": list(q.shape)}
        cases.append(
            _record(
                label, actual, reference, shape, "float16", page_size=page_size,
                cache_seqlens=lengths_cpu, block_table=tables_cpu,
            )
        )
        if label == "reverse-tail":
            flops, nbytes = gqa_decode_paged_roofline(
                batch=batch, heads=heads, heads_kv=heads_kv, seqlen_kv=lengths_cpu[0],
                dim=128, page_size=page_size, dtype="float16",
            )
            perf = _perf_case(label, shape, "float16", tested, baseline, flops, nbytes)
    cases.append(
        _rejection(
            OPS[2], lambda: build_gqa_decode_paged_kernel(1, 4, 2, 512, 128, 128, torch.bfloat16)
        )
    )
    return cases, [perf]


def _prefill_one(q_lens, old_lens, heads, heads_kv, page_size, tables, label):
    batch, dim = len(q_lens), 128
    max_pages = len(tables[0])
    physical = (max(max(row) for row in tables) + 1) * page_size
    total_q = sum(q_lens)
    torch.manual_seed(414 + physical)
    q_cpu = torch.randn(total_q, heads, dim, dtype=torch.float16)
    k_new_cpu = torch.randn(total_q, heads_kv, dim, dtype=torch.float16)
    v_new_cpu = torch.randn_like(k_new_cpu)
    old_k = [torch.randn(length, heads_kv, dim, dtype=torch.float16) for length in old_lens]
    old_v = [torch.randn_like(item) for item in old_k]
    k_pages_cpu = torch.zeros(physical, heads_kv, dim, dtype=torch.float16)
    v_pages_cpu = torch.zeros_like(k_pages_cpu)
    for bid in range(batch):
        k_pages_cpu += _pack_pages(old_k[bid], tables[bid], page_size, physical)
        v_pages_cpu += _pack_pages(old_v[bid], tables[bid], page_size, physical)
    cu_cpu = torch.tensor([0] + list(torch.tensor(q_lens).cumsum(0).tolist()), dtype=torch.int32)
    cache_cpu = torch.tensor(old_lens, dtype=torch.int32)
    table_cpu = torch.tensor(tables, dtype=torch.int32)
    q, k_new, v_new, k_pages, v_pages, cu, cache, table = [
        item.npu() for item in (q_cpu, k_new_cpu, v_new_cpu, k_pages_cpu, v_pages_cpu, cu_cpu, cache_cpu, table_cpu)
    ]
    kernel = build_gqa_prefill_paged_kernel(
        batch, heads, heads_kv, total_q, max(q_lens), physical, max_pages,
        page_size, dim, True, torch.float16,
    )
    def tested():
        return kernel(q, k_new, v_new, k_pages, v_pages, cu, cache, table)

    old_k_npu = [item.npu() for item in old_k]
    old_v_npu = [item.npu() for item in old_v]

    def baseline():
        return _prefill_reference(
            q, k_new, v_new, old_k_npu, old_v_npu, q_lens, old_lens
        )
    actual, reference = tested(), baseline()
    shape = {
        "q": list(q.shape), "k_new": list(k_new.shape), "v_new": list(v_new.shape),
        "k_pages": list(k_pages.shape), "v_pages": list(v_pages.shape), "out": list(q.shape),
    }
    record = _record(
        label, actual, reference, shape, "float16", page_size=page_size,
        cache_seqlens=old_lens, block_table=tables, q_lens=q_lens,
    )
    start = 0
    append_error = 0.0
    for bid, q_len in enumerate(q_lens):
        for offset in range(q_len):
            logical = old_lens[bid] + offset
            physical_pos = tables[bid][logical // page_size] * page_size + logical % page_size
            append_error = max(
                append_error,
                float((k_pages[physical_pos].float() - k_new[start + offset].float()).abs().max().cpu()),
                float((v_pages[physical_pos].float() - v_new[start + offset].float()).abs().max().cpu()),
            )
        start += q_len
    record["append_max_abs_err"] = append_error
    if append_error != 0.0:
        raise AssertionError(f"{label}: cache append mismatch {append_error}")
    return record, tested, baseline, shape


def _prefill_cases():
    small = _prefill_one(
        [2, 3], [65, 130], 4, 2, 128,
        [[3, 2, 1, 0], [7, 6, 5, 4]], "mixed-reverse-tail",
    )
    long = _prefill_one(
        [1], [32730], 1, 1, 256, [list(reversed(range(128)))], "long-cache-32k",
    )
    cases = [small[0], long[0]]
    cases.append(
        _rejection(
            OPS[3],
            lambda: build_gqa_prefill_paged_kernel(
                1, 4, 2, 2, 2, 512, 4, 128, 128, True, torch.bfloat16
            ),
        )
    )
    flops, nbytes = gqa_prefill_paged_with_kv_cache_fwd_roofline(
        total_q=5, batch=2, heads=4, heads_kv=2, dim=128,
        max_pages_per_req=4, page_size=128, max_seqlen_q=3,
        q_lens=[2, 3], cache_lens=[65, 130], is_causal=True, dtype="float16",
    )
    perf = _perf_case("mixed-reverse-tail", small[3], "float16", small[1], small[2], flops, nbytes)
    return cases, [perf]


def _result(op_name, correctness_cases, perf_cases):
    correctness_passed = all(case["passed"] for case in correctness_cases)
    mins, perf_ok, status, reason = summary(perf_cases, correctness_passed, common)
    return {
        "schema_version": common.SCHEMA_VERSION,
        "op": op_name,
        "target": "ascend",
        "status": status,
        "device": {
            "soc": "ascend910b1",
            "visible_devices": common.visible_devices(),
            "npu_smi": common.npu_smi_info(),
        },
        "correctness": {
            "passed": correctness_passed,
            "cmd": "cd /home/dyq/workspace && export ASCEND_RT_VISIBLE_DEVICES=3 && python TileOPs/tests/npu/t114_decode_variants.py --out tileops-ascend-harness/coverage --evidence docs/reports/R114-data/coverage",
            "per_case": correctness_cases,
        },
        "perf": perf_fields(perf_cases, mins),
        "baseline": {
            "tier": "vendor",
            "library": "Torch-NPU eager attention reference",
            "repo_commit": _command_output(["git", "-C", str(ROOT / "TileOPs"), "rev-parse", "HEAD"]),
            "source_path": str(Path(torch_npu.__file__).resolve()),
            "build_cmd": "installed Torch-NPU eager baseline",
            "semantic_match": "exact",
            "fusion_match": "paged references gather separately; non-paged references exact",
            "compile_error": None,
        },
        "verdict": {"perf_ok": perf_ok, "threshold": common.THRESHOLD, "reason": reason},
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "harness_commit": common.git_revision(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES") != "3":
        raise RuntimeError("T114 must run with ASCEND_RT_VISIBLE_DEVICES=3")
    builders = (_gqa_cases, _mla_cases, _paged_cases, _prefill_cases)
    args.out.mkdir(parents=True, exist_ok=True)
    args.evidence.mkdir(parents=True, exist_ok=True)
    for op_name, build in zip(OPS, builders, strict=True):
        correctness_cases, perf_cases = build()
        payload = _result(op_name, correctness_cases, perf_cases)
        common.validate_result(payload)
        encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        (args.out / f"{op_name}.json").write_text(encoded)
        (args.evidence / f"{op_name}.json").write_text(encoded)
        print(f"WROTE {args.out / f'{op_name}.json'}", flush=True)


if __name__ == "__main__":
    main()
