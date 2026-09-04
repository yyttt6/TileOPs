"""T108 correctness evidence for paged MHA decode."""

import importlib.util
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
import torch_npu

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "TileOPs"))
sys.path.insert(0, str(ROOT / "tileops-ascend-harness"))
import harnesslib as harness_common  # noqa: E402

from tileops.perf.formulas import mha_decode_roofline  # noqa: E402

KERNEL_PATH = ROOT / "TileOPs/src/tileops/kernels/attention_decode.py"
SPEC = importlib.util.spec_from_file_location("t108_attention_decode", KERNEL_PATH)
assert SPEC is not None and SPEC.loader is not None
KERNEL_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(KERNEL_MODULE)

ATOL = 5e-3
RTOL = 1e-5


def _paged_reference(q, k, v, lengths, block_table, page_size):
    outputs = []
    for bid, length in enumerate(lengths.tolist()):
        logical_k = torch.cat(
            [
                k[page * page_size : (page + 1) * page_size]
                for page in block_table[bid].tolist()
            ],
            dim=0,
        )[:length]
        logical_v = torch.cat(
            [
                v[page * page_size : (page + 1) * page_size]
                for page in block_table[bid].tolist()
            ],
            dim=0,
        )[:length]
        scores = torch.einsum("hd,khd->hk", q[bid, 0].float(), logical_k.float())
        probabilities = torch.softmax(scores / q.shape[-1] ** 0.5, dim=-1)
        outputs.append(torch.einsum("hk,khd->hd", probabilities, logical_v.float()))
    return torch.stack(outputs).unsqueeze(1)


def _run_case(label, batch, heads, physical_tokens, page_size, lengths, tables):
    torch.manual_seed(108 + len(label))
    dtype = torch.float16
    dim = 128
    q_cpu = torch.randn((batch, 1, heads, dim), dtype=dtype)
    k_cpu = torch.randn((physical_tokens, heads, dim), dtype=dtype)
    v_cpu = torch.randn((physical_tokens, heads, dim), dtype=dtype)
    lengths_cpu = torch.tensor(lengths, dtype=torch.int32)
    table_cpu = torch.tensor(tables, dtype=torch.int32)
    kernel = KERNEL_MODULE.build_mha_decode_paged_kernel(
        batch, heads, 1, physical_tokens, dim, page_size, False, dtype
    )
    actual = kernel(
        q_cpu.npu(),
        k_cpu.npu(),
        v_cpu.npu(),
        lengths_cpu.npu(),
        table_cpu.npu(),
    )
    torch.npu.synchronize()
    actual_cpu = actual.detach().cpu().float()
    expected = _paged_reference(q_cpu, k_cpu, v_cpu, lengths_cpu, table_cpu, page_size)
    delta = (actual_cpu - expected).abs()
    max_abs = float(delta.max())
    max_rel = float((delta / expected.abs().clamp_min(1e-12)).max())
    finite = bool(torch.isfinite(actual_cpu).all())
    passed = finite and bool(torch.allclose(actual_cpu, expected, atol=ATOL, rtol=RTOL))
    result = {
        "shape": {
            "q": list(q_cpu.shape),
            "k": list(k_cpu.shape),
            "v": list(v_cpu.shape),
            "out": list(actual_cpu.shape),
        },
        "dtype": "float16",
        "label": label,
        "page_size": page_size,
        "cache_seqlens": lengths,
        "block_table": tables,
        "max_abs_err": max_abs,
        "max_rel_err": max_rel,
        "tolerance": {"atol": ATOL, "rtol": RTOL},
        "passed": passed,
        "finite": finite,
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    if not passed:
        raise AssertionError(f"{label}: max_abs_err={max_abs}")
    return result


def _command_output(command):
    return subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    ).stdout.strip()


def _perf_case(label, heads, physical_tokens, page_size, length, table):
    torch.manual_seed(208 + len(label))
    dtype = torch.float16
    dim = 128
    q = torch.randn((1, 1, heads, dim), dtype=dtype).npu()
    k = torch.randn((physical_tokens, heads, dim), dtype=dtype).npu()
    v = torch.randn((physical_tokens, heads, dim), dtype=dtype).npu()
    lengths = torch.tensor([length], dtype=torch.int32).npu()
    block_table = torch.tensor([table], dtype=torch.int32).npu()
    page_offsets = torch.arange(page_size, dtype=torch.int64)
    logical_indices = (
        torch.tensor(table, dtype=torch.int64)[:, None] * page_size + page_offsets
    ).reshape(-1)[:length]
    logical_indices = logical_indices.npu()
    kernel = KERNEL_MODULE.build_mha_decode_paged_kernel(
        1, heads, 1, physical_tokens, dim, page_size, False, dtype
    )

    def tested():
        return kernel(q, k, v, lengths, block_table)

    def baseline():
        logical_k = k.index_select(0, logical_indices).unsqueeze(0)
        logical_v = v.index_select(0, logical_indices).unsqueeze(0)
        return F.scaled_dot_product_attention(
            q.transpose(1, 2),
            logical_k.permute(0, 2, 1, 3),
            logical_v.permute(0, 2, 1, 3),
            scale=dim**-0.5,
        ).transpose(1, 2)

    tested_samples, baseline_samples = harness_common.paired_samples_us(
        tested, baseline, torch.npu.synchronize
    )
    tested_stats = harness_common.sample_stats(tested_samples)
    baseline_stats = harness_common.sample_stats(baseline_samples)
    roofline_op = SimpleNamespace(
        batch=1,
        seqlen_q=1,
        heads=heads,
        seqlen_kv=length,
        dim=dim,
        dtype=dtype,
    )
    flops, nbytes = mha_decode_roofline(roofline_op)
    profile = harness_common.load_profile(harness_common.PROFILE_PATH)
    sol_hw, bound_hw = harness_common.sol_us(flops, nbytes, "float16", profile, "peak")
    sol_achieved, bound_achieved = harness_common.sol_us(
        flops, nbytes, "float16", profile, "achieved"
    )
    tested_gate = harness_common.gate_case(tested_stats["median"], sol_hw)
    baseline_gate = harness_common.gate_case(baseline_stats["median"], sol_hw)
    overall_gate = (
        "VIOLATION" if "VIOLATION" in {tested_gate, baseline_gate} else "pass"
    )
    return {
        "shape": {
            "q": [1, 1, heads, dim],
            "k": [physical_tokens, heads, dim],
            "v": [physical_tokens, heads, dim],
            "out": [1, 1, heads, dim],
        },
        "dtype": "float16",
        "label": label,
        "page_size": page_size,
        "cache_seqlens": [length],
        "block_table": [table],
        "tileops_us": tested_stats["median"],
        "baseline_us": baseline_stats["median"],
        "ratio": baseline_stats["median"] / tested_stats["median"],
        "flops": flops,
        "bytes": nbytes,
        "roofline_source": "mha_decode_roofline() (table bytes omitted conservatively)",
        "roofline_error": None,
        "sol_us_hw": sol_hw,
        "sol_bound_hw": bound_hw,
        "sol_us_achieved": sol_achieved,
        "sol_bound_achieved": bound_achieved,
        "efficiency_vs_hw_peak": sol_hw / tested_stats["median"],
        "efficiency_vs_vendor_achieved": sol_achieved / tested_stats["median"],
        "baseline_efficiency_vs_hw_peak": sol_hw / baseline_stats["median"],
        "baseline_efficiency_vs_vendor_achieved": sol_achieved
        / baseline_stats["median"],
        "sol_gate": overall_gate,
        "sol_gate_detail": {"tileops": tested_gate, "baseline": baseline_gate},
        "timing_samples_us": {
            "tileops": tested_samples,
            "baseline": baseline_samples,
        },
        "timing_stats_us": {
            "tileops": tested_stats,
            "baseline": baseline_stats,
        },
    }


def main():
    cases = [
        _run_case(
            "nonidentity-tail",
            1,
            4,
            512,
            128,
            [193],
            [[3, 2, 1, 0]],
        ),
        _run_case(
            "dynamic-varlen-tail",
            2,
            3,
            1024,
            256,
            [37, 701],
            [[3, 1, 0, 2], [1, 3, 2, 0]],
        ),
        _run_case(
            "long-cache-32k",
            1,
            1,
            32768,
            256,
            [32731],
            [list(reversed(range(128)))],
        ),
    ]
    perf_cases = [
        _perf_case("nonidentity-tail", 4, 512, 128, 193, [3, 2, 1, 0]),
        _perf_case("long-cache-32k", 1, 32768, 256, 32731, list(reversed(range(128)))),
    ]
    ratio_min = min(case["ratio"] for case in perf_cases)
    sol_passed = all(case["sol_gate"] == "pass" for case in perf_cases)
    visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
    harness = ROOT / "tileops-ascend-harness"
    result = {
        "schema_version": "1.0.0",
        "op": "MultiHeadAttentionDecodePagedWithKVCacheFwdOp",
        "target": "ascend",
        "status": "perf_measured" if sol_passed else "blocked",
        "device": {
            "soc": "ascend910b1",
            "visible_devices": visible,
            "npu_smi": _command_output(["npu-smi", "info"]),
        },
        "correctness": {
            "passed": True,
            "cmd": (
                "cd /home/dyq/workspace && "
                "export ASCEND_RT_VISIBLE_DEVICES=1 && "
                "python TileOPs/tests/npu/t108_paged_decode.py"
            ),
            "per_case": cases,
        },
        "perf": {
            "timing": "paired synchronized host timer",
            "warmup": harness_common.WARMUP,
            "repeats": harness_common.REPEATS,
            "statistic": "median",
            "l2_flushed": False,
            "cases": perf_cases,
            "ratio_min": ratio_min,
        },
        "baseline": {
            "tier": "vendor",
            "library": "Torch-NPU index_select plus scaled_dot_product_attention",
            "repo_commit": _command_output(
                ["git", "-C", str(ROOT / "TileOPs"), "rev-parse", "HEAD"]
            ),
            "source_path": str(Path(torch_npu.__file__).resolve()),
            "build_cmd": "installed Torch-NPU eager baseline",
            "semantic_match": "exact",
            "fusion_match": "mismatch(paged gather and attention are separate baseline ops)",
        },
        "verdict": {
            "perf_ok": False,
            "threshold": 0.7,
            "reason": (
                f"correctness and SOL pass; ratio_min={ratio_min:.6f}"
                if sol_passed
                else "reported latency is below hardware SOL"
            ),
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "harness_commit": _command_output(
            ["git", "-C", str(harness), "rev-parse", "HEAD"]
        ),
    }
    print("CANONICAL_JSON=" + json.dumps(result, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
