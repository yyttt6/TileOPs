"""Minimal T087 B-half sentinel probe; family registrations remain fail-closed."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch

from tileops.kernels import elementwise_mixed as mixed

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "docs/reports/R087-data"
ARTIFACTS = DATA / "artifacts"


def _artifact(op: str, kernel) -> dict:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS / f"{op}-4097-current.cu"
    source = kernel.get_kernel_source()
    path.write_text(source, encoding="utf-8")
    return {"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def probe_bitwise(kind: str) -> dict:
    n = 4097
    a = torch.arange(n, device="npu", dtype=torch.int32)
    b = (a * 3 + 5).to(torch.int32)
    ref_fn = {"and": torch.bitwise_and, "or": torch.bitwise_or, "xor": torch.bitwise_xor}[kind]
    ref = ref_fn(a.cpu(), b.cpu())
    kernel = mixed._compile_bitwise(n, n, n, (n,), (1,), (1,), "int32", kind)
    out = kernel(a, b)
    torch.npu.synchronize()
    out_cpu = out.cpu()
    changed_from_zero = bool((out_cpu != 0).any())
    equal = bool(torch.equal(out_cpu, ref))
    explicit_error = None
    sentinel = torch.full((n,), -777777, device="npu", dtype=torch.int32)
    try:
        kernel(a, b, sentinel)
        torch.npu.synchronize()
        explicit_changed = bool((sentinel.cpu() != -777777).any())
    except Exception as exc:  # output contract evidence
        explicit_changed = None
        explicit_error = f"{type(exc).__name__}: {exc}"
    return {"op": f"Bitwise{kind.title()}FwdOp", "dtype": "int32", "shape": [n],
            "changed_from_zero": changed_from_zero, "equal": equal,
            "tail": int(out_cpu[-1]), "explicit_sentinel_changed": explicit_changed,
            "explicit_error": explicit_error, "artifact": _artifact(f"bitwise-{kind}", kernel)}


def probe_prelu() -> dict:
    shape = (1, 4, 1025)
    n = 4100
    x = torch.linspace(-1, 1, n, device="npu", dtype=torch.float16).reshape(shape)
    w = torch.tensor([0.1, 0.2, 0.3, 0.4], device="npu", dtype=torch.float16)
    ref = torch.nn.functional.prelu(x.float().cpu(), w.float().cpu()).to(torch.float16)
    kernel = mixed._compile_prelu(n, 4, shape, "float16", "float16", "float16", 4, 1025)
    out = kernel(x.reshape(-1), w)
    torch.npu.synchronize()
    out_cpu = out.cpu().reshape(shape)
    delta = (out_cpu.float() - ref.float()).abs()
    sentinel = torch.full((n,), -777.0, device="npu", dtype=torch.float16)
    explicit_error = None
    try:
        kernel(x.reshape(-1), w, sentinel)
        torch.npu.synchronize()
        explicit_changed = bool((sentinel.cpu() != -777.0).any())
    except Exception as exc:
        explicit_changed = None
        explicit_error = f"{type(exc).__name__}: {exc}"
    return {"op": "PreluFwdOp", "dtype": "float16", "shape": list(shape),
            "changed_from_zero": bool((out_cpu != 0).any()), "equal": bool(torch.allclose(out_cpu, ref, atol=1e-2, rtol=1e-2)),
            "max_abs_err": float(delta.max()), "tail": float(out_cpu.flatten()[-1]),
            "explicit_sentinel_changed": explicit_changed, "explicit_error": explicit_error,
            "artifact": _artifact("prelu", kernel)}


def main() -> None:
    if not os.environ.get("ASCEND_RT_VISIBLE_DEVICES"):
        raise RuntimeError("ASCEND_RT_VISIBLE_DEVICES must be set")
    rows = [probe_bitwise(kind) for kind in ("and", "or", "xor")]
    rows.append(probe_prelu())
    (DATA / "b_probe.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    for row in rows:
        print(json.dumps(row, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
