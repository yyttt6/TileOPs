"""Fail-closed floor_divide/remainder sign and divisibility diagnostic."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import tilelang
import torch

from tileops.kernels import elementwise_binary_batch as binary

DTYPES = (torch.float16, torch.bfloat16, torch.float32)
OPS = ("floor_divide", "remainder")
N = 4097
SENTINEL = 123.0


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _tolerance(dtype: torch.dtype) -> tuple[float, float]:
    value = 1.6e-2 if dtype == torch.bfloat16 else 1e-3
    return value, value


def _reference(op: str, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    fn = torch.floor_divide if op == "floor_divide" else torch.remainder
    return fn(a, b)


def _controlled_input(
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, list[dict]]:
    records: list[dict] = []
    for sign_a in (1, -1):
        for sign_b in (1, -1):
            quadrant = ("+" if sign_a > 0 else "-") + ("+" if sign_b > 0 else "-")
            for b_abs in (2, 3, 4):
                for quotient_abs in (1, 2, 5):
                    records.append(
                        {
                            "a": float(sign_a * b_abs * quotient_abs),
                            "b": float(sign_b * b_abs),
                            "group": "quadrant",
                            "quadrant": quadrant,
                            "divisibility": "exact",
                        }
                    )
                    records.append(
                        {
                            "a": float(sign_a * (b_abs * quotient_abs + 1)),
                            "b": float(sign_b * b_abs),
                            "group": "quadrant",
                            "quadrant": quadrant,
                            "divisibility": "nondiv",
                        }
                    )
    records.extend(
        [
            {
                "a": 0.0,
                "b": 2.0,
                "group": "boundary",
                "quadrant": "0+",
                "divisibility": "exact",
            },
            {
                "a": 0.0,
                "b": -2.0,
                "group": "boundary",
                "quadrant": "0-",
                "divisibility": "exact",
            },
            {
                "a": -4.783865928649902,
                "b": -1.594622015953064,
                "group": "near_integer",
                "quadrant": "--",
                "divisibility": "nondiv",
            },
        ]
    )
    expanded = [records[index % len(records)] for index in range(N)]
    a = torch.tensor([row["a"] for row in expanded], dtype=dtype)
    b = torch.tensor([row["b"] for row in expanded], dtype=dtype)
    return a, b, expanded


def _random_input(dtype: torch.dtype, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    a = torch.rand(N, dtype=dtype, generator=generator) * 16 - 8
    b = torch.rand(N, dtype=dtype, generator=generator) * 3 + 0.5
    b.reshape(-1)[::2].neg_()
    return a, b


def _mismatch_mask(
    op: str, got: torch.Tensor, ref: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    if op == "floor_divide":
        return got != ref
    atol, rtol = _tolerance(dtype)
    return ~torch.isclose(got.float(), ref.float(), atol=atol, rtol=rtol)


def _details(
    mask: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    got: torch.Tensor,
    ref: torch.Tensor,
    limit: int = 8,
) -> list[dict]:
    result = []
    for index in torch.nonzero(mask).flatten()[:limit].tolist():
        result.append(
            {
                "index": index,
                "a": float(a[index]),
                "b": float(b[index]),
                "expected": float(ref[index]),
                "actual": float(got[index]),
                "abs_err": float((got[index].float() - ref[index].float()).abs()),
            }
        )
    return result


def _category_rows(
    records: list[dict],
    mismatch: torch.Tensor,
    got: torch.Tensor,
    ref: torch.Tensor,
) -> list[dict]:
    indices: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        if record["group"] == "quadrant":
            indices[(record["quadrant"], record["divisibility"])].append(index)
    rows = []
    delta = (got.float() - ref.float()).abs()
    for quadrant in ("++", "+-", "-+", "--"):
        for divisibility in ("exact", "nondiv"):
            selected = torch.tensor(indices[(quadrant, divisibility)], dtype=torch.long)
            rows.append(
                {
                    "quadrant": quadrant,
                    "divisibility": divisibility,
                    "count": int(selected.numel()),
                    "mismatch_count": int(mismatch[selected].sum()),
                    "max_abs_err": float(delta[selected].max()),
                }
            )
    return rows


def _special_rows(
    records: list[dict], got: torch.Tensor, ref: torch.Tensor
) -> list[dict]:
    rows = []
    seen: set[tuple[str, float, float]] = set()
    for index, record in enumerate(records):
        if record["group"] == "quadrant":
            continue
        key = (record["group"], record["a"], record["b"])
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "group": record["group"],
                "a": float(record["a"]),
                "b": float(record["b"]),
                "expected": float(ref[index]),
                "actual": float(got[index]),
                "expected_signbit": bool(torch.signbit(ref[index])),
                "actual_signbit": bool(torch.signbit(got[index])),
            }
        )
    return rows


def _random_rows(
    a: torch.Tensor,
    b: torch.Tensor,
    mismatch: torch.Tensor,
    got: torch.Tensor,
    ref: torch.Tensor,
) -> list[dict]:
    rows = []
    delta = (got.float() - ref.float()).abs()
    remainder = torch.remainder(a.float(), b.float())
    for sign_a, sign_b, quadrant in (
        (1, 1, "++"),
        (1, -1, "+-"),
        (-1, 1, "-+"),
        (-1, -1, "--"),
    ):
        quadrant_mask = (a > 0 if sign_a > 0 else a < 0) & (
            b > 0 if sign_b > 0 else b < 0
        )
        for name, divisible_mask in (
            ("exact", remainder == 0),
            ("nondiv", remainder != 0),
        ):
            selected = quadrant_mask & divisible_mask
            rows.append(
                {
                    "quadrant": quadrant,
                    "divisibility": name,
                    "count": int(selected.sum()),
                    "mismatch_count": int((mismatch & selected).sum()),
                    "max_abs_err": float(delta[selected].max())
                    if bool(selected.any())
                    else 0.0,
                }
            )
    return rows


def _run_case(op: str, dtype: torch.dtype, trial: int) -> dict:
    binary._compile_batch.cache_clear()
    tilelang.cache.clear_cache()

    a_cpu, b_cpu, records = _controlled_input(dtype)
    kernel = binary.build_batch_binary(
        (N,),
        (N,),
        dtype,
        op_kind=op,
        op_name=f"T053 {op}",
        diagnostic_sentinel=True,
    )
    a = a_cpu.npu()
    b = b_cpu.npu()
    output_first = kernel(a, b)
    torch.npu.synchronize()
    got_first = output_first.cpu()
    output_second = kernel(a, b)
    torch.npu.synchronize()
    got_second = output_second.cpu()
    ref = _reference(op, a_cpu, b_cpu)
    mismatch = _mismatch_mask(op, got_second, ref, dtype)

    random_a_cpu, random_b_cpu = _random_input(dtype, 5300 + trial)
    random_a = random_a_cpu.npu()
    random_b = random_b_cpu.npu()
    random_output = kernel(random_a, random_b)
    torch.npu.synchronize()
    random_got = random_output.cpu()
    random_ref = _reference(op, random_a_cpu, random_b_cpu)
    random_mismatch = _mismatch_mask(op, random_got, random_ref, dtype)

    write_protocol = {
        "returned_distinct_from_inputs": bool(
            output_first.data_ptr() not in {a.data_ptr(), b.data_ptr()}
            and output_second.data_ptr() not in {a.data_ptr(), b.data_ptr()}
        ),
        "two_outputs_distinct": output_first.data_ptr() != output_second.data_ptr(),
        "sentinel_remaining_first": int((got_first == SENTINEL).sum()),
        "sentinel_remaining_second": int((got_second == SENTINEL).sum()),
        "double_run_equal": torch.equal(got_first, got_second),
        "finite_first": bool(torch.isfinite(got_first).all()),
        "finite_second": bool(torch.isfinite(got_second).all()),
    }
    write_protocol["passed"] = bool(
        write_protocol["returned_distinct_from_inputs"]
        and write_protocol["two_outputs_distinct"]
        and write_protocol["sentinel_remaining_first"] == 0
        and write_protocol["sentinel_remaining_second"] == 0
        and write_protocol["double_run_equal"]
        and write_protocol["finite_first"]
        and write_protocol["finite_second"]
    )
    return {
        "op": op,
        "dtype": _dtype_name(dtype),
        "trial": trial,
        "cache_protocol": {"python_lru_cleared": True, "tilelang_disk_cleared": True},
        "write_protocol": write_protocol,
        "controlled": {
            "mismatch_count": int(mismatch.sum()),
            "max_abs_err": float((got_second.float() - ref.float()).abs().max()),
            "categories": _category_rows(records, mismatch, got_second, ref),
            "special_cases": _special_rows(records, got_second, ref),
            "counterexamples": _details(mismatch, a_cpu, b_cpu, got_second, ref),
        },
        "random": {
            "mismatch_count": int(random_mismatch.sum()),
            "max_abs_err": float((random_got.float() - random_ref.float()).abs().max()),
            "categories": _random_rows(
                random_a_cpu, random_b_cpu, random_mismatch, random_got, random_ref
            ),
            "counterexamples": _details(
                random_mismatch, random_a_cpu, random_b_cpu, random_got, random_ref
            ),
        },
    }


def _semantic_examples() -> list[dict]:
    pairs = (
        (6.0, 2.0),
        (7.0, 2.0),
        (6.0, -2.0),
        (7.0, -2.0),
        (-6.0, 2.0),
        (-7.0, 2.0),
        (-6.0, -2.0),
        (-7.0, -2.0),
        (0.0, 2.0),
        (0.0, -2.0),
    )
    rows = []
    for a_value, b_value in pairs:
        a = torch.tensor(a_value)
        b = torch.tensor(b_value)
        rows.append(
            {
                "a": a_value,
                "b": b_value,
                "floor_divide": float(torch.floor_divide(a, b)),
                "remainder": float(torch.remainder(a, b)),
                "identity_holds": bool(
                    a == torch.floor_divide(a, b) * b + torch.remainder(a, b)
                ),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expect-correct", action="store_true")
    parser.add_argument("--op", choices=OPS)
    parser.add_argument(
        "--dtype", choices=tuple(_dtype_name(dtype) for dtype in DTYPES)
    )
    parser.add_argument("--trials", type=int, default=2)
    args = parser.parse_args()
    print(
        json.dumps({"semantic_examples": _semantic_examples()}, sort_keys=True),
        flush=True,
    )
    results = []
    selected_ops = (args.op,) if args.op else OPS
    selected_dtypes = tuple(
        dtype
        for dtype in DTYPES
        if args.dtype is None or _dtype_name(dtype) == args.dtype
    )
    for op in selected_ops:
        for dtype in selected_dtypes:
            for trial in range(1, args.trials + 1):
                result = _run_case(op, dtype, trial)
                results.append(result)
                print(json.dumps(result, sort_keys=True), flush=True)
    summary = {
        "cases": len(results),
        "write_protocol_all_passed": all(
            row["write_protocol"]["passed"] for row in results
        ),
        "controlled_numeric_all_passed": all(
            row["controlled"]["mismatch_count"] == 0 for row in results
        ),
        "random_numeric_all_passed": all(
            row["random"]["mismatch_count"] == 0 for row in results
        ),
    }
    print(json.dumps({"SUMMARY": summary}, sort_keys=True), flush=True)
    if not summary["write_protocol_all_passed"]:
        return 3
    if args.expect_correct and not (
        summary["controlled_numeric_all_passed"]
        and summary["random_numeric_all_passed"]
    ):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
