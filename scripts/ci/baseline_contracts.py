"""Explicit v0.1.0 metadata aliases; never match changed tensor or scalar contracts.

These mappings preserve arithmetic: GEMM changes only its public name; grouped
GEMM renames the output and transpose flags; FP8 token/group quantization renames
its tensors (the public wrapper already reorders the kernel's scale/q pair);
GELU merges two public operators while retaining the exact/tanh distinction.
Measurement IDs, latency, protocol and original evidence remain unchanged.
"""

import json
import math


def normalize(value):
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    if isinstance(value, str) and value in ("inf", "-inf"):
        return "Infinity" if value == "inf" else "-Infinity"
    if isinstance(value, dict):
        return {k: normalize(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [normalize(v) for v in value]
    return value


def comparison_identity(op, case, params):
    params = dict(params)

    def rename(old, new):
        if old in params:
            if new in params:
                raise ValueError(f"Ambiguous baseline parameter aliases: {old}, {new}")
            params[new] = params.pop(old)

    if op == "ReluFwdOp":
        op = "ReLUFwdOp"
    elif op in ("GemmFwdOp", "BmmFwdOp"):
        op = {"GemmFwdOp": "GEMMFwdOp", "BmmFwdOp": "BMMFwdOp"}[op]
    elif op == "GemmINT8W8A8FwdOp":
        op = "GEMMINT8W8A8FwdOp"
        rename("scale_a", "a_scale")
        rename("scale_b", "b_scale")
    elif op == "GroupedGemmFwdOp":
        op = "GroupedGEMMFwdOp"
        for old, new in [("output", "d"), ("transpose_a", "trans_a"), ("transpose_b", "trans_b")]:
            rename(old, new)
    elif op == "GemmW4A8FwdOp":
        op = "GEMMW4A8FwdOp"
        rename("output", "d")
    elif op == "GemmW4A16FwdOp":
        op = "GEMMW4A16FwdOp"
        rename("output", "d")
    elif op == "GemmFP8FwdOp":
        op = "GEMMFP8FwdOp"
        rename("scale_a", "a_scale")
        rename("scale_b", "b_scale")
    elif op == "FP8QuantFwdOp":
        op = "FP8QuantPerTokenGroupFwdOp"
        for old, new in [("input_tensor", "x"), ("output_tensor", "q"), ("scale_tensor", "scale")]:
            rename(old, new)
    elif op in ("GeluAndMulFwdOp", "GeluTanhAndMulFwdOp"):
        mode = "none" if op == "GeluAndMulFwdOp" else "tanh"
        if "approximate" in params and params["approximate"] != mode:
            raise ValueError("Legacy GELU name contradicts its approximate parameter")
        rename("x", "input")
        params["approximate"] = mode
        op = "GELUAndMulFwdOp"
        if mode == "tanh":
            case = {
                "llama-8b-ffn-prefill-float16": "llama-8b-prefill-tanh-float16",
                "llama-8b-ffn-prefill-bfloat16": "llama-8b-prefill-tanh-bfloat16",
                "llama-8b-ffn-decode-bfloat16": "llama-8b-decode-tanh-bfloat16",
            }.get(case, case)
    if op == "AllFwdOp":
        # v0.1.0 only renamed torch.all's input; retain every scalar/tensor field.
        rename("x", "input")
    if op == "MoEGroupedGemmFP8FwdOp":
        op = "MoEGroupedGEMMFP8FwdOp"
    if op == "SSDChunkStateFwdOp":
        rename("Bmat", "B")
    if op in {"CumsumFwdOp", "CumprodFwdOp"}:
        # The old metadata used x/y and omitted the public dtype=None default.
        # Preserve axis, shapes and dtypes; infer None only for same-dtype float
        # outputs, never for a widening scan whose dtype must be explicit.
        rename("x", "input")
        rename("y", "output")
        source, output = params.get("input"), params.get("output")
        if (
            source is not None
            and output is not None
            and len(source) == len(output) == 2
            and source[1] == output[1]
            and source[1] in {"float16", "bfloat16", "float32"}
        ):
            params.setdefault("dtype", None)
    if op == "MergeAttentionStatesFwdOp":
        # The old public interface used natural-log LSE; absence never means log2.
        params.setdefault("lse_base", "e")
    return op, case, json.dumps(normalize(params), sort_keys=True)
