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

    # Explicit public spelling changes; no case-insensitive/fuzzy matching.
    op = {
        "EluFwdOp": "ELUFwdOp",
        "GeluFwdOp": "GELUFwdOp",
        "LeakyReluFwdOp": "LeakyReLUFwdOp",
        "PreluFwdOp": "PReLUFwdOp",
        "SeluFwdOp": "SELUFwdOp",
        "SiluFwdOp": "SiLUFwdOp",
        "SiluAndMulFwdOp": "SiLUAndMulFwdOp",
        # This legacy name describes 1x128 groups, not FP8's 128x128 blocks.
        "INT8QuantPerBlockFwdOp": "INT8QuantPerTokenGroupFwdOp",
    }.get(op, op)
    input_renames = {
        "AmaxFwdOp",
        "AminFwdOp",
        "AnyFwdOp",
        "ArgmaxFwdOp",
        "ArgminFwdOp",
        "CountNonzeroFwdOp",
        "FusedAddRMSNormFwdOp",
        "FusedAddRMSNormPerTokenQuantFwdOp",
        "GroupNormFwdOp",
        "InstanceNormFwdOp",
        "LayerNormFwdOp",
        "LogSoftmaxFwdOp",
        "LogSumExpFwdOp",
        "MeanFwdOp",
        "ProdFwdOp",
        "StdFwdOp",
        "SumFwdOp",
        "VarFwdOp",
        "VarMeanFwdOp",
        "ELUFwdOp",
        "GELUFwdOp",
        "LeakyReLUFwdOp",
        "PReLUFwdOp",
        "SELUFwdOp",
        "SiLUFwdOp",
        "SiLUAndMulFwdOp",
    }
    if op in input_renames:
        rename("x", "input")
    aliases = {
        "TopKSelectFwdOp": [("index_score", "logits"), ("indexes", "indices"), ("topk", "top_k")],
        "DeltaNetChunkFwdOp": [("S", "h")],
        "GLARecurrentFwdOp": [("gk", "g")],
        "MeanPoolingFwdOp": [("indices", "chunk_indices"), ("offsets", "cu_seqlens")],
        "NSATopKVarlenFwdOp": [
            ("bs", "block_size"),
            ("offsets", "cu_seqlens"),
            ("scale", "sm_scale"),
        ],
        "NSAVarlenFwdOp": [("o_slc", "o"), ("offsets", "cu_seqlens"), ("scale", "sm_scale")],
        "SSDChunkCouplingFwdOp": [("C_mat", "C"), ("B_mat", "B"), ("chunk_len", "chunk_size")],
        "SSDChunkCumsumFwdOp": [("chunk_len", "chunk_size")],
        "SelectiveScanFwdOp": [("B_in", "B"), ("C_in", "C"), ("D_in", "D")],
        "TopKMaskFwdOp": [("k", "top_k")],
        "TopKTopPMaskFwdOp": [("k", "top_k"), ("p", "top_p")],
        "TopPMaskFwdOp": [("p", "top_p")],
    }
    for old, new in aliases.get(op, []):
        rename(old, new)
    if op in {
        "GDNFwdOp",
        "KDAFwdOp",
        "GLAFwdOp",
        "DeltaNetFwdOp",
        "NSATopKVarlenFwdOp",
        "NSAVarlenFwdOp",
    }:
        # Public constructor default. True remains a distinct timing contract.
        params.setdefault("validate_inputs", False)
    # Legacy Engram metadata redundantly recorded dimensions. Remove only when
    # the saved scalars agree exactly with the tensor metadata; never guess.
    dimensions = {}
    if op == "EngramDecodeFwdOp" and "e_t" in params and "h_t" in params:
        dimensions = {
            "batch": params["e_t"][0][0],
            "d_mem": params["e_t"][0][1],
            "d": params["h_t"][0][1],
        }
    if op == "EngramGateConvFwdOp" and "H" in params:
        dimensions = dict(zip(("M", "seq_len", "d"), params["H"][0], strict=True))
    for name, value in dimensions.items():
        if name in params:
            if params[name] != value:
                raise ValueError(f"Contradictory Engram dimension: {name}")
            del params[name]
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
