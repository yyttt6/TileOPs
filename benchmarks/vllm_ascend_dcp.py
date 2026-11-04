"""Pinned vLLM-Ascend DCP baselines without importing its serving stack.

Pass an external source checkout explicitly or place it on Python's module path.
Its three dependencies must match the pinned file hashes; numerical functions
and their Triton decorators are loaded intact from those files. A compatible
Triton Ascend runtime is required separately. No source is downloaded by this adapter.
"""

import ast
import copy
import functools
import hashlib
import importlib.util
import sys
import types
import warnings
from pathlib import Path
from typing import Any

import torch

from benchmarks.api import Implementation
from benchmarks.baselines import optional_baseline

__all__ = ["dcp_implementations", "vllm_ascend_dcp_provenance"]

_COMMIT = "7d9e52da97606050f61f33275b9d9a0def86881a"
_SOURCES = {
    "vllm_ascend/ops/triton/dcp/dcp_a2a.py": "aaa0736cdf7e1b2e263fb8f6c2021e2de7b3bf67990adca1f640202366222f80",
    "vllm_ascend/ops/triton/dcp/dcp_a2a_batched.py": "3a95b96b47486ada157b0a858903c44cd6be2cadfd7c1ab7a0105469a54884cc",
    "vllm_ascend/ops/triton/triton_utils.py": "cd918c79138a5f93294a7b25a47a8e5fcda2f6291e726bca94c0a76237881225",
}
_PROVENANCE = None


def _omit(reason):
    warnings.warn(f"vllm-ascend-dcp baselines omitted: {reason}", stacklevel=3)


def _source_text(root, relative):
    path = root / relative
    payload = path.read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != _SOURCES[relative]:
        raise ValueError(
            f"vLLM-Ascend source hash mismatch: {path}; expected {_SOURCES[relative]} "
            f"from {_COMMIT}, got {actual}"
        )
    return payload.decode()


def _load_declarations(root, relative, names, bindings):
    source = _source_text(root, relative)
    tree = ast.parse(source)
    selected = []
    found = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            selected.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in names:
            selected.append(node)
            found.add(node.name)
        elif isinstance(node, ast.Assign):
            assigned = {target.id for target in node.targets if isinstance(target, ast.Name)}
            if assigned & names:
                if not assigned <= names:
                    raise ValueError(f"Unexpected shared assignment in {relative}: {assigned}")
                selected.append(node)
                found.update(assigned)
    if found != names:
        raise ValueError(f"Missing pinned declarations in {relative}: {sorted(names - found)}")
    identity = hashlib.sha256((str(root) + relative).encode()).hexdigest()[:16]
    name = f"benchmarks._vllm_ascend_dcp_{identity}"
    module = types.ModuleType(name)
    module.__file__ = str(root / relative)
    module.__dict__.update(bindings)
    sys.modules[name] = module
    try:
        exec(
            compile(ast.Module(body=selected, type_ignores=[]), module.__file__, "exec"),
            module.__dict__,
        )
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


@functools.lru_cache(maxsize=8)
def _device_is_950(index):
    return torch.npu.get_device_name(index).startswith("Ascend950")


def _runtime_is_950():
    # The source package normally gets this identity from its generated build metadata.
    return _device_is_950(torch.npu.current_device())


@functools.lru_cache(maxsize=2)
def _pinned_api(source_root, triton, tl):
    root = Path(source_root)
    utilities_path = "vllm_ascend/ops/triton/triton_utils.py"
    utilities_names = {
        "_NUM_AICORE",
        "_NUM_VECTORCORE",
        "_UB_SIZE_BYTES",
        "DEFAULT_UB_SIZE_BYTES",
        "_MIN_UB_SIZE_BYTES",
        "_UB_SIZE_PROPERTY_KEYS",
        "init_device_properties_triton",
        "get_vectorcore_num",
    }
    utilities = _load_declarations(
        root,
        utilities_path,
        utilities_names,
        {"torch": torch, "triton": triton, "HAS_TRITON": True, "Any": Any},
    )
    batched_path = "vllm_ascend/ops/triton/dcp/dcp_a2a_batched.py"
    batched_names = {"_pack_dcp_output_lse_batched_kernel", "_fused_dcp_lse_combine_batched_kernel"}
    batched = _load_declarations(root, batched_path, batched_names, {"triton": triton, "tl": tl})
    combine_path = "vllm_ascend/ops/triton/dcp/dcp_a2a.py"
    combine_names = {
        "_pack_dcp_output_lse_kernel",
        "_fused_dcp_lse_combine_kernel",
        "_lse_pack_dim",
        "_validate_dcp_inputs",
        "pack_dcp_output_lse",
        "fused_dcp_lse_combine",
    }
    combine = _load_declarations(
        root,
        combine_path,
        combine_names,
        {
            "torch": torch,
            "triton": triton,
            "tl": tl,
            "is_950": _runtime_is_950,
            "init_device_properties_triton": utilities.init_device_properties_triton,
            "get_vectorcore_num": utilities.get_vectorcore_num,
            **{name: getattr(batched, name) for name in batched_names},
        },
    )
    provenance = {
        "repository": "https://github.com/vllm-project/vllm-ascend",
        "commit": _COMMIT,
        "source_root": str(root),
        "source_sha256": dict(_SOURCES),
        "loaded_declarations": {
            utilities_path: sorted(utilities_names),
            batched_path: sorted(batched_names),
            combine_path: sorted(combine_names),
        },
        "loading": "Complete original AST declarations and decorators; numerical bodies are not rewritten.",
        "runtime": {
            "triton_version": getattr(triton, "__version__", None),
            "triton_file": triton.__file__,
        },
        "device_identity_bridge": "is_950 uses the actual torch.npu device name, cached per device index, instead of generated vLLM build metadata.",
        "grid": "Original init_device_properties_triton/get_vectorcore_num; no core-count override.",
        "dispatch": "FP32 receive buffer, scatter_dim=0, RETURN_LSE=True; original generic DCP kernel.",
    }
    return combine.pack_dcp_output_lse, combine.fused_dcp_lse_combine, provenance


def _available_api(source_root=None):
    resolution = "explicit source_root argument"
    if source_root is None:
        package = importlib.util.find_spec("vllm_ascend")
        if package is None or not package.submodule_search_locations:
            _omit(f"pass source_root or expose source commit {_COMMIT} on Python's module path")
            return None
        source_root = Path(next(iter(package.submodule_search_locations))).parent
        resolution = "top-level vllm_ascend package on Python's module path; package not imported"
    root = Path(source_root).expanduser().resolve()
    missing = [relative for relative in _SOURCES if not (root / relative).is_file()]
    if missing:
        _omit(f"missing pinned source files under {root}: {', '.join(missing)}")
        return None
    for relative in _SOURCES:
        _source_text(root, relative)
    triton = optional_baseline("triton", "triton-ascend")
    tl = optional_baseline("triton.language", "triton-ascend")
    backends = optional_baseline("triton.backends", "triton-ascend", "backends")
    if triton is None or tl is None or backends is None or "ascend" not in backends:
        _omit("a compatible Triton runtime with its Ascend backend is unavailable")
        return None
    pack, combine, receipt = _pinned_api(str(root), triton, tl)
    return pack, combine, {**receipt, "source_resolution": resolution}


def vllm_ascend_dcp_provenance():
    """Return the last loaded source/runtime receipt, or None when neither tag was selected."""
    return copy.deepcopy(_PROVENANCE)


def _merge_adapter(pack, combine, *, direct_pack):
    def run(v_a, s_a, v_b, s_b):
        dim = v_a.shape[-1]
        if direct_pack:
            recv = torch.empty((1, *v_b.shape[:2], dim + 1), dtype=torch.float32, device=v_b.device)
            recv[0, ..., :dim].copy_(v_b)
            recv[0, ..., dim].copy_(s_b)
        else:
            recv = pack(v_b.float(), s_b.unsqueeze(-1), 1, 0)
        merged = combine(
            recv, dim, 0, return_lse=True, local_output=v_a, local_lse=s_a.unsqueeze(-1)
        )
        return merged[..., :dim].to(v_a.dtype), merged[..., dim]

    return run


def dcp_implementations(inputs, *, source_root=None):
    """Two full-call DCP comparisons; all casts and packing occur inside each timed call."""
    global _PROVENANCE
    if any(dim == 0 for dim in inputs[0].shape):
        _omit("the original DCP API requires non-empty token, head and feature dimensions")
        return {}
    api = _available_api(source_root)
    if api is None:
        return {}
    pack, combine, receipt = api
    _PROVENANCE = receipt
    implementations = {}
    for tag, direct in (("vllm-ascend-dcp-fused", False), ("vllm-ascend-dcp-direct-pack", True)):
        run = _merge_adapter(pack, combine, direct_pack=direct)
        run.source_provenance = {
            **copy.deepcopy(receipt),
            "adapter": "two torch copy_ calls into FP32 packed storage; original combine"
            if direct
            else "original pack and combine wrappers with timed FP32 input cast",
        }
        implementations[tag] = Implementation(run)
    return implementations
