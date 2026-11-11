"""Validated, content-addressed reference outputs; optional acceleration only.

This module never changes numerical checks. Entries are published only after the
caller validates the complete case, including negative controls and probes.
"""

import fcntl
import hashlib
import importlib.metadata
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

PROTOCOL = "golden-output-v1"
POLICY = Path(__file__).with_name("golden_cache_policy.json")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tensor_tree(value, tensor):
    import torch

    if isinstance(value, torch.Tensor):
        return tensor(value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, tuple):
        return tuple(tensor_tree(x, tensor) for x in value)
    if isinstance(value, list):
        return [tensor_tree(x, tensor) for x in value]
    if isinstance(value, dict) and all(isinstance(k, str) for k in value):
        return {k: tensor_tree(v, tensor) for k, v in value.items()}
    raise TypeError(f"Unsupported cache value: {type(value)}")


def input_digest(args):
    import torch

    h = hashlib.sha256()

    def walk(v):
        if isinstance(v, torch.Tensor):
            if v.layout != torch.strided:
                raise TypeError("Only dense tensors are cacheable")
            h.update(
                canonical(
                    [
                        "tensor",
                        list(v.shape),
                        str(v.dtype),
                        str(v.device),
                        list(v.stride()),
                        v.requires_grad,
                    ]
                )
            )
            raw = v.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
            h.update(memoryview(raw))
        elif isinstance(v, (tuple, list)):
            h.update(canonical([type(v).__name__, len(v)]))
            for x in v:
                walk(x)
        elif isinstance(v, dict) and all(isinstance(k, str) for k in v):
            h.update(b"dict")
            for k in sorted(v):
                walk(k)
                walk(v[k])
        elif v is None or isinstance(v, (str, bool, int, float)):
            h.update(canonical([type(v).__name__, v]))
        else:
            raise TypeError(f"Unsupported input: {type(v)}")

    walk(args)
    return h.hexdigest()


def rng_digest():
    import torch

    h = hashlib.sha256(bytes(torch.get_rng_state().tolist()))
    if hasattr(torch, "npu") and torch.npu.is_initialized():
        h.update(bytes(torch.npu.get_rng_state().cpu().tolist()))
    return h.hexdigest()


def contract_identity(source):
    """Conservative invalidation includes all possible in-tree reference helpers.

    Source commit IDs are not keys: identical trees can share. Kernel edits also
    invalidate this first version; narrowing dependencies requires a separate audit.
    """
    import torch

    source = Path(source).resolve()
    h = hashlib.sha256()
    for folder in ("workloads", "benchmarks", "src/tileops"):
        for p in sorted((source / folder).rglob("*")):
            if p.suffix in {".py", ".yaml", ".json"} and p.is_file():
                h.update(str(p.relative_to(source)).encode())
                h.update(bytes.fromhex(digest_file(p)))
    lock = Path(
        os.environ.get(
            "ASCEND950_GOLDEN_TOOLCHAIN_LOCK",
            "/opt/tileops-ci/server2-20261030/config/toolchain-v1.lock.json",
        )
    )
    if not lock.is_file():
        raise ValueError("Missing verified toolchain lock")
    runtime = {}
    for name in ("torch", "torch-npu", "tilelang", "triton", "fla-core"):
        try:
            runtime[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            runtime[name] = None
    overlay = os.environ.get("ASCEND950_REFERENCE_ROOT")
    if overlay:
        for p in sorted(Path(overlay).rglob("*")):
            if p.is_file() and p.suffix in {".py", ".so"}:
                h.update(str(p.relative_to(overlay)).encode())
                h.update(bytes.fromhex(digest_file(p)))
    device = (
        torch.npu.get_device_name(0)
        if hasattr(torch, "npu") and torch.npu.is_initialized()
        else "cpu"
    )
    return dict(
        protocol=PROTOCOL,
        source=h.hexdigest(),
        toolchain=digest_file(lock),
        runtime=runtime,
        device=device,
        implementation=digest_file(__file__),
    )


class GoldenCache:
    def __init__(self, root, identity, *, max_bytes=128 << 30, min_free_bytes=200 << 30):
        self.root = Path(root)
        self.identity = identity
        self.max_bytes = max_bytes
        self.min_free_bytes = min_free_bytes
        self.pending = {}
        self.events = []

    def note(self, **event):
        self.events.append(event)

    def _load(self, key):
        import torch

        path = self.root / key
        meta = json.loads((path / "metadata.json").read_text())
        if meta["key"] != key or meta["identity"] != self.identity or meta["protocol"] != PROTOCOL:
            raise ValueError("Cache identity mismatch")
        if digest_file(path / "outputs.pt") != meta["sha256"]:
            raise ValueError("Cache checksum mismatch")
        saved = torch.load(path / "outputs.pt", map_location="cpu", weights_only=True)
        devices = iter(meta["devices"])
        result = tensor_tree(saved, lambda t: t.to(next(devices)))
        if next(devices, None) is not None:
            raise ValueError("Cache tensor structure mismatch")
        return result

    def wrap(self, reference):
        def run(*args):
            start = time.monotonic()
            try:
                before = input_digest(args)
                rng = rng_digest()
                key = hashlib.sha256(canonical([self.identity, before])).hexdigest()
            except (TypeError, ValueError, RuntimeError) as exc:
                self.note(status="bypass", reason=str(exc))
                return reference(*args)
            try:
                result = self._load(key)
            except Exception as exc:  # Corrupt/unsupported cache data is always a miss.
                self.note(
                    status="miss",
                    key=key,
                    reason=type(exc).__name__,
                    lookup_seconds=time.monotonic() - start,
                )
            else:
                self.note(status="hit", key=key, lookup_seconds=time.monotonic() - start)
                return result
            # Reference exceptions propagate unchanged. Never cache their results.
            result = reference(*args)
            try:
                if before != input_digest(args) or rng != rng_digest():
                    self.note(
                        status="not_stored", reason="reference changed inputs or RNG", key=key
                    )
                    return result
                devices = []

                def capture(t):
                    import torch

                    if (t.is_floating_point() or t.is_complex()) and not torch.isfinite(
                        t
                    ).all().item():
                        raise ValueError("Nonfinite golden is not persisted")
                    devices.append(str(t.device))
                    return t.detach().cpu().contiguous().clone()

                saved = tensor_tree(result, capture)
                if not devices:
                    raise ValueError("No tensor output")
                self.pending[key] = (saved, devices)
            except (TypeError, ValueError, RuntimeError) as exc:
                self.note(status="not_stored", key=key, reason=str(exc))
            return result

        return run

    def commit(self):
        """Call only after full verification and custom probes succeed."""
        import torch

        for key, (saved, devices) in self.pending.items():
            temp = None
            try:
                self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
                with (self.root / ".lock").open("a") as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    target = self.root / key
                    if target.exists():
                        # Existing corrupt entries are not silently overwritten.
                        continue
                    sizes = []
                    tensor_tree(
                        saved, lambda t, sizes=sizes: sizes.append(t.numel() * t.element_size())
                    )
                    estimate = sum(sizes) + (1 << 20)
                    used = sum(p.stat().st_size for p in self.root.glob("*/outputs.pt"))
                    if (
                        used + estimate > self.max_bytes
                        or shutil.disk_usage(self.root).free - estimate < self.min_free_bytes
                    ):
                        self.note(status="not_stored", key=key, reason="capacity limit")
                        continue
                    temp = Path(tempfile.mkdtemp(prefix=".pending-", dir=self.root))
                    with (temp / "outputs.pt").open("wb") as f:
                        torch.save(saved, f)
                        f.flush()
                        os.fsync(f.fileno())
                    meta = dict(
                        protocol=PROTOCOL,
                        key=key,
                        identity=self.identity,
                        devices=devices,
                        sha256=digest_file(temp / "outputs.pt"),
                    )
                    with (temp / "metadata.json").open("wb") as f:
                        f.write(canonical(meta))
                        f.flush()
                        os.fsync(f.fileno())
                    temp.rename(target)
                    temp = None
                    self.note(status="stored", key=key)
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                self.note(status="not_stored", key=key, reason=str(exc))
            finally:
                if temp is not None:
                    shutil.rmtree(temp, ignore_errors=True)
        self.pending.clear()

    def discard(self):
        self.pending.clear()


def for_case(source, op, case_id, identity):
    root = os.environ.get("ASCEND950_GOLDEN_CACHE_DIR")
    policy = json.loads(POLICY.read_text())
    if not root or policy["operators"].get(op) not in {"high", "medium"}:
        return None
    return GoldenCache(
        root,
        dict(identity, op=op, case=case_id),
        max_bytes=int(os.environ.get("ASCEND950_GOLDEN_CACHE_MAX_BYTES", 128 << 30)),
        min_free_bytes=int(os.environ.get("ASCEND950_GOLDEN_CACHE_MIN_FREE_BYTES", 200 << 30)),
    )
