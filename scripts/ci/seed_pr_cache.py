"""Seed a private PR cache from the matching nightly toolchain, without shared writable files."""

import argparse
import errno
import hashlib
import json
import re
import shutil
import tempfile
from pathlib import Path


def key(identity):
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def copy_entry(entry, target, kind):
    # Snapshot and validate only files described by TileLang's checksum manifest.
    index = "manifest.json" if kind == "kernels" else "metadata.json"
    metadata = (entry / index).read_bytes()
    manifest = json.loads(metadata)
    if kind == "kernels":
        files = manifest["files"]
        if (
            manifest["version"] != 1
            or not {"device_kernel.asc", "executable.so", "params.json"} <= files.keys()
        ):
            raise ValueError("incomplete kernel manifest")
    else:
        if manifest["format"] != "tilelang.ascend-binary-cache.v1":
            raise ValueError("unknown binary format")
        files = {"kernel.aibin": manifest}
    with tempfile.TemporaryDirectory(prefix=".pr-seed-", dir=target.parent) as tmp:
        stage = Path(tmp) / "entry"
        stage.mkdir()
        for name, descriptor in files.items():
            if Path(name).name != name or name in ("", ".", "..", index):
                raise ValueError("unexpected cache file")
            src = entry / name
            if src.is_symlink() or not src.is_file():
                raise ValueError("cache payload is not a regular file")
            dst = stage / name
            shutil.copy2(src, dst)
            if (
                dst.stat().st_size != descriptor["size"]
                or hashlib.sha256(dst.read_bytes()).hexdigest() != descriptor["sha256"]
            ):
                raise ValueError("cache checksum mismatch")
        (stage / index).write_bytes(metadata)
        try:
            stage.rename(target)
        except OSError as exc:
            if exc.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                raise
            return False
    return True


def seed(root, report):
    identity = dict(report)
    private_key = identity.pop("key")
    if private_key != key(identity) or identity["schema"] != "tileops.ascend-cache.v2":
        raise ValueError("unexpected PR cache identity")
    trusted = {k: identity[k] for k in ("lock_sha256", "toolchain")}
    trusted["schema"] = "tileops.ascend-trusted-program-cache.v1"
    origin = root / ("tilelang-v1-" + key(trusted))
    destination = root / ("tilelang-v1-" + private_key)
    stats = dict(copied=0, existing=0, invalid=0, trusted=origin.name, private=destination.name)
    version = identity["toolchain"]["packages"]["tilelang"]
    machine = identity["toolchain"]["machine"]
    for value in (version, machine):
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.+-]*", value):
            raise ValueError("unexpected cache namespace")
    for suffix in (
        Path(version) / ("linux-" + machine) / "kernels",
        Path(version) / "ascend-binaries",
    ):
        source = origin / suffix
        if not source.is_dir():
            continue
        parent = destination / suffix
        parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() != source or parent.resolve() != parent:
            raise ValueError("aliased cache namespace")
        for entry in source.iterdir():
            if (
                not re.fullmatch("[0-9a-f]{64}", entry.name)
                or entry.is_symlink()
                or not entry.is_dir()
            ):
                continue
            target = parent / entry.name
            if target.exists():
                stats["existing"] += 1
                continue
            try:
                copied = copy_entry(entry, target, suffix.name)
            except (ValueError, KeyError, FileNotFoundError, TypeError):
                stats["invalid"] += 1
                continue
            stats["copied" if copied else "existing"] += 1
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            seed(args.cache_root.resolve(), json.loads(args.report.read_bytes())), sort_keys=True
        )
    )
