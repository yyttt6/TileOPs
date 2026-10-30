#!/usr/bin/env python3
"""Validate the pinned CI toolchain without initializing an accelerator."""

import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lock",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "config/ascend950-toolchain.json",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    lock = json.loads(args.lock.read_text())
    observed = {
        "profile": lock["profile"],
        "machine": platform.machine(),
        "python": platform.python_version(),
        "packages": {},
        "files": {},
    }
    errors = []
    for key in ("machine", "python"):
        if observed[key] != lock[key]:
            errors.append(f"{key}: expected {lock[key]}, found {observed[key]}")
    for name, expected in lock["packages"].items():
        try:
            actual = metadata.version(name)
        except metadata.PackageNotFoundError:
            actual = None
        observed["packages"][name] = actual
        if actual != expected:
            errors.append(f"{name}: expected {expected}, found {actual}")
    cann = os.environ.get("ASCEND_HOME_PATH")
    cxx = os.environ.get("CXX", "g++")
    try:
        version = subprocess.check_output([cxx, "-dumpfullversion"], text=True).strip()
        observed["cxx_version"] = version
        if version != lock["cxx_version"]:
            errors.append(f"CXX: expected {lock['cxx_version']}, found {version}")
    except (OSError, subprocess.CalledProcessError) as exc:
        errors.append(f"CXX unavailable: {exc}")
    for item in lock["files"]:
        key = item["name"]
        try:
            if item.get("package"):
                path = Path(metadata.distribution(item["package"]).locate_file(item["path"]))
            else:
                if not cann:
                    raise RuntimeError(
                        "ASCEND_HOME_PATH is unset; source the canonical toolchain environment"
                    )
                root = os.environ.get(item.get("root_env", "ASCEND_HOME_PATH"))
                if not root:
                    raise RuntimeError(f"{item['root_env']} is unset")
                path = Path(root) / item["path"]
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            observed["files"][key] = {"path": str(path), "sha256": digest}
            if digest != item["sha256"]:
                errors.append(f"{key}: binary/header fingerprint differs from {lock['profile']}")
        except (OSError, RuntimeError, metadata.PackageNotFoundError) as exc:
            errors.append(f"{key}: {exc}")
    observed["status"] = "failed" if errors else "passed"
    observed["errors"] = errors
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(observed, indent=2) + "\n")
    print(json.dumps(observed, indent=2))
    if errors:
        print(
            "Toolchain mismatch: use docs/ascend950/toolchain.md before debugging the operator.",
            file=sys.stderr,
        )
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())
