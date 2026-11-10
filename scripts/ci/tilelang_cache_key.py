"""Stable persistent-cache namespace after the canonical toolchain check passes.

Identical source trees reuse a namespace across runs and merge commits. Different
source trees are isolated; sharing individual kernels across PRs is not promised.
TileLang still applies its own per-kernel cache keys inside this namespace.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def cache_identity(source: Path, lock: Path, verified_toolchain: Path) -> dict:
    lock_bytes = lock.read_bytes()
    if not lock_bytes.strip():
        raise ValueError("Toolchain lock is empty")
    json.loads(lock_bytes)
    report = json.loads(verified_toolchain.read_text())
    if report.get("status") != "passed" or report.get("errors") != []:
        raise ValueError("Toolchain verification must pass before reusing compiled kernels")
    if not report.get("files") or not report.get("packages"):
        raise ValueError("Toolchain report has no binary or package fingerprints")
    # Absolute package paths can contain the job's venv path. Hash their verified
    # contents instead, so moving an identical environment does not invalidate it.
    fingerprints = {name: item["sha256"] for name, item in report["files"].items()}
    if any(
        len(value) != 64 or any(c not in "0123456789abcdef" for c in value)
        for value in fingerprints.values()
    ):
        raise ValueError("Toolchain report contains an invalid SHA256 fingerprint")
    # The archive is verified clean before environment preparation. Reject later
    # tracked-source edits rather than lending them HEAD's persistent namespace.
    subprocess.run(["git", "diff", "--quiet", "HEAD", "--"], cwd=source, check=True)
    tree = subprocess.check_output(
        ["git", "rev-parse", "HEAD^{tree}"], cwd=source, text=True
    ).strip()
    return {
        "schema": "tileops.ascend-cache.v1",
        "source_tree": tree,
        "lock_sha256": hashlib.sha256(lock_bytes).hexdigest(),
        "toolchain": {
            **{key: report[key] for key in ("profile", "machine", "python", "cxx_version")},
            "packages": report["packages"],
            "files": fingerprints,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--verified-toolchain", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    identity = cache_identity(args.source, args.lock, args.verified_toolchain)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    args.output.write_text(json.dumps({"key": key, **identity}, indent=2) + "\n")
    print(key)


if __name__ == "__main__":
    main()
