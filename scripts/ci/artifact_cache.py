"""Publish immutable, checksum-verified archives for reuse by runners on one host."""

import argparse
import fcntl
import hashlib
import re
import shutil
import tempfile
import time
from pathlib import Path

FILES = {"source.commit", "source.tar.gz", "harness.tar.gz", "baseline.tar.gz"}


def digest_file(path):
    with Path(path).open("rb") as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
        return digest.hexdigest()


def verified(directory, digest):
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("Invalid archive manifest digest")
    directory = Path(directory)
    manifest = directory / "SHA256SUMS"
    if digest_file(manifest) != digest:
        raise ValueError("Archive manifest changed")
    entries = [line.split() for line in manifest.read_text().splitlines()]
    if len(entries) != len(FILES) or any(len(e) != 2 for e in entries):
        raise ValueError("Invalid archive manifest entries")
    if {name for _, name in entries} != FILES:
        raise ValueError("Unexpected archive members")
    for sha, name in entries:
        if not re.fullmatch(r"[0-9a-f]{64}", sha) or digest_file(directory / name) != sha:
            raise ValueError(f"Archive checksum mismatch: {name}")
    return directory


def publish(archives, cache_root, digest):
    archives = verified(archives, digest)
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    entry = cache_root / digest
    with (cache_root / f"{digest}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            verified(entry, digest)
            return "already-present"
        except (OSError, ValueError):
            pass
        with tempfile.TemporaryDirectory(prefix=f".{digest}-", dir=cache_root) as tmp:
            stage = Path(tmp) / "entry"
            stage.mkdir()
            for name in FILES | {"SHA256SUMS"}:
                shutil.copyfile(archives / name, stage / name)
            verified(stage, digest)
            if entry.exists():
                entry.rename(cache_root / f"{digest}.invalid-{time.time_ns()}")
            stage.rename(entry)
    return "published"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--archives", required=True, type=Path)
    p.add_argument("--cache-root", required=True, type=Path)
    p.add_argument("--digest", required=True)
    a = p.parse_args()
    print(publish(a.archives, a.cache_root, a.digest))


if __name__ == "__main__":
    main()
