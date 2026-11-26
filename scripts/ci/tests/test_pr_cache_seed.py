"""The PR can consume production objects but can only write its own copies."""

import hashlib
import json
from pathlib import Path

import pytest

from scripts.ci.seed_pr_cache import key, seed

pytestmark = pytest.mark.smoke


def test_trusted_objects_are_checked_and_private(tmp_path):
    common = dict(
        lock_sha256="test-lock", toolchain=dict(machine="aarch64", packages={"tilelang": "0.1.15"})
    )
    trusted_key = key(dict(common, schema="tileops.ascend-trusted-program-cache.v1"))
    private = dict(common, schema="tileops.ascend-cache.v2", compilation_sources="candidate")
    private["key"] = key(private)
    sub = Path("0.1.15/ascend-binaries")
    trusted = tmp_path / f"tilelang-v1-{trusted_key}" / sub
    payload = b"compiled"
    for name, sha in (("a" * 64, hashlib.sha256(payload).hexdigest()), ("b" * 64, "0" * 64)):
        entry = trusted / name
        entry.mkdir(parents=True)
        (entry / "kernel.aibin").write_bytes(payload)
        (entry / "metadata.json").write_text(
            json.dumps(
                dict(format="tilelang.ascend-binary-cache.v1", sha256=sha, size=len(payload))
            )
        )
    stats = seed(tmp_path, private)
    assert (stats["copied"], stats["invalid"]) == (1, 1)
    dst = tmp_path / ("tilelang-v1-" + private["key"]) / sub
    assert not (dst / ("b" * 64)).exists()
    (dst / ("a" * 64) / "kernel.aibin").write_bytes(b"pr")
    assert (trusted / ("a" * 64) / "kernel.aibin").read_bytes() == payload
    assert seed(tmp_path, private)["existing"] == 1
    changed = dict(private, lock_sha256="different")
    changed["key"] = key({k: v for k, v in changed.items() if k != "key"})
    assert seed(tmp_path, changed)["copied"] == 0
