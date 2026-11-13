"""An exact source-only exception cannot reuse affected math or changed runtime/input."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import golden_cache as cache
import reference_cache_compat as compat

pytestmark = pytest.mark.smoke


def identity(source):
    return dict(
        protocol=cache.PROTOCOL,
        source=compat._with_overlay(source),
        source_mode="independent-reference",
        toolchain="same-lock",
        runtime={"torch": "same-version"},
        device="cpu",
        implementation=cache.digest_file(cache.__file__),
        op="KDAFwdOp",
        case="small",
    )


@pytest.fixture
def unchanged(monkeypatch):
    monkeypatch.delenv("ASCEND950_REFERENCE_ROOT", raising=False)
    monkeypatch.setattr(
        compat,
        "reference_source_digest",
        lambda source: (compat.NEW_SOURCE, "independent-reference"),
    )


def test_only_source_changes_for_reviewed_operator(unchanged):
    new = identity(compat.NEW_SOURCE)
    assert compat.reuse_unchanged_reference_identity("source", "KDAFwdOp", new) == identity(
        compat.OLD_SOURCE
    )
    assert new["source"] != identity(compat.OLD_SOURCE)["source"]


@pytest.mark.parametrize(
    "op", ["SharedExpertMLPFwdOp", "FusedMoESharedExpertFwdOp", "IndexedExpertMLPFwdOp", "unknown"]
)
def test_affected_and_unreviewed_ops_keep_fresh_identity(unchanged, op):
    new = identity(compat.NEW_SOURCE)
    assert compat.reuse_unchanged_reference_identity("source", op, new) is new


@pytest.mark.parametrize("what", ["source", "mode", "identity", "overlay"])
def test_changed_source_or_identity_never_aliases(unchanged, monkeypatch, tmp_path, what):
    new = identity(compat.NEW_SOURCE)
    if what == "source":
        monkeypatch.setattr(
            compat, "reference_source_digest", lambda source: ("0" * 64, "independent-reference")
        )
    elif what == "mode":
        new["source_mode"] = "full-source"
    elif what == "identity":
        new["source"] = "0" * 64
    else:
        (tmp_path / "reference.py").write_text("changed oracle\n")
        monkeypatch.setenv("ASCEND950_REFERENCE_ROOT", str(tmp_path))
    assert compat.reuse_unchanged_reference_identity("source", "KDAFwdOp", new) is new


def test_overlay_fingerprints_preserved_in_old_and_new(unchanged, monkeypatch, tmp_path):
    file = tmp_path / "reference.py"
    file.write_bytes(b"same oracle")
    monkeypatch.setenv("ASCEND950_REFERENCE_ROOT", str(tmp_path))
    new = identity(compat.NEW_SOURCE)
    assert compat.reuse_unchanged_reference_identity("source", "KDAFwdOp", new) == identity(
        compat.OLD_SOURCE
    )
    file.write_bytes(b"different oracle")
    assert compat.reuse_unchanged_reference_identity("source", "KDAFwdOp", new) is new


def test_real_cache_hit_then_changed_runtime_and_input_miss(unchanged, tmp_path):
    import torch

    x = torch.tensor([1.0, 2.0], device="cpu")
    old = cache.GoldenCache(tmp_path, identity(compat.OLD_SOURCE), min_free_bytes=0)
    assert torch.equal(old.wrap(lambda value: value + 1)(x), x + 1)
    old.commit()
    new_identity = compat.reuse_unchanged_reference_identity(
        "source", "KDAFwdOp", identity(compat.NEW_SOURCE)
    )
    hit = cache.GoldenCache(tmp_path, new_identity, min_free_bytes=0)

    def forbidden(value):
        raise AssertionError("unchanged reference should not execute")

    assert torch.equal(hit.wrap(forbidden)(x), x + 1)
    assert hit.events[-1]["status"] == "hit"
    different = cache.GoldenCache(
        tmp_path, dict(new_identity, runtime={"torch": "changed-version"}), min_free_bytes=0
    )
    assert torch.equal(different.wrap(lambda value: value + 2)(x), x + 2)
    assert different.events[-1]["status"] == "miss"
    changed_input = cache.GoldenCache(tmp_path, new_identity, min_free_bytes=0)
    assert torch.equal(changed_input.wrap(lambda value: value + 2)(x + 1), x + 3)
    assert changed_input.events[-1]["status"] == "miss"
