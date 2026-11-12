"""No-device selection and real workflow archive restoration contracts."""

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.smoke
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/ci"))
from artifact_cache import FILES, publish, verified  # noqa: E402
from early_selection import selection  # noqa: E402


def git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


@pytest.fixture
def repo(tmp_path):
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.email", "test@example.invalid")
    git(tmp_path, "config", "user.name", "Test")
    p = tmp_path / "src/tileops/_kernels/ascend/elementwise/foo.py"
    p.parent.mkdir(parents=True)
    p.write_text('OP = "FooOp"\n')
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-qm", "base")
    return tmp_path


def test_docs_skip_but_kernel_changes_select(repo):
    base = git(repo, "rev-parse", "HEAD")
    (repo / "README.md").write_text("docs")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "docs")
    assert selection(repo, base)["operators"] == []
    (repo / "src/tileops/_kernels/ascend/elementwise/foo.py").write_text('OP = "FooOp"\nX = 2\n')
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "kernel")
    assert selection(repo, base)["operators"] == ["FooOp"]


def test_merge_uses_actual_first_parent_and_rejects_wrong_head(repo):
    original = git(repo, "rev-parse", "HEAD")
    git(repo, "checkout", "-qb", "pr")
    (repo / "README.md").write_text("docs")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "pr")
    head = git(repo, "rev-parse", "HEAD")
    git(repo, "checkout", "-qb", "base-new", original)
    (repo / "src/tileops/_kernels/ascend/elementwise/foo.py").write_text('OP = "FooOp"\nX = 3\n')
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "base-advanced")
    actual = git(repo, "rev-parse", "HEAD")
    git(repo, "merge", "--no-ff", "-m", "merge", "pr")
    result = selection(repo, original, head)
    assert result["base"] == actual and result["operators"] == []
    with pytest.raises(ValueError, match="parents"):
        selection(repo, original, original)


def archives(root):
    root.mkdir()
    for name in FILES:
        (root / name).write_bytes(name.encode())
    manifest = "".join(
        f"{hashlib.sha256((root / name).read_bytes()).hexdigest()}  {name}\n"
        for name in sorted(FILES)
    )
    (root / "SHA256SUMS").write_text(manifest)
    return hashlib.sha256(manifest.encode()).hexdigest()


def probe_code():
    jobs = yaml.safe_load((ROOT / ".github/workflows/ascend950.yml").read_text())["jobs"]
    steps = [
        next(s for s in jobs[j]["steps"] if s.get("id") == "archive-cache")
        for j in ["preflight", "pr-performance", "device"]
    ]
    assert steps[0]["run"] == steps[1]["run"] == steps[2]["run"]
    assert jobs["pr-performance"]["needs"] == ["sources", "preflight"]
    assert "performance_required == 'true'" in jobs["pr-performance"]["if"]
    return steps[0]["run"].split("python3 - <<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]


def restore(tmp_path, cache, digest, label):
    output = tmp_path / f"{label}.output"
    target = tmp_path / label
    env = dict(
        os.environ,
        CACHE_ROOT=str(cache),
        ARCHIVE_DIGEST=digest,
        ARCHIVES=str(target),
        GITHUB_OUTPUT=str(output),
    )
    subprocess.run([sys.executable, "-c", probe_code()], env=env, check=True)
    return output.read_text(), target


def test_real_workflow_miss_hit_and_corruption_fallback(tmp_path):
    source = tmp_path / "source"
    digest = archives(source)
    cache = tmp_path / "cache"
    assert restore(tmp_path, cache, digest, "miss")[0] == "hit=false\n"
    assert publish(source, cache, digest) == "published"
    result, target = restore(tmp_path, cache, digest, "hit")
    assert result == "hit=true\n"
    verified(target, digest)
    assert publish(source, cache, digest) == "already-present"
    (cache / digest / "source.tar.gz").write_bytes(b"corrupt")
    assert restore(tmp_path, cache, digest, "corrupt")[0] == "hit=false\n"
    assert publish(source, cache, digest) == "published"
    assert list(cache.glob("*.invalid-*"))
    assert restore(tmp_path, cache, digest, "repaired")[0] == "hit=true\n"


def test_wrong_manifest_digest_and_path_escape_are_rejected(tmp_path):
    source = tmp_path / "source"
    digest = archives(source)
    with pytest.raises(ValueError):
        publish(source, tmp_path / "cache", "0" * 64)
    (source / "SHA256SUMS").write_text("0" * 64 + "  ../outside\n")
    digest = hashlib.sha256((source / "SHA256SUMS").read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        publish(source, tmp_path / "cache", digest)
    assert restore(tmp_path, tmp_path / "cache", "../escape", "bad")[0] == "hit=false\n"


def test_old_pr_excludes_target_only_manifest_changes(repo):
    from pr_performance import select

    manifest = repo / "src/tileops/manifest/_workload.py"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("OLD_VALIDATION = True\n")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "shared ancestor manifest")
    common = git(repo, "rev-parse", "HEAD")
    git(repo, "checkout", "-qb", "target")
    manifest.write_text("NEW_VALIDATION = True\n")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "target-only shared changes")
    target = git(repo, "rev-parse", "HEAD")
    git(repo, "checkout", "-qb", "old-pr", common)
    kernel = repo / "src/tileops/_kernels/ascend/topk/select.py"
    kernel.parent.mkdir(parents=True)
    kernel.write_text('OP = "TopKSelectFwdOp"\n')
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "topk only")
    early = selection(repo, target)
    late = select(repo, target)
    assert early["operators"] == late["operators"] == ["TopKSelectFwdOp"]
    assert early["changed"] == ["src/tileops/_kernels/ascend/topk/select.py"]
    assert early["comparison_base"] == common
    assert early["target_base"] == target
    assert early["target_is_ancestor"] is False

    # Shared changes actually introduced by the PR must still expand coverage.
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("PR_VALIDATION = True\n")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "PR shared change")
    assert selection(repo, target)["operators"] == ["FooOp", "TopKSelectFwdOp"]


def test_manual_target_descendant_has_no_false_revert_changes(repo):
    common = git(repo, "rev-parse", "HEAD")
    p = repo / "src/tileops/_kernels/ascend/elementwise/foo.py"
    p.write_text('OP = "FooOp"\nX = 4\n')
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "target advances")
    target = git(repo, "rev-parse", "HEAD")
    git(repo, "checkout", "--detach", common)
    result = selection(repo, target)
    assert result["operators"] == []
    assert result["comparison_base"] == common
