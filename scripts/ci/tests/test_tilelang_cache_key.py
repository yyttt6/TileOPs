"""Cache reuse and invalidation contracts, without importing a device runtime."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.ci.tilelang_cache_key import cache_identity

pytestmark = pytest.mark.smoke
SCRIPT = Path(__file__).resolve().parents[1] / "tilelang_cache_key.py"


def git(repo, *args):
    return subprocess.check_output(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com", *args],
        cwd=repo,
        text=True,
        stderr=subprocess.PIPE,
    ).strip()


@pytest.fixture
def environment(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    git(source, "init")
    (source / "kernel.py").write_text("value = 1\n")
    git(source, "add", "kernel.py")
    git(source, "commit", "-m", "initial")
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps({"profile": "test", "compiler": "v1"}))
    report = tmp_path / "verified.json"
    report.write_text(
        json.dumps(
            {
                "status": "passed",
                "errors": [],
                "profile": "test",
                "machine": "aarch64",
                "python": "3.12.15",
                "cxx_version": "12.3.0",
                "packages": {"tilelang": "0.1.15"},
                "files": {"compiler": {"path": "/job-1/bisheng", "sha256": "a" * 64}},
            }
        )
    )
    return source, lock, report


def test_identical_tree_survives_new_commit_checkout_path_and_report_path(environment, tmp_path):
    source, lock, report = environment
    before = cache_identity(*environment)
    git(source, "commit", "--allow-empty", "-m", "different commit, same code")
    checkout = tmp_path / "another-run"
    git(tmp_path, "clone", str(source), str(checkout))
    data = json.loads(report.read_text())
    data["files"]["compiler"]["path"] = "/job-2/bisheng"
    report.write_text(json.dumps(data, indent=2))
    assert cache_identity(checkout, lock, report) == before


@pytest.mark.parametrize("changed", ["source", "lock", "binary", "package"])
def test_changed_compilation_inputs_invalidate_namespace(environment, changed):
    source, lock, report = environment
    before = cache_identity(*environment)
    if changed == "source":
        (source / "kernel.py").write_text("value = 2\n")
        git(source, "commit", "-am", "change kernel")
    elif changed == "lock":
        lock.write_text(json.dumps({"profile": "test", "compiler": "v2"}))
    else:
        data = json.loads(report.read_text())
        if changed == "binary":
            data["files"]["compiler"]["sha256"] = "b" * 64
        else:
            data["packages"]["tilelang"] = "0.1.16"
        report.write_text(json.dumps(data))
    assert cache_identity(*environment) != before


@pytest.mark.parametrize(
    "invalid",
    [
        "missing-lock",
        "empty-lock",
        "missing-report",
        "failed",
        "errors",
        "missing-fingerprints",
        "invalid-fingerprint",
    ],
)
def test_cli_fails_closed_without_emitting_a_cache_key(environment, tmp_path, invalid):
    source, lock, report = environment
    if invalid == "missing-lock":
        lock.unlink()
    elif invalid == "empty-lock":
        lock.write_text("")
    elif invalid == "missing-report":
        report.unlink()
    else:
        data = json.loads(report.read_text())
        if invalid == "failed":
            data["status"] = "failed"
        elif invalid == "errors":
            data["errors"] = ["compiler differs"]
        elif invalid == "missing-fingerprints":
            data["files"] = {}
        else:
            data["files"]["compiler"]["sha256"] = ""
        report.write_text(json.dumps(data))
    receipt = tmp_path / "receipt.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--source",
            str(source),
            "--lock",
            str(lock),
            "--verified-toolchain",
            str(report),
            "--output",
            str(receipt),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert result.stdout == ""
    assert not receipt.exists()


def test_cli_emits_reproducible_key_and_audit_receipt(environment, tmp_path):
    source, lock, report = environment
    keys = []
    for run in (1, 2):
        receipt = tmp_path / f"run-{run}.json"
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--source",
                str(source),
                "--lock",
                str(lock),
                "--verified-toolchain",
                str(report),
                "--output",
                str(receipt),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        key = result.stdout.strip()
        saved = json.loads(receipt.read_text())
        assert len(key) == 64
        assert saved.pop("key") == key
        assert saved == cache_identity(*environment)
        keys.append(key)
    assert keys[0] == keys[1]


@pytest.mark.parametrize("helper_location", ["trusted", "candidate", "missing"])
def test_workflow_cache_setup_supports_old_revisions_and_fails_closed(
    environment, tmp_path, monkeypatch, helper_location
):
    import os
    import shutil

    import yaml

    source, lock, report = environment
    workflow = SCRIPT.parents[2] / ".github/workflows/ascend950.yml"
    steps = yaml.safe_load(workflow.read_text())["jobs"]["pr-performance"]["steps"]
    body = next(
        s["run"] for s in steps if s.get("name", "").startswith("Changed operator performance")
    )
    # Test cache setup independently of how the subsequent harness is invoked.
    cache_end = 'mkdir -p "$TILELANG_CACHE_DIR"'
    body = body[body.index("CACHE_KEY_SCRIPT=") : body.index(cache_end) + len(cache_end)]
    body = body.replace("/opt/tileops-ci/server2-20261030/config/toolchain-v1.lock.json", str(lock))
    cache_root = tmp_path / "cache"
    body = body.replace("/opt/tileops-ci/server2-20261030/cache", str(cache_root))
    if helper_location != "missing":
        prefix = source / "ci-harness" if helper_location == "trusted" else source
        helper = prefix / "scripts/ci/tilelang_cache_key.py"
        helper.parent.mkdir(parents=True)
        shutil.copyfile(SCRIPT, helper)
    shutil.copyfile(report, tmp_path / "pr-toolchain.json")
    monkeypatch.setenv("RUNNER_TEMP", str(tmp_path))
    monkeypatch.setenv("PATH", str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"])
    result = subprocess.run(
        ["bash", "-c", "set -eo pipefail\n" + body],
        cwd=source,
        capture_output=True,
        text=True,
    )
    receipt = tmp_path / "pr-cache.json"
    if helper_location == "missing":
        assert result.returncode != 0
        assert not receipt.exists()
        assert not cache_root.exists()
    else:
        assert result.returncode == 0, result.stderr
        key = json.loads(receipt.read_text())["key"]
        assert (cache_root / f"tilelang-v1-{key}").is_dir()


def test_uncommitted_source_cannot_reuse_head_cache(environment):
    source, _, _ = environment
    (source / "kernel.py").write_text("uncommitted = True\n")
    with pytest.raises(subprocess.CalledProcessError):
        cache_identity(*environment)


@pytest.mark.parametrize("wrong_head", [False, True])
def test_workflow_compares_tested_merge_parent_not_stale_event_base(
    environment, tmp_path, wrong_head
):
    import os

    import yaml

    source, _, _ = environment
    old = git(source, "rev-parse", "HEAD")
    git(source, "checkout", "-b", "candidate")
    (source / "candidate.py").write_text("changed = True\n")
    git(source, "add", ".")
    git(source, "commit", "-m", "candidate")
    head = git(source, "rev-parse", "HEAD")
    git(source, "checkout", "-b", "base", old)
    (source / "unrelated.py").write_text("already_merged = True\n")
    git(source, "add", ".")
    git(source, "commit", "-m", "advance target")
    tested_base = git(source, "rev-parse", "HEAD")
    git(source, "merge", "--no-ff", "candidate", "-m", "test merge")
    steps = yaml.safe_load((SCRIPT.parents[2] / ".github/workflows/ascend950.yml").read_text())[
        "jobs"
    ]["pr-performance"]["steps"]
    body = next(
        s["run"] for s in steps if s.get("name", "").startswith("Changed operator performance")
    )
    body = body[body.index("# PR events") : body.index("# Prefer the trusted")]
    result = subprocess.run(
        ["bash", "-c", "set -eo pipefail\n" + body],
        cwd=source,
        env={
            **os.environ,
            "PR_HEAD": old if wrong_head else head,
            "PR_BASE": old,
            "RUNNER_TEMP": str(tmp_path),
        },
        capture_output=True,
        text=True,
    )
    if wrong_head:
        assert result.returncode != 0
        assert not (tmp_path / "pr-selection-base.txt").exists()
    else:
        assert result.returncode == 0, result.stderr
        selected = (tmp_path / "pr-selection-base.txt").read_text().strip()
        assert selected == tested_base
        assert git(source, "diff", "--name-only", selected, "HEAD") == "candidate.py"
