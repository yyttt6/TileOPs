"""CPU-only early selection, using the same selector and exact merge base as timing."""

import argparse
import json
import subprocess
from pathlib import Path

from pr_performance import select


def selection(source, base, pr_head=""):
    source = Path(source)

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=source, text=True).strip()

    commit = git("rev-parse", "HEAD")
    if pr_head:
        parents = git("show", "-s", "--format=%P", "HEAD").split()
        if len(parents) != 2 or parents[1] != pr_head:
            raise ValueError("Unexpected PR merge parents")
        base = parents[0]
    base = git("rev-parse", f"{base}^{{commit}}")
    return dict(commit=commit, base=base, **select(source, base))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--base", required=True)
    p.add_argument("--pr-head", default="")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--github-output", type=Path)
    a = p.parse_args()
    result = selection(a.source, a.base, a.pr_head)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    if a.github_output:
        with a.github_output.open("a") as stream:
            stream.write(f"performance_required={str(bool(result['operators'])).lower()}\n")


if __name__ == "__main__":
    main()
