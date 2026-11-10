"""The operators the performance job will measure, for the pre-compilation pass to warm.

The measurement step is the trusted harness's, and it selects from ``git diff <base> HEAD``.
The pass must pre-compile the *same* operators or it warms programs nobody asks for, so this
script calls that selection rather than re-implementing it, preferring the trusted harness
copy when the job has one checked out beside the source tree.

It deliberately stops at the direct selection: the harness expands it with composites whose
delegates changed, which needs the device-side registration listing. A composite left out
compiles lazily on its first call exactly as it does today, which costs time but never
correctness.

The ``--base`` argument is the job's *effective* base, not the event's base sha: the
checkout is a merge of the PR into the base branch as it stands now, so diffing against the
sha the PR was opened on would attribute every later base-branch merge to this PR.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load(root: Path):
    """The selection module: the trusted harness's if present, else the candidate's."""
    for path in (root / "ci-harness/scripts/ci/pr_performance.py", HERE / "pr_performance.py"):
        if path.is_file():
            spec = importlib.util.spec_from_file_location("pr_performance", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module, path
    raise SystemExit("no pr_performance.py to take the selection from")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path.cwd())
    parser.add_argument("--base", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.source.resolve()
    module, origin = _load(root)
    plan = module.select(root, args.base)
    receipt = dict(
        base=args.base,
        selection_source=str(origin.relative_to(root)),
        operators=plan["operators"],
        changed=plan["changed"],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print("\n".join(plan["operators"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())

