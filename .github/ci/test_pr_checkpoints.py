import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from pr_checkpoints import (
    archive_index,
    digest,
    fingerprint,
    measurement_sources,
    partitions,
    save,
    verify,
)


class ReservationTests(unittest.TestCase):
    def test_compile_fixture_changes_keep_only_execution_identity(self):
        old = {"workloads/op.py": "one", "scripts/ci/tests/test_selection.py": "before"}
        fresh = {**old, "scripts/ci/tests/test_selection.py": "after"}
        self.assertNotEqual(fingerprint(old), fingerprint(fresh))
        self.assertEqual(measurement_sources(old), measurement_sources(fresh))
        self.assertNotEqual(
            measurement_sources(old), measurement_sources({**fresh, "workloads/op.py": "two"})
        )

    def test_partition_after_a_partial_recovery(self):
        cases = {"a": ["one"], "b": [str(i) for i in range(278)], "c": ["x", "y"]}
        done = {("a", "one"), ("b", "1"), ("b", "45")}
        groups = list(partitions(list(cases), cases, done))
        actual = [(op, c) for group in groups for op, ids, partial in group for c in ids]
        self.assertEqual(set(actual), {(op, c) for op, ids in cases.items() for c in ids} - done)
        self.assertEqual(len(actual), len(set(actual)))
        self.assertTrue(all(sum(len(ids) for op, ids, scope in group) <= 24 for group in groups))
        self.assertTrue(all(len(group) == 1 and group[0][2] for group in groups[:-1]))

    def test_small_operators_share_bounded_reservations(self):
        cases = {f"op{i}": ["a", "b"] for i in range(17)}
        groups = list(partitions(list(cases), cases, set()))
        self.assertEqual([len(g) for g in groups], [8, 8, 1])
        self.assertTrue(all(not part for group in groups for op, ids, part in group))

    def test_compact_artifact_digest_is_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "checkpoints.zip"
            index = Path(tmp) / "prior.json"
            obj = {"schema": 1, "identity": {}, "records": []}
            with zipfile.ZipFile(archive, "w") as zipf:
                zipf.writestr("checkpoint-index.json", json.dumps(obj))
            with self.assertRaises(ValueError):
                archive_index(archive, "sha256:" + "0" * 64, index)
            archive_index(archive, digest(archive.read_bytes()), index)
            self.assertEqual(json.loads(index.read_text()), obj)

    def test_receipt_cases_source_and_original_trace_are_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "100-1"
            output = root / "parts/batch-0000/TestOp"
            output.mkdir(parents=True)
            guard = {
                "phase": "finished",
                "workload_completed": True,
                "workload_exit_code": 0,
                "exit_code": 0,
                "performance_valid": True,
                "visibility_lost": False,
                "conflict_detected": False,
                "owner": "pr/100/batch-0000",
            }
            receipt = root / "batch-0000.guard.json"
            save(receipt, guard)
            plan = root / "batch-0000.cases.json"
            save(plan, {"TestOp": ["b"]})
            save(
                root / "batch-0000.plan.json",
                {
                    "jobs": [{"op": "TestOp", "output": str(output)}],
                    "timer": "auto",
                    "case_plan": str(plan),
                },
            )
            save(
                root / "batch-0000.summary.json",
                {"records": [{"op": "TestOp", "exit_code": 0, "fatal": None}]},
            )
            provenance = {
                "commit": "old_commit",
                "base": "base",
                "correctness_protocol": "final-profile-output-v1",
                "harness_sources": {},
                "baseline_sha256": "baseline",
            }
            save(root / "provenance.json", provenance)
            save(output / "sources.json", {"code.py": "hash"})
            save(output / "hardware.json", {"device": "test"})
            save(
                output / "result.json",
                {
                    "status": "passed",
                    "source_changed": False,
                    "failed": 0,
                    "selection_scope": "case-sample",
                    "passed": 1,
                },
            )
            data = output / "results.jsonl"
            data.write_text(
                json.dumps(
                    {
                        "op": "TestOp",
                        "case_id": "b",
                        "status": "passed",
                        "input_preserved": True,
                        "correctness_source": "final-profile-repeat",
                        "correctness": {"total_outputs": 1, "checked_outputs": 1},
                        "measurement": {
                            "warmup": 3,
                            "repeats": 15,
                            "flush_bytes": 1 << 30,
                            "latency_samples_ms": [1] * 15,
                            "latency_ms": 1,
                            "profile_coverage_evidence": str(output / "coverage.json"),
                        },
                    }
                )
                + "\n"
            )
            record = {
                "op": "TestOp",
                "origin": "100-1",
                "commit": "old_commit",
                "cases": ["b"],
                "scope": "case-sample",
                "directory": str(output.relative_to(root)),
                "guard": receipt.name,
                "guard_sha256": digest(receipt.read_bytes()),
                "sources_sha256": fingerprint({"code.py": "hash"}),
                "rows_sha256": digest(data.read_bytes()),
            }

            def check(rec, src=None):
                return verify(
                    rec,
                    base,
                    ["a", "b"],
                    {"code.py": "hash"} if src is None else src,
                    {"device": "test"},
                    lambda m: m["latency_samples_ms"],
                    provenance,
                )

            self.assertEqual(check(record)[0]["case_id"], "b")
            for changed in (
                {"cases": ["a"]},
                {"commit": "other"},
                {"scope": "all-cases"},
                {"guard_sha256": "invalid"},
                {"directory": "../escape"},
            ):
                with self.assertRaises((ValueError, AssertionError)):
                    check({**record, **changed})
            with self.assertRaises(AssertionError):
                check(record, {"code.py": "changed"})
            save(
                receipt,
                {**guard, "exit_code": 124, "workload_exit_code": 143, "performance_valid": False},
            )
            with self.assertRaises(ValueError):
                check({**record, "guard_sha256": hashlib.sha256(receipt.read_bytes()).hexdigest()})


if __name__ == "__main__":
    unittest.main()
