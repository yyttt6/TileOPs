"""CPU-only regressions for exact workload matching across JSON encodings."""

import unittest

from pr_performance import canonical_contract, matching_baselines


class ContractTest(unittest.TestCase):
    def test_preserved_contracts_and_current_provider_precedence(self):
        def row(n, value):
            return dict(
                library="torch-eager",
                op="ExampleOp",
                case="same",
                workload={"params": {"n": n}},
                latency_us=value,
            )

        latest, old = row(64, 3), row(32, 5)
        snapshot = dict(rows=[latest], historical_rows=[old, row(64, 9)])
        self.assertEqual(matching_baselines(snapshot, "ExampleOp", "same", {"n": 32}), [old])
        self.assertEqual(matching_baselines(snapshot, "ExampleOp", "same", {"n": 64}), [latest])
        self.assertEqual(matching_baselines(snapshot, "ExampleOp", "same", {"n": 16}), [])

    def test_positive_and_negative_infinity(self):
        for raw in (float("inf"), "inf", "Infinity"):
            self.assertEqual(canonical_contract({"ord": raw}), {"ord": "Infinity"})
        self.assertEqual(canonical_contract([-float("inf")]), ["-Infinity"])
        self.assertNotEqual(canonical_contract(float("inf")), canonical_contract(-float("inf")))

    def test_different_workloads_remain_different(self):
        base = {"q": [[32, 128], "float16"], "scale": 0.125}
        for other in (
            {"q": [[64, 128], "float16"], "scale": 0.125},
            {"q": [[32, 128], "bfloat16"], "scale": 0.125},
            {"q": [[32, 128], "float16"], "scale": 0.25},
        ):
            self.assertNotEqual(canonical_contract(base), canonical_contract(other))

    def test_ssd_chunk_state_group_state_input_alias_preserves_contract(self):
        op = "SSDChunkStateFwdOp"
        case = "mamba2-1p3b-float16-float32"
        group_state = [[2, 32768, 1, 128], "float16"]
        seq_idx = [[2, 32768], "int32"]
        legacy = dict(
            library="torch-eager",
            op=op,
            case=case,
            workload={"params": {"Bmat": group_state, "seq_idx": seq_idx}},
            latency_us=181768.9,
        )
        baseline = {"rows": [legacy]}
        params = {"B": group_state, "seq_idx": seq_idx}
        self.assertEqual(matching_baselines(baseline, op, case, params), [legacy])
        self.assertEqual(matching_baselines(baseline, op, case, {"B": group_state}), [])
        grouped = {**params, "B": [[2, 32768, 2, 128], "float16"]}
        self.assertEqual(matching_baselines(baseline, op, case, grouped), [])
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            matching_baselines(baseline, op, case, {**params, "Bmat": group_state})


if __name__ == "__main__":
    unittest.main()
