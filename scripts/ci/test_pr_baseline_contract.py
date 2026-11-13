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

    def test_linear_attention_defaults_and_renames_preserve_contracts(self):
        from baseline_contracts import comparison_identity as key

        for op in ("GDNFwdOp", "KDAFwdOp", "GLAFwdOp", "DeltaNetFwdOp"):
            params = {"q": [[1, 128, 4, 64], "bfloat16"]}
            self.assertEqual(
                key(op, "c", params), key(op, "c", {**params, "validate_inputs": False})
            )
            self.assertNotEqual(
                key(op, "c", params), key(op, "c", {**params, "validate_inputs": True})
            )
        old = {"gk": [[1, 4, 64], "float16"], "scale": 0.125}
        current = {"g": old["gk"], "scale": 0.125}
        self.assertEqual(key("GLARecurrentFwdOp", "c", old), key("GLARecurrentFwdOp", "c", current))
        self.assertNotEqual(
            key("GLARecurrentFwdOp", "c", old),
            key("GLARecurrentFwdOp", "c", {**current, "scale": 0.25}),
        )
        with self.assertRaises(ValueError):
            key("GLARecurrentFwdOp", "c", {**old, **current})

    def test_topk_select_public_field_renames(self):
        from baseline_contracts import comparison_identity as key

        old = {
            "index_score": [[1, 1, 16384, 1], "float32"],
            "indexes": [[1, 1, 1, 2048], "int32"],
            "topk": 2048,
        }
        new = {"logits": old["index_score"], "indices": old["indexes"], "top_k": 2048}
        self.assertEqual(key("TopKSelectFwdOp", "c", old), key("TopKSelectFwdOp", "c", new))
        self.assertNotEqual(
            key("TopKSelectFwdOp", "c", old), key("TopKSelectFwdOp", "c", {**new, "top_k": 1024})
        )
        with self.assertRaises(ValueError):
            key("TopKSelectFwdOp", "c", {**new, "indexes": old["indexes"]})

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


if __name__ == "__main__":
    unittest.main()
