"""CPU-only regressions for exact workload matching across JSON encodings."""

import unittest

from pr_performance import canonical_contract


class ContractTest(unittest.TestCase):
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
