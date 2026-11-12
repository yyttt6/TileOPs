import unittest

from baseline_contracts import comparison_identity


class ScanAliasTest(unittest.TestCase):
    def test_only_names_and_missing_default_are_normalized(self):
        for op in ["CumsumFwdOp", "CumprodFwdOp"]:
            old = {"x": [[2, 65], "bfloat16"], "y": [[2, 65], "bfloat16"], "dim": -1}
            new = {
                "input": [[2, 65], "bfloat16"],
                "output": [[2, 65], "bfloat16"],
                "dim": -1,
                "dtype": None,
            }
            expected = comparison_identity(op, "same", old)
            self.assertEqual(expected, comparison_identity(op, "same", new))
            for change in [
                {"dim": 0},
                {"input": [[3, 65], "bfloat16"]},
                {"output": [[2, 65], "float32"]},
                {"dtype": "float32"},
            ]:
                self.assertNotEqual(expected, comparison_identity(op, "same", dict(new, **change)))
            self.assertNotEqual(expected, comparison_identity(op, "different", new))
            with self.assertRaises(ValueError):
                comparison_identity(op, "same", dict(old, input=new["input"]))

    def test_missing_dtype_does_not_imply_widening(self):
        a = {"x": [[2, 65], "bfloat16"], "y": [[2, 65], "float32"], "dim": -1}
        b = {
            "input": [[2, 65], "bfloat16"],
            "output": [[2, 65], "float32"],
            "dim": -1,
            "dtype": "float32",
        }
        self.assertNotEqual(
            comparison_identity("CumsumFwdOp", "same", a),
            comparison_identity("CumsumFwdOp", "same", b),
        )
