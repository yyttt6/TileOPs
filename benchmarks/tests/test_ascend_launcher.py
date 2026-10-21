"""Launcher routing tests: standard library only, no runtime imports or device work."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


class LauncherRoutingTest(unittest.TestCase):
    def test_site_guard_wraps_the_complete_runtime_command(self):
        root = Path(__file__).resolve().parents[2]
        spec = importlib.util.spec_from_file_location(
            "ascend_launcher", root / "scripts/benchmark_ascend.py"
        )
        launcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(launcher)
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            guard = temp / "host_guard.py"
            guard.write_text("# Site guard routing fixture; never executed.\n")
            run = temp / "run"
            argv = [
                "benchmark_ascend.py",
                "--run-dir",
                str(run),
                "--owner",
                "test/site-guard",
                "--guard",
                str(guard),
                "--verify-only",
                "--op",
                "all",
            ]

            def completed(command, **kwargs):
                self.assertEqual(command[:3], [sys.executable, str(guard), "run"])
                self.assertIn("benchmarks.ascend.runner", command)
                self.assertIn("--verify-only", command)
                self.assertEqual(command[-2:], ["--op", "all"])
                self.assertIn(str(root / "src"), kwargs["env"]["PYTHONPATH"])
                (run / "result.json").write_text(json.dumps({"status": "passed"}))
                return SimpleNamespace(returncode=0)

            with (
                patch.object(sys, "argv", argv),
                patch.object(launcher.subprocess, "run", completed),
            ):
                self.assertEqual(launcher.main(), 0)
            receipt = json.loads((run / "receipt.json").read_text())
            self.assertEqual(receipt["guard_exit_code"], 0)
            self.assertEqual(receipt["command"][1], str(guard))


if __name__ == "__main__":
    unittest.main()
