from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class FullInventoryScriptTest(unittest.TestCase):
    def test_build_full_inventory_artifacts_script(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            outdir = Path(td) / "full_inventory"
            cmd = [
                "python3",
                str(ROOT / "scripts" / "build_full_inventory_artifacts.py"),
                "--inventory",
                str(ROOT / "resources" / "paper_inventory.json"),
                "--outdir",
                str(outdir),
            ]
            subprocess.run(cmd, check=True)

            expected_files = [
                outdir / "paper_inventory.json",
                outdir / "artifact_index.csv",
                outdir / "artifact_values.csv",
            ]
            for path in expected_files:
                self.assertTrue(path.exists(), f"missing output: {path}")


if __name__ == "__main__":
    unittest.main()
