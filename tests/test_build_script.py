from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class BuildScriptTest(unittest.TestCase):
    def test_build_main_body_artifacts_script(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            outdir = Path(td) / "artifacts"
            cmd = [
                "python3",
                str(ROOT / "scripts" / "build_main_body_artifacts.py"),
                "--expected",
                str(ROOT / "repr" / "expected.json"),
                "--results",
                str(ROOT / "tests" / "fixtures" / "results_fixture.json"),
                "--outdir",
                str(outdir),
            ]
            subprocess.run(cmd, check=True)

            expected_files = [
                outdir / "table_hero_expected.csv",
                outdir / "table_scale_expected.csv",
                outdir / "table_logvol_main_expected.csv",
                outdir / "table_diag_main_expected.json",
                outdir / "inline_claims_expected.csv",
                outdir / "figure_headline_points_expected.csv",
                outdir / "figure_scale_points_expected.csv",
                outdir / "run_metrics_long.csv",
                outdir / "run_vs_expected_width_deltas.csv",
            ]
            for path in expected_files:
                self.assertTrue(path.exists(), f"missing output: {path}")


if __name__ == "__main__":
    unittest.main()
