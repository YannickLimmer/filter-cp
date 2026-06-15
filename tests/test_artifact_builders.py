from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from filter_cp_repro.artifact_builders import (  # noqa: E402
    build_main_artifact_parity_rows,
    build_figure_headline_points,
    build_figure_scale_points,
    build_inline_claims,
    compare_diag_to_expected,
    compare_logvol_to_expected,
    build_table_diag_main,
    build_table_hero,
    build_table_logvol_main,
    build_table_scale,
    compare_results_to_expected,
    flatten_run_results,
    load_expected,
    load_results,
    summarize_parity_rows,
)


class ArtifactBuildersTest(unittest.TestCase):
    def setUp(self) -> None:
        self.expected = load_expected(ROOT / "repr" / "expected.json")

    def test_table_hero_row_count(self) -> None:
        rows = build_table_hero(self.expected)
        # 2 cells x 7 methods in expected table_hero
        self.assertEqual(len(rows), 14)

    def test_table_scale_row_count(self) -> None:
        rows = build_table_scale(self.expected)
        # 4 cells x 7 methods in expected table_scale_gpvar
        self.assertEqual(len(rows), 28)

    def test_logvol_row_count(self) -> None:
        rows = build_table_logvol_main(self.expected)
        # 2 cells x 4 methods
        self.assertEqual(len(rows), 8)

    def test_diag_structure(self) -> None:
        diag = build_table_diag_main(self.expected)
        self.assertIn("rho_dG_range", diag)
        self.assertIn("tau_int_range", diag)

    def test_inline_claim_values(self) -> None:
        claims = {r["claim_id"]: r for r in build_inline_claims(self.expected)}
        self.assertAlmostEqual(claims["improvement_vs_static_metrla"]["percent"], 23.5588972431, places=6)
        self.assertAlmostEqual(claims["improvement_vs_static_pems_bay"]["percent"], 40.5405405405, places=6)
        self.assertAlmostEqual(claims["improvement_vs_best_nonfilter_metrla"]["percent"], 8.4084084084, places=6)
        self.assertAlmostEqual(claims["improvement_vs_best_nonfilter_pems_bay"]["percent"], 13.9130434782, places=6)

    def test_figure_points_nonempty(self) -> None:
        self.assertGreater(len(build_figure_headline_points(self.expected)), 0)
        self.assertGreater(len(build_figure_scale_points(self.expected)), 0)

    def test_flatten_and_compare(self) -> None:
        results = load_results(ROOT / "tests" / "fixtures" / "results_fixture.json")
        rows = flatten_run_results(results)
        self.assertGreater(len(rows), 0)
        comp = compare_results_to_expected(results, self.expected)
        self.assertGreater(len(comp), 0)
        for r in comp:
            self.assertIn("within_tolerance", r)
            self.assertTrue(math.isfinite(float(r["delta"])))

    def test_logvol_and_diag_parity(self) -> None:
        results = {
            "metrla": {
                "aggregate": {
                    "methods": {
                        "GNF": {"vhat_m": 1.20},
                        "AgACIGroupCGIF": {"vhat_m": 1.30},
                        "ACIPerGroupFactorCGIF": {"vhat_m": 1.35},
                        "StaticCGIF": {"vhat_m": 1.46},
                    },
                    "audit": {
                        "rho_dG": 0.455,
                        "rho_DL": 0.49,
                        "rho_score": 0.145,
                        "rho_ind": 0.84,
                        "tau_int": 12.0,
                    },
                }
            },
            "pems_bay": {
                "aggregate": {
                    "methods": {
                        "GNF": {"vhat_m": 0.82},
                        "AgACIGroupCGIF": {"vhat_m": 1.05},
                        "ACIPerGroupFactorCGIF": {"vhat_m": 1.06},
                        "StaticCGIF": {"vhat_m": 1.31},
                    },
                    "audit": {
                        "rho_dG": 0.458,
                        "rho_DL": 0.495,
                        "rho_score": 0.150,
                        "rho_ind": 0.86,
                        "tau_int": 11.0,
                    },
                }
            },
        }
        logvol = compare_logvol_to_expected(results, self.expected)
        self.assertEqual(len(logvol), 8)
        self.assertTrue(all("within_tolerance" in r for r in logvol))

        diag = compare_diag_to_expected(results, self.expected)
        self.assertGreaterEqual(len(diag), 10)
        self.assertTrue(all("within_tolerance" in r for r in diag))

    def test_main_body_parity_summary(self) -> None:
        results = load_results(ROOT / "tests" / "fixtures" / "results_fixture.json")
        rows = build_main_artifact_parity_rows(results, self.expected, "table_hero")
        self.assertGreater(len(rows), 0)
        summary = summarize_parity_rows(rows)
        self.assertEqual(summary["total"], len(rows))
        self.assertIn("table_hero", summary["by_artifact"])


if __name__ == "__main__":
    unittest.main()
