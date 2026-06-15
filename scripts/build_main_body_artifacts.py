#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from filter_cp_repro.artifact_builders import (
    build_figure_headline_points,
    build_figure_scale_points,
    build_inline_claims,
    build_table_diag_main,
    build_table_hero,
    build_table_logvol_main,
    build_table_scale,
    compare_results_to_expected,
    flatten_run_results,
    load_expected,
    load_results,
)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build machine-readable main-body artifacts from expected and run outputs.")
    p.add_argument(
        "--expected",
        type=Path,
        default=ROOT / "repr" / "expected.json",
        help="Path to expected reference json.",
    )
    p.add_argument(
        "--results",
        nargs="*",
        type=Path,
        default=[],
        help="Optional run results json files (results_<cells>.json).",
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=ROOT / "artifacts" / "main_body",
        help="Output directory.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    expected = load_expected(args.expected)

    write_csv(args.outdir / "table_hero_expected.csv", build_table_hero(expected))
    write_csv(args.outdir / "table_scale_expected.csv", build_table_scale(expected))
    write_csv(args.outdir / "table_logvol_main_expected.csv", build_table_logvol_main(expected))
    write_json(args.outdir / "table_diag_main_expected.json", build_table_diag_main(expected))
    write_csv(args.outdir / "inline_claims_expected.csv", build_inline_claims(expected))
    write_csv(args.outdir / "figure_headline_points_expected.csv", build_figure_headline_points(expected))
    write_csv(args.outdir / "figure_scale_points_expected.csv", build_figure_scale_points(expected))

    merged_results: dict[str, Any] = {}
    for path in args.results:
        data = load_results(path)
        merged_results.update(data)

    if merged_results:
        write_csv(args.outdir / "run_metrics_long.csv", flatten_run_results(merged_results))
        write_csv(
            args.outdir / "run_vs_expected_width_deltas.csv",
            compare_results_to_expected(merged_results, expected),
        )

    print(f"Wrote artifacts to {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
