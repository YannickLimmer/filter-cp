from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_expected(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_results(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_inventory(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _method_stats(cell_record: dict[str, Any], method: str) -> tuple[float, float] | None:
    val = cell_record.get(method)
    if not isinstance(val, list) or len(val) != 2:
        return None
    return float(val[0]), float(val[1])


def build_table_hero(expected: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    hero = expected["table_hero"]
    for cell, rec in hero.items():
        n_val = int(rec["N"])
        for method, val in rec.items():
            if method == "N":
                continue
            mean, std = float(val[0]), float(val[1])
            rows.append(
                {
                    "table": "tab:hero",
                    "cell": cell,
                    "N": n_val,
                    "method": method,
                    "width_mean": mean,
                    "width_std": std,
                }
            )
    rows.sort(key=lambda r: (r["cell"], r["method"]))
    return rows


def build_table_scale(expected: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    scale = expected["table_scale_gpvar"]
    for cell, rec in scale.items():
        n_val = int(rec["N"])
        for method, val in rec.items():
            if method == "N":
                continue
            mean, std = float(val[0]), float(val[1])
            rows.append(
                {
                    "table": "tab:scale",
                    "cell": cell,
                    "N": n_val,
                    "method": method,
                    "width_mean": mean,
                    "width_std": std,
                }
            )
    rows.sort(key=lambda r: (r["N"], r["cell"], r["method"]))
    return rows


def build_table_logvol_main(expected: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    table = expected["table_logvol_main"]
    for cell, rec in table.items():
        if cell.startswith("_"):
            continue
        n_val = int(rec["N"])
        for method, val in rec.items():
            if method in {"N"}:
                continue
            rows.append(
                {
                    "table": "tab:logvol-main",
                    "cell": cell,
                    "N": n_val,
                    "method": method,
                    "logvol_per_coord": float(val),
                }
            )
    rows.sort(key=lambda r: (r["cell"], r["method"]))
    return rows


def build_table_diag_main(expected: dict[str, Any]) -> dict[str, Any]:
    diag = expected["table_diag_main"]
    return {
        "table": "tab:diag-main",
        "rho_dG_range": [float(diag["rho_dG_range"][0]), float(diag["rho_dG_range"][1])],
        "rho_DL_range": [float(diag["rho_DL_range"][0]), float(diag["rho_DL_range"][1])],
        "rho_score_range": [float(diag["rho_score_range"][0]), float(diag["rho_score_range"][1])],
        "rho_ind_approx": float(diag["rho_ind_approx"]),
        "tau_int_range": [float(diag["tau_int_range"][0]), float(diag["tau_int_range"][1])],
    }


def _pct_improvement(numerator: float, denominator: float) -> float:
    return 100.0 * (1.0 - numerator / denominator)


def build_inline_claims(expected: dict[str, Any]) -> list[dict[str, Any]]:
    hero = expected["table_hero"]

    metrla = hero["metrla"]
    pems = hero["pems_bay"]

    rows = [
        {
            "claim_id": "improvement_vs_static_metrla",
            "dataset": "metrla",
            "numerator_method": "GNF",
            "denominator_method": "StaticCGIF",
            "percent": _pct_improvement(float(metrla["GNF"][0]), float(metrla["StaticCGIF"][0])),
        },
        {
            "claim_id": "improvement_vs_static_pems_bay",
            "dataset": "pems_bay",
            "numerator_method": "GNF",
            "denominator_method": "StaticCGIF",
            "percent": _pct_improvement(float(pems["GNF"][0]), float(pems["StaticCGIF"][0])),
        },
        {
            "claim_id": "improvement_vs_best_nonfilter_metrla",
            "dataset": "metrla",
            "numerator_method": "GNF",
            "denominator_method": "AgACIGroupCGIF",
            "percent": _pct_improvement(float(metrla["GNF"][0]), float(metrla["AgACIGroupCGIF"][0])),
        },
        {
            "claim_id": "improvement_vs_best_nonfilter_pems_bay",
            "dataset": "pems_bay",
            "numerator_method": "GNF",
            "denominator_method": "ACIPerGroupFactorCGIF",
            "percent": _pct_improvement(float(pems["GNF"][0]), float(pems["ACIPerGroupFactorCGIF"][0])),
        },
    ]
    return rows


def build_figure_headline_points(expected: dict[str, Any]) -> list[dict[str, Any]]:
    """Machine-readable points for Figure 2-like panels from expected references."""
    # Panel mapping in the manuscript narrative:
    # panel 1: metrla (N=20), panel 2: pems_bay (N=50),
    # panel 3: metrla_full (N=207), panel 4: pems_bay_full (N=325).
    hero = expected["table_hero"]
    scale = expected["table_scale_gpvar"]

    panels = [
        (1, "metrla", hero["metrla"]),
        (2, "pems_bay", hero["pems_bay"]),
        (3, "metrla_full", scale["metrla_full"]),
        (4, "pems_bay_full", scale["pems_bay_full"]),
    ]

    rows: list[dict[str, Any]] = []
    for panel, cell, rec in panels:
        n_val = int(rec["N"])
        for method, val in rec.items():
            if method == "N":
                continue
            rows.append(
                {
                    "figure": "fig:headline",
                    "panel": panel,
                    "cell": cell,
                    "N": n_val,
                    "method": method,
                    "width_mean": float(val[0]),
                    "width_std": float(val[1]),
                }
            )
    rows.sort(key=lambda r: (r["panel"], r["method"]))
    return rows


def build_figure_scale_points(expected: dict[str, Any]) -> list[dict[str, Any]]:
    """Machine-readable points for Figure 3-like scale plot from expected references."""
    scale = expected["table_scale_gpvar"]
    rows: list[dict[str, Any]] = []
    for cell, rec in scale.items():
        n_val = int(rec["N"])
        for method, val in rec.items():
            if method == "N":
                continue
            rows.append(
                {
                    "figure": "fig:scale",
                    "cell": cell,
                    "N": n_val,
                    "method": method,
                    "width_mean": float(val[0]),
                    "width_std": float(val[1]),
                }
            )
    rows.sort(key=lambda r: (r["method"], r["N"], r["cell"]))
    return rows


def flatten_run_results(results: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cell, payload in results.items():
        agg = payload.get("aggregate", {})
        methods = agg.get("methods", {})
        for method, stats in methods.items():
            row: dict[str, Any] = {
                "cell": cell,
                "method": method,
            }
            if "error" in stats:
                row["error"] = stats["error"]
                rows.append(row)
                continue
            for key in (
                "coverage",
                "coverage_gap",
                "joint_coverage",
                "mean_width",
                "mean_width_std",
                "mean_halfwidth",
                "mean_halfwidth_std",
                "vhat_m",
                "vhat_m_std",
                "winkler",
                "winkler_std",
                "_time",
                "_n_seeds_ok",
                "_n_seeds_failed",
            ):
                if key in stats:
                    row[key] = stats[key]
            rows.append(row)
    rows.sort(key=lambda r: (r["cell"], r["method"]))
    return rows


def compare_results_to_expected(
    results: dict[str, Any],
    expected: dict[str, Any],
    tolerance_multiplier: float = 2.0,
    absolute_floor: float = 0.10,
) -> list[dict[str, Any]]:
    """Compare run widths to expected table_hero and table_scale_gpvar references."""
    expected_tables = {
        **expected.get("table_hero", {}),
        **expected.get("table_scale_gpvar", {}),
    }
    out: list[dict[str, Any]] = []
    for cell, payload in results.items():
        exp_cell = expected_tables.get(cell)
        if not exp_cell:
            continue
        methods = payload.get("aggregate", {}).get("methods", {})
        for method, stats in methods.items():
            if "error" in stats:
                continue
            exp_stats = _method_stats(exp_cell, method)
            if exp_stats is None:
                continue
            exp_mean, exp_std = exp_stats
            obs_mean = float(stats["mean_width"])
            delta = obs_mean - exp_mean
            tol = max(tolerance_multiplier * exp_std, absolute_floor)
            out.append(
                {
                    "cell": cell,
                    "method": method,
                    "observed_width_mean": obs_mean,
                    "expected_width_mean": exp_mean,
                    "expected_width_std": exp_std,
                    "delta": delta,
                    "tolerance": tol,
                    "within_tolerance": abs(delta) <= tol,
                }
            )
    out.sort(key=lambda r: (r["cell"], r["method"]))
    return out


def compare_logvol_to_expected(
    results: dict[str, Any],
    expected: dict[str, Any],
    absolute_tolerance: float = 0.15,
) -> list[dict[str, Any]]:
    """Compare run Vhat_m to expected table_logvol_main references."""
    exp_table = expected.get("table_logvol_main", {})
    out: list[dict[str, Any]] = []
    for cell, exp_row in exp_table.items():
        if cell.startswith("_"):
            continue
        payload = results.get(cell, {})
        methods = payload.get("aggregate", {}).get("methods", {})
        for method, exp_val in exp_row.items():
            if method == "N":
                continue
            stats = methods.get(method, {})
            if "error" in stats or "vhat_m" not in stats:
                continue
            observed = float(stats["vhat_m"])
            expected_val = float(exp_val)
            delta = observed - expected_val
            out.append(
                {
                    "cell": cell,
                    "method": method,
                    "metric": "vhat_m",
                    "observed_value": observed,
                    "expected_value": expected_val,
                    "delta": delta,
                    "tolerance": float(absolute_tolerance),
                    "within_tolerance": abs(delta) <= absolute_tolerance,
                }
            )
    out.sort(key=lambda r: (r["cell"], r["method"]))
    return out


def compare_diag_to_expected(
    results: dict[str, Any],
    expected: dict[str, Any],
    range_slack: float = 0.01,
    rho_ind_tolerance: float = 0.05,
) -> list[dict[str, Any]]:
    """Compare audited diagnostics against expected tab:diag-main ranges."""
    diag = expected.get("table_diag_main", {})
    out: list[dict[str, Any]] = []
    rho_dg_range = diag.get("rho_dG_range", [float("-inf"), float("inf")])
    rho_dl_range = diag.get("rho_DL_range", [float("-inf"), float("inf")])
    rho_score_range = diag.get("rho_score_range", [float("-inf"), float("inf")])
    tau_int_range = diag.get("tau_int_range", [float("-inf"), float("inf")])
    rho_ind_target = diag.get("rho_ind_approx", None)

    for cell, payload in results.items():
        audit = payload.get("aggregate", {}).get("audit") or {}
        if not audit or "error" in audit:
            continue

        def _add_range(metric: str, bounds: list[float]) -> None:
            if metric not in audit:
                return
            observed = float(audit[metric])
            lo = float(bounds[0]) - range_slack
            hi = float(bounds[1]) + range_slack
            tol = range_slack
            out.append(
                {
                    "cell": cell,
                    "method": "__audit__",
                    "metric": metric,
                    "observed_value": observed,
                    "expected_value": f"[{float(bounds[0])}, {float(bounds[1])}]",
                    "delta": 0.0 if lo <= observed <= hi else min(abs(observed - lo), abs(observed - hi)),
                    "tolerance": tol,
                    "within_tolerance": lo <= observed <= hi,
                }
            )

        _add_range("rho_dG", rho_dg_range)
        _add_range("rho_DL", rho_dl_range)
        _add_range("rho_score", rho_score_range)
        _add_range("tau_int", tau_int_range)
        if rho_ind_target is not None and "rho_ind" in audit:
            observed = float(audit["rho_ind"])
            expected_val = float(rho_ind_target)
            delta = observed - expected_val
            out.append(
                {
                    "cell": cell,
                    "method": "__audit__",
                    "metric": "rho_ind",
                    "observed_value": observed,
                    "expected_value": expected_val,
                    "delta": delta,
                    "tolerance": float(rho_ind_tolerance),
                    "within_tolerance": abs(delta) <= rho_ind_tolerance,
                }
            )
    out.sort(key=lambda r: (r["cell"], r["metric"]))
    return out


def build_main_artifact_parity_rows(
    results: dict[str, Any],
    expected: dict[str, Any],
    artifact: str,
) -> list[dict[str, Any]]:
    artifact = artifact.strip()
    width_rows = compare_results_to_expected(results, expected)
    logvol_rows = compare_logvol_to_expected(results, expected)
    diag_rows = compare_diag_to_expected(results, expected)

    if artifact == "table_hero":
        allowed = set(expected.get("table_hero", {}).keys())
        rows = [r for r in width_rows if r["cell"] in allowed]
        for r in rows:
            r["artifact"] = artifact
            r["metric"] = "mean_width"
            r["observed_value"] = r.pop("observed_width_mean")
            r["expected_value"] = r.pop("expected_width_mean")
            r.pop("expected_width_std", None)
        return rows

    if artifact in {"table_scale", "figure_scale"}:
        allowed = set(expected.get("table_scale_gpvar", {}).keys())
        rows = [r for r in width_rows if r["cell"] in allowed]
        for r in rows:
            r["artifact"] = artifact
            r["metric"] = "mean_width"
            r["observed_value"] = r.pop("observed_width_mean")
            r["expected_value"] = r.pop("expected_width_mean")
            r.pop("expected_width_std", None)
        return rows

    if artifact == "figure_headline":
        allowed = set(expected.get("table_hero", {}).keys()) | set(expected.get("table_scale_gpvar", {}).keys())
        rows = [r for r in width_rows if r["cell"] in allowed]
        for r in rows:
            r["artifact"] = artifact
            r["metric"] = "mean_width"
            r["observed_value"] = r.pop("observed_width_mean")
            r["expected_value"] = r.pop("expected_width_mean")
            r.pop("expected_width_std", None)
        return rows

    if artifact == "table_logvol_main":
        for r in logvol_rows:
            r["artifact"] = artifact
        return logvol_rows

    if artifact == "table_diag_main":
        for r in diag_rows:
            r["artifact"] = artifact
        return diag_rows

    if artifact == "main_body":
        merged: list[dict[str, Any]] = []
        for r in build_main_artifact_parity_rows(results, expected, "table_hero"):
            merged.append(r)
        for r in build_main_artifact_parity_rows(results, expected, "table_scale"):
            merged.append(r)
        for r in build_main_artifact_parity_rows(results, expected, "table_logvol_main"):
            merged.append(r)
        for r in build_main_artifact_parity_rows(results, expected, "table_diag_main"):
            merged.append(r)
        merged.sort(key=lambda r: (str(r.get("artifact")), str(r.get("cell")), str(r.get("method")), str(r.get("metric"))))
        return merged

    raise ValueError(
        f"Unknown artifact {artifact!r}; expected one of "
        "'table_hero', 'table_scale', 'figure_scale', 'figure_headline', "
        "'table_logvol_main', 'table_diag_main', 'main_body'."
    )


def summarize_parity_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    passed = sum(1 for r in rows if bool(r.get("within_tolerance")))
    failed = total - passed
    by_artifact: dict[str, dict[str, int]] = {}
    for row in rows:
        artifact = str(row.get("artifact", "unknown"))
        bucket = by_artifact.setdefault(artifact, {"total": 0, "passed": 0, "failed": 0})
        bucket["total"] += 1
        if bool(row.get("within_tolerance")):
            bucket["passed"] += 1
        else:
            bucket["failed"] += 1
    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": float(passed / total) if total else 0.0,
        "all_passed": failed == 0 and total > 0,
        "by_artifact": by_artifact,
    }


def build_inventory_index(inventory: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for art in inventory.get("artifacts", []):
        rows.append(
            {
                "artifact_id": art.get("artifact_id"),
                "artifact_type": art.get("artifact_type"),
                "paper_location": art.get("paper_location"),
                "status": art.get("status"),
                "config_id": art.get("config_id"),
            }
        )
    return rows


def build_artifact_value_rows(inventory: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for art in inventory.get("artifacts", []):
        art_id = art.get("artifact_id")
        values = art.get("all_reported_numbers")
        if isinstance(values, dict):
            rows.append(
                {
                    "artifact_id": art_id,
                    "value_key": "__json__",
                    "value_json": json.dumps(values, sort_keys=True),
                }
            )
        elif values is not None:
            rows.append(
                {
                    "artifact_id": art_id,
                    "value_key": "__text__",
                    "value_json": str(values),
                }
            )
    return rows
