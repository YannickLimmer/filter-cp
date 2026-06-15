#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from filter_cp_repro.artifact_builders import (  # noqa: E402
    build_main_artifact_parity_rows,
    load_expected,
    load_results,
    summarize_parity_rows,
)


ARTIFACT_TO_CELLS = {
    "table_hero": ["hero"],
    "table_logvol_main": ["hero"],
    "table_diag_main": ["hero"],
    "table_scale": ["gpvar_scale"],
    "figure_scale": ["gpvar_scale"],
    "figure_headline": ["gpvar_scale"],
    "main_body": ["gpvar_scale"],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="One-command artifact run + parity report pipeline.")
    p.add_argument(
        "--artifact",
        required=True,
        choices=sorted(ARTIFACT_TO_CELLS.keys()),
        help="Main artifact target.",
    )
    p.add_argument("--python", default="python", help="Python executable.")
    p.add_argument("--cache", default=str(ROOT / ".cache"), help="Dataset cache directory.")
    p.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    p.add_argument("--n-seeds", type=int, default=None, help="Override seed count.")
    p.add_argument("--seed-start", type=int, default=None, help="Override seed start.")
    p.add_argument("--include-vendored", action="store_true", help="Include vendored baselines.")
    p.add_argument(
        "--deterministic-config",
        type=Path,
        default=ROOT / "config" / "deterministic_execution.json",
        help="Deterministic execution profile.",
    )
    p.add_argument(
        "--expected",
        type=Path,
        default=ROOT / "repr" / "expected.json",
        help="Expected reference json.",
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        default=ROOT / "results",
        help="Directory for raw results outputs.",
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=ROOT / "artifacts" / "parity",
        help="Directory for parity outputs.",
    )
    p.add_argument("--strict", action="store_true", help="Exit nonzero if any parity row fails.")
    return p.parse_args()


def load_deterministic_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def apply_deterministic_env(base_env: dict[str, str], cfg: dict[str, Any]) -> dict[str, str]:
    env = dict(base_env)
    env["PYTHONHASHSEED"] = str(cfg.get("python_hash_seed", 0))
    for group in ("thread_env", "torch_env"):
        for k, v in (cfg.get(group) or {}).items():
            env[str(k)] = str(v)
    return env


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


def flatten_per_seed(results: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cell, payload in results.items():
        seeds = payload.get("seeds", [])
        per_seed = payload.get("per_seed", [])
        for idx, seed_payload in enumerate(per_seed):
            seed = seeds[idx] if idx < len(seeds) else idx
            methods = seed_payload.get("methods", {})
            for method, stats in methods.items():
                row: dict[str, Any] = {
                    "cell": cell,
                    "seed": int(seed),
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
                    "mean_halfwidth",
                    "vhat_m",
                    "winkler",
                    "_time",
                ):
                    if key in stats:
                        row[key] = stats[key]
                rows.append(row)
            audit = seed_payload.get("audit") or {}
            if audit and "error" not in audit:
                for metric in ("rho_dG", "rho_DL", "rho_score", "rho_ind", "tau_int"):
                    if metric in audit:
                        rows.append(
                            {
                                "cell": cell,
                                "seed": int(seed),
                                "method": "__audit__",
                                "metric": metric,
                                "value": float(audit[metric]),
                            }
                        )
    rows.sort(key=lambda r: (str(r.get("cell")), int(r.get("seed", 0)), str(r.get("method"))))
    return rows


def runtime_envelope() -> dict[str, Any]:
    env = {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
    }
    try:
        import torch  # type: ignore

        env["torch"] = {
            "version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
            "num_threads": int(torch.get_num_threads()),
        }
    except Exception as exc:
        env["torch"] = {"error": f"{type(exc).__name__}: {exc}"}
    return env


def main() -> int:
    args = parse_args()
    det_cfg = load_deterministic_config(args.deterministic_config)

    reproduce_defaults = det_cfg.get("reproduce_defaults", {})
    audit = bool(reproduce_defaults.get("audit", True))
    include_vendored = bool(reproduce_defaults.get("include_vendored_baselines", False))

    if args.include_vendored:
        include_vendored = True

    cells = ARTIFACT_TO_CELLS[args.artifact]
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.outdir.mkdir(parents=True, exist_ok=True)

    n_seeds = args.n_seeds
    if n_seeds is None and args.mode == "full":
        n_seeds = 10

    seed_start = args.seed_start
    if seed_start is None:
        seed_start = int(reproduce_defaults.get("seed_start", 0))

    run_name = f"{args.artifact}_{args.mode}_n{n_seeds if n_seeds is not None else 'default'}"
    result_path = args.results_dir / f"results_{run_name}.json"

    cmd = [
        args.python,
        str(ROOT / "repr" / "reproduce.py"),
        "--mode",
        args.mode,
        "--cells",
        *cells,
        "--output",
        str(result_path),
        "--check-expected",
    ]
    if n_seeds is not None:
        cmd.extend(["--n-seeds", str(n_seeds)])
    if seed_start is not None:
        cmd.extend(["--seed-start", str(seed_start)])
    if not include_vendored:
        cmd.append("--no-vendored-baselines")
    if audit:
        cmd.append("--audit")

    env = apply_deterministic_env(os.environ, det_cfg)
    env["CAPFACTOR_REPR_CACHE"] = args.cache

    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)

    expected = load_expected(args.expected)
    results = load_results(result_path)
    parity_rows = build_main_artifact_parity_rows(results, expected, args.artifact)
    summary = summarize_parity_rows(parity_rows)

    per_seed_rows = flatten_per_seed(results)

    parity_csv = args.outdir / f"parity_{run_name}.csv"
    parity_json = args.outdir / f"parity_{run_name}.json"
    per_seed_csv = args.outdir / f"per_seed_{run_name}.csv"
    manifest_json = args.outdir / f"manifest_{run_name}.json"

    write_csv(parity_csv, parity_rows)
    write_csv(per_seed_csv, per_seed_rows)
    parity_json.write_text(json.dumps({"summary": summary, "rows": parity_rows}, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact": args.artifact,
        "mode": args.mode,
        "cells": cells,
        "n_seeds": n_seeds,
        "seed_start": seed_start,
        "result_file": str(result_path),
        "parity_files": {
            "csv": str(parity_csv),
            "json": str(parity_json),
            "per_seed_csv": str(per_seed_csv),
        },
        "parity_summary": summary,
        "deterministic_config": json.loads(args.deterministic_config.read_text(encoding="utf-8")),
        "runtime_envelope": runtime_envelope(),
        "environment_snapshot": {
            "PYTHONHASHSEED": env.get("PYTHONHASHSEED"),
            "OMP_NUM_THREADS": env.get("OMP_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": env.get("OPENBLAS_NUM_THREADS"),
            "MKL_NUM_THREADS": env.get("MKL_NUM_THREADS"),
            "VECLIB_MAXIMUM_THREADS": env.get("VECLIB_MAXIMUM_THREADS"),
            "NUMEXPR_NUM_THREADS": env.get("NUMEXPR_NUM_THREADS"),
            "CUBLAS_WORKSPACE_CONFIG": env.get("CUBLAS_WORKSPACE_CONFIG"),
            "PYTORCH_ENABLE_MPS_FALLBACK": env.get("PYTORCH_ENABLE_MPS_FALLBACK"),
            "CAPFACTOR_REPR_CACHE": env.get("CAPFACTOR_REPR_CACHE"),
        },
    }
    manifest_json.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Wrote {parity_csv}")
    print(f"Wrote {parity_json}")
    print(f"Wrote {per_seed_csv}")
    print(f"Wrote {manifest_json}")
    print(f"Parity: total={summary['total']} passed={summary['passed']} failed={summary['failed']}")

    if args.strict and not summary["all_passed"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
