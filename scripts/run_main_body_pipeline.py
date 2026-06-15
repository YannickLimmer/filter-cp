#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a practical main-body pipeline and build artifact files.")
    p.add_argument("--python", default="python", help="Python executable to use.")
    p.add_argument("--cache", default=str(ROOT / ".cache"), help="Cache directory for dataset downloads.")
    p.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    p.add_argument("--include-vendored", action="store_true", help="Include vendored baselines in repr runs.")
    p.add_argument("--run-metrla", action="store_true", help="Run METR-LA main-body cell.")
    p.add_argument("--run-pems", action="store_true", help="Run PEMS-BAY main-body cell.")
    p.add_argument("--run-synthetic", action="store_true", help="Run synthetic smoke sanity cell.")
    p.add_argument("--audit", action="store_true", help="Enable audit outputs.")
    return p.parse_args()


def run(cmd: list[str], env: dict[str, str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def main() -> int:
    args = parse_args()
    targets: list[tuple[str, str]] = []
    if args.run_synthetic:
        targets.append(("synthetic", str(ROOT / "results_synthetic_pipeline.json")))
    if args.run_metrla:
        targets.append(("metrla", str(ROOT / "results_metrla_pipeline.json")))
    if args.run_pems:
        targets.append(("pems_bay", str(ROOT / "results_pems_bay_pipeline.json")))

    if not targets:
        targets = [
            ("synthetic", str(ROOT / "results_synthetic_pipeline.json")),
            ("metrla", str(ROOT / "results_metrla_pipeline.json")),
        ]

    env = dict(**__import__("os").environ)
    env["CAPFACTOR_REPR_CACHE"] = args.cache

    for cell, out in targets:
        cmd = [
            args.python,
            str(ROOT / "repr" / "reproduce.py"),
            "--mode",
            args.mode,
            "--cells",
            cell,
            "--output",
            out,
        ]
        if not args.include_vendored:
            cmd.append("--no-vendored-baselines")
        if args.audit:
            cmd.append("--audit")
        if cell != "synthetic":
            cmd.append("--check-expected")
        run(cmd, env)

    results_files = [out for _, out in targets]
    build_cmd = [
        args.python,
        str(ROOT / "scripts" / "build_main_body_artifacts.py"),
        "--outdir",
        str(ROOT / "artifacts" / "main_body"),
        "--results",
        *results_files,
    ]
    run(build_cmd, env)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
