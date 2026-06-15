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

from filter_cp_repro.artifact_builders import (  # noqa: E402
    build_artifact_value_rows,
    build_inventory_index,
    load_inventory,
)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({k for r in rows for k in r.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Emit machine-readable artifact catalog from paper inventory.")
    p.add_argument(
        "--inventory",
        type=Path,
        default=ROOT / "resources" / "paper_inventory.json",
        help="Path to paper inventory json.",
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=ROOT / "artifacts" / "full_inventory",
        help="Output directory.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    inv = load_inventory(args.inventory)

    write_json(args.outdir / "paper_inventory.json", inv)
    write_csv(args.outdir / "artifact_index.csv", build_inventory_index(inv))
    write_csv(args.outdir / "artifact_values.csv", build_artifact_value_rows(inv))

    print(f"Wrote full inventory artifacts to {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
