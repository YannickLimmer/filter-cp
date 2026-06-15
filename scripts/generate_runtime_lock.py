#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as md
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path


def _sha256_distribution(dist: md.Distribution) -> str:
    h = hashlib.sha256()
    files = dist.files or []
    for rel in sorted(files, key=lambda p: str(p)):
        path = dist.locate_file(rel)
        if not path.is_file():
            continue
        h.update(str(rel).encode("utf-8"))
        h.update(b"\0")
        try:
            h.update(path.read_bytes())
        except OSError:
            continue
        h.update(b"\0")
    return h.hexdigest()


def _normalise_name(name: str) -> str:
    return name.lower().replace("_", "-")


def _parse_requirements(path: Path) -> list[str]:
    names: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        token = raw.split("#", 1)[0].strip()
        if ">=" in token:
            name = token.split(">=", 1)[0].strip()
        elif "==" in token:
            name = token.split("==", 1)[0].strip()
        else:
            name = token
        names.append(_normalise_name(name))
    return names


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate strict runtime lock with exact versions + content hashes.")
    p.add_argument(
        "--requirements",
        type=Path,
        default=Path("repr/requirements.txt"),
        help="Input requirements file used by repr runs.",
    )
    p.add_argument(
        "--out-json",
        type=Path,
        default=Path("lock/python-runtime.lock.json"),
        help="Output JSON lock file.",
    )
    p.add_argument(
        "--out-requirements",
        type=Path,
        default=Path("lock/requirements-exact.txt"),
        help="Pinned requirements output.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    req_names = set(_parse_requirements(args.requirements))
    all_dists = list(md.distributions())
    dist_by_name = {}
    for dist in all_dists:
        name = dist.metadata.get("Name") or dist.metadata.get("Summary") or ""
        if not name:
            continue
        dist_by_name[_normalise_name(name)] = dist

    locked_runtime: list[dict[str, str]] = []
    missing: list[str] = []
    for req_name in sorted(req_names):
        dist = dist_by_name.get(req_name)
        if dist is None:
            missing.append(req_name)
            continue
        locked_runtime.append(
            {
                "name": req_name,
                "version": dist.version,
                "sha256": _sha256_distribution(dist),
            }
        )

    if missing:
        raise RuntimeError(f"Missing required packages in active environment: {missing}")

    all_packages: list[dict[str, str]] = []
    for dist in sorted(all_dists, key=lambda d: _normalise_name(d.metadata.get("Name", ""))):
        name = dist.metadata.get("Name")
        if not name:
            continue
        all_packages.append(
            {
                "name": _normalise_name(name),
                "version": dist.version,
                "sha256": _sha256_distribution(dist),
            }
        )

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": Path(sys.executable).name,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "requirements_source": str(args.requirements),
        "runtime_requirements": locked_runtime,
        "all_installed_packages": all_packages,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Auto-generated exact requirement lock.",
        "# Verify with: python scripts/verify_runtime_lock.py",
        "",
    ]
    for item in locked_runtime:
        lines.append(f"{item['name']}=={item['version']}  # sha256:{item['sha256']}")
    args.out_requirements.parent.mkdir(parents=True, exist_ok=True)
    args.out_requirements.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_requirements}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
