#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as md
import json
from pathlib import Path


def _normalise_name(name: str) -> str:
    return name.lower().replace("_", "-")


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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Verify exact versions and hashes from runtime lock.")
    p.add_argument(
        "--lock",
        type=Path,
        default=Path("lock/python-runtime.lock.json"),
        help="Lock JSON produced by generate_runtime_lock.py",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    lock = json.loads(args.lock.read_text(encoding="utf-8"))

    dist_by_name = {}
    for dist in md.distributions():
        name = dist.metadata.get("Name")
        if not name:
            continue
        dist_by_name[_normalise_name(name)] = dist

    failures: list[str] = []
    for req in lock.get("runtime_requirements", []):
        name = _normalise_name(req["name"])
        expected_version = req["version"]
        expected_hash = req["sha256"]
        dist = dist_by_name.get(name)
        if dist is None:
            failures.append(f"{name}: missing package")
            continue
        if dist.version != expected_version:
            failures.append(f"{name}: version mismatch (expected {expected_version}, found {dist.version})")
            continue
        observed_hash = _sha256_distribution(dist)
        if observed_hash != expected_hash:
            failures.append(f"{name}: hash mismatch (expected {expected_hash}, found {observed_hash})")

    if failures:
        print("LOCK VERIFICATION FAILED")
        for line in failures:
            print(f"- {line}")
        return 1

    print("LOCK VERIFICATION PASSED")
    print(f"Checked {len(lock.get('runtime_requirements', []))} runtime requirement packages.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
