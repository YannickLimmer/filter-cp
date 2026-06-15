# Filter-CP Reproducibility Repository

This repository contains source code to reproduce the numeric outputs used by the paper `filter_cp.pdf`:

- table metrics (coverage, joint coverage, width, winkler),
- diagnostic metrics (`rho_*`, `tau_int`, log-volume),
- machine-readable JSON outputs for downstream plotting.

The entry point is:

- `repr/reproduce.py`

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r repr/requirements.txt

# Fast synthetic smoke check
python repr/reproduce.py --mode smoke --cells synthetic --no-vendored-baselines
```

## Test

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

## Output

Runs write machine-readable result files:

- `results_<cells>.json`

These files are the canonical raw numeric outputs used for tables and plot data generation.

## Build Main-Body Artifact Files

Generate machine-readable table and figure data files from the canonical expected references and optional run outputs:

```bash
python scripts/build_main_body_artifacts.py \
  --expected repr/expected.json \
  --results results_metrla_smoke.json results_synthetic_smoke.json \
  --outdir artifacts/main_body
```

Outputs include:

- `table_hero_expected.csv`
- `table_scale_expected.csv`
- `table_logvol_main_expected.csv`
- `table_diag_main_expected.json`
- `inline_claims_expected.csv`
- `figure_headline_points_expected.csv`
- `figure_scale_points_expected.csv`
- `run_metrics_long.csv` (if run outputs are provided)
- `run_vs_expected_width_deltas.csv` (if run outputs are provided)

## Build Full Artifact Inventory Files

```bash
python scripts/build_full_inventory_artifacts.py \
  --inventory resources/paper_inventory.json \
  --outdir artifacts/full_inventory
```

Outputs:

- `paper_inventory.json` (canonical artifact inventory)
- `artifact_index.csv` (one row per paper artifact)
- `artifact_values.csv` (reported numeric payloads by artifact)

## End-to-End Sample Pipeline

```bash
python scripts/run_main_body_pipeline.py --mode smoke --run-synthetic --run-metrla
```

## Strict Lock and Deterministic Runtime

```bash
# Generate lock from active environment (exact versions + content hashes)
python scripts/generate_runtime_lock.py

# Verify lock before long runs
python scripts/verify_runtime_lock.py
```

Deterministic execution profile:

- `config/deterministic_execution.json`

Hardware/runtime envelope and reproducibility pipeline docs:

- `docs/runtime_envelope.md`

## One-Command Artifact Parity Pipelines

Each command runs the experiment cell set, writes raw per-seed outputs,
and emits pass/fail parity reports against `repr/expected.json`.

```bash
# Main-body aggregate (tables + figure numerics)
python scripts/run_artifact_parity.py --artifact main_body --mode full --n-seeds 10 --strict

# Individual main artifacts
python scripts/run_artifact_parity.py --artifact table_hero --mode full --n-seeds 10 --strict
python scripts/run_artifact_parity.py --artifact table_scale --mode full --n-seeds 10 --strict
python scripts/run_artifact_parity.py --artifact table_logvol_main --mode full --n-seeds 10 --strict
python scripts/run_artifact_parity.py --artifact table_diag_main --mode full --n-seeds 10 --strict
python scripts/run_artifact_parity.py --artifact figure_headline --mode full --n-seeds 10 --strict
python scripts/run_artifact_parity.py --artifact figure_scale --mode full --n-seeds 10 --strict
```

## Heavier Verification Profiles

Examples used for stronger checks beyond minimal smoke:

```bash
# Full-profile synthetic multi-seed run + audit
CAPFACTOR_REPR_CACHE=.cache \
python repr/reproduce.py \
  --mode full --cells synthetic --n-seeds 3 --audit \
  --no-vendored-baselines \
  --output results/results_synthetic_full_n3_audit.json

# Full-profile METR-LA run (single seed) + audit
CAPFACTOR_REPR_CACHE=.cache \
python repr/reproduce.py \
  --mode full --cells metrla --n-seeds 1 --audit \
  --no-vendored-baselines --check-expected \
  --output results/results_metrla_full_n1_audit.json

# Bounded heavier PEMS-BAY run (multi-seed smoke) + audit
CAPFACTOR_REPR_CACHE=.cache \
python repr/reproduce.py \
  --mode smoke --cells pems_bay --n-seeds 2 --audit \
  --no-vendored-baselines --check-expected \
  --output results/results_pems_bay_smoke_n2_audit.json
```
