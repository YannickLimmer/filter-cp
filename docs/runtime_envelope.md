# Runtime Envelope and Determinism

## Deterministic Profile

`config/deterministic_execution.json` defines deterministic defaults:

- `PYTHONHASHSEED=0`
- thread caps (`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `VECLIB_MAXIMUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`)
- torch reproducibility env (`CUBLAS_WORKSPACE_CONFIG=:4096:8`, `PYTORCH_ENABLE_MPS_FALLBACK=1`)
- default seed policy (`seed_start=0`)

All one-command pipelines consume this config automatically.

## Strict Lock

- Generate exact version + hash lock from the active environment:
  - `python scripts/generate_runtime_lock.py`
- Verify lock before execution:
  - `python scripts/verify_runtime_lock.py`

Generated files:

- `lock/python-runtime.lock.json`
- `lock/requirements-exact.txt`

## One-Command Main Artifact Pipelines

Each command regenerates raw per-seed outputs, parity rows, and a runtime manifest.

- Table 2 (`tab:hero`):
  - `python scripts/run_artifact_parity.py --artifact table_hero --mode full --n-seeds 10 --strict`
- Table 3 (`tab:logvol-main`):
  - `python scripts/run_artifact_parity.py --artifact table_logvol_main --mode full --n-seeds 10 --strict`
- Table 4 (`tab:diag-main`):
  - `python scripts/run_artifact_parity.py --artifact table_diag_main --mode full --n-seeds 10 --strict`
- Table 9 / Figure 3 scale:
  - `python scripts/run_artifact_parity.py --artifact table_scale --mode full --n-seeds 10 --strict`
  - `python scripts/run_artifact_parity.py --artifact figure_scale --mode full --n-seeds 10 --strict`
- Figure 2 headline:
  - `python scripts/run_artifact_parity.py --artifact figure_headline --mode full --n-seeds 10 --strict`

Aggregate main-body run:

- `python scripts/run_artifact_parity.py --artifact main_body --mode full --n-seeds 10 --strict`

## Output Conventions

Outputs are written under `results/` and `artifacts/parity/`:

- `results_<artifact>_<mode>_n<k>.json` (raw run output, includes per-seed payloads)
- `per_seed_<artifact>_<mode>_n<k>.csv` (seed x method rows)
- `parity_<artifact>_<mode>_n<k>.csv` and `.json` (pass/fail rows vs reported values)
- `manifest_<artifact>_<mode>_n<k>.json` (runtime envelope + parity summary)
