# Graph-Filtered Conformal Prediction Reproducibility Suite

This folder is a self-contained reproducibility artefact for the
paper's main empirical claim: a learned graph-convolutional GRU
filter (the **GNF** row) gives sharper at-target ellipsoids than every
non-filter baseline on the moderate-N graph-native traffic cells, and
the **GNF + ACI** wrap restores at-target coverage at the full-graph
scale.

Every datapoint of the paper's main section can be reproduced from
this folder, with no external project imports and no credentials:

* **Table tab:hero** — moderate-N METR-LA-20 / PEMS-BAY-50, 7 methods.
* **Table tab:scale** — GPVAR scale sweep on N ∈ {20, 207, 50, 325},
  7 methods.
* **Table tab:logvol-main** — per-coord normalised log-volume on the
  hero cells (Theorem thm:logvol).
* **Table tab:diag-main** — empirical contraction-rate diagnostics
  ``rho_dG`` / ``rho_DL`` / ``rho_score`` / ``rho_ind`` / ``tau_int``
  (``--audit`` flag).

Method names follow the paper-facing registry used in this
repository (`GNF`, `KalmanFCP`, `StaticCGIF`, `DiagGRU`,
`GCNrankzero` and related baselines).

## Layout

```
repr/
├── reproduce.py        # top-level CLI driver
├── expected.json       # paper Table tab:hero / tab:scale numbers
├── requirements.txt    # pinned dependencies
├── README.md
└── _lib/               # self-contained implementation package
    ├── core.py            # quantiles, splits, metrics, seeding
    ├── graph.py           # normalised shift, busiest-k, spectral, k-NN
    ├── backbone.py        # GraphPolyVAR forecaster
    ├── filters.py         # GraphLGSSM / NeuralDiagGaussianFilter
    │                      # / GraphNeuralSSMFilter
    ├── methods.py         # 13 methods by publication name
    ├── _lib_helpers.py    # factor + per-community fit primitives
    ├── audit.py           # rho_dG / rho_DL / rho_score / log-volume
    ├── vendored_copula.py # CopulaCPTS (Sun & Yu, ICLR 2024)
    ├── vendored_spci.py   # MultiDimSPCI (Xu & Xie, ICML 2024)
    ├── _copulacpts_*.py   # CopulaCPTS upstream sources
    ├── _multidimspci_*.py # MultiDimSPCI upstream sources
    └── data/              # one loader per primary cell
        ├── metrla.py / pems_bay.py        — DCRNN HF mirrors
        ├── aqi.py / elec.py               — UCI archives
        ├── ett.py / solar.py / jena.py / loop_seattle.py
        │                                    — GiftEval HF mirror
        ├── largest.py                     — LargeST GBA (HF, ~6.9 GB)
        └── synthetic.py                   — offline GraphPolyVAR
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Smoke test on the offline synthetic cell (a few minutes on CPU,
# under a minute on a single GPU).
python reproduce.py --mode smoke --cells synthetic

# Smoke METR-LA-20 with reported-number cross-check.
python reproduce.py --mode smoke --cells metrla --check-expected

# Full Table tab:hero reproduction.
python reproduce.py --mode full --cells hero --check-expected

# Full Table tab:scale reproduction (GPVAR backbone, N up to 325).
python reproduce.py --mode full --cells gpvar_scale --check-expected

# Online-calibration assumption-audit / rho-diagnostics row.
python reproduce.py --mode full --cells hero --audit

# Every cell in the paper.
python reproduce.py --mode full --cells all
```

Results are printed as a Markdown table per cell and also written to
`results_<cells>.json` with per-seed and aggregate numbers.

## What is reproduced

### Datasets

| Cell | N | Source | Subgraph |
|---|---:|---|---|
| `metrla`        | 20   | `witgaw/METR-LA` (HF) | busiest-20 |
| `metrla_full`   | 207  | `witgaw/METR-LA` (HF) | full graph |
| `pems_bay`      | 50   | `witgaw/PEMS-BAY` (HF) | busiest-50 |
| `pems_bay_full` | 325  | `witgaw/PEMS-BAY` (HF) | full graph |
| `aqi`           | 12   | UCI 501 | full graph |
| `elec`          | 20   | UCI 321 | first 20 clients |
| `ett`           | 7    | `Salesforce/GiftEval` (HF) | full |
| `solar`         | 20   | `Salesforce/GiftEval` (HF) | busiest-20 |
| `jena`          | 18   | `Salesforce/GiftEval` (HF) | busiest-18 |
| `loop_seattle`  | 20   | `Salesforce/GiftEval` (HF) | busiest-20 |
| `largest`       | 2327 | LargeST GBA (HF) | district 4 |
| `synthetic`     | 20   | offline GraphPolyVAR | sanity check |

The shorthand `--cells hero` selects `metrla, pems_bay`;
`--cells gpvar_scale` selects all four METR-LA / PEMS-BAY cells of
Table tab:scale; `--cells all` runs every dataset above.

### Methods (publication names)

| Publication name | Role |
|---|---|
| `GNF` (**hero**) | graph-conv GRU + structured Σ_t = diag + LL^T |
| `GNF + ACI` (**deployed**) | `GNF` with Algorithm 1 (Gibbs–Candès) ACI wrap |
| `KalmanFCP` | linear graph LG-SSM + filtered CP (honest negative) |
| `DiagGRU` | plain GRU + diagonal Σ_t (no graph mixing) |
| `GCNrankzero` | GCN-GRU with rank-0 covariance head (graph only) |
| `StaticCGIF` | static-Σ̂ Mahalanobis CP (reference) |
| `FactorCGIF` (r = 4) | factor + residual-max |
| `AgACIGroupCGIF` | AgACI expert-aggregation on max-over-communities |
| `ACIPerGroupFactorCGIF` | per-community factor + residual-max ACI |
| `EWMACovCGIF` | causal time-varying Σ_t (EWMA) |
| `RollingCovCGIF` | causal time-varying Σ_t (rolling window) |
| `CopulaCPTS` | Wang & Yu, ICLR 2024 — vendored upstream |
| `MultiDimSPCI` | Xu & Xie, ICML 2024 — vendored upstream |

The two vendored upstream rows can be skipped with
`--no-vendored-baselines` (helpful for reviewers without scikit-learn
or who want to avoid the CopulaCPTS torch optimisation).

### Diagnostics (`--audit`)

`--audit` adds a row of empirical contraction-rate diagnostics
mirroring Table tab:diag-main:

* `rho_dG` — emitted-law contraction (Theorem thm:obs-contract).
* `rho_DL` — finite-horizon observability proxy
  (Assumption ass:obs (O3)).
* `rho_score` — score-CDF forgetting rate (Theorem thm:C);
  paper reports `[0.140, 0.153]` across cells.
* `rho_ind`, `tau_int` — block-mixing indicators
  (Theorem thm:learned-validity (a), Assumption ass:bernstein).
* `Vhat_m_static` — per-coord normalised log-volume of the
  StaticCGIF ellipsoid (Theorem thm:logvol).

The diagnostics are computed post-hoc from the `GNF + ACI` filter's
score stream, so they cost only one extra filter run per seed.

### Expected-numbers cross-check (`--check-expected`)

`expected.json` carries the paper's `(mean, std)` ellipsoid widths
from Table tab:hero and Table tab:scale, plus Table tab:logvol-main
and the rho-diagnostic ranges of Table tab:diag-main, all keyed by
publication name.  When `--check-expected` is on, each cell's table
prints an extra **Δ** column comparing your run against the
reference, with a `✓` marker when the absolute delta is within
`2 × seed-std` (Monte-Carlo tolerance) and `!` otherwise.  Smoke
mode is expected to differ — the marker exists so 10-seed full-mode
runs surface any divergence cleanly.

## Reviewer notes

* **No credentials, no proprietary data.**  Every loader pulls from a
  public source — UCI (HTTP) or Hugging Face (`huggingface_hub`).
  LargeST GBA is the largest payload (~6.9 GB H5 on first call).
* **No external project imports.**  The suite is self-contained under
  `repr/` and does not depend on non-public code.
* **Caching.**  All raw downloads + reshape products are written to
  `$CAPFACTOR_REPR_CACHE` (default `~/.cache/capfactor_repr/`).
* **CPU vs GPU.**  PyTorch is a hard requirement (the three filter
  rows are core methods, not optional baselines).  On CPU a 10-seed
  full run on the two hero cells takes a couple of hours; a single
  CUDA GPU brings it to under an hour.  No multi-GPU parallelism is
  needed.
* **Validity regime.**  `GNF + ACI` carries the paper's Theorem
  thm:learned-validity (a) certificate (online long-run joint-miss
  calibration with explicit dependence on `rho_ind` and `tau_int`),
  *not* a static split-CP guarantee — see the audit row's
  `rho_ind` / `tau_int` for the threshold-mixing regime each cell
  occupies.
* **Scope.**  This bundle covers Tables tab:hero, tab:scale,
  tab:logvol-main, and tab:diag-main of the paper's main section.
  The multi-backbone crossover (Table tab:backbone-crossover) uses
  modern forecasters (DLinear / NLinear / PatchTST / iTransformer /
  STGNN); their full implementations and the rho-validation
  appendix tables are outside the scope of this numeric verification bundle.

## Command-line reference

```text
usage: reproduce.py [-h] [--mode {smoke,full}]
                    [--cells CELLS [CELLS ...]] [--n-seeds N]
                    [--seed-start S] [--alpha A]
                    [--subgraph-size N] [--max-test-steps N]
                    [--n-groups N]
                    [--gnf-hidden D] [--gnf-low-rank R] [--gnf-epochs E]
                    [--diaggru-hidden D] [--diaggru-epochs E]
                    [--kalman-rho R] [--factor-rank R]
                    [--aci-gamma G] [--aci-window W]
                    [--cond-half-life HL] [--cond-window W]
                    [--warmup-steps W]
                    [--no-vendored-baselines] [--audit] [--check-expected]
                    [--copula-epochs N] [--output PATH]
```

* `--mode smoke|full` — mode-specific defaults for seeds / test
  length / GRU epochs / copula epochs.
* `--cells` — one or more dataset names plus the shorthands
  `all` (every cell), `primary` (every cell except LargeST),
  `hero` (METR-LA-20 + PEMS-BAY-50, Table tab:hero),
  `gpvar_scale` (the four cells of Table tab:scale).
* `--audit` — adds the contraction-rate diagnostic row per cell.
* `--check-expected` — adds a Δ column comparing against
  `expected.json`.
* `--no-vendored-baselines` — skips CopulaCPTS / MultiDimSPCI rows.
* `--gnf-hidden / --gnf-low-rank / --gnf-epochs / --aci-gamma / --aci-window` —
  GNF / GNF+ACI hyperparameters.  Defaults match the paper:
  `hidden = 32`, `low_rank = 4`, `epochs = 60`, `γ = 5·10⁻³`,
  ACI rolling window = 400.
* `--kalman-rho` — GraphLGSSM spectral-radius scale (default 0.8).
