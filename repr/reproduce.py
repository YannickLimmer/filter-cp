"""Reproducibility driver for graph-filtered conformal prediction.

Top-level CLI for reproducing every datapoint of the paper's main
section: Table tab:hero (moderate-N graph-native), Table tab:scale
(GPVAR scale sweep on full-graph METR-LA / PEMS-BAY), Table
tab:logvol-main (normalised log-volume on the headline cells), and
Table tab:diag-main (empirical contraction-rate diagnostics).

Methods follow the paper's publication-facing names (``GNF``,
``KalmanFCP``, ``StaticCGIF``, ``DiagGRU``, ``GCNrankzero``).
All thirteen methods run unconditionally on
every invocation; ``--no-vendored-baselines`` is the only opt-out and
only skips the two upstream-vendored joint-CP rows (CopulaCPTS,
MultiDimSPCI).  PyTorch is a hard requirement (the three filter rows
are core methods).

Coverage
--------
Datasets (``--cells all``):

    metrla         — METR-LA busiest-20 subgraph (Table tab:hero)
    metrla_full    — METR-LA full graph N=207 (Table tab:scale)
    pems_bay       — PEMS-BAY busiest-50 subgraph (Table tab:hero)
    pems_bay_full  — PEMS-BAY full graph N=325 (Table tab:scale)
    aqi / elec / ett / solar / jena / loop_seattle
                   — six additional correlated-sensor benchmarks
                     used by Table tab:multibackbone-nontraffic and
                     the ρ̂-validation Table tab:rho-full
    largest        — LargeST-GBA scale stress test (N≈2327)
    synthetic      — offline GraphPolyVAR fallback (no network)

Methods (publication names; all run on every invocation):

    GNF                      graph-conv GRU + structured Σ_t  (hero)
    GNF + ACI                + Algorithm 1 ACI wrap          (deployed)
    KalmanFCP                linear graph LG-SSM             (honest-neg)
    DiagGRU                  plain GRU + diag Σ_t             (no graph)
    GCNrankzero              GCN-GRU with rank-0 cov head    (graph-only)
    StaticCGIF               static Σ̂ Mahalanobis CP         (reference)
    FactorCGIF (r=4)         factor + residual-max
    AgACIGroupCGIF           AgACI on max-over-communities
    ACIPerGroupFactorCGIF    per-community factor + res-max ACI
    EWMACovCGIF              causal EWMA Σ_t
    RollingCovCGIF           causal rolling-window Σ_t
    CopulaCPTS               vendored upstream (Wang & Yu, ICLR 2024)
    MultiDimSPCI             vendored upstream (Xu & Xie, ICML 2024)

Diagnostics (``--audit``)
-------------------------
Empirical contraction-rate diagnostics from Table tab:diag-main:
``rho_dG`` (emitted-law contraction), ``rho_DL`` (observability proxy),
``rho_score`` (CDF-forgetting rate), ``rho_ind`` and ``tau_int``
(threshold-mixing), plus the per-coord normalised log-volume
``Vhat_m`` of Theorem thm:logvol on the headline cells.

Modes
-----
::

    # Smoke test on the offline synthetic cell (a few minutes on CPU,
    # well under a minute on a GPU — the GRU filters dominate).
    python reproduce.py --mode smoke --cells synthetic

    # Smoke METR-LA, paper-number cross-check (~5 min CPU).
    python reproduce.py --mode smoke --cells metrla --check-expected

    # Full Table tab:hero reproduction (10 seeds × 2 cells × all
    # methods).  Hours on CPU; ~30-60 min on a single GPU.
    python reproduce.py --mode full --cells hero --check-expected

    # Full Table tab:scale (4 cells, GPVAR backbone).
    python reproduce.py --mode full --cells gpvar_scale --check-expected

    # All eleven cells × every method.
    python reproduce.py --mode full --cells all

Reviewer notes
--------------
* ``--check-expected`` prints a Δ column per row vs the reference
  numbers in ``expected.json``; ``✓`` marks deltas inside
  ``2 × seed-std``.
* ``--audit`` adds the contraction-rate diagnostic row per cell.
* ``--no-vendored-baselines`` skips CopulaCPTS / MultiDimSPCI rows.
* Caching: every dataset download is written to
  ``$CAPFACTOR_REPR_CACHE`` (default ``~/.cache/capfactor_repr/``) and
  reused on later runs.

No external project imports. No credentials.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _lib.audit import log_volume_per_coord, rho_diagnostics
from _lib.backbone import GraphPolyVAR
from _lib.core import (
    cache_dir, empirical_coverage, four_way_split, joint_coverage,
    mean_pi_halfwidth, seed_everything, slice4, winkler_score, zscore,
)
from _lib.data import CELL_LOADERS
from _lib.graph import (
    kmeans_node_groups, normalised_shift, select_busiest_subgraph,
)
from _lib.methods import (
    acipergroupfactorcgif_predict, agacigroupcgif_predict,
    diaggru_predict, ewmacov_predict, factorcgif_predict,
    gcnrankzero_predict, gnf_aci_predict, gnf_predict,
    kalmanfcp_predict, rollingcov_predict, staticcgif_predict,
)


HERO_CELLS = ["metrla", "pems_bay"]
GPVAR_SCALE_CELLS = ["metrla", "metrla_full", "pems_bay", "pems_bay_full"]
PRIMARY_CELLS = [
    "metrla", "metrla_full", "pems_bay", "pems_bay_full",
    "aqi", "elec", "ett", "solar", "jena", "loop_seattle",
]
ALL_CELLS = PRIMARY_CELLS + ["largest"]
DEFAULT_CELLS = ["synthetic"]

METHOD_ORDER = [
    "GNF", "GNF+ACI",
    "KalmanFCP", "DiagGRU", "GCNrankzero",
    "StaticCGIF", "FactorCGIF",
    "AgACIGroupCGIF", "ACIPerGroupFactorCGIF",
    "EWMACovCGIF", "RollingCovCGIF",
    "CopulaCPTS", "MultiDimSPCI",
]


@dataclass
class CellConfig:
    cell: str
    n_seeds: int
    seed_start: int
    alpha: float
    subgraph_size: int
    max_test_steps: int
    n_groups: int
    var_K: int
    var_L: int
    ridge_lam: float
    cgif_shrinkage: float
    gnf_hidden: int
    gnf_low_rank: int
    gnf_epochs: int
    diaggru_hidden: int
    diaggru_epochs: int
    kalman_rho: float
    factor_rank: int
    aci_gamma: float
    aci_window: int | None
    cond_half_life: float
    cond_window: int
    warmup_steps: int
    include_vendored: bool
    audit: bool
    copula_epochs: int


def run_one_seed(cfg: CellConfig, Y: np.ndarray, A_sym: np.ndarray, seed: int) -> dict:
    subgraph_rng = None if seed == cfg.seed_start else seed
    idx, A_sub = select_busiest_subgraph(Y, A_sym, cfg.subgraph_size, rng_seed=subgraph_rng)
    Y_sub = Y[:, idx]
    split = four_way_split(Y_sub.shape[0])
    Y_tr, Y_pi, Y_ca, Y_te = slice4(Y_sub, split)
    if Y_te.shape[0] > cfg.max_test_steps:
        Y_te = Y_te[: cfg.max_test_steps]
    Y_tr_z, mu, sd = zscore(Y_tr)
    Y_pi_z = (Y_pi - mu) / sd
    Y_ca_z = (Y_ca - mu) / sd
    Y_te_z = (Y_te - mu) / sd

    seed_everything(seed)
    S_sub = normalised_shift(A_sub)
    bb = GraphPolyVAR(S=S_sub, K=cfg.var_K, L=cfg.var_L, lam=cfg.ridge_lam).fit(Y_tr_z)
    Yp_pi = bb.predict_one_step(Y_pi_z)
    Yp_ca = bb.predict_one_step(Y_ca_z)
    Yp_te = bb.predict_one_step(Y_te_z)
    groups = kmeans_node_groups(A_sub,
                                n_groups=min(cfg.n_groups, cfg.subgraph_size),
                                seed=seed)

    methods: dict[str, dict] = {}

    def _t(name, fn, *a, **k):
        t0 = time.time()
        try:
            res = fn(*a, **k)
        except Exception as exc:
            res = {"error": f"{type(exc).__name__}: {exc}",
                   "lo": None, "hi": None, "inside_ellipsoid": None}
        res["_time"] = time.time() - t0
        methods[name] = res

    # --- Filter rows --------------------------------------------------------
    # The three filter rows pass the train block as the warm-up source; CGIFFiltered
    # uses Y_pi as a long warm-up because the paper's split has Y_train short.
    Y_warmup_train = np.concatenate([Y_tr_z, Y_pi_z], axis=0)
    _t("GNF", gnf_predict, Y_warmup_train, Y_ca_z, Y_te_z,
       S_sub=S_sub, alpha=cfg.alpha,
       hidden=cfg.gnf_hidden, low_rank=cfg.gnf_low_rank,
       epochs=cfg.gnf_epochs, warmup_steps=cfg.warmup_steps, seed=seed)
    _t("GNF+ACI", gnf_aci_predict, Y_warmup_train, Y_ca_z, Y_te_z,
       S_sub=S_sub, alpha=cfg.alpha,
       hidden=cfg.gnf_hidden, low_rank=cfg.gnf_low_rank,
       epochs=cfg.gnf_epochs, gamma=cfg.aci_gamma, window=cfg.aci_window,
       warmup_steps=cfg.warmup_steps, seed=seed)
    _t("KalmanFCP", kalmanfcp_predict, Y_warmup_train, Y_ca_z, Y_te_z,
       S_sub=S_sub, alpha=cfg.alpha, rho_scale=cfg.kalman_rho,
       warmup_steps=cfg.warmup_steps)
    _t("DiagGRU", diaggru_predict, Y_warmup_train, Y_ca_z, Y_te_z,
       S_sub=S_sub, alpha=cfg.alpha,
       hidden=cfg.diaggru_hidden, epochs=cfg.diaggru_epochs,
       warmup_steps=cfg.warmup_steps, seed=seed)
    _t("GCNrankzero", gcnrankzero_predict, Y_warmup_train, Y_ca_z, Y_te_z,
       S_sub=S_sub, alpha=cfg.alpha,
       hidden=cfg.gnf_hidden, epochs=cfg.gnf_epochs,
       warmup_steps=cfg.warmup_steps, seed=seed)

    # --- Static / non-filter rows ------------------------------------------
    _t("StaticCGIF", staticcgif_predict,
       Y_pi_z, Yp_pi, Y_ca_z, Yp_ca, Y_te_z, Yp_te,
       alpha=cfg.alpha, shrinkage=cfg.cgif_shrinkage)
    _t("FactorCGIF", factorcgif_predict,
       Y_pi_z, Yp_pi, Y_ca_z, Yp_ca, Y_te_z, Yp_te,
       alpha=cfg.alpha, rank=cfg.factor_rank)
    _t("AgACIGroupCGIF", agacigroupcgif_predict,
       groups, Y_pi_z, Yp_pi, Y_ca_z, Yp_ca, Y_te_z, Yp_te,
       alpha=cfg.alpha, window=cfg.aci_window)
    _t("ACIPerGroupFactorCGIF", acipergroupfactorcgif_predict,
       groups, Y_pi_z, Yp_pi, Y_ca_z, Yp_ca, Y_te_z, Yp_te,
       alpha=cfg.alpha, rank=cfg.factor_rank,
       gamma=cfg.aci_gamma, window=cfg.aci_window)
    _t("EWMACovCGIF", ewmacov_predict,
       Y_pi_z, Yp_pi, Y_ca_z, Yp_ca, Y_te_z, Yp_te,
       alpha=cfg.alpha, half_life=cfg.cond_half_life)
    _t("RollingCovCGIF", rollingcov_predict,
       Y_pi_z, Yp_pi, Y_ca_z, Yp_ca, Y_te_z, Yp_te,
       alpha=cfg.alpha, window=cfg.cond_window)

    # --- Vendored upstream rows --------------------------------------------
    if cfg.include_vendored:
        from _lib.vendored_copula import copulacpts_predict
        from _lib.vendored_spci import multidimspci_predict
        _t("CopulaCPTS", copulacpts_predict,
           Y_pi_z, Yp_pi, Y_ca_z, Yp_ca, Y_te_z, Yp_te,
           alpha=cfg.alpha, copula_epochs=cfg.copula_epochs)
        _t("MultiDimSPCI", multidimspci_predict,
           Y_pi_z, Yp_pi, Y_ca_z, Yp_ca, Y_te_z, Yp_te, alpha=cfg.alpha)
    else:
        for nm in ("CopulaCPTS", "MultiDimSPCI"):
            methods[nm] = {"error": "skipped (--no-vendored-baselines)",
                           "lo": None, "hi": None, "inside_ellipsoid": None,
                           "_time": 0.0}

    # --- Audit row (rho diagnostics + log-volume) --------------------------
    audit = None
    if cfg.audit:
        try:
            # Use GNF+ACI's score stream for the rho diagnostics.
            from _lib.methods import fcp_aci_predict
            from _lib.filters import GraphNeuralSSMFilter
            filt = GraphNeuralSSMFilter(
                hidden_size=cfg.gnf_hidden, low_rank=cfg.gnf_low_rank,
                epochs=cfg.gnf_epochs, seed=seed,
            ).fit(Y_warmup_train, S_sub)
            # Re-run to capture per-step Sigma_t for rho_dG.
            warmup_len = min(cfg.warmup_steps, Y_warmup_train.shape[0])
            Y_w = Y_warmup_train[-warmup_len:] if warmup_len > 0 else None
            h, P = filt.initial_state(Y_w)
            T_cal = Y_ca_z.shape[0]
            s_cal = np.empty(T_cal, dtype=np.float64)
            for t in range(T_cal):
                h, P, mu_t, Sigma_t = filt.step(h, P, Y_ca_z[t])
                e = Y_ca_z[t] - mu_t
                try:
                    v = np.linalg.solve(Sigma_t, e)
                    s_cal[t] = float(np.sqrt(np.dot(e, v)))
                except np.linalg.LinAlgError:
                    s_cal[t] = float(np.linalg.norm(e))
            T_test = Y_te_z.shape[0]
            mu_seq = np.empty_like(Y_te_z)
            Sigma_seq = np.empty((T_test, *Sigma_t.shape), dtype=np.float64)
            s_test = np.empty(T_test, dtype=np.float64)
            h_t, P_t = h.copy(), P.copy()
            for t in range(T_test):
                h_t, P_t, mu_t, Sigma_t = filt.step(h_t, P_t, Y_te_z[t])
                mu_seq[t] = mu_t
                Sigma_seq[t] = Sigma_t
                e = Y_te_z[t] - mu_t
                try:
                    v = np.linalg.solve(Sigma_t, e)
                    s_test[t] = float(np.sqrt(np.dot(e, v)))
                except np.linalg.LinAlgError:
                    s_test[t] = float(np.linalg.norm(e))
            audit = rho_diagnostics(s_cal, s_test, Y_te_z, mu_seq, max_lag=50)
            # Add Vhat_m for the static-Sigma reference.
            Sigma_static = np.cov(Y_pi_z - Yp_pi, rowvar=False)
            audit["Vhat_m_static"] = log_volume_per_coord(
                np.diag(Sigma_static), q=methods["StaticCGIF"].get("q", 1.0)
            )
        except Exception as exc:
            audit = {"error": f"{type(exc).__name__}: {exc}"}

    # --- Score every method ------------------------------------------------
    scored: dict[str, dict] = {}
    for name, res in methods.items():
        if res.get("error") is not None or res.get("lo") is None:
            scored[name] = {"error": res.get("error", "missing"),
                            "_time": res.get("_time", 0.0)}
            continue
        lo = np.asarray(res["lo"], dtype=np.float64)
        hi = np.asarray(res["hi"], dtype=np.float64)
        if np.any(hi < lo):
            hi = np.maximum(hi, lo)
        cov = empirical_coverage(Y_te_z, lo, hi)
        if "inside_ellipsoid" in res and res["inside_ellipsoid"] is not None:
            joint = float(np.asarray(res["inside_ellipsoid"], dtype=bool).mean())
        else:
            joint = joint_coverage(Y_te_z, lo, hi)
        # Paper convention: full ellipsoid width (mean (hi - lo)).
        width = float(np.maximum(hi - lo, 0.0).mean())
        halfwidth = 0.5 * np.maximum(hi - lo, 0.0)
        vhat_m = float(np.mean(np.log(np.maximum(halfwidth, 1e-12))))
        wink = winkler_score(Y_te_z, lo, hi, cfg.alpha)
        scored[name] = {
            "coverage": float(cov),
            "joint_coverage": float(joint),
            "mean_width": float(width),
            "mean_halfwidth": float(0.5 * width),
            "vhat_m": vhat_m,
            "winkler": float(wink),
            "coverage_gap": float(cov - (1.0 - cfg.alpha)),
            "_time": float(res.get("_time", 0.0)),
        }
    return {"methods": scored, "audit": audit, "N_used": int(A_sub.shape[0])}


def aggregate_seeds(seed_runs: list[dict]) -> dict:
    if len(seed_runs) == 1:
        return seed_runs[0]
    methods_agg: dict[str, dict] = {}
    method_names: set[str] = set()
    for s in seed_runs:
        method_names.update(s["methods"].keys())
    for m in method_names:
        per_seed = [s["methods"].get(m, {}) for s in seed_runs]
        good = [s for s in per_seed if "error" not in s]
        if not good:
            methods_agg[m] = {"error": "all seeds failed"}
            continue
        agg = {}
        for k in ("coverage", "coverage_gap", "joint_coverage",
                  "mean_width", "mean_halfwidth", "vhat_m", "winkler"):
            v = np.array([s[k] for s in good], dtype=np.float64)
            agg[k] = float(v.mean())
            agg[f"{k}_std"] = float(v.std())
        agg["_time"] = float(np.mean([s.get("_time", 0.0) for s in good]))
        agg["_n_seeds_ok"] = len(good)
        if len(good) < len(per_seed):
            agg["_n_seeds_failed"] = len(per_seed) - len(good)
        methods_agg[m] = agg
    audit_runs = [s.get("audit") for s in seed_runs
                  if s.get("audit") and "error" not in s.get("audit", {})]
    audit = None
    if audit_runs:
        keys = sorted({k for a in audit_runs for k in a.keys()})
        audit = {}
        for k in keys:
            vals = [a.get(k) for a in audit_runs if isinstance(a.get(k), (int, float))]
            if vals:
                audit[k] = float(np.mean(vals))
    return {"methods": methods_agg, "audit": audit, "N_used": seed_runs[0].get("N_used")}


def print_cell_table(cell_name: str, agg: dict, *, expected: dict | None,
                     n_seeds: int, mode: str) -> None:
    methods = agg["methods"]
    print()
    print(f"### {cell_name}  (N_used={agg.get('N_used', '?')}, "
          f"n_seeds={n_seeds}, mode={mode})")
    cols = ["method", "coverage", "joint", "width", "winkler"]
    if expected:
        cols.extend(["paper", "Δ"])
    widths = {"method": 24, "coverage": 9, "joint": 9, "width": 14,
              "winkler": 9, "paper": 14, "Δ": 8}
    hdr = "| " + " | ".join(
        f"{c:>{widths[c]}}" if c != "method" else f"{c:<{widths[c]}}"
        for c in cols) + " |"
    sep = "|" + "|".join("-" * (widths[c] + 2) for c in cols) + "|"
    print(hdr); print(sep)
    for m in METHOD_ORDER:
        if m not in methods:
            continue
        s = methods[m]
        if "error" in s:
            row = [m, "(skip)" if "skipped" in str(s["error"]) else "ERR",
                   "—", "—", "—"]
            if expected:
                row.extend(["—", "—"])
            print("| " + " | ".join(
                f"{v:>{widths[cols[i]]}}" if i > 0 else f"{v:<{widths[cols[i]]}}"
                for i, v in enumerate(row)) + " |")
            continue
        w_std = s.get("mean_width_std", 0.0)
        w_s = (f"{s['mean_width']:.3f}±{w_std:.3f}" if n_seeds > 1
               else f"{s['mean_width']:.3f}")
        row = [m, f"{s['coverage']:.3f}", f"{s['joint_coverage']:.3f}",
               w_s, f"{s['winkler']:.2f}"]
        if expected:
            paper = expected.get(m)
            if paper is None:
                row.extend(["n/a", "—"])
            else:
                p_mean, p_std = paper
                row.append(f"{p_mean:.2f}±{p_std:.2f}")
                delta = s["mean_width"] - p_mean
                tol = max(2.0 * p_std, 0.10)
                marker = "✓" if abs(delta) <= tol else "!"
                row.append(f"{delta:+.2f}{marker}")
        print("| " + " | ".join(
            f"{v:>{widths[cols[i]]}}" if i > 0 else f"{v:<{widths[cols[i]]}}"
            for i, v in enumerate(row)) + " |")
    print()
    audit = agg.get("audit")
    if audit:
        print(f"### audit  ({cell_name})")
        for k, v in audit.items():
            if isinstance(v, (int, float)):
                print(f"  {k:<22s} = {v:.4f}")
            else:
                print(f"  {k:<22s} = {v}")
        print()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Graph-filtered conformal prediction reproducibility: "
                    "every datapoint of the paper's main section.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    p.add_argument("--cells", nargs="+", default=DEFAULT_CELLS,
                   help="dataset names from CELL_LOADERS, or one of "
                        "'all' / 'primary' / 'hero' / 'gpvar_scale'.")
    p.add_argument("--n-seeds", type=int, default=None)
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--subgraph-size", type=int, default=None)
    p.add_argument("--max-test-steps", type=int, default=None)
    p.add_argument("--n-groups", type=int, default=4)
    p.add_argument("--gnf-hidden", type=int, default=32)
    p.add_argument("--gnf-low-rank", type=int, default=4)
    p.add_argument("--gnf-epochs", type=int, default=None)
    p.add_argument("--diaggru-hidden", type=int, default=64)
    p.add_argument("--diaggru-epochs", type=int, default=None)
    p.add_argument("--kalman-rho", type=float, default=0.8)
    p.add_argument("--factor-rank", type=int, default=4)
    p.add_argument("--aci-gamma", type=float, default=5e-3)
    p.add_argument("--aci-window", type=int, default=400)
    p.add_argument("--cond-half-life", type=float, default=288.0)
    p.add_argument("--cond-window", type=int, default=288)
    p.add_argument("--warmup-steps", type=int, default=50)
    p.add_argument("--no-vendored-baselines", action="store_true")
    p.add_argument("--audit", action="store_true")
    p.add_argument("--check-expected", action="store_true")
    p.add_argument("--copula-epochs", type=int, default=None)
    p.add_argument("--output", type=Path, default=None)
    return p.parse_args()


def expand_cells(cells: list[str]) -> list[str]:
    out: list[str] = []
    for c in cells:
        if c == "all":
            out.extend(ALL_CELLS)
        elif c == "primary":
            out.extend(PRIMARY_CELLS)
        elif c == "hero":
            out.extend(HERO_CELLS)
        elif c == "gpvar_scale":
            out.extend(GPVAR_SCALE_CELLS)
        elif c in CELL_LOADERS:
            out.append(c)
        else:
            raise ValueError(f"Unknown cell {c!r}; choose from "
                             f"{list(CELL_LOADERS)} or 'all'/'primary'/"
                             f"'hero'/'gpvar_scale'.")
    seen: set[str] = set()
    return [c for c in out if not (c in seen or seen.add(c))]


def make_config(args: argparse.Namespace, cell: str) -> CellConfig:
    smoke = args.mode == "smoke"
    _, _, default_N = CELL_LOADERS[cell]
    subgraph_size = args.subgraph_size or default_N
    if args.n_seeds is not None:
        n_seeds = args.n_seeds
    else:
        n_seeds = 1 if smoke else 10
    if args.max_test_steps is not None:
        max_test = args.max_test_steps
    else:
        max_test = 500 if smoke else 4000
    gnf_epochs = args.gnf_epochs if args.gnf_epochs is not None else (10 if smoke else 60)
    diag_epochs = args.diaggru_epochs if args.diaggru_epochs is not None else (10 if smoke else 80)
    copula_epochs = args.copula_epochs if args.copula_epochs is not None else (50 if smoke else 800)
    return CellConfig(
        cell=cell, n_seeds=n_seeds, seed_start=args.seed_start,
        alpha=args.alpha, subgraph_size=subgraph_size, max_test_steps=max_test,
        n_groups=args.n_groups, var_K=2, var_L=2, ridge_lam=1e-2,
        cgif_shrinkage=0.10,
        gnf_hidden=args.gnf_hidden, gnf_low_rank=args.gnf_low_rank,
        gnf_epochs=gnf_epochs,
        diaggru_hidden=args.diaggru_hidden, diaggru_epochs=diag_epochs,
        kalman_rho=args.kalman_rho, factor_rank=args.factor_rank,
        aci_gamma=args.aci_gamma, aci_window=args.aci_window,
        cond_half_life=args.cond_half_life, cond_window=args.cond_window,
        warmup_steps=args.warmup_steps,
        include_vendored=not args.no_vendored_baselines,
        audit=args.audit, copula_epochs=copula_epochs,
    )


def main() -> int:
    args = parse_args()
    cells = expand_cells(args.cells)
    expected = None
    if args.check_expected:
        with open(Path(__file__).resolve().parent / "expected.json") as f:
            expected = json.load(f)
    print(f"\n=== graph-filtered CP reproducibility ({args.mode}, cells={cells}) ===")
    print(f"   cache directory: {cache_dir()}")
    results_all: dict[str, dict] = {}
    for cell in cells:
        cfg = make_config(args, cell)
        print(f"\n=== Loading {cell} ===")
        loader_name, loader, _ = CELL_LOADERS[cell]
        try:
            if cell == "synthetic":
                Y, A = loader(T=6000, N=max(cfg.subgraph_size, 20), seed=0)
            else:
                Y, A = loader()
        except Exception as exc:
            print(f"   load failed: {type(exc).__name__}: {exc}")
            results_all[cell] = {"error": str(exc), "config": asdict(cfg)}
            continue
        print(f"   Y: shape={Y.shape}  A: shape={A.shape}  "
              f"subgraph_size={cfg.subgraph_size}")
        A_sym = np.maximum(A, A.T)
        seeds = list(range(cfg.seed_start, cfg.seed_start + cfg.n_seeds))
        seed_runs: list[dict] = []
        for i, seed in enumerate(seeds):
            print(f"-- seed {seed} ({i + 1}/{len(seeds)}) --")
            t0 = time.time()
            scored = run_one_seed(cfg, Y, A_sym, seed=seed)
            dt = time.time() - t0
            n_methods = sum(1 for v in scored["methods"].values() if "error" not in v)
            print(f"   done in {dt:.1f} s; {n_methods}/{len(scored['methods'])} methods OK.")
            seed_runs.append(scored)
        agg = aggregate_seeds(seed_runs)
        # Pick the right reference table for the Δ column.
        cell_expected = None
        if expected is not None:
            for tbl in ("table_hero", "table_scale_gpvar"):
                if cell in expected.get(tbl, {}):
                    cell_expected = {k: v for k, v in expected[tbl][cell].items()
                                     if k != "N" and isinstance(v, list)}
                    break
        print_cell_table(cell, agg,
                         expected=cell_expected, n_seeds=len(seeds), mode=args.mode)
        results_all[cell] = {
            "config": asdict(cfg),
            "n_seeds_run": len(seeds),
            "seeds": seeds,
            "per_seed": seed_runs,
            "aggregate": agg,
        }
    out_path = args.output or Path(f"results_{'_'.join(cells)[:80]}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results_all, f, indent=2, default=float)
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
