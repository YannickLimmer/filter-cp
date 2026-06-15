"""Core primitives: quantiles, covariance estimators, splits, metrics, seeding."""

from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------- #
# Cache directory                                                             #
# --------------------------------------------------------------------------- #
def cache_dir() -> Path:
    """Persistent cache for downloaded datasets.

    Defaults to ``~/.cache/capfactor_repr/`` unless ``$CAPFACTOR_REPR_CACHE``
    is set.  Created on first access.
    """
    p = Path(os.environ.get("CAPFACTOR_REPR_CACHE",
                            Path.home() / ".cache" / "capfactor_repr"))
    p.mkdir(parents=True, exist_ok=True)
    return p


# --------------------------------------------------------------------------- #
# Quantile + covariance primitives                                            #
# --------------------------------------------------------------------------- #
def split_conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """``ceil((n+1)(1-alpha))``-th order statistic (Vovk-style inflation)."""
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
    s = np.asarray(scores, dtype=np.float64).ravel()
    n = s.size
    if n == 0:
        raise ValueError("Need at least one calibration score.")
    rank = max(1, min(math.ceil((n + 1) * (1.0 - alpha)), n))
    return float(np.sort(s)[rank - 1])


def safe_cholesky(M: np.ndarray, jitter: float = 1e-8) -> np.ndarray:
    """Cholesky with progressively larger jitter until PSD."""
    j = jitter
    for _ in range(8):
        try:
            return np.linalg.cholesky(M + j * np.eye(M.shape[0]))
        except np.linalg.LinAlgError:
            j *= 10.0
    raise np.linalg.LinAlgError(f"Matrix not PSD after jitter={j:.1e}.")


def shrinkage_cov(residuals: np.ndarray, *, shrinkage: float = 0.10) -> np.ndarray:
    """``(1 - λ) S + λ diag(S)`` with sample covariance ``S``."""
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError(f"shrinkage must lie in [0, 1], got {shrinkage}.")
    e = np.asarray(residuals, dtype=np.float64)
    n = e.shape[0]
    S = (e.T @ e) / max(n - 1, 1)
    return (1.0 - shrinkage) * S + shrinkage * np.diag(np.diag(S))


def ledoit_wolf_diagonal(e: np.ndarray) -> tuple[np.ndarray, float]:
    """Schäfer-Strimmer (2005) closed-form LW shrinkage to ``diag(S)``."""
    e = np.asarray(e, dtype=np.float64)
    n, N = e.shape
    if n < 2:
        raise ValueError(f"need n >= 2, got n={n}.")
    S = (e.T @ e) / (n - 1)
    e_sq = e * e
    M2 = (e_sq.T @ e_sq) / n
    var_w = M2 - S * S
    var_s = var_w / n
    iu = np.triu_indices(N, k=1)
    num = 2.0 * float(var_s[iu].sum())
    den = 2.0 * float((S[iu] ** 2).sum())
    lam = 1.0 if den <= 0.0 else float(np.clip(num / den, 0.0, 1.0))
    return (1.0 - lam) * S + lam * np.diag(np.diag(S)), lam


# --------------------------------------------------------------------------- #
# Splits and z-score                                                          #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FourWaySplit:
    """``train / pilot / calibration / test`` chronological split.

    Defaults to 70 / 10 / 10 / 10.  The pilot block hosts every
    pilot-measurable shape choice (factor subspace, per-node scales,
    Huber cap, community partition); the calibration block hosts the
    split-CP quantiles.  Theorem F-safe.
    """

    n: int
    train_end: int
    pilot_end: int
    cal_end: int


def four_way_split(
    n: int, *, train_frac: float = 0.7, pilot_frac: float = 0.1, cal_frac: float = 0.1,
) -> FourWaySplit:
    if not (0.0 < train_frac < 1.0 and 0.0 < pilot_frac < 1.0 and 0.0 < cal_frac < 1.0):
        raise ValueError("fractions must lie in (0, 1).")
    if train_frac + pilot_frac + cal_frac >= 1.0:
        raise ValueError("train + pilot + cal must leave room for the test block.")
    te = max(1, round(train_frac * n))
    pe = max(te + 1, round((train_frac + pilot_frac) * n))
    ce = max(pe + 1, round((train_frac + pilot_frac + cal_frac) * n))
    te = min(te, n - 3)
    pe = min(pe, n - 2)
    ce = min(ce, n - 1)
    return FourWaySplit(n=n, train_end=te, pilot_end=pe, cal_end=ce)


def slice4(Y: np.ndarray, s: FourWaySplit
           ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (Y[: s.train_end], Y[s.train_end : s.pilot_end],
            Y[s.pilot_end : s.cal_end], Y[s.cal_end :])


def zscore(Y_train: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = np.nanmean(Y_train, axis=0)
    sd = np.nanstd(Y_train, axis=0).clip(min=1e-6)
    return (Y_train - mu) / sd, mu, sd


# --------------------------------------------------------------------------- #
# Metrics                                                                     #
# --------------------------------------------------------------------------- #
def empirical_coverage(y_true, lo, hi) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    return float(((y_true >= lo) & (y_true <= hi)).mean())


def joint_coverage(y_true, lo, hi) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    inside = (y_true >= lo) & (y_true <= hi)
    all_in = inside.reshape(inside.shape[0], -1).all(axis=1)
    return float(all_in.mean())


def mean_pi_halfwidth(lo, hi) -> float:
    """Half-width ``(hi - lo) / 2`` averaged over ``(t, i)``.

    Matches the paper's per-coord ``h_i`` convention.
    """
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    w = np.maximum(hi - lo, 0.0)
    return float(0.5 * w.mean())


def winkler_score(y_true, lo, hi, alpha: float) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    width = hi - lo
    under = np.maximum(lo - y_true, 0.0)
    over = np.maximum(y_true - hi, 0.0)
    s = width + (2.0 / alpha) * (under + over)
    return float(s.mean())


# --------------------------------------------------------------------------- #
# Misc                                                                        #
# --------------------------------------------------------------------------- #
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def fill_nans_columnwise(Y: np.ndarray) -> np.ndarray:
    """Forward-fill NaNs per column, then back-fill with column mean."""
    out = np.array(Y, dtype=np.float64, copy=True)
    T, N = out.shape
    for j in range(N):
        col = out[:, j]
        mask = np.isnan(col)
        if not mask.any():
            continue
        idx = np.where(~mask, np.arange(T), -1)
        np.maximum.accumulate(idx, out=idx)
        valid = idx >= 0
        col_ffill = np.where(valid, col[np.where(valid, idx, 0)], np.nan)
        leading = np.isnan(col_ffill)
        if leading.any():
            col_mean = float(np.nanmean(col)) if np.isfinite(col).any() else 0.0
            col_ffill = np.where(leading, col_mean, col_ffill)
        out[:, j] = col_ffill
    return out


def chronological_70_10_20(T: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Loader-side 70/10/20 split.  Composed with ``four_way_split`` later."""
    n_train = int(T * 0.70)
    n_val = int(T * 0.10)
    return (np.arange(0, n_train, dtype=np.int64),
            np.arange(n_train, n_train + n_val, dtype=np.int64),
            np.arange(n_train + n_val, T, dtype=np.int64))
