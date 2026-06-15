"""Factor-model + per-community helpers used by ``methods.py``."""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_triangular

from .core import ledoit_wolf_diagonal, safe_cholesky


def fit_factor_loadings(
    residuals: np.ndarray, rank: int, sigma_floor: float = 1e-3
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Top-``rank`` SVD factor model ``e ≈ B F + R``.

    Returns ``(B, Sigma_F, sigma_R)`` of shapes ``(N, r)``, ``(r, r)``,
    ``(N,)``.
    """
    e = np.asarray(residuals, dtype=np.float64)
    n, N = e.shape
    rank = int(min(max(1, rank), min(n - 1, N)))
    e_c = e - e.mean(axis=0, keepdims=True)
    _V, s, Wt = np.linalg.svd(e_c, full_matrices=False)
    B = Wt[:rank, :].T
    F = e_c @ B
    Sigma_F = (F.T @ F) / max(n - 1, 1)
    R = e_c - F @ B.T
    sigma_R = np.sqrt(np.maximum((R * R).sum(axis=0) / max(n - 1, 1), sigma_floor ** 2))
    return B, Sigma_F, sigma_R


def factor_quad_form(F: np.ndarray, Sigma_F: np.ndarray) -> np.ndarray:
    """Per-row Mahalanobis ``sqrt(F Sigma_F^{-1} F^T)``."""
    F = np.atleast_2d(np.asarray(F, dtype=np.float64))
    Sigma_F = np.asarray(Sigma_F, dtype=np.float64)
    r = Sigma_F.shape[0]
    Sigma_F = 0.5 * (Sigma_F + Sigma_F.T)
    try:
        L = np.linalg.cholesky(Sigma_F + 1e-10 * np.eye(r))
    except np.linalg.LinAlgError:
        L = np.linalg.cholesky(Sigma_F + 1e-6 * np.eye(r))
    z = np.linalg.solve(L, F.T)
    return np.sqrt((z * z).sum(axis=0))


def fit_group_covariance(
    R_group: np.ndarray, sigma_floor: float = 1e-3
) -> tuple[np.ndarray, np.ndarray, float]:
    """Ledoit-Wolf diagonal-target shrinkage for one community."""
    Sigma, lam = ledoit_wolf_diagonal(R_group)
    diag_S = np.maximum(np.diag(Sigma), sigma_floor ** 2)
    Sigma = Sigma.copy()
    np.fill_diagonal(Sigma, diag_S)
    L = safe_cholesky(Sigma)
    return Sigma, L, lam


def group_mahalanobis(R_group: np.ndarray, L_group: np.ndarray) -> np.ndarray:
    R_group = np.atleast_2d(np.asarray(R_group, dtype=np.float64))
    z = solve_triangular(L_group, R_group.T, lower=True)
    return np.sqrt((z * z).sum(axis=0))
