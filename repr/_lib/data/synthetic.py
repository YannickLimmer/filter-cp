"""Offline synthetic GraphPolyVAR — pipeline sanity check (no network)."""

from __future__ import annotations

import numpy as np

from ..graph import normalised_shift


def synthesise_gpvar(*, T: int = 6000, N: int = 20, seed: int = 0,
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Hand-picked stable GraphPolyVAR(K=2, L=2) on a 4-community SBM."""
    rng = np.random.default_rng(seed)
    positions = rng.uniform(size=(N, 2))
    communities = rng.integers(0, 4, size=N)
    same = communities[:, None] == communities[None, :]
    probs = np.where(same, 0.30, 0.02)
    np.fill_diagonal(probs, 0.0)
    edges = (rng.uniform(size=(N, N)) < probs).astype(np.float64)
    edges = np.triu(edges, k=1)
    diff = positions[:, None, :] - positions[None, :, :]
    kernel = np.exp(-(diff * diff).sum(-1) / 0.30 ** 2)
    A = edges * kernel
    A = A + A.T
    np.fill_diagonal(A, 0.0)
    S = normalised_shift(A)
    theta = np.array([[0.70, -0.25], [0.15, -0.05]])
    Y = np.zeros((T, N), dtype=np.float64)
    sigma_eps = 0.5
    powers = [np.eye(N), S]
    K, L = 2, 2
    for t in range(L, T):
        y_t = np.zeros(N, dtype=np.float64)
        for l in range(L):
            for k in range(K):
                y_t = y_t + theta[k, l] * (powers[k] @ Y[t - 1 - l])
        Y[t] = y_t + sigma_eps * rng.standard_normal(N)
    return Y[100:], A
