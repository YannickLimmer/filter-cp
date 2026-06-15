"""Graph utilities — normalised shift, busiest-k, spectral clusters, k-NN."""

from __future__ import annotations

import numpy as np


def normalised_shift(A: np.ndarray) -> np.ndarray:
    """``D^{-1/2} (A + I) D^{-1/2}`` — symmetric, spectral radius 1."""
    N = A.shape[0]
    A_hat = A + np.eye(N, dtype=A.dtype)
    deg = A_hat.sum(axis=1)
    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
    return (d_inv_sqrt[:, None] * A_hat) * d_inv_sqrt[None, :]


def select_busiest_subgraph(
    Y: np.ndarray, A: np.ndarray, k: int, *, rng_seed: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Pick ``k`` columns by variance; deterministic top-k for ``rng_seed=None``."""
    if k >= Y.shape[1]:
        return np.arange(Y.shape[1]), A
    var = np.nanvar(Y, axis=0)
    if rng_seed is None:
        idx = np.argsort(-var)[:k]
    else:
        pool_size = min(2 * k, Y.shape[1])
        pool = np.argsort(-var)[:pool_size]
        rng = np.random.default_rng(rng_seed)
        idx = np.sort(rng.choice(pool, size=k, replace=False))
    return idx, A[np.ix_(idx, idx)]


def kmeans_node_groups(
    A: np.ndarray, n_groups: int, *, seed: int = 0, n_iter: int = 50
) -> list[np.ndarray]:
    """Spectral clustering: bottom-``n_groups`` eigvecs of normalised Laplacian → k-means++."""
    A = 0.5 * (np.asarray(A, dtype=np.float64) + np.asarray(A, dtype=np.float64).T)
    N = A.shape[0]
    if not 1 <= n_groups <= N:
        raise ValueError(f"n_groups must lie in [1, {N}], got {n_groups}.")
    d = A.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        d_safe = np.where(d > 0, 1.0 / np.sqrt(np.maximum(d, 1e-12)), 0.0)
    L = np.eye(N) - (A * d_safe[:, None]) * d_safe[None, :]
    L = 0.5 * (L + L.T)
    eigvals, eigvecs = np.linalg.eigh(L)
    idx = np.argsort(eigvals)[:n_groups]
    X = eigvecs[:, idx]
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / np.where(norms > 0, norms, 1.0)
    rng = np.random.default_rng(seed)
    centers = np.empty((n_groups, X.shape[1]))
    centers[0] = X[rng.integers(0, N)]
    for k in range(1, n_groups):
        d2 = ((X[:, None, :] - centers[None, :k, :]) ** 2).sum(axis=2).min(axis=1)
        if d2.sum() <= 0.0:
            centers[k] = X[rng.integers(0, N)]
        else:
            probs = d2 / d2.sum()
            centers[k] = X[rng.choice(N, p=probs)]
    labels = np.zeros(N, dtype=np.intp)
    for _ in range(n_iter):
        d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = d2.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for k in range(n_groups):
            mask = labels == k
            if mask.any():
                centers[k] = X[mask].mean(axis=0)
    groups: list[np.ndarray] = []
    for k in range(n_groups):
        idx_k = np.where(labels == k)[0]
        if idx_k.size == 0:
            d2_c = ((X - centers[k]) ** 2).sum(axis=1)
            idx_k = np.array([int(d2_c.argmin())], dtype=np.intp)
        groups.append(idx_k.astype(np.intp))
    return groups


def knn_correlation_graph(
    Y_train: np.ndarray, *, k: int = 8, eps: float = 1e-6,
) -> np.ndarray:
    """Symmetric k-NN Pearson-correlation similarity graph on training rows."""
    if Y_train.ndim != 2:
        raise ValueError(f"Y_train must be 2-D, got shape {Y_train.shape}.")
    if k <= 0:
        raise ValueError(f"k must be >= 1, got {k}.")
    Y = np.asarray(Y_train, dtype=np.float64)
    Y = Y - Y.mean(axis=0, keepdims=True)
    sd = Y.std(axis=0, keepdims=True)
    Y = Y / np.where(sd > eps, sd, 1.0)
    C = (Y.T @ Y) / Y.shape[0]
    np.fill_diagonal(C, 0.0)
    N = C.shape[0]
    A = np.zeros_like(C)
    abs_C = np.abs(C)
    if k >= N:
        idx = np.argsort(-abs_C, axis=1)
    else:
        idx = np.argpartition(-abs_C, kth=k, axis=1)[:, :k]
    rows = np.repeat(np.arange(N), idx.shape[1])
    cols = idx.ravel()
    A[rows, cols] = np.maximum(C[rows, cols], 0.0)
    A = np.maximum(A, A.T)
    np.fill_diagonal(A, 0.0)
    return A


def haversine_km(coords_a: np.ndarray, coords_b: np.ndarray) -> np.ndarray:
    """Pairwise great-circle distance in km between two ``(N, 2)`` ``(lat, lon)`` arrays."""
    R = 6371.0088
    lat1 = np.deg2rad(coords_a[:, None, 0])
    lon1 = np.deg2rad(coords_a[:, None, 1])
    lat2 = np.deg2rad(coords_b[None, :, 0])
    lon2 = np.deg2rad(coords_b[None, :, 1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def gaussian_kernel_adjacency(
    coords: np.ndarray, *, sigma_km: float | None = None,
) -> np.ndarray:
    """``A_ij = exp(-d_ij^2 / sigma^2)`` on Haversine distance; median bandwidth default."""
    D = haversine_km(coords, coords)
    np.fill_diagonal(D, 0.0)
    if sigma_km is None:
        triu = D[np.triu_indices_from(D, k=1)]
        sigma_km = float(np.median(triu)) if triu.size else 1.0
    A = np.exp(-(D ** 2) / (sigma_km ** 2 + 1e-12))
    np.fill_diagonal(A, 0.0)
    return A
