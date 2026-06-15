"""Jena Weather — MPI Jena Climate (21 channels) via GiftEval mirror."""

from __future__ import annotations

import numpy as np

from ..core import cache_dir, fill_nans_columnwise
from ..graph import knn_correlation_graph
from ._common import gifteval_panel


def load_jena(*, knn_k: int = 4) -> tuple[np.ndarray, np.ndarray]:
    cache = cache_dir() / "jena_weather"
    cache.mkdir(parents=True, exist_ok=True)
    tag = cache / f"jena_H_k{knn_k}_v1.npz"
    if tag.exists():
        with np.load(tag, allow_pickle=False) as d:
            return d["Y"].astype(np.float64), d["A"].astype(np.float64)
    Y_raw, _ = gifteval_panel("jena_weather", freq="H")
    Y = fill_nans_columnwise(Y_raw)
    n_train = int(Y.shape[0] * 0.70)
    A = knn_correlation_graph(Y[:n_train], k=knn_k)
    np.savez(tag, Y=Y, A=A)
    return Y, A
