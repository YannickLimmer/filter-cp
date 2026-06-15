"""Loop-Seattle — Seattle freeway loop sensors (323) via GiftEval mirror."""

from __future__ import annotations

import numpy as np

from ..core import cache_dir, fill_nans_columnwise
from ..graph import knn_correlation_graph
from ._common import gifteval_panel


def load_loop_seattle(*, knn_k: int = 8) -> tuple[np.ndarray, np.ndarray]:
    cache = cache_dir() / "loop_seattle"
    cache.mkdir(parents=True, exist_ok=True)
    tag = cache / f"loop_seattle_H_k{knn_k}_v1.npz"
    if tag.exists():
        with np.load(tag, allow_pickle=False) as d:
            return d["Y"].astype(np.float64), d["A"].astype(np.float64)
    Y_raw, _ = gifteval_panel("LOOP_SEATTLE", freq="H")
    Y = fill_nans_columnwise(Y_raw)
    n_train = int(Y.shape[0] * 0.70)
    A = knn_correlation_graph(Y[:n_train], k=knn_k)
    np.savez(tag, Y=Y, A=A)
    return Y, A
