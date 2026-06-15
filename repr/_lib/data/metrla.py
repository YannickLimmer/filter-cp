"""METR-LA — DCRNN traffic-speed benchmark on the LA freeway network.

Public mirror: ``witgaw/METR-LA``.  Returns ``(Y, A)`` with
``Y`` shape ``(34272, 207)`` and a symmetric Gaussian-kernel adjacency.
"""

from __future__ import annotations

import numpy as np

from ._common import hf_download


def load_metrla() -> tuple[np.ndarray, np.ndarray]:
    try:
        import pandas as pd
    except ImportError as e:
        raise RuntimeError(
            "pandas + pyarrow are required for METR-LA.  See requirements.txt."
        ) from e

    def _df(filename: str):
        path = hf_download("witgaw/METR-LA", filename)
        df = pd.read_parquet(path, columns=["node_id", "t0_timestamp", "x_t+0_d0"])
        df = df.rename(columns={"x_t+0_d0": "value"})
        w = df.pivot(index="t0_timestamp", columns="node_id", values="value").sort_index()
        w.columns = [str(int(c)) for c in w.columns]
        return w

    d_tr = _df("train.parquet")
    d_va = _df("val.parquet")
    d_te = _df("test.parquet")
    cols = sorted(set(d_tr.columns) | set(d_va.columns) | set(d_te.columns))
    d_full = pd.concat([
        d_tr.reindex(columns=cols), d_va.reindex(columns=cols), d_te.reindex(columns=cols),
    ]).sort_index()
    Y = d_full.to_numpy(dtype=np.float64)
    A = np.load(hf_download("witgaw/METR-LA", "sensor_graph/adj_mx.npy")).astype(np.float64)
    A = 0.5 * (A + A.T)
    np.fill_diagonal(A, 0.0)
    return Y, A
