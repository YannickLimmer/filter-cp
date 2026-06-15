"""ELEC — UCI Electricity Load Diagrams 2011-2014 (UCI #321).

Hourly aggregate of the 15-minute kWh consumption for 370 anonymous
Portuguese clients.  k-NN Pearson-correlation graph on the training
slice supplies the adjacency.

Source: https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014
"""

from __future__ import annotations

import io
import zipfile

import numpy as np

from ..core import cache_dir
from ..graph import knn_correlation_graph
from ._common import download_to


UCI_URL = (
    "https://archive.ics.uci.edu/static/public/321/"
    "electricityloaddiagrams20112014.zip"
)


def load_elec(*, n_clients: int = 20, knn_k: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Load the first ``n_clients`` clients (sorted alphabetically).

    The paper's main-cell ``elec-20`` is the default 20-client subset.
    """
    try:
        import pandas as pd
    except ImportError as e:
        raise RuntimeError("pandas is required for ELEC.") from e

    cache = cache_dir() / "elec"
    cache.mkdir(parents=True, exist_ok=True)
    tag = cache / f"elec_hourly_v1_n{n_clients}_k{knn_k}.npz"
    if tag.exists():
        with np.load(tag, allow_pickle=False) as d:
            return d["Y"].astype(np.float64), d["A"].astype(np.float64)

    zip_path = cache / "elec_uci_321.zip"
    download_to(UCI_URL, zip_path)
    with zipfile.ZipFile(zip_path) as outer:
        names = outer.namelist()
        target = next((n for n in names if n.endswith(".txt")), None)
        if target is None:
            nested = next((n for n in names if n.endswith(".zip")), None)
            if nested is None:
                raise RuntimeError(f"Cannot find LD2011_2014.txt in {zip_path}.")
            with zipfile.ZipFile(io.BytesIO(outer.read(nested))) as inner:
                target = next((n for n in inner.namelist() if n.endswith(".txt")), None)
                raw = inner.read(target) if target else b""
        else:
            raw = outer.read(target)
    df = pd.read_csv(io.BytesIO(raw),
                     sep=";", decimal=",", index_col=0, parse_dates=True, dayfirst=False)
    df.index = pd.to_datetime(df.index)
    df = df.astype(np.float64).resample("1h").sum(min_count=1).sort_index()
    df = df.dropna(how="all").fillna(0.0)
    cols = sorted(df.columns)[:n_clients]
    df = df[cols]
    Y = df.to_numpy(dtype=np.float64)
    n_train = int(Y.shape[0] * 0.70)
    A = knn_correlation_graph(Y[:n_train], k=knn_k)
    np.savez(tag, Y=Y, A=A)
    return Y, A
