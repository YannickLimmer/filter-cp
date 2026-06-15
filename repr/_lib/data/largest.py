"""LargeST — Greater Bay Area (GBA) sub-panel of Liu et al. (NeurIPS 2023).

Public mirror: ``ttydsc30/TrafficForecast-LargeST-Project-Dataset``.
The full California 8 600-sensor block ships as per-year H5 files
(~6.9 GB each); we read only the GBA district subset for ``year=2019``
and aggregate from 5-minute to hourly cadence.  Dead-sensor pruning
follows the standard LargeST/DCRNN convention.
"""

from __future__ import annotations

import numpy as np

from ..core import cache_dir, fill_nans_columnwise
from ._common import hf_download


# Caltrans district 4 is the canonical "Greater Bay Area" subset.
GBA_DISTRICTS: list[int] = [4]


def _aggregate_hourly(
    Y_5min: np.ndarray, timestamps_5min: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    step = 12
    T = Y_5min.shape[0]
    T_trunc = (T // step) * step
    Y_trunc = Y_5min[:T_trunc]
    Y_agg = Y_trunc.reshape(T_trunc // step, step, -1).mean(axis=1)
    ts_agg = timestamps_5min[:T_trunc:step]
    return Y_agg, ts_agg


def load_largest(*, year: int = 2019) -> tuple[np.ndarray, np.ndarray]:
    """Load LargeST GBA at the paper's default ``year=2019``, hourly.

    Returns ``(Y, A)`` with ``Y`` shape ``(T, 2327)`` (after dead-sensor
    pruning) and a symmetric road-network adjacency.
    """
    try:
        import pandas as pd
        import h5py
    except ImportError as e:
        raise RuntimeError(
            "pandas + h5py are required for LargeST.  "
            "Install via `pip install h5py` (already in requirements.txt)."
        ) from e

    cache = cache_dir() / "largest"
    cache.mkdir(parents=True, exist_ok=True)
    tag = cache / f"largest_gba_{year}_H_v1.npz"
    if tag.exists():
        with np.load(tag, allow_pickle=False) as d:
            return d["Y"].astype(np.float64), d["A"].astype(np.float64)

    print(f"  [largest] downloading raw {year} block (~6.9 GB) on first use ...")
    meta = pd.read_csv(hf_download(
        "ttydsc30/TrafficForecast-LargeST-Project-Dataset",
        "data/raw/metadata.csv",
    ))
    mask = meta["District"].isin(GBA_DISTRICTS).to_numpy()
    idx = np.nonzero(mask)[0].astype(np.int64)
    if idx.size == 0:
        raise RuntimeError(f"GBA subset matched zero sensors in metadata.")

    h5_path = hf_download(
        "ttydsc30/TrafficForecast-LargeST-Project-Dataset",
        f"data/raw/{year}.h5",
    )
    with h5py.File(h5_path, "r") as f:
        block = f["t"]["block0_values"]
        axis1 = f["t"]["axis1"][:]
        Y_raw = block[:, idx].astype(np.float64)
    timestamps_5min = np.asarray(axis1, dtype="datetime64[ns]")
    Y_5min = fill_nans_columnwise(Y_raw)
    Y, _timestamps = _aggregate_hourly(Y_5min, timestamps_5min)

    A_full = np.load(hf_download(
        "ttydsc30/TrafficForecast-LargeST-Project-Dataset",
        "data/raw/adjacency.npy",
    )).astype(np.float64)
    A = A_full[np.ix_(idx, idx)]
    A = 0.5 * (A + A.T)
    np.fill_diagonal(A, 0.0)

    # Dead-sensor pruning: drop sensors with std < 1e-6 on the train slice.
    train_end = int(0.4 * Y.shape[0])
    sd_train = np.std(Y[:train_end], axis=0)
    live = sd_train >= 1e-6
    if (~live).any():
        Y = Y[:, live]
        A = A[np.ix_(live, live)]

    np.savez(tag, Y=Y, A=A)
    return Y, A
