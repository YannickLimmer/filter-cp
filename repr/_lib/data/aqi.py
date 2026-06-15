"""AQI — UCI Beijing 12-site Multi-Site Air-Quality dataset (UCI #501).

Hourly PM2.5 from 12 monitoring stations in Beijing, 2013-03 to 2017-02.
Adjacency built from published station coordinates with a Gaussian
kernel on Haversine distance.

Source: https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data
"""

from __future__ import annotations

import io
import zipfile

import numpy as np

from ..core import cache_dir, fill_nans_columnwise
from ..graph import gaussian_kernel_adjacency
from ._common import download_to


UCI_URL = (
    "https://archive.ics.uci.edu/static/public/501/"
    "beijing+multi+site+air+quality+data.zip"
)

# Published station coordinates (decimal degrees) — Zhang et al. 2017.
STATION_COORDS: dict[str, tuple[float, float]] = {
    "Aotizhongxin":  (39.982, 116.397),
    "Changping":     (40.220, 116.230),
    "Dingling":      (40.292, 116.220),
    "Dongsi":        (39.929, 116.417),
    "Guanyuan":      (39.929, 116.339),
    "Gucheng":       (39.914, 116.184),
    "Huairou":       (40.328, 116.628),
    "Nongzhanguan":  (39.937, 116.461),
    "Shunyi":        (40.127, 116.655),
    "Tiantan":       (39.886, 116.407),
    "Wanliu":        (39.987, 116.287),
    "Wanshouxigong": (39.878, 116.352),
}


def load_aqi() -> tuple[np.ndarray, np.ndarray]:
    try:
        import pandas as pd
    except ImportError as e:
        raise RuntimeError(
            "pandas is required for AQI.  See requirements.txt."
        ) from e

    cache = cache_dir() / "aqi"
    cache.mkdir(parents=True, exist_ok=True)
    npz = cache / "aqi_pm25_v1.npz"
    if npz.exists():
        with np.load(npz, allow_pickle=False) as d:
            return d["Y"].astype(np.float64), d["A"].astype(np.float64)

    zip_path = cache / "aqi_uci_501.zip"
    download_to(UCI_URL, zip_path)
    pieces: dict[str, "pd.Series"] = {}
    with zipfile.ZipFile(zip_path) as outer:
        nested_name = next(
            (n for n in outer.namelist() if n.endswith(".zip") and "PRSA2017" in n),
            None,
        )
        if nested_name is None:
            raise RuntimeError(f"No nested PRSA2017 zip in {zip_path}.")
        with zipfile.ZipFile(io.BytesIO(outer.read(nested_name))) as zf:
            csv_names = [n for n in zf.namelist() if n.endswith(".csv") and "PRSA_Data_" in n]
            for n in sorted(csv_names):
                base = n.rsplit("/", 1)[-1]
                station = base.split("PRSA_Data_")[1].split("_")[0]
                with zf.open(n) as f:
                    df = pd.read_csv(io.BytesIO(f.read()))
                ts = pd.to_datetime(dict(year=df["year"], month=df["month"],
                                         day=df["day"], hour=df["hour"]))
                df = df.set_index(ts).sort_index()
                pieces[station] = df["PM2.5"].astype(np.float64)
    sensors = [s for s in STATION_COORDS if s in pieces]
    panel = pd.concat({s: pieces[s] for s in sensors}, axis=1).sort_index()
    panel = panel.interpolate(method="linear", limit_direction="both", axis=0)
    panel = panel.fillna(panel.mean())
    Y = panel.to_numpy(dtype=np.float64)
    coords = np.asarray([STATION_COORDS[s] for s in sensors], dtype=np.float64)
    A = gaussian_kernel_adjacency(coords)
    np.savez(npz, Y=Y, A=A)
    return Y, A
