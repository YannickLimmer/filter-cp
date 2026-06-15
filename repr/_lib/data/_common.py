"""Shared dataset utilities — atomic download, GiftEval reader, types."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..core import cache_dir


@dataclass(frozen=True)
class LoadedCell:
    """``(Y, A)`` plus optional per-dataset metadata."""

    Y: np.ndarray
    A: np.ndarray


def download_to(url: str, dest: Path, *, force: bool = False, timeout: int = 300) -> Path:
    """Atomically download ``url`` to ``dest``.  Idempotent."""
    if dest.exists() and not force:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        import requests
    except ImportError as e:
        raise RuntimeError(
            "requests is required for dataset downloads.  Install via "
            "`pip install -r requirements.txt`."
        ) from e
    print(f"  [download] {url} -> {dest}")
    r = requests.get(url, timeout=timeout, stream=True)
    r.raise_for_status()
    tmp = dest.with_suffix(dest.suffix + ".part")
    with open(tmp, "wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 20):
            if chunk:
                f.write(chunk)
    tmp.rename(dest)
    return dest


def hf_download(repo_id: str, filename: str) -> Path:
    """``huggingface_hub`` proxy — caches under HF cache dir."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise RuntimeError(
            f"huggingface_hub is required to load {repo_id}.  Install via "
            "`pip install -r requirements.txt`."
        ) from e
    return Path(hf_hub_download(
        repo_id=repo_id, repo_type="dataset", filename=filename, token=False,
    ))


def gifteval_panel(
    task: str, *, freq: str = "H", n_items: int | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Load one Salesforce/GiftEval task as a ``(T, N)`` numpy panel.

    Handles both storage flavours used by the mirror: a single-row
    fixed-size-list (multivariate-as-row, ETT / Jena) and N rows of
    list (univariate-per-row, Solar / Loop-Seattle).
    """
    try:
        import pyarrow as pa
        import pandas as pd
    except ImportError as e:
        raise RuntimeError(
            "pyarrow and pandas are required for GiftEval datasets.  "
            "Install via `pip install -r requirements.txt`."
        ) from e
    arrow_path = hf_download(
        "Salesforce/GiftEval", f"{task}/{freq}/data-00000-of-00001.arrow",
    )
    table = pa.ipc.open_stream(arrow_path).read_all()
    target_field = table.schema.field("target")
    target_type = target_field.type
    if pa.types.is_fixed_size_list(target_type):
        if table.num_rows != 1:
            raise RuntimeError(
                f"Expected 1 row for multivariate task {task}/{freq}, "
                f"got {table.num_rows}."
            )
        row = table.slice(0, 1).to_pylist()[0]
        Y = np.asarray(row["target"], dtype=np.float64)
        if Y.ndim != 2:
            raise RuntimeError(f"Multivariate target shape {Y.shape}.")
        Y = Y.T
        N = Y.shape[1]
        names = [f"ch_{i:03d}" for i in range(N)]
        if n_items is not None:
            N = min(N, n_items)
            Y = Y[:, :N]
            names = names[:N]
        return Y, names
    if pa.types.is_list(target_type):
        df = table.to_pandas().sort_values("item_id").reset_index(drop=True)
        if n_items is not None:
            df = df.head(n_items).reset_index(drop=True)
        targets = [np.asarray(t, dtype=np.float64) for t in df["target"].tolist()]
        T_panel = min(t.shape[0] for t in targets)
        Y = np.stack([t[:T_panel] for t in targets], axis=1)
        names = df["item_id"].astype(str).tolist()
        return Y, names
    raise RuntimeError(f"Unsupported target type for {task}/{freq}: {target_type}.")


def npz_cached(path: Path):
    """Decorator-style helper: load NPZ if present, else build."""
    return path
