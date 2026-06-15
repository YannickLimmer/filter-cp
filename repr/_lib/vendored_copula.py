"""CopulaCPTS (Wang et al., ICLR 2024) — vendored full-fidelity reference.

Wraps the upstream ``copulaCPTS`` class so it returns ``(lo, hi)``
intervals matching the paper's row.  The two-stage copula optimisation
is identical to the authors' implementation; only the model adapter
(``_PrecomputedPredictor``) differs, since the calibrator consumes
pre-computed point predictions rather than re-running a forecaster.

Paper: https://proceedings.iclr.cc/paper_files/paper/2024/hash/8707924df5e207fa496f729f49069446-Abstract-Conference.html
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _torch_or_raise():
    try:
        import torch
        return torch
    except ImportError as e:
        raise RuntimeError(
            "CopulaCPTS requires PyTorch.  Install via "
            "`pip install -r requirements.txt`."
        ) from e


class _PrecomputedPredictor:
    """Adapter: ``predict(idx) -> (k, N, 1) tensor`` from stored ``y_pred_cal``."""

    def __init__(self, y_pred_cal: np.ndarray) -> None:
        self._preds = y_pred_cal

    def predict(self, idx):
        torch = _torch_or_raise()
        p = self._preds[idx]
        return torch.tensor(p[:, :, None], dtype=torch.float32)


@dataclass
class CopulaCPTSVendored:
    """Joint copula CP — per-node ``radius`` from the copula optimisation."""

    cali_split: float = 0.6
    copula_epochs: int = 800

    _radius: np.ndarray | None = field(default=None, init=False)

    def fit(self, y_cal: np.ndarray, y_pred_cal: np.ndarray, alpha: float
            ) -> "CopulaCPTSVendored":
        torch = _torch_or_raise()
        from ._copulacpts_core import copulaCPTS

        y_cal = np.asarray(y_cal, dtype=np.float64)
        y_pred_cal = np.asarray(y_pred_cal, dtype=np.float64)
        n_cal = y_cal.shape[0]
        model = _PrecomputedPredictor(y_pred_cal)
        idx_all = np.arange(n_cal)
        y_cal_3d = y_cal[:, :, None]
        cp = copulaCPTS(model, idx_all, torch.tensor(y_cal_3d, dtype=torch.float32))
        cp.calibrate()
        _, radius = cp.predict(cp.copula_x, epsilon=alpha)
        self._radius = np.asarray(radius, dtype=np.float64)
        return self

    def predict_intervals(self, y_pred: np.ndarray
                          ) -> tuple[np.ndarray, np.ndarray]:
        if self._radius is None:
            raise RuntimeError("CopulaCPTSVendored.fit must be called first.")
        y_pred = np.asarray(y_pred, dtype=np.float64)
        return y_pred - self._radius[None, :], y_pred + self._radius[None, :]


def copulacpts_predict(
    Y_pilot: np.ndarray, Y_pred_pilot: np.ndarray,
    Y_cal: np.ndarray, Y_pred_cal: np.ndarray,
    Y_test: np.ndarray, Y_pred_test: np.ndarray, *,
    alpha: float, copula_epochs: int = 800,
) -> dict:
    """Reproducibility-suite-shaped wrapper around CopulaCPTS.

    Concatenates pilot + cal as the calibration block (the upstream
    method already does its own internal split via ``cali_split``);
    returns the ``{'lo', 'hi', 'inside_ellipsoid'}`` dict the runner
    expects.  ``inside_ellipsoid`` is the box-coverage indicator since
    CopulaCPTS does not expose a Mahalanobis radius.
    """
    Y_combined = np.concatenate([Y_pilot, Y_cal], axis=0)
    Yp_combined = np.concatenate([Y_pred_pilot, Y_pred_cal], axis=0)
    cp = CopulaCPTSVendored(copula_epochs=copula_epochs)
    cp.fit(Y_combined, Yp_combined, alpha=alpha)
    lo, hi = cp.predict_intervals(Y_pred_test)
    inside = np.all((Y_test >= lo) & (Y_test <= hi), axis=1)
    return {"lo": lo, "hi": hi, "inside_ellipsoid": inside}


__all__ = ["CopulaCPTSVendored", "copulacpts_predict"]
