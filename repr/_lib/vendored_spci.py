"""MultiDimSPCI (Xu & Xie, ICML 2024) — vendored full-fidelity reference.

The upstream implementation routes the Mahalanobis nonconformity score
through ``SPCI_and_EnbPI.get_et``; we keep it intact and only bypass the
bootstrap forecaster training (we feed pre-computed residuals directly).

Paper: https://proceedings.mlr.press/v235/xu24m.html
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class MultiDimSPCIVendored:
    past_window: int = 100

    global_cov_: np.ndarray | None = field(default=None, init=False)
    global_cov_inv_: np.ndarray | None = field(default=None, init=False)
    radius_: float | None = field(default=None, init=False)

    def fit(self, y_cal: np.ndarray, y_pred_cal: np.ndarray, alpha: float
            ) -> "MultiDimSPCIVendored":
        from ._multidimspci_core import SPCI_and_EnbPI

        y_cal = np.asarray(y_cal, dtype=np.float64)
        y_pred_cal = np.asarray(y_pred_cal, dtype=np.float64)
        residuals = y_cal - y_pred_cal
        n_cal, _n_nodes = residuals.shape

        dummy_X = np.zeros((n_cal, 1))
        dummy_Y = residuals
        spci = SPCI_and_EnbPI(
            dummy_X, dummy_X[:1], dummy_Y, dummy_Y[:1], fit_func=lambda *a: None,
        )
        spci.Ensemble_online_resid = np.vstack([residuals, residuals[:1]])
        spci.get_test_et = False
        cal_et = spci.get_et(residuals)
        n = len(cal_et)
        q_level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
        self.radius_ = float(np.quantile(cal_et, q_level))
        self.global_cov_ = spci.global_cov.copy()
        self.global_cov_inv_ = spci.global_cov_inv.copy()
        return self

    def predict_intervals(self, y_pred: np.ndarray
                          ) -> tuple[np.ndarray, np.ndarray]:
        if self.radius_ is None or self.global_cov_ is None:
            raise RuntimeError("MultiDimSPCIVendored.fit must be called first.")
        y_pred = np.asarray(y_pred, dtype=np.float64)
        half = self.radius_ * np.sqrt(np.diag(self.global_cov_))
        return y_pred - half[None, :], y_pred + half[None, :]

    def ellipsoid_joint_coverage(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if self.global_cov_inv_ is None or self.radius_ is None:
            raise RuntimeError("MultiDimSPCIVendored.fit must be called first.")
        e = np.asarray(y_true) - np.asarray(y_pred)
        maha_sq = np.einsum("ti,ij,tj->t", e, self.global_cov_inv_, e)
        return float((maha_sq <= self.radius_ ** 2).mean())


def multidimspci_predict(
    Y_pilot: np.ndarray, Y_pred_pilot: np.ndarray,
    Y_cal: np.ndarray, Y_pred_cal: np.ndarray,
    Y_test: np.ndarray, Y_pred_test: np.ndarray, *,
    alpha: float,
) -> dict:
    """Reproducibility-suite-shaped wrapper around MultiDimSPCI."""
    Y_combined = np.concatenate([Y_pilot, Y_cal], axis=0)
    Yp_combined = np.concatenate([Y_pred_pilot, Y_pred_cal], axis=0)
    sp = MultiDimSPCIVendored()
    sp.fit(Y_combined, Yp_combined, alpha=alpha)
    lo, hi = sp.predict_intervals(Y_pred_test)
    inside_box = np.all((Y_test >= lo) & (Y_test <= hi), axis=1)
    inside_ell = sp.ellipsoid_joint_coverage(Y_test, Y_pred_test)
    return {
        "lo": lo, "hi": hi, "inside_ellipsoid": inside_box,
        "ellipsoid_joint": float(inside_ell),
    }


__all__ = ["MultiDimSPCIVendored", "multidimspci_predict"]
