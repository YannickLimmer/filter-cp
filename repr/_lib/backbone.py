"""GraphPolyVAR forecaster — the shared backbone used by every method."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class GraphPolyVAR:
    """``y_t = sum_{k, l} theta_{k, l} (S^k y_{t-l}) + c``, fit by ridge OLS.

    Matches the ``\\mu(\\cdot)`` used in every conformal row of the
    paper's main table; cross-row variation is calibration-only.
    """

    S: np.ndarray
    K: int = 2
    L: int = 2
    lam: float = 1e-2

    Theta_: np.ndarray = field(default_factory=lambda: np.zeros(0), init=False)
    intercept_: float = field(default=0.0, init=False)
    fallback_: np.ndarray = field(default_factory=lambda: np.zeros(0), init=False)
    _powers: list[np.ndarray] = field(default_factory=list, init=False, repr=False)

    def _ensure_powers(self) -> None:
        if self._powers and len(self._powers) == self.K:
            return
        N = self.S.shape[0]
        P = [np.eye(N, dtype=np.float64)]
        for _ in range(self.K - 1):
            P.append(P[-1] @ self.S)
        self._powers = P

    def fit(self, Y_train: np.ndarray) -> "GraphPolyVAR":
        self._ensure_powers()
        T, N = Y_train.shape
        rows_X, rows_y = [], []
        for t in range(self.L, T):
            for_t = np.empty((N, self.K * self.L), dtype=np.float64)
            for l in range(self.L):
                y_lag = Y_train[t - 1 - l]
                for k in range(self.K):
                    for_t[:, l * self.K + k] = self._powers[k] @ y_lag
            rows_X.append(for_t)
            rows_y.append(Y_train[t])
        X = np.concatenate(rows_X, axis=0)
        y = np.concatenate(rows_y, axis=0)
        X_ = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
        d = self.K * self.L
        I_pen = np.eye(d + 1, dtype=np.float64) * self.lam
        I_pen[-1, -1] = 0.0
        beta = np.linalg.solve(X_.T @ X_ + I_pen, X_.T @ y)
        self.Theta_ = beta[:-1].reshape(self.L, self.K).T
        self.intercept_ = float(beta[-1])
        self.fallback_ = Y_train.mean(axis=0)
        return self

    def predict_one_step(self, Y_history: np.ndarray) -> np.ndarray:
        T, N = Y_history.shape
        self._ensure_powers()
        out = np.empty_like(Y_history, dtype=np.float64)
        out[: self.L] = self.fallback_
        for t in range(self.L, T):
            y_t = np.full(N, self.intercept_, dtype=np.float64)
            for l in range(self.L):
                y_lag = Y_history[t - 1 - l]
                for k in range(self.K):
                    y_t = y_t + self.Theta_[k, l] * (self._powers[k] @ y_lag)
            out[t] = y_t
        return out
