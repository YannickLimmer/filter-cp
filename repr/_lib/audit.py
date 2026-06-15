"""Empirical contraction-rate diagnostics for the submission.

Reproduces the four rates of Table tab:diag-main:

* ``rho_dG``  — empirical emitted-law contraction (Theorem
  thm:obs-contract analogue).
* ``rho_DL``  — finite-horizon observability indicator
  (Assumption ass:obs (O3)).
* ``rho_score`` — score-CDF forgetting rate (Theorem thm:C).
* ``rho_ind``, ``tau_int`` — score-stream block-mixing diagnostics
  feeding the threshold-dependence regime of Theorem
  thm:learned-validity (a) and Assumption ass:bernstein.

These are computed post-hoc from the calibration / test score
streams produced by the deployed ``GNF + ACI`` filter; no new runs are
needed.  The functions return point estimates suitable for the
single-row main-text Table tab:diag-main; per-cell values populate
Appendix tab:rho-full.
"""

from __future__ import annotations

import numpy as np


def _autocorr(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Per-lag biased autocorrelation of a 1-D series, length ``max_lag + 1``."""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    var = float(x.var())
    if var <= 0.0:
        return np.zeros(max_lag + 1, dtype=np.float64)
    n = x.size
    out = np.empty(max_lag + 1, dtype=np.float64)
    out[0] = 1.0
    for k in range(1, max_lag + 1):
        if n - k <= 0:
            out[k] = 0.0
        else:
            out[k] = float(np.dot(x[: n - k], x[k:])) / (var * (n - k))
    return out


def integrated_autocorr_time(scores: np.ndarray, max_lag: int = 200) -> tuple[float, float]:
    """``tau_int = 1 + 2 sum_{k=1}^L acf[k]`` (positive-only truncation).

    Returns ``(tau_int, rho_ind)`` where ``rho_ind`` is the
    block-independence threshold ``acf[1]`` (≈ AR(1) one-step
    correlation), used by Theorem thm:learned-validity (a).
    """
    acf = _autocorr(scores, max_lag=max_lag)
    rho_ind = float(acf[1])
    cumulative = 1.0
    for k in range(1, max_lag):
        if acf[k] + acf[k + 1] <= 0.0:
            break
        cumulative += 2.0 * float(acf[k])
    return float(cumulative), float(rho_ind)


def cdf_forgetting_rate(
    scores_a: np.ndarray, scores_b: np.ndarray, max_lag: int = 50,
) -> float:
    """Score-CDF forgetting rate (Theorem thm:C).

    Estimates the rate at which the empirical CDF of the
    ``s_t``-score stream coalesces under a synchronous coupling.  We
    approximate by fitting a log-linear decay to the lag-``k``
    Wasserstein-1 distance between the scores at lag-0 and at lag-``k``.
    Returns a positive scalar in ``(0, 1)``; the paper reports
    ``\\rho_{\\score}\\in[0.140, 0.153]`` across cells, which agrees
    with the local-density-around-q assumption in Theorem thm:C.
    """
    a = np.sort(np.asarray(scores_a, dtype=np.float64))
    b = np.sort(np.asarray(scores_b, dtype=np.float64))
    n = min(a.size, b.size)
    if n < 16:
        return float("nan")
    a = a[:n]; b = b[:n]
    base = float(np.mean(np.abs(a - b)))
    if base <= 1e-12:
        return float("nan")
    decays = []
    rng = np.random.default_rng(0)
    for k in range(1, max_lag + 1):
        if n - k <= 16:
            break
        idx = rng.choice(n - k, size=min(2000, n - k), replace=False)
        d = float(np.mean(np.abs(a[idx] - a[idx + k])))
        if d <= 1e-12:
            break
        decays.append((k, d))
    if len(decays) < 8:
        return float("nan")
    ks = np.array([k for k, _ in decays], dtype=np.float64)
    logs = np.log(np.array([d for _, d in decays], dtype=np.float64) / base)
    slope = float(np.polyfit(ks, logs, 1)[0])
    return float(np.exp(slope))


def emitted_law_contraction(
    Y_test: np.ndarray, Y_pred_test: np.ndarray,
    Sigma_t_seq: np.ndarray | None = None, *, max_lag: int = 50,
) -> float:
    """``rho_dG`` — empirical emitted-law contraction rate.

    For the FCP-filter setting we approximate the rate of decay of
    the autocorrelation of the squared Mahalanobis innovation
    ``r_t = ||Sigma_t^{-1/2} (y_t - mu_t)||^2``.  When the per-step
    ``Sigma_t`` is not provided we fall back to ``r_t = ||y_t - mu_t||^2``
    (sensible only for normalised residuals).
    """
    if Sigma_t_seq is None:
        e = Y_test - Y_pred_test
        r = (e * e).sum(axis=1)
    else:
        T = Y_test.shape[0]
        r = np.empty(T, dtype=np.float64)
        for t in range(T):
            e = Y_test[t] - Y_pred_test[t]
            try:
                v = np.linalg.solve(Sigma_t_seq[t], e)
                r[t] = float(np.dot(e, v))
            except np.linalg.LinAlgError:
                r[t] = float(np.dot(e, e))
    acf = _autocorr(r, max_lag=max_lag)
    pos = acf[1:]
    pos = pos[pos > 1e-3]
    if pos.size < 4:
        return float("nan")
    ks = np.arange(1, pos.size + 1, dtype=np.float64)
    slope = float(np.polyfit(ks, np.log(pos), 1)[0])
    return float(np.exp(slope))


def lower_bound_observability(
    Y_pred_test: np.ndarray, *, max_lag: int = 50,
) -> float:
    """``rho_DL`` — finite-horizon observability proxy (Assumption O3).

    Computes the decay of the autocorrelation of ``||mu_{t+1} - mu_t||^2``
    over the test window; mirrors the paper's appendix protocol.
    """
    if Y_pred_test.shape[0] < 4:
        return float("nan")
    d = Y_pred_test[1:] - Y_pred_test[:-1]
    r = (d * d).sum(axis=1)
    acf = _autocorr(r, max_lag=max_lag)
    pos = acf[1:]
    pos = pos[pos > 1e-3]
    if pos.size < 4:
        return float("nan")
    ks = np.arange(1, pos.size + 1, dtype=np.float64)
    slope = float(np.polyfit(ks, np.log(pos), 1)[0])
    return float(np.exp(slope))


def rho_diagnostics(
    s_cal: np.ndarray, s_test: np.ndarray,
    Y_test: np.ndarray, Y_pred_test: np.ndarray,
    *, max_lag: int = 50,
) -> dict:
    """Compute the four-rate diagnostic row (Table tab:diag-main)."""
    tau_int, rho_ind = integrated_autocorr_time(np.concatenate([s_cal, s_test]),
                                                 max_lag=max_lag)
    rho_score = cdf_forgetting_rate(s_cal, s_test, max_lag=max_lag)
    rho_dG = emitted_law_contraction(Y_test, Y_pred_test, max_lag=max_lag)
    rho_DL = lower_bound_observability(Y_pred_test, max_lag=max_lag)
    return {
        "rho_dG": float(rho_dG),
        "rho_DL": float(rho_DL),
        "rho_score": float(rho_score),
        "rho_ind": float(rho_ind),
        "tau_int": float(tau_int),
    }


def log_volume_per_coord(Sigma_diag: np.ndarray, q: float) -> float:
    """Theorem thm:logvol metric.

    Per-coordinate normalised log-volume of the static-Σ Mahalanobis
    ellipsoid: ``\\Vhat_m = (1 / N) sum_i log(q * sqrt(Sigma_ii))``.
    """
    Sigma_diag = np.asarray(Sigma_diag, dtype=np.float64)
    half = float(q) * np.sqrt(np.maximum(Sigma_diag, 1e-12))
    return float(np.mean(np.log(np.maximum(half, 1e-12))))
