"""Conformal methods for the submission's publication-name registry.

Each method returns a dict with keys ``lo, hi, inside_ellipsoid``;
:func:`reproduce.run_one_seed` scores them uniformly into
``coverage / joint / mean_halfwidth / winkler``.

Publication names follow the paper's macros (``GNF``, ``KalmanFCP``,
``StaticCGIF``, ``DiagGRU``, ``GCNrankzero``); the factor / group
ablations keep their codebase names where the paper's main table also
does.

Filter rows
~~~~~~~~~~~
* :func:`gnf_predict` — **GNF**: graph-conv GRU + structured Σ_t,
  CGIFFiltered wrapper.
* :func:`gnf_aci_predict` — **GNF + ACI wrap**: same filter, ACI-driven
  α_t.  Algorithm 1 of the paper.
* :func:`kalmanfcp_predict` — **KalmanFCP**: linear graph LG-SSM, the
  honest-negative comparator.
* :func:`diaggru_predict` — **DiagGRU**: plain GRU + diagonal Σ_t (no
  graph mixing, no low-rank factor).
* :func:`gcnrankzero_predict` — **GCN-rank-0**: graph mixing retained,
  covariance head collapsed to diagonal.

Static / non-filter rows
~~~~~~~~~~~~~~~~~~~~~~~~
* :func:`staticcgif_predict` — **StaticCGIF**: full Mahalanobis CP with
  a single empirical Σ̂.
* :func:`factorcgif_predict` — **FactorCGIF** (rank 4): factor +
  residual-max.
* :func:`agacigroupcgif_predict` — **AgACIGroupCGIF**: AgACI
  expert-aggregation over γ values for the max-over-communities
  Mahalanobis score.
* :func:`acipergroupfactorcgif_predict` — **ACIPerGroupFactorCGIF**:
  per-community factor + residual-max ACI.
* :func:`ewmacov_predict` — **EWMACovCGIF**: causal time-varying Σ_t
  via EWMA.
* :func:`rollingcov_predict` — **RollingCovCGIF**: causal time-varying
  Σ_t via a rolling window.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.linalg import solve_triangular

from ._lib_helpers import (
    factor_quad_form, fit_factor_loadings, fit_group_covariance,
    group_mahalanobis,
)
from .core import (
    safe_cholesky, shrinkage_cov, split_conformal_quantile,
)
from .filters import (
    GraphLGSSM, GraphNeuralSSMFilter, NeuralDiagGaussianFilter,
)


# --------------------------------------------------------------------------- #
# 1. Filter wrappers — CGIFFiltered (split-CP) and CGIFFiltered+ACI online    #
# --------------------------------------------------------------------------- #
def fcp_predict(
    filt, Y_train: np.ndarray, Y_cal: np.ndarray, Y_test: np.ndarray,
    *, alpha: float, warmup_steps: int = 50,
) -> dict:
    """CGIFFiltered: per-step Mahalanobis CP with the filter's ``(μ_t, Σ_t)``.

    Calibration scores are
    ``s_t = sqrt((y_t - μ_t)^T Σ_t^{-1} (y_t - μ_t))``;
    ``q`` is the ``ceil((n+1)(1-α))``-th order statistic.  Test runs
    online (filter observes each ``y_test[t]`` before predicting
    ``t + 1``).
    """
    Y_train = np.asarray(Y_train, dtype=np.float64)
    Y_cal = np.asarray(Y_cal, dtype=np.float64)
    Y_test = np.asarray(Y_test, dtype=np.float64)
    warmup_len = min(warmup_steps, Y_train.shape[0])
    Y_warmup = Y_train[-warmup_len:] if warmup_len > 0 else None
    h, P = filt.initial_state(Y_warmup)
    T_cal, N = Y_cal.shape
    scores = np.empty(T_cal, dtype=np.float64)
    for t in range(T_cal):
        h, P, mu_t, Sigma_t = filt.step(h, P, Y_cal[t])
        e = Y_cal[t] - mu_t
        try:
            v = np.linalg.solve(Sigma_t, e)
            scores[t] = float(np.sqrt(np.dot(e, v)))
        except np.linalg.LinAlgError:
            scores[t] = float(np.linalg.norm(e))
    q = split_conformal_quantile(scores, alpha)
    T_test = Y_test.shape[0]
    lo = np.empty((T_test, N), dtype=np.float64)
    hi = np.empty((T_test, N), dtype=np.float64)
    inside = np.empty(T_test, dtype=bool)
    h_t, P_t = h.copy(), P.copy()
    for t in range(T_test):
        h_t, P_t, mu_t, Sigma_t = filt.step(h_t, P_t, Y_test[t])
        std = np.sqrt(np.clip(np.diag(Sigma_t), 0.0, None))
        half = q * std
        lo[t] = mu_t - half
        hi[t] = mu_t + half
        e = Y_test[t] - mu_t
        try:
            v = np.linalg.solve(Sigma_t, e)
            maha = float(np.sqrt(np.dot(e, v)))
        except np.linalg.LinAlgError:
            maha = float(np.linalg.norm(e))
        inside[t] = maha <= q
    return {"lo": lo, "hi": hi, "inside_ellipsoid": inside, "q": float(q)}


def fcp_aci_predict(
    filt, Y_train: np.ndarray, Y_cal: np.ndarray, Y_test: np.ndarray,
    *, alpha: float, gamma: float = 5e-3,
    window: int | None = None, warmup_steps: int = 50,
) -> dict:
    """ACI-wrapped filtered CP — α_t adapts on joint mis-coverage.

    Pilot-frozen filter; conformal radius rolled from
    ``calibration scores ∪ accumulated test scores`` at adaptive α_t
    (Gibbs--Candès update).
    """
    Y_train = np.asarray(Y_train, dtype=np.float64)
    Y_cal = np.asarray(Y_cal, dtype=np.float64)
    Y_test = np.asarray(Y_test, dtype=np.float64)
    warmup_len = min(warmup_steps, Y_train.shape[0])
    Y_warmup = Y_train[-warmup_len:] if warmup_len > 0 else None
    h, P = filt.initial_state(Y_warmup)
    T_cal, N = Y_cal.shape
    s_cal = np.empty(T_cal, dtype=np.float64)
    for t in range(T_cal):
        h, P, mu_t, Sigma_t = filt.step(h, P, Y_cal[t])
        e = Y_cal[t] - mu_t
        try:
            v = np.linalg.solve(Sigma_t, e)
            s_cal[t] = float(np.sqrt(np.dot(e, v)))
        except np.linalg.LinAlgError:
            s_cal[t] = float(np.linalg.norm(e))

    T_test = Y_test.shape[0]
    lo = np.empty((T_test, N), dtype=np.float64)
    hi = np.empty((T_test, N), dtype=np.float64)
    inside = np.empty(T_test, dtype=bool)
    alpha_t = float(alpha)
    s_hist: list[float] = []
    h_t, P_t = h.copy(), P.copy()
    for t in range(T_test):
        if window is not None and window > 0 and s_hist:
            pool = np.concatenate([s_cal, np.asarray(s_hist[-window:])])
        elif s_hist:
            pool = np.concatenate([s_cal, np.asarray(s_hist)])
        else:
            pool = s_cal
        if alpha_t >= 1.0 - 1e-9:
            q_t = float(np.inf)
        elif alpha_t <= 1e-9:
            q_t = float(np.max(pool))
        else:
            q_t = split_conformal_quantile(pool, alpha_t)
        h_t, P_t, mu_t, Sigma_t = filt.step(h_t, P_t, Y_test[t])
        std = np.sqrt(np.clip(np.diag(Sigma_t), 0.0, None))
        half = q_t * std
        lo[t] = mu_t - half
        hi[t] = mu_t + half
        e = Y_test[t] - mu_t
        try:
            v = np.linalg.solve(Sigma_t, e)
            maha = float(np.sqrt(np.dot(e, v)))
        except np.linalg.LinAlgError:
            maha = float(np.linalg.norm(e))
        ok = maha <= q_t
        inside[t] = ok
        alpha_t = alpha_t + gamma * (float(alpha) - (0.0 if ok else 1.0))
        alpha_t = float(np.clip(alpha_t, 1e-4, 1.0 - 1e-4))
        s_hist.append(float(maha))
    return {"lo": lo, "hi": hi, "inside_ellipsoid": inside}


# --------------------------------------------------------------------------- #
# 2. The five filter rows                                                     #
# --------------------------------------------------------------------------- #
def gnf_predict(
    Y_train_z: np.ndarray, Y_cal_z: np.ndarray, Y_test_z: np.ndarray,
    *, S_sub: np.ndarray, alpha: float, hidden: int = 32, low_rank: int = 4,
    epochs: int = 60, warmup_steps: int = 50, seed: int = 0,
) -> dict:
    """**GNF** — graph-conv GRU + structured Σ_t = diag + LL^T."""
    filt = GraphNeuralSSMFilter(
        hidden_size=hidden, low_rank=low_rank, epochs=epochs, seed=seed,
    ).fit(Y_train_z, S_sub)
    return fcp_predict(filt, Y_train_z, Y_cal_z, Y_test_z,
                       alpha=alpha, warmup_steps=warmup_steps)


def gnf_aci_predict(
    Y_train_z: np.ndarray, Y_cal_z: np.ndarray, Y_test_z: np.ndarray,
    *, S_sub: np.ndarray, alpha: float, hidden: int = 32, low_rank: int = 4,
    epochs: int = 60, gamma: float = 5e-3, window: int | None = None,
    warmup_steps: int = 50, seed: int = 0,
) -> dict:
    """**GNF + ACI** — Algorithm 1 (deployed online)."""
    filt = GraphNeuralSSMFilter(
        hidden_size=hidden, low_rank=low_rank, epochs=epochs, seed=seed,
    ).fit(Y_train_z, S_sub)
    return fcp_aci_predict(filt, Y_train_z, Y_cal_z, Y_test_z,
                           alpha=alpha, gamma=gamma, window=window,
                           warmup_steps=warmup_steps)


def kalmanfcp_predict(
    Y_train_z: np.ndarray, Y_cal_z: np.ndarray, Y_test_z: np.ndarray,
    *, S_sub: np.ndarray, alpha: float, rho_scale: float = 0.8,
    warmup_steps: int = 50,
) -> dict:
    """**KalmanFCP** — linear graph LG-SSM, the honest-negative row."""
    filt = GraphLGSSM(rho_scale=rho_scale).fit(Y_train_z, S_sub)
    return fcp_predict(filt, Y_train_z, Y_cal_z, Y_test_z,
                       alpha=alpha, warmup_steps=warmup_steps)


def diaggru_predict(
    Y_train_z: np.ndarray, Y_cal_z: np.ndarray, Y_test_z: np.ndarray,
    *, S_sub: np.ndarray, alpha: float, hidden: int = 64,
    epochs: int = 80, warmup_steps: int = 50, seed: int = 0,
) -> dict:
    """**DiagGRU** — plain GRU + diagonal Σ_t."""
    filt = NeuralDiagGaussianFilter(
        hidden_size=hidden, epochs=epochs, seed=seed,
    ).fit(Y_train_z, S_sub)
    return fcp_predict(filt, Y_train_z, Y_cal_z, Y_test_z,
                       alpha=alpha, warmup_steps=warmup_steps)


def gcnrankzero_predict(
    Y_train_z: np.ndarray, Y_cal_z: np.ndarray, Y_test_z: np.ndarray,
    *, S_sub: np.ndarray, alpha: float, hidden: int = 32,
    epochs: int = 60, warmup_steps: int = 50, seed: int = 0,
) -> dict:
    """**GCN-rank-0** — graph mixing retained, low-rank head collapsed.

    Isolates the covariance rank at fixed graph mixing
    (Appendix tab:rank in the paper).
    """
    filt = GraphNeuralSSMFilter(
        hidden_size=hidden, low_rank=0, epochs=epochs, seed=seed,
    ).fit(Y_train_z, S_sub)
    return fcp_predict(filt, Y_train_z, Y_cal_z, Y_test_z,
                       alpha=alpha, warmup_steps=warmup_steps)


# --------------------------------------------------------------------------- #
# 3. Static / non-filter rows                                                 #
# --------------------------------------------------------------------------- #
def staticcgif_predict(
    Y_pilot: np.ndarray, Y_pred_pilot: np.ndarray,
    Y_cal: np.ndarray, Y_pred_cal: np.ndarray,
    Y_test: np.ndarray, Y_pred_test: np.ndarray, *,
    alpha: float, shrinkage: float = 0.10,
) -> dict:
    """**StaticCGIF** — full Mahalanobis CP with a single empirical Σ̂."""
    e_pilot = Y_pilot - Y_pred_pilot
    Sigma = shrinkage_cov(e_pilot, shrinkage=shrinkage)
    L = safe_cholesky(Sigma)
    L_inv = solve_triangular(L, np.eye(L.shape[0]), lower=True)
    e_cal = Y_cal - Y_pred_cal
    z_cal = e_cal @ L_inv.T
    s_cal = np.sqrt((z_cal * z_cal).sum(axis=1))
    q = split_conformal_quantile(s_cal, alpha)
    std = np.sqrt(np.diag(Sigma))
    half = q * std
    lo = Y_pred_test - half[None, :]
    hi = Y_pred_test + half[None, :]
    e_test = Y_test - Y_pred_test
    z_test = e_test @ L_inv.T
    s_test = np.sqrt((z_test * z_test).sum(axis=1))
    return {"lo": lo, "hi": hi,
            "inside_ellipsoid": (s_test <= q).astype(np.bool_),
            "q": float(q)}


def factorcgif_predict(
    Y_pilot: np.ndarray, Y_pred_pilot: np.ndarray,
    Y_cal: np.ndarray, Y_pred_cal: np.ndarray,
    Y_test: np.ndarray, Y_pred_test: np.ndarray, *,
    alpha: float, rank: int = 4, alpha_factor_share: float = 0.5,
    sigma_floor: float = 1e-3,
) -> dict:
    """**FactorCGIF** (rank ``r``) — factor + residual-max CP, static."""
    e_pilot = Y_pilot - Y_pred_pilot
    e_mean = e_pilot.mean(axis=0)
    B, Sigma_F, sigma_R = fit_factor_loadings(e_pilot, rank=rank, sigma_floor=sigma_floor)
    sigma_R = np.maximum(sigma_R, sigma_floor)
    e_cal = Y_cal - Y_pred_cal
    ec = e_cal - e_mean[None, :]
    F_cal = ec @ B
    R_cal = ec - F_cal @ B.T
    sF_cal = factor_quad_form(F_cal, Sigma_F)
    sR_cal = (np.abs(R_cal) / sigma_R[None, :]).max(axis=1)
    aF = alpha_factor_share * alpha
    aR = alpha - aF
    qF = split_conformal_quantile(sF_cal, aF)
    qR = split_conformal_quantile(sR_cal, aR)
    node_factor_var = np.einsum("ij,jk,ik->i", B, Sigma_F, B)
    node_factor_std = np.sqrt(np.maximum(node_factor_var, 0.0))
    half = qF * node_factor_std + qR * sigma_R
    centre = Y_pred_test + e_mean[None, :]
    lo = centre - half[None, :]
    hi = centre + half[None, :]
    e_test = Y_test - Y_pred_test
    ect = e_test - e_mean[None, :]
    F_t = ect @ B
    R_t = ect - F_t @ B.T
    sF_t = factor_quad_form(F_t, Sigma_F)
    sR_t = (np.abs(R_t) / sigma_R[None, :]).max(axis=1)
    inside = (sF_t <= qF) & (sR_t <= qR)
    return {"lo": lo, "hi": hi, "inside_ellipsoid": inside}


def _aci_q(scores: np.ndarray, alpha_t: float) -> float:
    if alpha_t >= 1.0 - 1e-9:
        return float(np.inf)
    if alpha_t <= 1e-9:
        return float(np.max(scores)) if len(scores) > 0 else 0.0
    return split_conformal_quantile(scores, alpha_t)


def acipergroupfactorcgif_predict(
    groups: Sequence[np.ndarray],
    Y_pilot: np.ndarray, Y_pred_pilot: np.ndarray,
    Y_cal: np.ndarray, Y_pred_cal: np.ndarray,
    Y_test: np.ndarray, Y_pred_test: np.ndarray, *,
    alpha: float, rank: int = 4, alpha_factor_share: float = 0.5,
    gamma: float = 5e-3, window: int | None = None,
    sigma_floor: float = 1e-3,
) -> dict:
    """**ACIPerGroupFactorCGIF** — per-community factor + residual-max ACI."""
    if not 0.0 < alpha_factor_share < 1.0:
        raise ValueError("alpha_factor_share must lie in (0, 1).")
    theta_F = float(alpha_factor_share)
    theta_R = 1.0 - theta_F

    e_pilot = Y_pilot - Y_pred_pilot
    n, N = e_pilot.shape
    e_mean = e_pilot.mean(axis=0)
    B_list: list[np.ndarray] = []
    Sigma_F_list: list[np.ndarray] = []
    sigma_R = np.zeros(N, dtype=np.float64)
    for g in groups:
        g = np.asarray(g, dtype=np.intp)
        r_eff = int(min(max(1, rank), min(g.size - 1, n - 1)))
        B_c, Sigma_Fc, sigma_Rc = fit_factor_loadings(e_pilot[:, g], rank=r_eff, sigma_floor=sigma_floor)
        sigma_Rc = np.maximum(sigma_Rc, sigma_floor)
        B_list.append(B_c)
        Sigma_F_list.append(Sigma_Fc)
        sigma_R[g] = sigma_Rc
    node_factor_std = np.zeros(N, dtype=np.float64)
    for g_idx, g in enumerate(groups):
        g = np.asarray(g, dtype=np.intp)
        var_c = np.einsum("ij,jk,ik->i", B_list[g_idx], Sigma_F_list[g_idx], B_list[g_idx])
        node_factor_std[g] = np.sqrt(np.maximum(var_c, 0.0))

    def _scores(Y_obs, Y_pred):
        ect = (Y_obs - Y_pred) - e_mean[None, :]
        n_t = ect.shape[0]
        sF = np.zeros((n_t, len(groups)), dtype=np.float64)
        sR = np.zeros((n_t, len(groups)), dtype=np.float64)
        for gi, g in enumerate(groups):
            g = np.asarray(g, dtype=np.intp)
            e_g = ect[:, g]
            F_c = e_g @ B_list[gi]
            R_c = e_g - F_c @ B_list[gi].T
            sF[:, gi] = factor_quad_form(F_c, Sigma_F_list[gi])
            sR[:, gi] = (np.abs(R_c) / sigma_R[g][None, :]).max(axis=1)
        return sF.max(axis=1), sR.max(axis=1)

    sF_cal, sR_cal = _scores(Y_cal, Y_pred_cal)
    T_test = Y_test.shape[0]
    lo = np.empty_like(Y_pred_test)
    hi = np.empty_like(Y_pred_test)
    inside = np.empty(T_test, dtype=bool)
    alpha_t = float(alpha)
    sF_hist: list[float] = []
    sR_hist: list[float] = []
    for t in range(T_test):
        if window is not None and window > 0 and sF_hist:
            pool_F = np.concatenate([sF_cal, np.asarray(sF_hist[-window:])])
            pool_R = np.concatenate([sR_cal, np.asarray(sR_hist[-window:])])
        elif sF_hist:
            pool_F = np.concatenate([sF_cal, np.asarray(sF_hist)])
            pool_R = np.concatenate([sR_cal, np.asarray(sR_hist)])
        else:
            pool_F, pool_R = sF_cal, sR_cal
        q_F = _aci_q(pool_F, theta_F * alpha_t)
        q_R = _aci_q(pool_R, theta_R * alpha_t)
        half = q_F * node_factor_std + q_R * sigma_R
        centre = Y_pred_test[t] + e_mean
        lo[t] = centre - half
        hi[t] = centre + half
        sF_row, sR_row = _scores(Y_test[t : t + 1], Y_pred_test[t : t + 1])
        sF_t = float(sF_row[0]); sR_t = float(sR_row[0])
        ok = (sF_t <= q_F) and (sR_t <= q_R)
        inside[t] = ok
        alpha_t = alpha_t + gamma * (float(alpha) - (0.0 if ok else 1.0))
        alpha_t = float(np.clip(alpha_t, 1e-4, 1.0 - 1e-4))
        sF_hist.append(sF_t); sR_hist.append(sR_t)
    return {"lo": lo, "hi": hi, "inside_ellipsoid": inside}


def agacigroupcgif_predict(
    groups: Sequence[np.ndarray],
    Y_pilot: np.ndarray, Y_pred_pilot: np.ndarray,
    Y_cal: np.ndarray, Y_pred_cal: np.ndarray,
    Y_test: np.ndarray, Y_pred_test: np.ndarray, *,
    alpha: float,
    gammas: Sequence[float] = (1e-3, 5e-3, 1e-2, 2e-2, 5e-2),
    eta: float = 1.0, window: int | None = None,
    sigma_floor: float = 1e-3,
) -> dict:
    """**AgACIGroupCGIF** — AgACI expert aggregation over γ on max-over-communities Mahalanobis.

    K experts each run ACI-GroupCGIF with a different ``γ``; per-step
    the half-widths are mixed by Bernstein-style multiplicative
    weights with rate ``η`` on the realised mis-coverage cost.
    """
    e_pilot = Y_pilot - Y_pred_pilot
    n, N = e_pilot.shape
    e_mean = e_pilot.mean(axis=0)
    L_group: list[np.ndarray] = []
    diag_per_node = np.zeros(N, dtype=np.float64)
    for g in groups:
        g = np.asarray(g, dtype=np.intp)
        Sigma_c, L_c, _ = fit_group_covariance(
            e_pilot[:, g] - e_mean[g][None, :], sigma_floor=sigma_floor,
        )
        L_group.append(L_c)
        diag_per_node[g] = np.sqrt(np.maximum(np.diag(Sigma_c), 0.0))

    def _scores(Y_obs, Y_pred):
        ect = (Y_obs - Y_pred) - e_mean[None, :]
        n_t = ect.shape[0]
        s = np.zeros((n_t, len(groups)), dtype=np.float64)
        for gi, g in enumerate(groups):
            g = np.asarray(g, dtype=np.intp)
            s[:, gi] = group_mahalanobis(ect[:, g], L_group[gi])
        return s.max(axis=1)

    s_cal = _scores(Y_cal, Y_pred_cal)
    K = len(gammas)
    alpha_t = np.full(K, float(alpha), dtype=np.float64)
    weights = np.full(K, 1.0 / K, dtype=np.float64)
    T_test = Y_test.shape[0]
    lo = np.empty_like(Y_pred_test)
    hi = np.empty_like(Y_pred_test)
    inside = np.empty(T_test, dtype=bool)
    s_hist: list[float] = []

    for t in range(T_test):
        if window is not None and window > 0 and s_hist:
            pool = np.concatenate([s_cal, np.asarray(s_hist[-window:])])
        elif s_hist:
            pool = np.concatenate([s_cal, np.asarray(s_hist)])
        else:
            pool = s_cal
        q_per_expert = np.array([_aci_q(pool, a) for a in alpha_t], dtype=np.float64)
        # Mix radii by current weights.
        finite = np.isfinite(q_per_expert)
        if not finite.any():
            q_t = float(np.max(pool))
        else:
            q_per_expert = np.where(finite, q_per_expert, q_per_expert[finite].max())
            q_t = float(np.dot(weights, q_per_expert))
        half = q_t * diag_per_node
        centre = Y_pred_test[t] + e_mean
        lo[t] = centre - half
        hi[t] = centre + half
        s_t = float(_scores(Y_test[t : t + 1], Y_pred_test[t : t + 1])[0])
        ok = s_t <= q_t
        inside[t] = ok
        # Per-expert miss + interval cost (interval score with α)
        miss = np.array([1.0 if s_t > q_e else 0.0 for q_e in q_per_expert], dtype=np.float64)
        loss = np.where(finite, q_per_expert + (2.0 / float(alpha)) * miss * (s_t - q_per_expert).clip(min=0.0),
                        q_per_expert.max() * 2.0)
        loss = loss - loss.min()
        weights = weights * np.exp(-eta * loss)
        weights = weights / weights.sum()
        # ACI update per expert.
        for k, g_k in enumerate(gammas):
            alpha_t[k] = alpha_t[k] + g_k * (float(alpha) - (0.0 if (s_t <= q_per_expert[k]) else 1.0))
            alpha_t[k] = float(np.clip(alpha_t[k], 1e-4, 1.0 - 1e-4))
        s_hist.append(s_t)
    return {"lo": lo, "hi": hi, "inside_ellipsoid": inside}


# --------------------------------------------------------------------------- #
# 4. EWMA / Rolling time-varying-Sigma baselines (appendix tab:hero-full)    #
# --------------------------------------------------------------------------- #
def _mahalanobis(e: np.ndarray, Sigma: np.ndarray) -> float:
    try:
        L = safe_cholesky(Sigma)
    except np.linalg.LinAlgError:
        return float(np.linalg.norm(e))
    z = np.linalg.solve(L, e)
    return float(np.sqrt(float(z @ z)))


def _condcov_predict(
    kind: str,
    Y_pilot: np.ndarray, Y_pred_pilot: np.ndarray,
    Y_cal: np.ndarray, Y_pred_cal: np.ndarray,
    Y_test: np.ndarray, Y_pred_test: np.ndarray, *,
    alpha: float, half_life: float = 288.0, window: int = 288,
    shrinkage: float = 0.05, jitter: float = 1e-6,
) -> dict:
    e_pilot = Y_pilot - Y_pred_pilot
    e_cal = Y_cal - Y_pred_cal
    e_test = Y_test - Y_pred_test
    N = e_cal.shape[1]
    Sigma_init = shrinkage_cov(e_pilot, shrinkage=shrinkage)
    if kind == "ewma":
        w = 1.0 - 0.5 ** (1.0 / float(half_life))
        Sigma_t = Sigma_init.copy()
        for e in e_pilot:
            Sigma_t = (1.0 - w) * Sigma_t + w * np.outer(e, e)
    else:
        history: list[np.ndarray] = list(e_pilot[max(0, e_pilot.shape[0] - window):])
        Sigma_t = Sigma_init.copy()
    T_cal = e_cal.shape[0]
    scores = np.empty(T_cal, dtype=np.float64)
    for t in range(T_cal):
        if kind == "rolling":
            if history:
                Sigma_t = shrinkage_cov(np.asarray(history), shrinkage=shrinkage)
            else:
                Sigma_t = Sigma_init
        Sigma_reg = Sigma_t + jitter * np.eye(N)
        scores[t] = _mahalanobis(e_cal[t], Sigma_reg)
        if kind == "ewma":
            Sigma_t = (1.0 - w) * Sigma_t + w * np.outer(e_cal[t], e_cal[t])
        else:
            history.append(e_cal[t])
            if len(history) > window:
                history = history[-window:]
    q = split_conformal_quantile(scores, alpha)
    T_test = e_test.shape[0]
    lo = np.empty((T_test, N), dtype=np.float64)
    hi = np.empty((T_test, N), dtype=np.float64)
    inside = np.empty(T_test, dtype=bool)
    for t in range(T_test):
        if kind == "rolling":
            if history:
                Sigma_t = shrinkage_cov(np.asarray(history), shrinkage=shrinkage)
            else:
                Sigma_t = Sigma_init
        Sigma_reg = Sigma_t + jitter * np.eye(N)
        std = np.sqrt(np.clip(np.diag(Sigma_reg), 0.0, None))
        half = q * std
        lo[t] = Y_pred_test[t] - half
        hi[t] = Y_pred_test[t] + half
        inside[t] = _mahalanobis(e_test[t], Sigma_reg) <= q
        if kind == "ewma":
            Sigma_t = (1.0 - w) * Sigma_t + w * np.outer(e_test[t], e_test[t])
        else:
            history.append(e_test[t])
            if len(history) > window:
                history = history[-window:]
    return {"lo": lo, "hi": hi, "inside_ellipsoid": inside, "q": float(q)}


def ewmacov_predict(
    Y_pilot: np.ndarray, Y_pred_pilot: np.ndarray,
    Y_cal: np.ndarray, Y_pred_cal: np.ndarray,
    Y_test: np.ndarray, Y_pred_test: np.ndarray, *,
    alpha: float, half_life: float = 288.0, shrinkage: float = 0.05,
) -> dict:
    """**EWMACovCGIF** — exponentially-weighted Σ_t."""
    return _condcov_predict("ewma", Y_pilot, Y_pred_pilot, Y_cal, Y_pred_cal,
                            Y_test, Y_pred_test, alpha=alpha,
                            half_life=half_life, shrinkage=shrinkage)


def rollingcov_predict(
    Y_pilot: np.ndarray, Y_pred_pilot: np.ndarray,
    Y_cal: np.ndarray, Y_pred_cal: np.ndarray,
    Y_test: np.ndarray, Y_pred_test: np.ndarray, *,
    alpha: float, window: int = 288, shrinkage: float = 0.05,
) -> dict:
    """**RollingCovCGIF** — rolling-window Σ_t."""
    return _condcov_predict("rolling", Y_pilot, Y_pred_pilot, Y_cal, Y_pred_cal,
                            Y_test, Y_pred_test, alpha=alpha,
                            window=window, shrinkage=shrinkage)
