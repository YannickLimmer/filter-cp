"""Latent filters that drive the filtered-CP (FCP) wrapper.

Three filters share a common interface — ``fit(Y, S)`` /
``initial_state(Y_warmup)`` / ``step(h, P, y) -> (h_new, P_new, mu_pred,
Sigma_pred)`` — so the same :func:`fcp_predict` wrapper plugs into all
three.

* :class:`GraphLGSSM` — linear graph Kalman filter
  (``KalmanFCP`` row in the paper).  Pure NumPy.
* :class:`NeuralDiagGaussianFilter` — plain GRU + diagonal Σ_t
  (``DiagGRU`` row in the paper).  PyTorch.
* :class:`GraphNeuralSSMFilter` — graph-convolutional GRU + structured
  Σ_t = diag(d_t) + L_t L_t^T (the paper's hero, ``GNF`` row).
  PyTorch.

A ``low_rank=0`` :class:`GraphNeuralSSMFilter` produces the paper's
``GCNrankzero`` ablation: graph mixing retained, covariance head
collapsed to diagonal.

PyTorch is a hard requirement here; the script aborts at startup if
``torch`` is missing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError as _exc:
    raise RuntimeError(
        "PyTorch is required — the GraphNeuralSSM and NeuralDiagGaussian "
        "filters are core submission methods, not optional baselines.  "
        "Install via `pip install -r requirements.txt`."
    ) from _exc


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
# 1. GraphLGSSM (NumPy)                                                       #
# --------------------------------------------------------------------------- #
def _spectral_radius(M: np.ndarray) -> float:
    return float(np.max(np.abs(np.linalg.eigvals(M))))


def _build_graph_poly_F(S: np.ndarray, K: int, rho_scale: float) -> np.ndarray:
    """``F`` with spectral radius ``rho_scale`` along the dominant graph direction."""
    N = S.shape[0]
    base = S.copy()
    if K > 1:
        Sk = S.copy()
        for _ in range(1, K):
            Sk = Sk @ S
            base = base + Sk / K
    sr = _spectral_radius(base)
    if sr < 1e-10:
        return rho_scale * np.eye(N, dtype=np.float64)
    return (rho_scale / sr) * base


@dataclass
class GraphLGSSM:
    """Graph-polynomial linear Gaussian SSM with Kalman-filter inference.

    Transition ``F`` is a scaled graph polynomial with spectral radius
    ``rho_scale``; ``Q`` and ``R`` are isotropic, calibrated to the
    empirical innovation variance on the training set.  Drives the
    paper's ``KalmanFCP`` row.
    """

    rho_scale: float = 0.8
    K: int = 1
    obs_noise_frac: float = 0.1

    F_: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)), init=False)
    Q_: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)), init=False)
    R_: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)), init=False)
    P0_: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)), init=False)
    _N: int = field(default=0, init=False)

    def fit(self, Y_train: np.ndarray, S: np.ndarray) -> "GraphLGSSM":
        Y = np.asarray(Y_train, dtype=np.float64)
        S = np.asarray(S, dtype=np.float64)
        T, N = Y.shape
        if not 0.0 < self.rho_scale < 1.0:
            raise ValueError(f"rho_scale must be in (0, 1), got {self.rho_scale}.")
        self._N = N
        self.F_ = _build_graph_poly_F(S, self.K, self.rho_scale)
        sig2 = float(np.mean(np.var(Y, axis=0)))
        q2 = sig2 * (1.0 - self.rho_scale ** 2)
        r2 = sig2 * self.obs_noise_frac
        Q_init = q2 * np.eye(N, dtype=np.float64)
        R_init = r2 * np.eye(N, dtype=np.float64)
        warmup = max(int(0.2 * T), 10)
        h = np.zeros(N, dtype=np.float64)
        P = sig2 * np.eye(N, dtype=np.float64)
        innov_sq, sigma_tr, n_cal = 0.0, 0.0, 0
        for t in range(T):
            h_prior = self.F_ @ h
            P_prior = self.F_ @ P @ self.F_.T + Q_init
            Sigma_pred = P_prior + R_init
            innov = Y[t] - h_prior
            K_ = np.linalg.solve(Sigma_pred.T, P_prior.T).T
            h = h_prior + K_ @ innov
            P = (np.eye(N) - K_) @ P_prior
            if t >= warmup:
                innov_sq += float(np.dot(innov, innov))
                sigma_tr += float(np.trace(Sigma_pred))
                n_cal += 1
        scale = 1.0
        if n_cal > 0 and sigma_tr > 1e-12:
            scale = (innov_sq / n_cal) / (sigma_tr / n_cal)
        self.Q_ = (q2 * scale) * np.eye(N, dtype=np.float64)
        self.R_ = (r2 * scale) * np.eye(N, dtype=np.float64)
        self.P0_ = sig2 * np.eye(N, dtype=np.float64)
        return self

    def initial_state(self, Y_warmup: np.ndarray | None = None):
        if self.F_.size == 0:
            raise RuntimeError("GraphLGSSM.fit must be called first.")
        h = np.zeros(self._N, dtype=np.float64)
        P = self.P0_.copy()
        if Y_warmup is not None:
            for t in range(Y_warmup.shape[0]):
                h, P, _, _ = self.step(h, P, Y_warmup[t])
        return h, P

    def step(self, h, P, y):
        F, Q, R = self.F_, self.Q_, self.R_
        N = len(h)
        h_prior = F @ h
        P_prior = F @ P @ F.T + Q
        mu_pred = h_prior
        Sigma_pred = P_prior + R
        K_ = np.linalg.solve(Sigma_pred.T, P_prior.T).T
        h_new = h_prior + K_ @ (y - h_prior)
        P_new = (np.eye(N, dtype=np.float64) - K_) @ P_prior
        return h_new, P_new, mu_pred, Sigma_pred


# --------------------------------------------------------------------------- #
# 2. NeuralDiagGaussianFilter (PyTorch) — DiagGRU                             #
# --------------------------------------------------------------------------- #
class _GRUHead(nn.Module):
    """GRU + (mean, log-var) heads for diagonal-Gaussian output."""

    def __init__(self, N: int, hidden_size: int, num_layers: int) -> None:
        super().__init__()
        self.gru = nn.GRU(N, hidden_size, num_layers, batch_first=True)
        self.mean_head = nn.Linear(hidden_size, N)
        self.log_var_head = nn.Linear(hidden_size, N)

    def forward(self, y_t, h):
        feat = h[-1]
        mu = self.mean_head(feat)
        log_sigma2 = self.log_var_head(feat)
        _, h_new = self.gru(y_t.unsqueeze(1), h)
        return h_new, mu, log_sigma2


@dataclass
class NeuralDiagGaussianFilter:
    """Plain GRU + diagonal Σ_t (the paper's ``DiagGRU`` row)."""

    hidden_size: int = 64
    num_layers: int = 1
    epochs: int = 80
    lr: float = 1e-3
    weight_decay: float = 1e-4
    min_sigma: float = 1e-3
    warmup_frac: float = 0.1
    seed: int = 0

    _model: object = field(default=None, init=False, repr=False)
    _N: int = field(default=0, init=False)

    def fit(self, Y_train: np.ndarray, S: np.ndarray) -> "NeuralDiagGaussianFilter":
        torch.manual_seed(int(self.seed))
        Y = np.asarray(Y_train, dtype=np.float32)
        T, N = Y.shape
        self._N = N
        dev = _device()
        model = _GRUHead(N, self.hidden_size, self.num_layers).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        warmup = max(int(self.warmup_frac * T), 5)
        Y_t = torch.tensor(Y, device=dev)
        BPTT = 64
        model.train()
        for _ in range(self.epochs):
            h = torch.zeros(self.num_layers, 1, self.hidden_size, device=dev).detach()
            win_nll = torch.tensor(0.0, device=dev)
            win_n = 0
            for t in range(T):
                y_t = Y_t[t].unsqueeze(0)
                h, mu, log_s2 = model(y_t, h)
                if t >= warmup:
                    s2 = torch.exp(log_s2).clamp(min=self.min_sigma ** 2)
                    nll = 0.5 * (torch.log(s2).sum() + ((Y_t[t] - mu[0]) ** 2 / s2).sum())
                    win_nll = win_nll + nll
                    win_n += 1
                if (t + 1) % BPTT == 0 or t == T - 1:
                    if win_n > 0:
                        loss = win_nll / win_n
                        opt.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                        opt.step()
                    h = h.detach()
                    win_nll = torch.tensor(0.0, device=dev)
                    win_n = 0
        self._model = model.eval()
        return self

    def initial_state(self, Y_warmup: np.ndarray | None = None):
        if self._model is None:
            raise RuntimeError("fit must be called first.")
        dev = _device()
        h = torch.zeros(self.num_layers, 1, self.hidden_size, device=dev)
        if Y_warmup is not None:
            with torch.no_grad():
                for t in range(Y_warmup.shape[0]):
                    y_t = torch.tensor(Y_warmup[t].astype(np.float32), device=dev).unsqueeze(0)
                    h, _, _ = self._model(y_t, h)
        h_np = h.squeeze(1).cpu().numpy().flatten()
        return h_np, np.eye(self._N, dtype=np.float64)

    def step(self, h, P, y):
        dev = _device()
        h_t = torch.tensor(
            h.reshape(self.num_layers, 1, self.hidden_size),
            dtype=torch.float32, device=dev,
        )
        y_t = torch.tensor(np.asarray(y, dtype=np.float32), device=dev).unsqueeze(0)
        with torch.no_grad():
            h_new_t, mu_t, log_s2_t = self._model(y_t, h_t)
        mu_pred = mu_t[0].cpu().numpy().astype(np.float64)
        s2 = np.exp(log_s2_t[0].cpu().numpy().astype(np.float64)).clip(self.min_sigma ** 2)
        Sigma_pred = np.diag(s2)
        h_new = h_new_t.squeeze(1).cpu().numpy().flatten().astype(np.float64)
        return h_new, np.eye(self._N, dtype=np.float64), mu_pred, Sigma_pred


# --------------------------------------------------------------------------- #
# 3. GraphNeuralSSMFilter (PyTorch) — GNF and (low_rank=0) GCNrankzero        #
# --------------------------------------------------------------------------- #
class _GraphConvGRUCell(nn.Module):
    """Per-node GCN-GRU cell (GCRN-M1 of Li et al. 2018)."""

    def __init__(self, in_features: int, hidden_size: int) -> None:
        super().__init__()
        d, f = hidden_size, in_features
        inp = 2 * d + f
        self.W_z = nn.Linear(inp, d)
        self.W_r = nn.Linear(inp, d)
        self.W_n = nn.Linear(inp, d)

    def forward(self, y, h, S):
        Sh = S @ h
        xh = torch.cat([h, Sh, y], dim=-1)
        z = torch.sigmoid(self.W_z(xh))
        r = torch.sigmoid(self.W_r(xh))
        rh = r * h
        xrh = torch.cat([rh, S @ rh, y], dim=-1)
        n = torch.tanh(self.W_n(xrh))
        return (1 - z) * h + z * n


class _StructuredHeads(nn.Module):
    """Per-node heads producing ``mu``, ``log d``, low-rank factor ``L``."""

    def __init__(self, hidden_size: int, low_rank: int) -> None:
        super().__init__()
        self.mean_head = nn.Linear(hidden_size, 1)
        self.log_diag_head = nn.Linear(hidden_size, 1)
        self.low_rank = low_rank
        if low_rank > 0:
            self.low_rank_head = nn.Linear(hidden_size, low_rank)
        else:
            self.low_rank_head = None  # GCNrankzero ablation

    def forward(self, h):
        mu = self.mean_head(h).squeeze(-1)
        log_d = self.log_diag_head(h).squeeze(-1)
        if self.low_rank_head is not None:
            L = self.low_rank_head(h)
        else:
            L = torch.zeros(h.shape[0], 0, device=h.device)
        return mu, log_d, L


@dataclass
class GraphNeuralSSMFilter:
    """Graph-convolutional GRU + structured Σ_t = diag(d_t) + L_t L_t^T.

    The paper's hero filter (``GNF`` row).  Set ``low_rank=0`` for the
    ``GCNrankzero`` ablation (graph mixing retained, covariance head
    collapsed to diagonal).
    """

    hidden_size: int = 32
    low_rank: int = 4
    epochs: int = 60
    lr: float = 1e-3
    weight_decay: float = 1e-4
    min_diag: float = 1e-4
    warmup_frac: float = 0.1
    seed: int = 0

    _cell: object = field(default=None, init=False, repr=False)
    _heads: object = field(default=None, init=False, repr=False)
    _N: int = field(default=0, init=False)
    _S_fit: np.ndarray | None = field(default=None, init=False, repr=False)

    def fit(self, Y_train: np.ndarray, S: np.ndarray) -> "GraphNeuralSSMFilter":
        torch.manual_seed(int(self.seed))
        Y = np.asarray(Y_train, dtype=np.float32)
        T, N = Y.shape
        self._N = N
        S_arr = np.asarray(S, dtype=np.float32)
        self._S_fit = np.asarray(S, dtype=np.float64)
        dev = _device()
        cell = _GraphConvGRUCell(1, self.hidden_size).to(dev)
        heads = _StructuredHeads(self.hidden_size, self.low_rank).to(dev)
        params = list(cell.parameters()) + list(heads.parameters())
        opt = torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
        S_t = torch.tensor(S_arr, device=dev)
        Y_t = torch.tensor(Y, device=dev)
        r = self.low_rank
        warmup = max(int(self.warmup_frac * T), 5)
        BPTT = 64
        cell.train(); heads.train()
        for _ in range(self.epochs):
            h = torch.zeros(N, self.hidden_size, device=dev).detach()
            win_nll = torch.tensor(0.0, device=dev)
            win_n = 0
            for t in range(T):
                y_t = Y_t[t].unsqueeze(-1)
                mu, log_d, L = heads(h)
                h = cell(y_t, h, S_t)
                if t >= warmup:
                    d = torch.exp(log_d).clamp(min=self.min_diag)
                    e = Y_t[t] - mu
                    if r > 0 and L.shape[-1] > 0:
                        Dinv_L = L / d.unsqueeze(1)
                        M = torch.eye(r, device=dev) + L.t() @ Dinv_L
                        M_chol = torch.linalg.cholesky(M + 1e-6 * torch.eye(r, device=dev))
                        Dinv_e = e / d
                        LtDinv_e = L.t() @ Dinv_e
                        v = torch.cholesky_solve(LtDinv_e.unsqueeze(1), M_chol).squeeze(1)
                        Sigma_inv_e = Dinv_e - Dinv_L @ v
                        quad = (e * Sigma_inv_e).sum()
                        log_det = d.log().sum() + 2.0 * M_chol.diagonal().log().sum()
                    else:
                        quad = (e * e / d).sum()
                        log_det = d.log().sum()
                    nll = 0.5 * (log_det + quad)
                    win_nll = win_nll + nll
                    win_n += 1
                if (t + 1) % BPTT == 0 or t == T - 1:
                    if win_n > 0:
                        loss = win_nll / win_n
                        opt.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(params, 5.0)
                        opt.step()
                    h = h.detach()
                    win_nll = torch.tensor(0.0, device=dev)
                    win_n = 0
        cell.eval(); heads.eval()
        self._cell, self._heads = cell, heads
        return self

    def initial_state(self, Y_warmup: np.ndarray | None = None):
        if self._cell is None or self._heads is None:
            raise RuntimeError("fit must be called first.")
        N, d = self._N, self.hidden_size
        dev = _device()
        h = torch.zeros(N, d, device=dev)
        if Y_warmup is not None:
            S_t = torch.tensor(self._S_fit.astype(np.float32), device=dev)
            with torch.no_grad():
                for t in range(Y_warmup.shape[0]):
                    y_t = torch.tensor(Y_warmup[t].astype(np.float32), device=dev).unsqueeze(-1)
                    h = self._cell(y_t, h, S_t)
        h_np = h.cpu().numpy().flatten().astype(np.float64)
        return h_np, np.eye(N, dtype=np.float64)

    def step(self, h, P, y):
        N, d = self._N, self.hidden_size
        dev = _device()
        h_t = torch.tensor(h.reshape(N, d), dtype=torch.float32, device=dev)
        y_t = torch.tensor(np.asarray(y, dtype=np.float32), device=dev).unsqueeze(-1)
        S_t = torch.tensor(self._S_fit.astype(np.float32), device=dev)
        with torch.no_grad():
            mu_t, log_d_t, L_t = self._heads(h_t)
            h_new_t = self._cell(y_t, h_t, S_t)
        mu_pred = mu_t.cpu().numpy().astype(np.float64)
        d_np = np.exp(log_d_t.cpu().numpy().astype(np.float64)).clip(self.min_diag)
        if L_t.shape[-1] > 0:
            L_np = L_t.cpu().numpy().astype(np.float64)
            Sigma_pred = np.diag(d_np) + L_np @ L_np.T
        else:
            Sigma_pred = np.diag(d_np)
        h_new = h_new_t.cpu().numpy().flatten().astype(np.float64)
        return h_new, np.eye(N, dtype=np.float64), mu_pred, Sigma_pred
