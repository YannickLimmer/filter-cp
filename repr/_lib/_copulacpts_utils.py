# Vendored from Rose-STL-Lab/CopulaCPTS @ 56f29640
# Commit: 56f29640edfe5e19e37efa0eb5eb372d5c3a9c87
# Fetch date: 2026-04-25
# Patches:
#   - Removed `from copulae import GumbelCopula` (not installed; not needed here).
#   - Removed `from copulae.core import pseudo_obs`; replaced with scipy-based
#     implementation to avoid the copulae dependency.

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import rankdata
from tqdm import trange


def pseudo_obs(data: np.ndarray) -> np.ndarray:
    """Empirical CDF at each observation (replaces copulae.core.pseudo_obs).

    Returns values in (0, 1) using average ranks divided by (n+1).
    """
    n = data.shape[0]
    return rankdata(data, method="average", axis=0) / (n + 1)


def gumbel_copula_loss(x, cop, data, epsilon):
    return np.fabs(cop.cdf([x] * data.shape[1]) - 1 + epsilon)


def empirical_copula_loss(x, data, epsilon):
    pseudo_data = pseudo_obs(data)
    return np.fabs(
        np.mean(
            np.all(
                np.less_equal(pseudo_data, np.array([x] * pseudo_data.shape[1])),
                axis=1,
            )
        )
        - 1
        + epsilon
    )


def empirical_copula_loss_new(x, data, epsilon):
    pseudo_data = pseudo_obs(data)
    return (
        np.mean(
            np.all(
                np.less_equal(pseudo_data, np.array([x] * pseudo_data.shape[1])),
                axis=1,
            )
        )
        - 1
        + epsilon
    )


class CP(nn.Module):
    def __init__(self, dimension, epsilon):
        super(CP, self).__init__()
        self.alphas = nn.Parameter(torch.ones(dimension))
        self.epsilon = epsilon
        self.relu = torch.nn.ReLU()

    def forward(self, pseudo_data):
        coverage = torch.mean(
            torch.relu(
                torch.prod(torch.sigmoid((self.alphas - pseudo_data) * 1000), dim=1)
            )
        )
        return torch.abs(coverage - 1 + self.epsilon)


def search_alpha(alpha_input, epsilon, epochs=500):
    pseudo_data = torch.tensor(alpha_input)
    dim = alpha_input.shape[-1]
    cp = CP(dim, epsilon)
    optimizer = torch.optim.Adam(cp.parameters(), weight_decay=1e-4)

    with trange(epochs, desc="copula-alpha", unit="epoch", leave=False) as pbar:
        for _ in pbar:
            optimizer.zero_grad()
            loss = cp(pseudo_data)
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=float(loss.detach()))

    return cp.alphas.detach().numpy()
