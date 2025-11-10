"""Variational inference routines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

LogProbFn = Callable[[np.ndarray], float]
GradLogProbFn = Callable[[np.ndarray], np.ndarray]


@dataclass
class VariationalResult:
    mean: np.ndarray
    cov: np.ndarray
    elbo_history: list[float]


def mean_field_gaussian_vi(
    log_prob: LogProbFn,
    grad_log_prob: GradLogProbFn,
    initial_mean: np.ndarray,
    initial_log_std: np.ndarray,
    n_iterations: int = 500,
    lr: float = 0.05,
    n_mc_samples: int = 5,
    rng: np.random.Generator | None = None,
) -> VariationalResult:
    """Diagonal Gaussian variational inference via stochastic gradient ascent."""

    if n_iterations <= 0:
        raise ValueError("n_iterations must be positive.")
    if n_mc_samples <= 0:
        raise ValueError("n_mc_samples must be positive.")

    rng = rng or np.random.default_rng()
    mean = np.array(initial_mean, dtype=float)
    log_std = np.array(initial_log_std, dtype=float)
    dim = mean.shape[0]
    elbo_history: list[float] = []

    for _ in range(n_iterations):
        std = np.exp(log_std)
        eps = rng.normal(size=(n_mc_samples, dim))
        samples = mean + eps * std

        log_probs = np.array([log_prob(sample) for sample in samples])
        # Entropy of diagonal Gaussian
        entropy = float(np.sum(log_std) + 0.5 * dim * (1 + np.log(2 * np.pi)))
        elbo = float(np.mean(log_probs) + entropy)
        elbo_history.append(elbo)

        grads = np.array([grad_log_prob(sample) for sample in samples])
        grad_mean = np.mean(grads, axis=0)
        grad_log_std = np.mean(grads * eps * std, axis=0) + 1.0

        mean = mean + lr * grad_mean
        log_std = log_std + lr * grad_log_std

    cov = np.diag(np.exp(2.0 * log_std))
    return VariationalResult(mean=mean, cov=cov, elbo_history=elbo_history)
