"""Helper utilities for evaluating common likelihoods."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from .distributions import Normal


def log_likelihood(distribution, data: Iterable[float]) -> float:
    """Return the summed log-likelihood under a given distribution."""

    values = np.asarray(list(data), dtype=float)
    return float(np.sum(distribution.log_prob(values)))


def log_marginal_likelihood_gaussian(
    X: np.ndarray,
    y: np.ndarray,
    prior_mean: np.ndarray,
    prior_cov: np.ndarray,
    noise_variance: float,
) -> float:
    """Closed form log marginal likelihood for Bayesian linear regression."""

    if noise_variance <= 0:
        raise ValueError("Noise variance must be positive.")
    n, d = X.shape
    sigma_inv = np.linalg.inv(prior_cov)
    precision = sigma_inv + (X.T @ X) / noise_variance
    cov_post = np.linalg.inv(precision)
    mean_post = cov_post @ (sigma_inv @ prior_mean + (X.T @ y) / noise_variance)

    residual = y - X @ mean_post
    log_det_prior = np.linalg.slogdet(prior_cov)[1]
    log_det_post = np.linalg.slogdet(cov_post)[1]
    quadratic = residual @ residual / noise_variance + (
        (mean_post - prior_mean).T @ sigma_inv @ (mean_post - prior_mean)
    )
    const = -0.5 * n * math.log(2 * math.pi * noise_variance)
    return float(const + 0.5 * (log_det_prior - log_det_post) - 0.5 * quadratic)


def predictive_normal(
    x_new: np.ndarray,
    posterior_mean: np.ndarray,
    posterior_cov: np.ndarray,
    noise_variance: float,
) -> Normal:
    """Return the predictive distribution for Bayesian linear regression."""

    predictive_mean = float(x_new @ posterior_mean)
    predictive_var = float(x_new @ posterior_cov @ x_new + noise_variance)
    return Normal(predictive_mean, math.sqrt(predictive_var))
