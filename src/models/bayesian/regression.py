"""Bayesian regression models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from ...inference import hamiltonian_monte_carlo, mean_field_gaussian_vi, metropolis_hastings
from ...probability import log_marginal_likelihood_gaussian, predictive_normal


@dataclass
class BayesianLinearRegression:
    """Bayesian linear regression with Gaussian prior and noise."""

    prior_mean: np.ndarray
    prior_cov: np.ndarray
    noise_variance: float

    posterior_mean: np.ndarray | None = None
    posterior_cov: np.ndarray | None = None

    def fit_conjugate(self, X: np.ndarray, y: np.ndarray) -> None:
        """Compute the closed-form posterior."""

        if self.noise_variance <= 0:
            raise ValueError("Noise variance must be positive.")

        sigma_inv = np.linalg.inv(self.prior_cov)
        precision = sigma_inv + (X.T @ X) / self.noise_variance
        cov_post = np.linalg.inv(precision)
        mean_post = cov_post @ (sigma_inv @ self.prior_mean + (X.T @ y) / self.noise_variance)

        self.posterior_mean = mean_post
        self.posterior_cov = cov_post

    def log_marginal_likelihood(self, X: np.ndarray, y: np.ndarray) -> float:
        return log_marginal_likelihood_gaussian(X, y, self.prior_mean, self.prior_cov, self.noise_variance)

    def predict_mean(self, X: np.ndarray) -> np.ndarray:
        if self.posterior_mean is None:
            raise RuntimeError("Model must be fitted before calling predict_mean().")
        return X @ self.posterior_mean

    def predictive_distribution(self, x_new: np.ndarray):
        if self.posterior_mean is None or self.posterior_cov is None:
            raise RuntimeError("Model must be fitted before building predictive distribution.")
        return predictive_normal(x_new, self.posterior_mean, self.posterior_cov, self.noise_variance)

    def _log_posterior(self, X: np.ndarray, y: np.ndarray) -> Callable[[np.ndarray], float]:
        sigma_inv = np.linalg.inv(self.prior_cov)
        def logp(w: np.ndarray) -> float:
            resid = y - X @ w
            ll = -0.5 * (resid @ resid) / self.noise_variance
            prior = -0.5 * ((w - self.prior_mean).T @ sigma_inv @ (w - self.prior_mean))
            const = -0.5 * X.shape[0] * np.log(2 * np.pi * self.noise_variance)
            norm_const = -0.5 * (
                self.prior_mean.size * np.log(2 * np.pi)
                + np.linalg.slogdet(self.prior_cov)[1]
            )
            return float(ll + prior + const + norm_const)
        return logp

    def _grad_log_posterior(self, X: np.ndarray, y: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
        sigma_inv = np.linalg.inv(self.prior_cov)
        def grad(w: np.ndarray) -> np.ndarray:
            resid = y - X @ w
            grad_ll = (X.T @ resid) / self.noise_variance
            grad_prior = -sigma_inv @ (w - self.prior_mean)
            return grad_ll + grad_prior
        return grad

    def sample_posterior_mh(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_samples: int = 1_000,
        proposal_cov: np.ndarray | None = None,
        burn_in: int | float = 0.1,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        logp = self._log_posterior(X, y)
        result = metropolis_hastings(
            logp,
            initial_state=self.prior_mean,
            n_samples=n_samples,
            proposal_cov=proposal_cov,
            burn_in=burn_in,
            rng=rng,
        )
        return result.samples

    def sample_posterior_hmc(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_samples: int = 1_000,
        step_size: float = 0.05,
        n_steps: int = 10,
        burn_in: int | float = 0.1,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        logp = self._log_posterior(X, y)
        grad = self._grad_log_posterior(X, y)
        result = hamiltonian_monte_carlo(
            logp,
            grad,
            initial_state=self.prior_mean,
            n_samples=n_samples,
            step_size=step_size,
            n_steps=n_steps,
            burn_in=burn_in,
            rng=rng,
        )
        return result.samples

    def variational_posterior(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_iterations: int = 500,
        lr: float = 0.05,
        n_mc_samples: int = 5,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        logp = self._log_posterior(X, y)
        grad = self._grad_log_posterior(X, y)
        result = mean_field_gaussian_vi(
            logp,
            grad,
            initial_mean=self.prior_mean,
            initial_log_std=np.log(np.sqrt(np.diag(self.prior_cov))),
            n_iterations=n_iterations,
            lr=lr,
            n_mc_samples=n_mc_samples,
            rng=rng,
        )
        return result.mean, result.cov
