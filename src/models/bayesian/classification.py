"""Bayesian classification models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from ...inference import hamiltonian_monte_carlo, mean_field_gaussian_vi, metropolis_hastings


@dataclass
class BayesianLogisticRegression:
    """Bayesian logistic regression with a Gaussian prior."""

    prior_mean: np.ndarray
    prior_cov: np.ndarray

    posterior_mean: np.ndarray | None = None
    posterior_cov: np.ndarray | None = None

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-z))

    def _log_posterior(self, X: np.ndarray, y: np.ndarray) -> Callable[[np.ndarray], float]:
        sigma_inv = np.linalg.inv(self.prior_cov)
        def logp(w: np.ndarray) -> float:
            logits = X @ w
            probs = 1.0 / (1.0 + np.exp(-logits))
            probs = np.clip(probs, 1e-9, 1.0 - 1e-9)
            ll = np.sum(y * np.log(probs) + (1.0 - y) * np.log(1.0 - probs))
            prior = -0.5 * ((w - self.prior_mean).T @ sigma_inv @ (w - self.prior_mean))
            norm_const = -0.5 * (
                self.prior_mean.size * np.log(2 * np.pi)
                + np.linalg.slogdet(self.prior_cov)[1]
            )
            return float(ll + prior + norm_const)
        return logp

    def _grad_log_posterior(self, X: np.ndarray, y: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
        sigma_inv = np.linalg.inv(self.prior_cov)
        def grad(w: np.ndarray) -> np.ndarray:
            probs = 1.0 / (1.0 + np.exp(-(X @ w)))
            grad_ll = X.T @ (y - probs)
            grad_prior = -sigma_inv @ (w - self.prior_mean)
            return grad_ll + grad_prior
        return grad

    def _hessian_log_posterior(self, X: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
        sigma_inv = np.linalg.inv(self.prior_cov)
        def hess(w: np.ndarray) -> np.ndarray:
            probs = 1.0 / (1.0 + np.exp(-(X @ w)))
            W = probs * (1.0 - probs)
            hess_ll = -(X.T @ (W[:, None] * X))
            return hess_ll - sigma_inv
        return hess

    def fit_map(self, X: np.ndarray, y: np.ndarray, n_iter: int = 25, tol: float = 1e-6) -> None:
        """Find the maximum a posteriori estimate via Newton's method."""

        w = self.prior_mean.copy()
        grad_fn = self._grad_log_posterior(X, y)
        hess_fn = self._hessian_log_posterior(X)

        for _ in range(n_iter):
            grad = grad_fn(w)
            hess = hess_fn(w)
            step = np.linalg.solve(-hess, grad)
            w_new = w + step
            if np.linalg.norm(step) < tol:
                w = w_new
                break
            w = w_new

        self.posterior_mean = w
        self.posterior_cov = np.linalg.inv(-hess_fn(w))

    def fit_variational(
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
        self.posterior_mean = result.mean
        self.posterior_cov = result.cov
        return result.mean, result.cov

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

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.posterior_mean is None:
            raise RuntimeError("Model must be fitted before calling predict_proba().")
        logits = X @ self.posterior_mean
        return self._sigmoid(logits)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)
