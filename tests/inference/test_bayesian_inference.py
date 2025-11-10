"""Validation tests for Bayesian inference utilities."""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from src.models.bayesian import BayesianLinearRegression, BayesianLogisticRegression
from src.inference import hamiltonian_monte_carlo, metropolis_hastings


def generate_linear_data(rng: np.random.Generator, n: int = 50) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = rng.normal(size=(n, 2))
    weights = np.array([1.5, -2.0])
    noise = rng.normal(scale=0.3, size=n)
    y = X @ weights + noise
    return X, y, weights


def test_linear_regression_conjugate_posterior_matches_closed_form():
    rng = np.random.default_rng(0)
    X, y, _ = generate_linear_data(rng)
    prior_mean = np.zeros(2)
    prior_cov = np.eye(2)
    model = BayesianLinearRegression(prior_mean, prior_cov, noise_variance=0.3**2)
    model.fit_conjugate(X, y)

    sigma_inv = np.linalg.inv(prior_cov)
    precision = sigma_inv + (X.T @ X) / (0.3**2)
    cov_expected = np.linalg.inv(precision)
    mean_expected = cov_expected @ (sigma_inv @ prior_mean + (X.T @ y) / (0.3**2))

    assert np.allclose(model.posterior_mean, mean_expected)
    assert np.allclose(model.posterior_cov, cov_expected)


def test_metropolis_hastings_recovers_linear_posterior_mean():
    rng = np.random.default_rng(1)
    X, y, _ = generate_linear_data(rng)
    prior_mean = np.zeros(2)
    prior_cov = np.eye(2)
    model = BayesianLinearRegression(prior_mean, prior_cov, noise_variance=0.3**2)
    model.fit_conjugate(X, y)

    logp = model._log_posterior(X, y)
    result = metropolis_hastings(
        logp,
        initial_state=prior_mean,
        n_samples=4000,
        burn_in=1000,
        proposal_cov=0.1 * np.eye(2),
        rng=np.random.default_rng(2),
    )
    sampled_mean = result.samples.mean(axis=0)

    assert np.allclose(sampled_mean, model.posterior_mean, atol=0.1)
    assert 0.1 < result.acceptance_rate < 0.9


def test_hmc_matches_linear_posterior_mean():
    rng = np.random.default_rng(2)
    X, y, _ = generate_linear_data(rng)
    prior_mean = np.zeros(2)
    prior_cov = np.eye(2)
    model = BayesianLinearRegression(prior_mean, prior_cov, noise_variance=0.3**2)
    model.fit_conjugate(X, y)

    logp = model._log_posterior(X, y)
    grad = model._grad_log_posterior(X, y)
    result = hamiltonian_monte_carlo(
        logp,
        grad,
        initial_state=prior_mean,
        n_samples=2000,
        step_size=0.05,
        n_steps=15,
        burn_in=500,
        rng=np.random.default_rng(3),
    )
    sampled_mean = result.samples.mean(axis=0)

    assert np.allclose(sampled_mean, model.posterior_mean, atol=0.1)
    assert result.acceptance_rate > 0.5


def test_variational_logistic_regression_reasonable_accuracy():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(300, 3))
    true_w = np.array([0.8, -1.0, 0.6])
    logits = X @ true_w
    probs = 1 / (1 + np.exp(-logits))
    y = rng.binomial(1, probs)

    prior_mean = np.zeros(3)
    prior_cov = np.eye(3)
    model = BayesianLogisticRegression(prior_mean, prior_cov)
    model.fit_variational(X, y, n_iterations=200, lr=0.05, n_mc_samples=10, rng=np.random.default_rng(5))

    preds = model.predict(X)
    accuracy = np.mean(preds == y)
    assert accuracy > 0.7
