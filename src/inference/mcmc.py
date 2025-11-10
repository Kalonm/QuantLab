"""Markov chain Monte Carlo routines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

LogProbFn = Callable[[np.ndarray], float]
GradLogProbFn = Callable[[np.ndarray], np.ndarray]


@dataclass
class MCMCResult:
    samples: np.ndarray
    acceptance_rate: float


def metropolis_hastings(
    log_prob: LogProbFn,
    initial_state: np.ndarray,
    n_samples: int,
    proposal_cov: np.ndarray | None = None,
    burn_in: int | float = 0.1,
    rng: np.random.Generator | None = None,
) -> MCMCResult:
    """Basic Gaussian random-walk Metropolis-Hastings sampler."""

    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    rng = rng or np.random.default_rng()
    dim = initial_state.shape[0]
    proposal_cov = proposal_cov if proposal_cov is not None else np.eye(dim)
    chol = np.linalg.cholesky(proposal_cov)

    chain = np.empty((n_samples, dim))
    current = np.array(initial_state, dtype=float)
    current_lp = float(log_prob(current))
    acceptances = 0

    for idx in range(n_samples):
        proposal = current + chol @ rng.normal(size=dim)
        proposal_lp = float(log_prob(proposal))
        log_accept_ratio = proposal_lp - current_lp
        if np.log(rng.uniform()) < log_accept_ratio:
            current = proposal
            current_lp = proposal_lp
            acceptances += 1
        chain[idx] = current

    burn = int(burn_in * n_samples) if isinstance(burn_in, float) else int(burn_in)
    burn = max(0, min(burn, n_samples - 1))
    kept = chain[burn:]
    return MCMCResult(kept, acceptances / n_samples)


def hamiltonian_monte_carlo(
    log_prob: LogProbFn,
    grad_log_prob: GradLogProbFn,
    initial_state: np.ndarray,
    n_samples: int,
    step_size: float = 0.1,
    n_steps: int = 5,
    burn_in: int | float = 0.1,
    rng: np.random.Generator | None = None,
) -> MCMCResult:
    """Simple Hamiltonian Monte Carlo sampler."""

    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if step_size <= 0:
        raise ValueError("step_size must be positive.")
    if n_steps <= 0:
        raise ValueError("n_steps must be positive.")

    rng = rng or np.random.default_rng()
    dim = initial_state.shape[0]
    current = np.array(initial_state, dtype=float)
    current_lp = float(log_prob(current))
    current_grad = np.array(grad_log_prob(current), dtype=float)

    chain = np.empty((n_samples, dim))
    acceptances = 0

    for idx in range(n_samples):
        momentum = rng.normal(size=dim)
        position = current.copy()
        grad = current_grad.copy()

        # Leapfrog integration
        proposed_momentum = momentum + 0.5 * step_size * grad
        proposed_position = position + step_size * proposed_momentum
        for _ in range(n_steps - 1):
            grad = np.array(grad_log_prob(proposed_position))
            proposed_momentum = proposed_momentum + step_size * grad
            proposed_position = proposed_position + step_size * proposed_momentum
        grad = np.array(grad_log_prob(proposed_position))
        proposed_momentum = proposed_momentum + 0.5 * step_size * grad
        proposed_momentum = -proposed_momentum

        proposed_lp = float(log_prob(proposed_position))

        current_hamiltonian = -current_lp + 0.5 * np.dot(momentum, momentum)
        proposed_hamiltonian = -proposed_lp + 0.5 * np.dot(proposed_momentum, proposed_momentum)
        log_accept_ratio = -(proposed_hamiltonian - current_hamiltonian)

        if np.log(rng.uniform()) < log_accept_ratio:
            current = proposed_position
            current_lp = proposed_lp
            current_grad = grad
            acceptances += 1

        chain[idx] = current

    burn = int(burn_in * n_samples) if isinstance(burn_in, float) else int(burn_in)
    burn = max(0, min(burn, n_samples - 1))
    kept = chain[burn:]
    return MCMCResult(kept, acceptances / n_samples)
