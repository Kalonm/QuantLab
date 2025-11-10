"""Point process estimators for market microstructure research.

This module focuses on light‑weight estimators that are fast enough for
scenario testing yet expressive enough for prototyping quantitative
strategies.  The emphasis is on reproducible, deterministic code that does
not require specialised third‑party solvers.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np


EventSeries = Sequence[float]


class BasePointProcess:
    """Minimal interface for simple point process models.

    The estimators implemented here accept a sequence of event timestamps
    (assumed to be strictly increasing) expressed in seconds.  The
    interface mirrors the scikit-learn API to ease integration in research
    notebooks and testing pipelines.
    """

    def fit(self, events: EventSeries) -> "BasePointProcess":
        raise NotImplementedError

    def intensity(self, times: Sequence[float]) -> np.ndarray:
        """Return the conditional intensity evaluated at ``times``."""

        raise NotImplementedError

    def sample(
        self,
        horizon: float,
        rng: Optional[np.random.Generator] = None,
        start: float = 0.0,
    ) -> np.ndarray:
        """Generate event times up to ``start + horizon`` using Ogata's thinning.

        Sub-classes should override this if a closed form sampler is
        available.  The base implementation relies on a generic thinning
        scheme that repeatedly draws exponential proposals and accepts them
        according to the current intensity upper bound.
        """

        raise NotImplementedError


class PoissonProcess(BasePointProcess):
    """Homogeneous Poisson process with maximum likelihood estimator."""

    def __init__(self, rate: Optional[float] = None):
        self.rate: Optional[float] = rate

    def fit(self, events: EventSeries) -> "PoissonProcess":
        events = np.asarray(events, dtype=float)
        if events.ndim != 1:
            raise ValueError("events must be a one-dimensional sequence")
        if len(events) < 2:
            raise ValueError("at least two events are required to estimate the rate")

        span = events[-1] - events[0]
        if span <= 0:
            raise ValueError("events must be strictly increasing")

        self.rate = (len(events) - 1) / span
        return self

    def intensity(self, times: Sequence[float]) -> np.ndarray:
        if self.rate is None:
            raise RuntimeError("model must be fitted before calling intensity")
        times = np.asarray(times, dtype=float)
        return np.full_like(times, fill_value=self.rate, dtype=float)

    def sample(
        self,
        horizon: float,
        rng: Optional[np.random.Generator] = None,
        start: float = 0.0,
    ) -> np.ndarray:
        if self.rate is None:
            raise RuntimeError("model must be fitted before sampling")
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        if rng is None:
            rng = np.random.default_rng()

        events: List[float] = []
        t = start
        while True:
            t += rng.exponential(1.0 / self.rate)
            if t > start + horizon:
                break
            events.append(t)
        return np.array(events, dtype=float)


@dataclass
class HawkesParameters:
    base_intensity: float
    excitation: float
    decay: float


class HawkesProcess(BasePointProcess):
    """Simple Hawkes process with an exponential kernel.

    The estimator implements a fast moment-matching routine tailored for
    high-frequency trading data.  It approximates the conditional intensity
    from inverse inter-arrival times which provides robust estimates even
    with relatively few events.  The goal is not statistical efficiency but
    rather a deterministic calibration suitable for regression testing.
    """

    def __init__(self, params: Optional[HawkesParameters] = None):
        self.params = params
        self._event_times: Optional[np.ndarray] = None

    def fit(self, events: EventSeries) -> "HawkesProcess":
        times = np.asarray(events, dtype=float)
        if times.ndim != 1:
            raise ValueError("events must be one-dimensional")
        if len(times) < 3:
            raise ValueError("at least three events are required for Hawkes calibration")
        if not np.all(np.diff(times) > 0):
            raise ValueError("events must be strictly increasing")

        inter_arrivals = np.diff(times)
        beta = 1.0 / max(np.median(inter_arrivals), 1e-8)

        decayed_history = np.zeros_like(times)
        for i in range(1, len(times)):
            deltas = times[i] - times[:i]
            decayed_history[i] = np.exp(-beta * deltas).sum()

        # Target intensity approximated by inverse inter-arrival times.
        target = np.zeros_like(times)
        target[1:] = 1.0 / np.maximum(inter_arrivals, 1e-8)
        design = np.column_stack([np.ones_like(times), decayed_history])
        coeffs, *_ = np.linalg.lstsq(design, target, rcond=None)
        mu = max(coeffs[0], 1e-6)
        alpha = max(coeffs[1], 1e-6)

        # Normalise alpha to ensure stationarity.
        branching_ratio = min(alpha / beta, 0.95)
        alpha = branching_ratio * beta

        self.params = HawkesParameters(mu, alpha, beta)
        self._event_times = times
        return self

    def intensity(self, times: Sequence[float]) -> np.ndarray:
        if self.params is None or self._event_times is None:
            raise RuntimeError("model must be fitted before calling intensity")
        times = np.asarray(times, dtype=float)
        mu, alpha, beta = self.params.base_intensity, self.params.excitation, self.params.decay
        intensities = np.full_like(times, mu, dtype=float)
        history = self._event_times
        for i, t in enumerate(times):
            mask = history < t
            if not np.any(mask):
                continue
            intensities[i] += alpha * np.exp(-beta * (t - history[mask])).sum()
        return intensities

    def sample(
        self,
        horizon: float,
        rng: Optional[np.random.Generator] = None,
        start: float = 0.0,
    ) -> np.ndarray:
        if self.params is None:
            raise RuntimeError("model must be fitted before sampling")
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        if rng is None:
            rng = np.random.default_rng()

        mu, alpha, beta = (
            self.params.base_intensity,
            self.params.excitation,
            self.params.decay,
        )
        events: List[float] = []
        t = start
        lambda_bar = mu
        while True:
            # Draw proposal time under the dominating intensity.
            w = rng.exponential(1.0 / lambda_bar)
            t_candidate = t + w
            if t_candidate > start + horizon:
                break

            # Update intensity upper bound with past events.
            decayed = np.exp(-beta * (t_candidate - np.array(events, dtype=float)))
            lambda_t = mu + alpha * decayed.sum() if events else mu
            if lambda_t < 0:
                lambda_t = mu
            accept_prob = lambda_t / lambda_bar if lambda_bar > 0 else 1.0
            if rng.uniform() <= accept_prob:
                events.append(t_candidate)
                t = t_candidate
                lambda_bar = max(lambda_t, mu)
            else:
                t = t_candidate
                lambda_bar = max(lambda_t, mu)

        return np.array(events, dtype=float)


__all__ = [
    "BasePointProcess",
    "PoissonProcess",
    "HawkesParameters",
    "HawkesProcess",
]
