"""Core probability distributions used throughout QuantLab."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

ArrayLike = np.ndarray | float


@dataclass
class Distribution:
    """Abstract base class for probability distributions."""

    def log_prob(self, value: ArrayLike) -> np.ndarray:
        raise NotImplementedError

    def sample(self, rng: np.random.Generator, size: Optional[int] = None) -> np.ndarray:
        raise NotImplementedError

    def mean(self) -> float:
        raise NotImplementedError

    def variance(self) -> float:
        raise NotImplementedError


@dataclass
class Normal(Distribution):
    """Univariate normal distribution."""

    mean_: float
    std: float

    def __post_init__(self) -> None:
        if self.std <= 0:
            raise ValueError("Standard deviation must be positive.")

    def log_prob(self, value: ArrayLike) -> np.ndarray:
        value_arr = np.asarray(value, dtype=float)
        var = self.std**2
        return -0.5 * ((value_arr - self.mean_) ** 2 / var + math.log(2 * math.pi * var))

    def sample(self, rng: np.random.Generator, size: Optional[int] = None) -> np.ndarray:
        return rng.normal(loc=self.mean_, scale=self.std, size=size)

    def mean(self) -> float:
        return float(self.mean_)

    def variance(self) -> float:
        return float(self.std**2)


@dataclass
class Bernoulli(Distribution):
    """Bernoulli distribution."""

    p: float

    def __post_init__(self) -> None:
        if not 0.0 < self.p < 1.0:
            raise ValueError("Probability p must be in (0, 1).")

    def log_prob(self, value: ArrayLike) -> np.ndarray:
        value_arr = np.asarray(value, dtype=float)
        return value_arr * math.log(self.p) + (1.0 - value_arr) * math.log(1.0 - self.p)

    def sample(self, rng: np.random.Generator, size: Optional[int] = None) -> np.ndarray:
        return rng.binomial(n=1, p=self.p, size=size)

    def mean(self) -> float:
        return float(self.p)

    def variance(self) -> float:
        return float(self.p * (1.0 - self.p))


@dataclass
class Beta(Distribution):
    """Beta distribution."""

    alpha: float
    beta: float

    def __post_init__(self) -> None:
        if self.alpha <= 0 or self.beta <= 0:
            raise ValueError("Alpha and beta must be positive.")

    def log_prob(self, value: ArrayLike) -> np.ndarray:
        value_arr = np.asarray(value, dtype=float)
        if np.any((value_arr <= 0) | (value_arr >= 1)):
            return np.full_like(value_arr, -np.inf, dtype=float)
        log_beta_fn = math.lgamma(self.alpha) + math.lgamma(self.beta) - math.lgamma(self.alpha + self.beta)
        return (
            (self.alpha - 1) * np.log(value_arr)
            + (self.beta - 1) * np.log(1.0 - value_arr)
            - log_beta_fn
        )

    def sample(self, rng: np.random.Generator, size: Optional[int] = None) -> np.ndarray:
        return rng.beta(self.alpha, self.beta, size=size)

    def mean(self) -> float:
        return float(self.alpha / (self.alpha + self.beta))

    def variance(self) -> float:
        denom = (self.alpha + self.beta) ** 2 * (self.alpha + self.beta + 1)
        return float(self.alpha * self.beta / denom)


@dataclass
class Gamma(Distribution):
    """Gamma distribution parameterised by shape and rate."""

    shape: float
    rate: float

    def __post_init__(self) -> None:
        if self.shape <= 0 or self.rate <= 0:
            raise ValueError("Shape and rate must be positive.")

    def log_prob(self, value: ArrayLike) -> np.ndarray:
        value_arr = np.asarray(value, dtype=float)
        if np.any(value_arr <= 0):
            return np.full_like(value_arr, -np.inf, dtype=float)
        return (
            (self.shape - 1) * np.log(value_arr)
            - self.rate * value_arr
            + self.shape * math.log(self.rate)
            - math.lgamma(self.shape)
        )

    def sample(self, rng: np.random.Generator, size: Optional[int] = None) -> np.ndarray:
        return rng.gamma(shape=self.shape, scale=1.0 / self.rate, size=size)

    def mean(self) -> float:
        return float(self.shape / self.rate)

    def variance(self) -> float:
        return float(self.shape / (self.rate**2))
