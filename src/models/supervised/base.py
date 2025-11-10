"""Base utilities for supervised learning wrappers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, Optional, Protocol, TypeVar

import numpy as np


class SupportsFit(Protocol):
    """Protocol representing estimators implementing ``fit`` and ``predict``."""

    def fit(self, X: np.ndarray, y: np.ndarray):  # pragma: no cover - Protocol signature
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:  # pragma: no cover - Protocol signature
        ...

    def score(self, X: np.ndarray, y: np.ndarray) -> float:  # pragma: no cover
        ...


EstimatorT = TypeVar("EstimatorT", bound=SupportsFit)


@dataclass
class SupervisedModel(Generic[EstimatorT]):
    """Light-weight wrapper around estimators to provide metric-aware scoring."""

    estimator: EstimatorT
    metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SupervisedModel[EstimatorT]":
        self.estimator.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.estimator.predict(X))

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        predictions = self.predict(X)
        if self.metric is not None:
            return float(self.metric(y, predictions))
        return float(self.estimator.score(X, y))


__all__ = ["SupervisedModel"]
