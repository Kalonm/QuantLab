"""k-nearest neighbour model wrappers."""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from .base import SupervisedModel


class KNNRegressor(SupervisedModel[KNeighborsRegressor]):
    """kNN regression wrapper."""

    def __init__(
        self,
        n_neighbors: int = 5,
        weights: str | Callable[[np.ndarray], np.ndarray] = "uniform",
        metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    ) -> None:
        estimator = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights)
        super().__init__(estimator=estimator, metric=metric)


class KNNClassifier(SupervisedModel[KNeighborsClassifier]):
    """kNN classification wrapper."""

    def __init__(
        self,
        n_neighbors: int = 5,
        weights: str | Callable[[np.ndarray], np.ndarray] = "uniform",
        metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    ) -> None:
        estimator = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
        super().__init__(estimator=estimator, metric=metric or accuracy_score)


__all__ = ["KNNClassifier", "KNNRegressor"]
