"""Kernel-based supervised learning wrappers."""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC, SVR

from .base import SupervisedModel


class SVMRegressor(SupervisedModel[SVR]):
    """Support vector regression wrapper."""

    def __init__(
        self,
        kernel: str = "rbf",
        C: float = 1.0,
        epsilon: float = 0.1,
        gamma: str | float = "scale",
        metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    ) -> None:
        estimator = SVR(kernel=kernel, C=C, epsilon=epsilon, gamma=gamma)
        super().__init__(estimator=estimator, metric=metric)


class SVMClassifier(SupervisedModel[SVC]):
    """Support vector machine classifier wrapper."""

    def __init__(
        self,
        kernel: str = "rbf",
        C: float = 1.0,
        gamma: str | float = "scale",
        probability: bool = True,
        metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    ) -> None:
        estimator = SVC(kernel=kernel, C=C, gamma=gamma, probability=probability)
        super().__init__(estimator=estimator, metric=metric or accuracy_score)


__all__ = ["SVMClassifier", "SVMRegressor"]
