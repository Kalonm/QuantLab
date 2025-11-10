"""Linear model wrappers for supervised learning tasks."""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge

from .base import SupervisedModel


class OLSRegressor(SupervisedModel[LinearRegression]):
    """Ordinary least squares regression wrapper."""

    def __init__(
        self,
        fit_intercept: bool = True,
        metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    ) -> None:
        estimator = LinearRegression(fit_intercept=fit_intercept)
        super().__init__(estimator=estimator, metric=metric)


class RidgeRegressor(SupervisedModel[Ridge]):
    """Ridge regression wrapper with optional metric override."""

    def __init__(
        self,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        solver: str = "auto",
        metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    ) -> None:
        estimator = Ridge(alpha=alpha, fit_intercept=fit_intercept, solver=solver)
        super().__init__(estimator=estimator, metric=metric)


class LassoRegressor(SupervisedModel[Lasso]):
    """LASSO regression wrapper."""

    def __init__(
        self,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        max_iter: int = 1000,
        metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    ) -> None:
        estimator = Lasso(alpha=alpha, fit_intercept=fit_intercept, max_iter=max_iter)
        super().__init__(estimator=estimator, metric=metric)


class ElasticNetRegressor(SupervisedModel[ElasticNet]):
    """Elastic-net regression wrapper."""

    def __init__(
        self,
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        fit_intercept: bool = True,
        max_iter: int = 1000,
        metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    ) -> None:
        estimator = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
        )
        super().__init__(estimator=estimator, metric=metric)


__all__ = [
    "ElasticNetRegressor",
    "LassoRegressor",
    "OLSRegressor",
    "RidgeRegressor",
]
