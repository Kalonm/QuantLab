"""Wrappers around statsmodels GLM estimators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np
import statsmodels.api as sm
from sklearn.metrics import r2_score

from .base import SupervisedModel


@dataclass
class _GLMEstimator:
    """Internal estimator bridging statsmodels GLM with sklearn-like API."""

    family: sm.families.Family
    fit_kwargs: Optional[Dict[str, object]] = None

    def __post_init__(self) -> None:
        self.fit_kwargs = dict(self.fit_kwargs or {})
        self._result = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_GLMEstimator":
        X_const = sm.add_constant(X, has_constant="add")
        model = sm.GLM(y, X_const, family=self.family)
        self._result = model.fit(**self.fit_kwargs)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._result is None:
            raise RuntimeError("Model must be fitted before calling predict().")
        X_const = sm.add_constant(X, has_constant="add")
        return np.asarray(self._result.predict(X_const))

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        predictions = self.predict(X)
        return float(r2_score(y, predictions))


class GLMRegressor(SupervisedModel[_GLMEstimator]):
    """Generalised linear model wrapper supporting arbitrary families."""

    def __init__(
        self,
        family: sm.families.Family,
        fit_kwargs: Optional[Dict[str, object]] = None,
        metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    ) -> None:
        estimator = _GLMEstimator(family=family, fit_kwargs=fit_kwargs)
        super().__init__(estimator=estimator, metric=metric or r2_score)


__all__ = ["GLMRegressor"]
