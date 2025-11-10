"""Naive Bayes model wrappers."""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

from .base import SupervisedModel


class NaiveBayesClassifier(SupervisedModel[GaussianNB]):
    """Gaussian Naive Bayes classifier wrapper."""

    def __init__(
        self,
        var_smoothing: float = 1e-9,
        metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    ) -> None:
        estimator = GaussianNB(var_smoothing=var_smoothing)
        super().__init__(estimator=estimator, metric=metric or accuracy_score)


__all__ = ["NaiveBayesClassifier"]
