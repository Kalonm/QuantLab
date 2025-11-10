"""Scaling utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.preprocessing import StandardScaler


@dataclass
class StandardizationResult:
    """Container for scaled datasets and the fitted scaler."""

    X_train: np.ndarray
    X_test: Optional[np.ndarray]
    scaler: StandardScaler


def standardize_data(
    X_train: np.ndarray,
    X_test: Optional[np.ndarray] = None,
    *,
    with_mean: bool = True,
    with_std: bool = True,
) -> StandardizationResult:
    """Standardise feature matrices using :class:`~sklearn.preprocessing.StandardScaler`.

    Parameters
    ----------
    X_train:
        Training feature matrix.
    X_test:
        Optional testing feature matrix. When provided, it will be transformed using
        the scaler fitted on ``X_train``.
    with_mean, with_std:
        Forwarded to :class:`~sklearn.preprocessing.StandardScaler`.
    """

    scaler = StandardScaler(with_mean=with_mean, with_std=with_std)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test) if X_test is not None else None
    return StandardizationResult(
        X_train=X_train_scaled,
        X_test=X_test_scaled,
        scaler=scaler,
    )


__all__ = ["standardize_data", "StandardizationResult"]
