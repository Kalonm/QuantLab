"""Cross-validation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split


@dataclass
class SplitResult:
    """Container for a single train/test split."""

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray


def kfold_splitter(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int | None = 42,
) -> Iterator[SplitResult]:
    """Yield :class:`SplitResult` objects using K-fold cross-validation."""

    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    for train_idx, test_idx in kf.split(X, y):
        yield SplitResult(X[train_idx], X[test_idx], y[train_idx], y[test_idx])


def stratified_kfold_splitter(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int | None = 42,
) -> Iterator[SplitResult]:
    """Yield :class:`SplitResult` objects using stratified K-fold CV."""

    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    for train_idx, test_idx in skf.split(X, y):
        yield SplitResult(X[train_idx], X[test_idx], y[train_idx], y[test_idx])


def holdout_split(
    X: np.ndarray,
    y: np.ndarray,
    *,
    test_size: float = 0.2,
    random_state: int | None = 42,
    stratify: Sequence[int] | None = None,
) -> SplitResult:
    """Perform a single train/test split returning a :class:`SplitResult`."""

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    return SplitResult(X_train, X_test, y_train, y_test)


__all__ = ["SplitResult", "holdout_split", "kfold_splitter", "stratified_kfold_splitter"]
