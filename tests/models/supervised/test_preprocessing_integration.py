"""Integration tests between preprocessing utilities and model wrappers."""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
sklearn_datasets = pytest.importorskip("sklearn.datasets")

from src.models.supervised import RidgeRegressor
from src.preprocessing import kfold_splitter, polynomial_features, standardize_data

make_regression = sklearn_datasets.make_regression


def test_preprocessing_pipeline_with_ridge() -> None:
    X, y = make_regression(
        n_samples=150,
        n_features=4,
        noise=0.5,
        random_state=1,
    )
    scaled = standardize_data(X)
    X_poly, _ = polynomial_features(scaled.X_train, degree=3, include_bias=False)

    splits = list(kfold_splitter(X_poly, y, n_splits=3, random_state=0))
    assert len(splits) == 3

    model = RidgeRegressor(alpha=1.0)
    scores = []
    for split in splits:
        model.fit(split.X_train, split.y_train)
        scores.append(model.score(split.X_test, split.y_test))

    assert np.mean(scores) > 0.8
