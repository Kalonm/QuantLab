"""Regression model wrapper tests."""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
sm = pytest.importorskip("statsmodels.api")
sklearn_datasets = pytest.importorskip("sklearn.datasets")
sklearn_metrics = pytest.importorskip("sklearn.metrics")

from src.models.supervised import (
    ElasticNetRegressor,
    GLMRegressor,
    LassoRegressor,
    OLSRegressor,
    RidgeRegressor,
)

make_regression = sklearn_datasets.make_regression
r2_score = sklearn_metrics.r2_score


@pytest.fixture(scope="module")
def regression_dataset() -> tuple[np.ndarray, np.ndarray]:
    X, y = make_regression(
        n_samples=128,
        n_features=5,
        n_informative=5,
        noise=0.1,
        random_state=0,
    )
    return X, y


def test_linear_regressors_fit_predict_score(regression_dataset: tuple[np.ndarray, np.ndarray]) -> None:
    X, y = regression_dataset
    models = [
        OLSRegressor(),
        RidgeRegressor(alpha=0.5),
        LassoRegressor(alpha=0.01),
        ElasticNetRegressor(alpha=0.1, l1_ratio=0.7),
    ]
    for model in models:
        fitted = model.fit(X, y)
        assert fitted is model
        predictions = model.predict(X)
        assert predictions.shape == y.shape
        score = model.score(X, y)
        assert score == pytest.approx(r2_score(y, predictions), rel=1e-6)


def test_glm_regressor_scores_like_r2(regression_dataset: tuple[np.ndarray, np.ndarray]) -> None:
    X, y = regression_dataset
    glm = GLMRegressor(family=sm.families.Gaussian())
    glm.fit(X, y)
    preds = glm.predict(X)
    assert preds.shape == y.shape
    assert glm.score(X, y) == pytest.approx(r2_score(y, preds), rel=1e-6)


def test_glm_predict_before_fit_raises() -> None:
    glm = GLMRegressor(family=sm.families.Gaussian())
    with pytest.raises(RuntimeError):
        glm.predict(np.zeros((1, 2)))
