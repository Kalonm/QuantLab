"""Classification model wrapper tests."""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
sklearn_datasets = pytest.importorskip("sklearn.datasets")
sklearn_metrics = pytest.importorskip("sklearn.metrics")

from src.models.supervised import KNNClassifier, NaiveBayesClassifier, SVMClassifier

make_classification = sklearn_datasets.make_classification
accuracy_score = sklearn_metrics.accuracy_score
f1_score = sklearn_metrics.f1_score


@pytest.fixture(scope="module")
def classification_dataset() -> tuple[np.ndarray, np.ndarray]:
    X, y = make_classification(
        n_samples=200,
        n_features=6,
        n_informative=5,
        n_redundant=1,
        n_clusters_per_class=1,
        class_sep=2.0,
        random_state=0,
    )
    return X, y


def test_classifiers_fit_predict_score(classification_dataset: tuple[np.ndarray, np.ndarray]) -> None:
    X, y = classification_dataset
    models = [
        NaiveBayesClassifier(),
        SVMClassifier(C=1.0),
        KNNClassifier(n_neighbors=3),
    ]
    for model in models:
        fitted = model.fit(X, y)
        assert fitted is model
        predictions = model.predict(X)
        assert predictions.shape == y.shape
        score = model.score(X, y)
        assert score == pytest.approx(accuracy_score(y, predictions), rel=1e-6)


def test_classifier_custom_metric(classification_dataset: tuple[np.ndarray, np.ndarray]) -> None:
    X, y = classification_dataset
    metric = lambda yt, yp: f1_score(yt, yp)
    svm = SVMClassifier(C=0.5, metric=metric)
    svm.fit(X, y)
    assert svm.score(X, y) == pytest.approx(f1_score(y, svm.predict(X)), rel=1e-6)
