"""Classification workflow demonstration using QuantLab utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.models.supervised import (
    KNNClassifier,
    NaiveBayesClassifier,
    SVMClassifier,
)
from src.preprocessing import holdout_split, standardize_data


@dataclass
class ClassificationResult:
    name: str
    parameters: Dict[str, float]
    accuracy: float
    precision: float
    recall: float
    f1: float


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }


def run_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    svm_cs: Iterable[float] = (0.1, 1.0, 10.0),
    knn_neighbors: Iterable[int] = (3, 5, 9),
) -> list[ClassificationResult]:
    """Train a collection of classifiers and compute hold-out metrics."""

    results: list[ClassificationResult] = []

    nb = NaiveBayesClassifier()
    nb.fit(X_train, y_train)
    nb_pred = nb.predict(X_test)
    nb_metrics = compute_metrics(y_test, nb_pred)
    results.append(
        ClassificationResult(
            name="NaiveBayes",
            parameters={},
            accuracy=nb_metrics["accuracy"],
            precision=nb_metrics["precision"],
            recall=nb_metrics["recall"],
            f1=nb_metrics["f1"],
        )
    )

    for C in svm_cs:
        svm = SVMClassifier(C=C)
        svm.fit(X_train, y_train)
        preds = svm.predict(X_test)
        metrics = compute_metrics(y_test, preds)
        results.append(
            ClassificationResult(
                name="SVM",
                parameters={"C": C},
                accuracy=metrics["accuracy"],
                precision=metrics["precision"],
                recall=metrics["recall"],
                f1=metrics["f1"],
            )
        )

    for n in knn_neighbors:
        knn = KNNClassifier(n_neighbors=n)
        knn.fit(X_train, y_train)
        preds = knn.predict(X_test)
        metrics = compute_metrics(y_test, preds)
        results.append(
            ClassificationResult(
                name="kNN",
                parameters={"n_neighbors": n},
                accuracy=metrics["accuracy"],
                precision=metrics["precision"],
                recall=metrics["recall"],
                f1=metrics["f1"],
            )
        )

    return results


def main() -> None:
    dataset = load_breast_cancer()
    X, y = dataset.data, dataset.target

    standardised = standardize_data(X)
    X_scaled = standardised.X_train

    split = holdout_split(X_scaled, y, test_size=0.25, stratify=y)

    results = run_models(split.X_train, split.y_train, split.X_test, split.y_test)

    print("Classification workflow summary:\n")
    for res in results:
        params = ", ".join(f"{k}={v}" for k, v in res.parameters.items()) or "default"
        print(
            f"{res.name:<12} | params: {params:<25} | Accuracy={res.accuracy:.3f} | "
            f"Precision={res.precision:.3f} | Recall={res.recall:.3f} | F1={res.f1:.3f}"
        )


if __name__ == "__main__":  # pragma: no cover - example script
    main()
