"""Demonstration of regression workflows using QuantLab wrappers.

This script acts as a light-weight notebook replacement illustrating how to:

* load a dataset,
* perform standardisation and feature engineering,
* train multiple models with hyper-parameter sweeps, and
* report regression metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.models.supervised import ElasticNetRegressor, OLSRegressor, RidgeRegressor
from src.preprocessing import polynomial_features, standardize_data


@dataclass
class RegressionRunResult:
    name: str
    parameters: Dict[str, float]
    r2: float
    rmse: float
    mae: float


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Return standard regression metrics for a set of predictions."""

    return {
        "r2": r2_score(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": mean_absolute_error(y_true, y_pred),
    }


def run_model_grid(
    X: np.ndarray,
    y: np.ndarray,
    *,
    ridge_alphas: Iterable[float] = (0.1, 1.0, 10.0),
    elasticnet_configs: Iterable[tuple[float, float]] = ((0.1, 0.5), (1.0, 0.5)),
) -> list[RegressionRunResult]:
    """Train a selection of models and compute evaluation metrics."""

    results: list[RegressionRunResult] = []

    ols = OLSRegressor()
    ols.fit(X, y)
    ols_pred = ols.predict(X)
    ols_metrics = evaluate_predictions(y, ols_pred)
    results.append(
        RegressionRunResult(
            name="OLS",
            parameters={},
            r2=ols_metrics["r2"],
            rmse=ols_metrics["rmse"],
            mae=ols_metrics["mae"],
        )
    )

    for alpha in ridge_alphas:
        ridge = RidgeRegressor(alpha=alpha)
        ridge.fit(X, y)
        preds = ridge.predict(X)
        metrics = evaluate_predictions(y, preds)
        results.append(
            RegressionRunResult(
                name="Ridge",
                parameters={"alpha": alpha},
                r2=metrics["r2"],
                rmse=metrics["rmse"],
                mae=metrics["mae"],
            )
        )

    for alpha, l1_ratio in elasticnet_configs:
        enet = ElasticNetRegressor(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000)
        enet.fit(X, y)
        preds = enet.predict(X)
        metrics = evaluate_predictions(y, preds)
        results.append(
            RegressionRunResult(
                name="ElasticNet",
                parameters={"alpha": alpha, "l1_ratio": l1_ratio},
                r2=metrics["r2"],
                rmse=metrics["rmse"],
                mae=metrics["mae"],
            )
        )

    return results


def main() -> None:
    dataset = load_diabetes()
    X_raw, y = dataset.data, dataset.target

    scaling_result = standardize_data(X_raw)
    X_scaled = scaling_result.X_train

    X_poly, feature_names = polynomial_features(X_scaled, degree=2, include_bias=False)

    results = run_model_grid(X_poly, y)

    print("Regression workflow summary:\n")
    for run in results:
        params = ", ".join(f"{k}={v}" for k, v in run.parameters.items()) or "default"
        print(f"{run.name:<10} | params: {params:<30} | R^2={run.r2:.3f} | "
              f"RMSE={run.rmse:.3f} | MAE={run.mae:.3f}")

    if feature_names is not None:
        print(f"\nGenerated {len(feature_names)} polynomial features.")


if __name__ == "__main__":  # pragma: no cover - example script
    main()
