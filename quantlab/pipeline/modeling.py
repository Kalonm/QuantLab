"""Model training stages inspired by the roadmap."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from .base import PipelineContext


@dataclass
class BaselineRegressionStage:
    """Train baseline linear models (OLS, Ridge, LASSO, Elastic Net)."""

    test_size: float = 0.2
    random_state: int = 42
    include_regularised: bool = True
    name: str = "baseline_regression"
    description: str = "Fit roadmap baseline supervised models on selected features."
    domain: str = "Machine Learning & AI"
    focus_area: str = "1.1 Supervised Learning"
    prerequisites: List[str] = field(default_factory=lambda: [
        "Numeric feature matrix prepared",
        "Target vector available",
    ])

    def run(self, context: PipelineContext) -> None:
        features: np.ndarray | None = context.get_artifact("feature_matrix")
        target: np.ndarray | None = context.get_artifact("target_vector")
        feature_names: List[str] | None = context.get_artifact("feature_names")

        if features is None or target is None or feature_names is None:
            raise ValueError("Required artifacts missing for model training.")

        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=self.test_size, random_state=self.random_state
        )

        models: Dict[str, LinearRegression] = {
            "ols": LinearRegression(),
        }
        if self.include_regularised:
            models.update(
                {
                    "ridge": Ridge(alpha=1.0),
                    "lasso": Lasso(alpha=0.001, max_iter=10000),
                    "elastic_net": ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=10000),
                }
            )

        fitted_models: Dict[str, Dict[str, float]] = {}
        for key, model in models.items():
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            metrics = {
                "mae": float(mean_absolute_error(y_test, predictions)),
                "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
                "r2": float(r2_score(y_test, predictions)),
            }
            fitted_models[key] = {
                "model": model,
                "metrics": metrics,
            }

        context.add_artifact("trained_models", fitted_models)
        context.add_metadata("feature_names", feature_names)
