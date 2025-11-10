"""Tree and ensemble learning models for QuantLab experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    BaggingClassifier,
    BaggingRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    StackingClassifier,
    StackingRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor

from ...pipeline.base import PipelineContext
from .supervised import BaselineRegressionStage

__all__ = ["TreeAndEnsembleStage"]


@dataclass
class TreeAndEnsembleStage:
    """Train tree-based and ensemble models maintained in the QuantLab backlog."""

    test_size: float = 0.2
    random_state: int = 42
    decision_tree_max_depth: int | None = None
    random_forest_n_estimators: int = 300
    random_forest_max_depth: int | None = None
    gradient_boosting_learning_rate: float = 0.05
    gradient_boosting_n_estimators: int = 300
    histogram_max_depth: int | None = None
    bagging_n_estimators: int = 50
    stacking_passthrough: bool = True
    xgboost_n_estimators: int = 400
    xgboost_learning_rate: float = 0.05
    lightgbm_n_estimators: int = 400
    lightgbm_learning_rate: float = 0.05
    catboost_depth: int = 6
    catboost_learning_rate: float = 0.05
    catboost_iterations: int = 400
    name: str = "tree_ensemble_models"
    description: str = "Fit tree-based and ensemble models spanning the QuantLab backlog."
    domain: str = "Machine Learning & AI"
    focus_area: str = "1.2 Tree & Ensemble Methods"
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

        task_type: str = context.get_metadata("target_task", "regression")
        n_classes: int | None = context.get_metadata("n_classes")

        stratify = target if task_type == "classification" else None

        X_train, X_test, y_train, y_test = train_test_split(
            features,
            target,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify,
        )

        if task_type == "classification":
            models = self._build_classification_models(n_classes)
            metric_fn = BaselineRegressionStage._compute_classification_metrics
        else:
            models = self._build_regression_models()
            metric_fn = BaselineRegressionStage._compute_regression_metrics

        fitted_models: Dict[str, Dict[str, object]] = {}
        for key, model in models.items():
            model.fit(X_train, y_train)
            raw_predictions = np.asarray(model.predict(X_test))
            predictions = raw_predictions.reshape(-1)
            if task_type == "classification":
                predictions = self._post_process_classification_predictions(
                    predictions, raw_predictions, n_classes
                )
            fitted_models[key] = {
                "model": model,
                "metrics": metric_fn(y_test, predictions),
            }

        context.add_artifact("tree_ensemble_models", fitted_models)
        context.add_metadata("feature_names", feature_names)

    def _build_regression_models(self) -> Dict[str, BaseEstimator]:
        base_regressors = {
            "decision_tree_regressor": DecisionTreeRegressor(
                max_depth=self.decision_tree_max_depth,
                random_state=self.random_state,
            ),
            "random_forest_regressor": RandomForestRegressor(
                n_estimators=self.random_forest_n_estimators,
                max_depth=self.random_forest_max_depth,
                random_state=self.random_state,
                n_jobs=-1,
            ),
            "gradient_boosting_regressor": GradientBoostingRegressor(
                learning_rate=self.gradient_boosting_learning_rate,
                n_estimators=self.gradient_boosting_n_estimators,
                random_state=self.random_state,
            ),
            "hist_gradient_boosting_regressor": HistGradientBoostingRegressor(
                learning_rate=self.gradient_boosting_learning_rate,
                max_depth=self.histogram_max_depth,
                random_state=self.random_state,
            ),
            "bagging_regressor": BaggingRegressor(
                n_estimators=self.bagging_n_estimators,
                random_state=self.random_state,
                n_jobs=-1,
            ),
            "xgboost_regressor": XGBRegressor(
                n_estimators=self.xgboost_n_estimators,
                learning_rate=self.xgboost_learning_rate,
                objective="reg:squarederror",
                tree_method="hist",
                random_state=self.random_state,
                n_jobs=-1,
                verbosity=0,
            ),
            "lightgbm_regressor": LGBMRegressor(
                n_estimators=self.lightgbm_n_estimators,
                learning_rate=self.lightgbm_learning_rate,
                random_state=self.random_state,
                n_jobs=-1,
            ),
            "catboost_regressor": CatBoostRegressor(
                iterations=self.catboost_iterations,
                depth=self.catboost_depth,
                learning_rate=self.catboost_learning_rate,
                random_seed=self.random_state,
                loss_function="RMSE",
                verbose=False,
            ),
        }

        stacking_estimators = [
            (
                "decision_tree",
                DecisionTreeRegressor(
                    max_depth=self.decision_tree_max_depth,
                    random_state=self.random_state,
                ),
            ),
            (
                "random_forest",
                RandomForestRegressor(
                    n_estimators=self.random_forest_n_estimators,
                    max_depth=self.random_forest_max_depth,
                    random_state=self.random_state,
                    n_jobs=-1,
                ),
            ),
            (
                "gradient_boosting",
                GradientBoostingRegressor(
                    learning_rate=self.gradient_boosting_learning_rate,
                    n_estimators=self.gradient_boosting_n_estimators,
                    random_state=self.random_state,
                ),
            ),
        ]

        base_regressors["stacking_regressor"] = StackingRegressor(
            estimators=stacking_estimators,
            final_estimator=LinearRegression(),
            passthrough=self.stacking_passthrough,
            n_jobs=-1,
        )

        return base_regressors

    def _build_classification_models(self, n_classes: int | None) -> Dict[str, BaseEstimator]:
        base_classifiers = {
            "decision_tree_classifier": DecisionTreeClassifier(
                max_depth=self.decision_tree_max_depth,
                random_state=self.random_state,
            ),
            "random_forest_classifier": RandomForestClassifier(
                n_estimators=self.random_forest_n_estimators,
                max_depth=self.random_forest_max_depth,
                random_state=self.random_state,
                n_jobs=-1,
            ),
            "gradient_boosting_classifier": GradientBoostingClassifier(
                learning_rate=self.gradient_boosting_learning_rate,
                n_estimators=self.gradient_boosting_n_estimators,
                random_state=self.random_state,
            ),
            "hist_gradient_boosting_classifier": HistGradientBoostingClassifier(
                learning_rate=self.gradient_boosting_learning_rate,
                max_depth=self.histogram_max_depth,
                random_state=self.random_state,
            ),
            "bagging_classifier": BaggingClassifier(
                n_estimators=self.bagging_n_estimators,
                random_state=self.random_state,
                n_jobs=-1,
            ),
            "xgboost_classifier": self._build_xgboost_classifier(n_classes),
            "lightgbm_classifier": self._build_lightgbm_classifier(n_classes),
            "catboost_classifier": self._build_catboost_classifier(n_classes),
        }

        stacking_estimators = [
            (
                "decision_tree",
                DecisionTreeClassifier(
                    max_depth=self.decision_tree_max_depth,
                    random_state=self.random_state,
                ),
            ),
            (
                "random_forest",
                RandomForestClassifier(
                    n_estimators=self.random_forest_n_estimators,
                    max_depth=self.random_forest_max_depth,
                    random_state=self.random_state,
                    n_jobs=-1,
                ),
            ),
            (
                "gradient_boosting",
                GradientBoostingClassifier(
                    learning_rate=self.gradient_boosting_learning_rate,
                    n_estimators=self.gradient_boosting_n_estimators,
                    random_state=self.random_state,
                ),
            ),
        ]

        base_classifiers["stacking_classifier"] = StackingClassifier(
            estimators=stacking_estimators,
            final_estimator=LogisticRegression(max_iter=1000, random_state=self.random_state),
            passthrough=self.stacking_passthrough,
            n_jobs=-1,
        )

        return base_classifiers

    def _build_xgboost_classifier(self, n_classes: int | None) -> XGBClassifier:
        classifier_kwargs = {
            "n_estimators": self.xgboost_n_estimators,
            "learning_rate": self.xgboost_learning_rate,
            "tree_method": "hist",
            "random_state": self.random_state,
            "use_label_encoder": False,
            "n_jobs": -1,
            "verbosity": 0,
        }

        if n_classes and n_classes > 2:
            classifier_kwargs["objective"] = "multi:softprob"
            classifier_kwargs["num_class"] = n_classes
            classifier_kwargs["eval_metric"] = "mlogloss"
        else:
            classifier_kwargs["objective"] = "binary:logistic"
            classifier_kwargs["eval_metric"] = "logloss"

        return XGBClassifier(**classifier_kwargs)

    def _build_lightgbm_classifier(self, n_classes: int | None) -> LGBMClassifier:
        objective = "binary" if not n_classes or n_classes <= 2 else "multiclass"
        kwargs = {
            "n_estimators": self.lightgbm_n_estimators,
            "learning_rate": self.lightgbm_learning_rate,
            "objective": objective,
            "random_state": self.random_state,
            "n_jobs": -1,
        }
        if n_classes and n_classes > 2:
            kwargs["num_class"] = n_classes
        return LGBMClassifier(**kwargs)

    def _build_catboost_classifier(self, n_classes: int | None) -> CatBoostClassifier:
        loss_function = "Logloss" if not n_classes or n_classes <= 2 else "MultiClass"
        kwargs = {
            "iterations": self.catboost_iterations,
            "depth": self.catboost_depth,
            "learning_rate": self.catboost_learning_rate,
            "random_seed": self.random_state,
            "verbose": False,
            "loss_function": loss_function,
        }
        if n_classes and n_classes > 2:
            kwargs["classes_count"] = n_classes
        return CatBoostClassifier(**kwargs)

    @staticmethod
    def _post_process_classification_predictions(
        flattened: np.ndarray, raw: np.ndarray, n_classes: int | None
    ) -> np.ndarray:
        """Convert raw model outputs into discrete class predictions."""

        if raw.ndim > 1 and raw.shape[1] > 1:
            return np.argmax(raw, axis=1)

        if flattened.dtype.kind == "f":
            if not n_classes or n_classes <= 2:
                return (flattened >= 0.5).astype(int)
            return np.round(flattened).astype(int)

        return flattened
