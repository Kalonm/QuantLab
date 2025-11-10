"""Model training stages inspired by the roadmap."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import statsmodels.api as sm
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
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
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    PoissonRegressor,
    Ridge,
)
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from statsmodels.discrete.discrete_model import Probit

from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor

from .base import PipelineContext


class StatsmodelsGLMRegressor(BaseEstimator, RegressorMixin):
    """Wrap statsmodels GLM models with a scikit-learn compatible API."""

    def __init__(self, family: sm.families.Family, fit_kwargs: Dict[str, object] | None = None):
        self.family = family
        self.fit_kwargs = fit_kwargs or {}
        self._result = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "StatsmodelsGLMRegressor":
        X_const = sm.add_constant(X, has_constant="add")
        model = sm.GLM(y, X_const, family=self.family)
        self._result = model.fit(**self.fit_kwargs)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._result is None:
            raise RuntimeError("Model must be fitted before calling predict().")
        X_const = sm.add_constant(X, has_constant="add")
        predictions = self._result.predict(X_const)
        return np.asarray(predictions)


class StatsmodelsProbitClassifier(BaseEstimator, ClassifierMixin):
    """Simple Probit classifier leveraging statsmodels."""

    def __init__(self) -> None:
        self._result = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "StatsmodelsProbitClassifier":
        X_const = sm.add_constant(X, has_constant="add")
        model = Probit(y, X_const)
        self._result = model.fit(disp=False)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._result is None:
            raise RuntimeError("Model must be fitted before calling predict().")
        probabilities = self.predict_proba(X)[:, 1]
        return (probabilities >= 0.5).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._result is None:
            raise RuntimeError("Model must be fitted before calling predict_proba().")
        X_const = sm.add_constant(X, has_constant="add")
        probs = np.asarray(self._result.predict(X_const))
        stacked = np.column_stack([1.0 - probs, probs])
        return stacked


@dataclass
class BaselineRegressionStage:
    """Train baseline supervised learning models spanning regression and classification."""

    test_size: float = 0.2
    random_state: int = 42
    include_regularised: bool = True
    ridge_alpha: float = 1.0
    lasso_alpha: float = 0.001
    lasso_max_iter: int = 10000
    elastic_net_alpha: float = 0.001
    elastic_net_l1_ratio: float = 0.5
    elastic_net_max_iter: int = 10000
    logistic_max_iter: int = 1000
    svm_regression_kernel: str = "rbf"
    svm_regression_c: float = 1.0
    svm_regression_epsilon: float = 0.1
    svm_classification_kernel: str = "rbf"
    svm_classification_c: float = 1.0
    knn_neighbors: int = 5
    poisson_alpha: float = 0.0
    poisson_max_iter: int = 1000
    negative_binomial_alpha: float = 1.0
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

        task_type: str = context.get_metadata("target_task", "regression")
        n_classes: int | None = context.get_metadata("n_classes")
        is_count_target: bool = bool(context.get_metadata("target_is_count", False))

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
            metric_fn = self._compute_classification_metrics
        else:
            models = self._build_regression_models(is_count_target)
            metric_fn = self._compute_regression_metrics

        fitted_models: Dict[str, Dict[str, object]] = {}
        for key, model in models.items():
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            metrics = metric_fn(y_test, predictions)
            fitted_models[key] = {
                "model": model,
                "metrics": metrics,
            }

        context.add_artifact("trained_models", fitted_models)
        context.add_metadata("feature_names", feature_names)

    def _build_regression_models(self, is_count_target: bool) -> Dict[str, BaseEstimator]:
        models: Dict[str, BaseEstimator] = {
            "ols": LinearRegression(),
        }

        if self.include_regularised:
            models.update(
                {
                    "ridge": Ridge(alpha=self.ridge_alpha),
                    "lasso": Lasso(alpha=self.lasso_alpha, max_iter=self.lasso_max_iter),
                    "elastic_net": ElasticNet(
                        alpha=self.elastic_net_alpha,
                        l1_ratio=self.elastic_net_l1_ratio,
                        max_iter=self.elastic_net_max_iter,
                    ),
                }
            )

        models["svm_regression"] = SVR(
            kernel=self.svm_regression_kernel,
            C=self.svm_regression_c,
            epsilon=self.svm_regression_epsilon,
        )
        models["knn_regression"] = KNeighborsRegressor(n_neighbors=self.knn_neighbors)

        if is_count_target:
            models["poisson_regression"] = PoissonRegressor(
                alpha=self.poisson_alpha,
                max_iter=self.poisson_max_iter,
            )
            models["negative_binomial_regression"] = StatsmodelsGLMRegressor(
                sm.families.NegativeBinomial(alpha=self.negative_binomial_alpha)
            )

        return models

    def _build_classification_models(self, n_classes: int | None) -> Dict[str, BaseEstimator]:
        models: Dict[str, BaseEstimator] = {
            "logistic_regression": LogisticRegression(
                max_iter=self.logistic_max_iter,
                random_state=self.random_state,
            ),
            "svm_classifier": SVC(
                kernel=self.svm_classification_kernel,
                C=self.svm_classification_c,
                probability=True,
                random_state=self.random_state,
            ),
            "knn_classifier": KNeighborsClassifier(n_neighbors=self.knn_neighbors),
            "naive_bayes_classifier": GaussianNB(),
        }

        if n_classes == 2:
            models["probit_regression"] = StatsmodelsProbitClassifier()

        return models

    @staticmethod
    def _compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        return {
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "r2": float(r2_score(y_true, y_pred)),
        }

    @staticmethod
    def _compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )
        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }


@dataclass
class TreeAndEnsembleStage:
    """Train tree-based and ensemble models from the roadmap backlog."""

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
    description: str = "Fit tree-based and ensemble models spanning roadmap section 1.2."
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
