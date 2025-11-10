"""Feature engineering stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd
from pandas.api import types as ptypes
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .base import PipelineContext


@dataclass
class NumericFeatureSelectorStage:
    """Select numeric features and handle missing values."""

    feature_subset: List[str] | None = None
    classification_max_unique: int = 20
    name: str = "numeric_feature_selection"
    description: str = "Filter numeric predictors and standardise inputs."
    domain: str = "Machine Learning & AI"
    focus_area: str = "1.1 Supervised Learning"
    prerequisites: List[str] = field(default_factory=lambda: [
        "Raw dataframe ingested",
        "Numeric columns identified",
    ])

    def run(self, context: PipelineContext) -> None:
        df: pd.DataFrame = context.get_artifact("raw_dataframe")
        if df is None:
            raise ValueError("Dataframe not found in context. Ensure ingestion stage ran successfully.")

        numeric_cols = self.feature_subset or context.get_metadata("numeric_features", [])
        target_col = context.get_metadata("target_column")

        numeric_cols = [col for col in numeric_cols if col != target_col]

        features = df[numeric_cols]
        target = df[target_col]

        valid_mask = target.notna()
        features = features.loc[valid_mask]
        target = target.loc[valid_mask]

        if target.empty:
            raise ValueError("Target column contains only missing values after filtering.")

        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()

        imputed = imputer.fit_transform(features)
        scaled = scaler.fit_transform(imputed)

        task_type = self._infer_task_type(target)

        context.add_artifact("feature_matrix", scaled)
        context.add_artifact("feature_names", numeric_cols)
        context.add_artifact("feature_imputer", imputer)
        context.add_artifact("feature_scaler", scaler)
        context.add_artifact("target_series", target)

        if task_type == "classification":
            encoder = LabelEncoder()
            encoded_target = encoder.fit_transform(target)
            context.add_artifact("target_vector", encoded_target.astype(np.int64))
            context.add_artifact("target_encoder", encoder)
            context.add_metadata("target_task", "classification")
            context.add_metadata("target_classes", encoder.classes_.tolist())
            context.add_metadata("n_classes", int(len(encoder.classes_)))
        else:
            numeric_target = target.to_numpy(dtype=np.float64)
            context.add_artifact("target_vector", numeric_target)
            context.add_metadata("target_task", "regression")
            context.add_metadata("n_classes", None)

        context.add_metadata("target_is_count", self._is_count_target(target))

    def _infer_task_type(self, target: pd.Series) -> str:
        dtype = target.dtype
        unique_values = target.unique()
        unique_count = len(unique_values)

        if ptypes.is_bool_dtype(dtype) or ptypes.is_categorical_dtype(dtype):
            return "classification"

        if not ptypes.is_numeric_dtype(dtype):
            return "classification"

        if unique_count <= 1:
            return "regression"

        if ptypes.is_integer_dtype(dtype):
            if unique_count <= self.classification_max_unique:
                return "classification"
            return "regression"

        if unique_count <= self.classification_max_unique and np.allclose(unique_values, np.round(unique_values)):
            return "classification"

        return "regression"

    @staticmethod
    def _is_count_target(target: pd.Series) -> bool:
        if not ptypes.is_numeric_dtype(target.dtype):
            return False

        values = target.to_numpy(dtype=np.float64)
        if values.size == 0:
            return False

        non_negative = np.all(values >= 0)
        integer_like = np.allclose(values, np.round(values))
        return bool(non_negative and integer_like)
