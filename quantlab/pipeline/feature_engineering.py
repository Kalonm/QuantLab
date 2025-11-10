"""Feature engineering stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from .base import PipelineContext


@dataclass
class NumericFeatureSelectorStage:
    """Select numeric features and handle missing values."""

    feature_subset: List[str] | None = None
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

        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()

        imputed = imputer.fit_transform(features)
        scaled = scaler.fit_transform(imputed)

        context.add_artifact("feature_matrix", scaled)
        context.add_artifact("feature_names", numeric_cols)
        context.add_artifact("target_vector", target.to_numpy(dtype=np.float64))
        context.add_artifact("feature_imputer", imputer)
        context.add_artifact("feature_scaler", scaler)
