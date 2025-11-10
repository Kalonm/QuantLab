"""Data ingestion stages for QuantLab."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import pandas as pd

from .base import PipelineContext, PipelineStage


@dataclass
class CSVIngestionStage:
    """Load tabular data from CSV files into the pipeline context."""

    source_path: Path
    target_column: str
    numeric_features: List[str] | None = None
    name: str = "csv_ingestion"
    description: str = "Load processed pair scan data and identify modelling target."
    domain: str = "Machine Learning & AI"
    focus_area: str = "1.1 Supervised Learning"
    prerequisites: List[str] = field(default_factory=lambda: [
        "Clean processed dataset available",
        "Target column selected",
    ])

    def run(self, context: PipelineContext) -> None:
        if not self.source_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.source_path}")

        data = pd.read_csv(self.source_path)
        context.add_artifact("raw_dataframe", data)
        context.add_metadata("target_column", self.target_column)
        if self.numeric_features is not None:
            context.add_metadata("numeric_features", self.numeric_features)
        else:
            numeric_cols = data.select_dtypes(include=["number"]).columns.tolist()
            context.add_metadata("numeric_features", numeric_cols)
