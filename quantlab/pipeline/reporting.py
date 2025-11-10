"""Reporting stages for pipeline outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from .base import PipelineContext


@dataclass
class MetricsReportingStage:
    """Persist model metrics to disk for transparency."""

    output_path: Path
    name: str = "metrics_reporting"
    description: str = "Record baseline regression metrics for roadmap tracking."
    domain: str = "Machine Learning & AI"
    focus_area: str = "Operational Reporting"
    prerequisites: List[str] = field(default_factory=lambda: [
        "Models trained and evaluated",
    ])

    def run(self, context: PipelineContext) -> None:
        trained_models: Dict[str, Dict[str, Dict[str, float]]] | None = context.get_artifact("trained_models")
        feature_names: List[str] | None = context.get_artifact("feature_names")
        if trained_models is None or feature_names is None:
            raise ValueError("No models available to report metrics.")

        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "feature_names": feature_names,
            "models": {},
        }

        for model_key, payload in trained_models.items():
            metrics = payload["metrics"]
            report["models"][model_key] = {
                "metrics": metrics,
            }

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)

        context.add_artifact("metrics_report", report)
