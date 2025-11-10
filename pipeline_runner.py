"""Entry point for executing the QuantLab modelling pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from quantlab.pipeline import Pipeline, PipelineContext
from quantlab.pipeline.data_ingestion import CSVIngestionStage
from quantlab.pipeline.feature_engineering import NumericFeatureSelectorStage
from quantlab.roadmap.machine_learning.supervised import BaselineRegressionStage
from quantlab.pipeline.reporting import MetricsReportingStage


def build_pipeline(dataset: Path, target: str, output: Path) -> Pipeline:
    """Construct the sequential pipeline stages."""

    ingestion = CSVIngestionStage(source_path=dataset, target_column=target)
    feature_select = NumericFeatureSelectorStage()
    modelling = BaselineRegressionStage()
    reporting = MetricsReportingStage(output_path=output)

    return Pipeline(stages=[ingestion, feature_select, modelling, reporting])


def parse_args() -> argparse.Namespace:
    """CLI argument parser for the pipeline runner."""

    parser = argparse.ArgumentParser(description="Run the QuantLab roadmap-aligned pipeline.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/processed/pair_scan_summary.csv"),
        help="Path to the processed dataset used for baseline modelling.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="half_life_bars",
        help="Name of the target column for regression tasks.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/baseline_regression_metrics.json"),
        help="Location where model metrics will be written.",
    )
    return parser.parse_args()


def main() -> None:
    """Execute the pipeline and print a concise summary."""

    args = parse_args()
    pipeline = build_pipeline(dataset=args.dataset, target=args.target, output=args.output)
    context = pipeline.run(PipelineContext())
    report = context.get_artifact("metrics_report")
    if report:
        print("Pipeline completed. Saved metrics report to:", args.output)
        for model_name, payload in report["models"].items():
            metrics = payload["metrics"]
            metric_summary = ", ".join(f"{key}={value:.4f}" for key, value in metrics.items())
            print(f" - {model_name}: {metric_summary}")


if __name__ == "__main__":
    main()
