"""Foundational abstractions for the QuantLab modelling pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol


class PipelineStage(Protocol):
    """Protocol for individual pipeline stages."""

    name: str
    description: str
    domain: str
    focus_area: str
    prerequisites: List[str]

    def run(self, context: "PipelineContext") -> None:
        """Execute the stage and mutate the provided context in-place."""


@dataclass
class PipelineContext:
    """Mutable context that threads state through the pipeline."""

    artifacts: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_artifact(self, key: str, value: Any) -> None:
        self.artifacts[key] = value

    def get_artifact(self, key: str, default: Optional[Any] = None) -> Any:
        return self.artifacts.get(key, default)

    def add_metadata(self, key: str, value: Any) -> None:
        self.metadata[key] = value

    def get_metadata(self, key: str, default: Optional[Any] = None) -> Any:
        return self.metadata.get(key, default)


@dataclass
class Pipeline:
    """Simple sequential pipeline runner."""

    stages: List[PipelineStage]

    def run(self, context: Optional[PipelineContext] = None) -> PipelineContext:
        ctx = context or PipelineContext()
        for stage in self.stages:
            stage.run(ctx)
        return ctx
