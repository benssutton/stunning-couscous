"""Orchestrates adjacency matrix computation: query -> infer -> persist."""

from __future__ import annotations

from uuid import uuid4

from pydantic import BaseModel

from services.clickhouse_service import ClickHouseService
from services.inference import Edge, InferenceMethod, PearsonInference


class AdjacencyResult(BaseModel):
    run_id: str
    method: str
    max_pval: float
    edges: list[Edge]
    root_events: list[str]


class AdjacencyService:
    def __init__(self, ch_svc: ClickHouseService):
        self.ch_svc = ch_svc
        self.methods: dict[str, InferenceMethod] = {
            "pearson": PearsonInference(),
        }

    def compute(self, method: str = "pearson", max_pval: float = 0.05) -> AdjacencyResult:
        """Run the full adjacency matrix pipeline.

        Raises KeyError if the method is not registered.
        """
        inference = self.methods[method]  # KeyError if unknown

        matrix = self.ch_svc.query_timestamp_matrix()
        event_labels = [c for c in matrix.columns if c != "chain_id"]

        edges = inference.infer(matrix, event_labels, max_pval)

        targets = {e.target for e in edges}
        root_events = [l for l in event_labels if l not in targets]

        run_id = uuid4()
        self.ch_svc.insert_adjacency_result(run_id, edges, method, max_pval)

        return AdjacencyResult(
            run_id=str(run_id),
            method=method,
            max_pval=max_pval,
            edges=edges,
            root_events=root_events,
        )
