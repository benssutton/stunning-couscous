"""Inference methods for event dependency tree discovery.

The InferenceMethod protocol defines the interface.  Implementations receive a
wide-format Polars DataFrame (rows = chains, columns = event timestamps as
float64 ms) and return a list of Edge dataclasses describing the inferred
dependency graph.
"""

from __future__ import annotations

import math
from typing import Protocol

import polars as pl
import polars_ds as pds
from pydantic import BaseModel
from scipy.stats import t as t_dist


class Edge(BaseModel):
    source: str
    target: str
    correlation: float
    p_value: float
    mean_delta_ms: float
    std_delta_ms: float
    max_delta_ms: float
    min_delta_ms: float
    sample_count: int


class InferenceMethod(Protocol):
    def infer(
        self,
        matrix: pl.DataFrame,
        event_labels: list[str],
        max_pval: float,
    ) -> list[Edge]: ...


class PearsonInference:
    """Pearson correlation-based dependency inference.

    Ported from notebook cells 17-18.  For each ordered pair (event, dependency):
      1. Drop rows where either timestamp is null.
      2. Verify temporal ordering — every instance of event >= dependency.
      3. Compute Pearson correlation via polars-ds.
      4. Accept only if one-tailed p-value < max_pval.
    Then for each event pick the dependency with the highest correlation.
    """

    def infer(
        self,
        matrix: pl.DataFrame,
        event_labels: list[str],
        max_pval: float = 0.05,
    ) -> list[Edge]:
        num_events = len(event_labels)
        # NxN correlation matrix (rows=event, cols=potential dependency)
        corr = [[0.0] * num_events for _ in range(num_events)]
        pvals = [[1.0] * num_events for _ in range(num_events)]
        sample_counts = [[0] * num_events for _ in range(num_events)]

        for e_idx in range(num_events):
            for d_idx in range(num_events):
                if e_idx == d_idx:
                    continue

                e_col = event_labels[e_idx]
                d_col = event_labels[d_idx]

                # Drop rows where either is null
                pair = matrix.select(e_col, d_col).drop_nulls()
                n = pair.height
                if n < 3:
                    continue

                e_series = pair[e_col]
                d_series = pair[d_col]

                # Temporal ordering: all instances of e must be >= d
                if (e_series < d_series).any():
                    continue

                # Pearson correlation via polars-ds
                r = pair.select(
                    pds.corr(e_col, d_col, method="pearson")
                ).item()

                if r is None or math.isnan(r):
                    continue

                p = self._pearson_p_value(r, n)
                if p < max_pval:
                    corr[e_idx][d_idx] = r
                    pvals[e_idx][d_idx] = p
                    sample_counts[e_idx][d_idx] = n

        # Extract edges: for each event, pick dependency with highest correlation
        edges: list[Edge] = []
        for e_idx in range(num_events):
            row = corr[e_idx]
            if all(v == 0.0 for v in row):
                continue  # root event

            d_idx = max(range(num_events), key=lambda i: row[i])
            e_col = event_labels[e_idx]
            d_col = event_labels[d_idx]

            # Compute latency delta statistics
            pair = matrix.select(e_col, d_col).drop_nulls()
            delta = (pair[e_col] - pair[d_col]).cast(pl.Float64)

            edges.append(Edge(
                source=d_col,
                target=e_col,
                correlation=corr[e_idx][d_idx],
                p_value=pvals[e_idx][d_idx],
                mean_delta_ms=delta.mean(),
                std_delta_ms=delta.std(),
                max_delta_ms=delta.max(),
                min_delta_ms=delta.min(),
                sample_count=sample_counts[e_idx][d_idx],
            ))

        return edges

    @staticmethod
    def _pearson_p_value(r: float, n: int) -> float:
        """One-tailed p-value for Pearson r (alternative='greater').

        Uses the t-distribution transformation:
            t = r * sqrt((n-2) / (1 - r^2))
        matching scipy.stats.pearsonr(alternative="greater").
        """
        if n <= 2:
            return 1.0
        if abs(r) >= 1.0:
            return 0.0
        t_stat = r * math.sqrt((n - 2) / (1 - r * r))
        return 1.0 - t_dist.cdf(t_stat, df=n - 2)

