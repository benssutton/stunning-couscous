"""Unit tests for PearsonInference — no external services required."""

import polars as pl

from services.inference import Edge, PearsonInference


def _linear_chain_matrix() -> tuple[pl.DataFrame, list[str]]:
    """Build a known A->B->C chain as a wide Polars DataFrame.

    Timestamps increase linearly with fixed offsets:
      A = base, B = base + 100ms, C = base + 250ms
    All correlations should be ~1.0 and temporal ordering holds.
    """
    n = 50
    base = [1000.0 + i * 1000 for i in range(n)]  # 1s, 2s, 3s ...
    a = base
    b = [t + 100.0 for t in base]   # A + 100ms
    c = [t + 250.0 for t in base]   # A + 250ms (B + 150ms)
    df = pl.DataFrame({
        "chain_id": [f"chain_{i}" for i in range(n)],
        "A": a,
        "B": b,
        "C": c,
    })
    return df, ["A", "B", "C"]


def test_linear_chain_produces_correct_edges():
    """A->B->C chain should infer edges A->B and B->C (or A->C)."""
    matrix, labels = _linear_chain_matrix()
    inf = PearsonInference()
    edges = inf.infer(matrix, labels, max_pval=0.05)

    targets = {e.target for e in edges}
    sources = {e.source for e in edges}

    # A is a root — should not appear as a target
    assert "A" not in targets
    # B and C should appear as targets
    assert "B" in targets
    assert "C" in targets
    # All correlations should be positive
    assert all(e.correlation > 0.9 for e in edges)
    # All p-values should be very small
    assert all(e.p_value < 0.001 for e in edges)
    # Mean deltas should be positive
    assert all(e.mean_delta_ms > 0 for e in edges)


def test_single_event_is_root():
    """A single event type should produce no edges."""
    df = pl.DataFrame({
        "chain_id": ["c1", "c2", "c3"],
        "X": [100.0, 200.0, 300.0],
    })
    inf = PearsonInference()
    edges = inf.infer(df, ["X"], max_pval=0.05)
    assert edges == []


def test_temporal_violation_no_edge():
    """If B sometimes precedes A, no A->B edge should be inferred."""
    df = pl.DataFrame({
        "chain_id": ["c1", "c2", "c3", "c4", "c5"],
        "A": [100.0, 200.0, 300.0, 400.0, 500.0],
        "B": [150.0, 180.0, 350.0, 380.0, 550.0],  # B < A at index 1,3
    })
    inf = PearsonInference()
    edges = inf.infer(df, ["A", "B"], max_pval=0.05)
    # Neither direction should produce an edge since both violate ordering
    assert edges == []


def test_edge_latency_stats():
    """Verify delta statistics are computed correctly."""
    n = 100
    base = [float(i * 1000) for i in range(n)]
    a = base
    b = [t + 50.0 for t in base]  # constant 50ms offset
    df = pl.DataFrame({
        "chain_id": [f"c{i}" for i in range(n)],
        "A": a,
        "B": b,
    })
    inf = PearsonInference()
    edges = inf.infer(df, ["A", "B"], max_pval=0.05)

    assert len(edges) == 1
    edge = edges[0]
    assert edge.source == "A"
    assert edge.target == "B"
    assert abs(edge.mean_delta_ms - 50.0) < 0.01
    assert edge.min_delta_ms == 50.0
    assert edge.max_delta_ms == 50.0
    assert edge.sample_count == n


def test_null_values_handled():
    """Rows with null timestamps should be excluded, not cause errors."""
    df = pl.DataFrame({
        "chain_id": ["c1", "c2", "c3", "c4", "c5"],
        "A": [100.0, 200.0, None, 400.0, 500.0],
        "B": [150.0, None, None, 450.0, 550.0],
    })
    inf = PearsonInference()
    # Should not raise — nulls are dropped
    edges = inf.infer(df, ["A", "B"], max_pval=0.05)
    # Only 3 valid pairs (indices 0, 3, 4) — may or may not produce edge
    # depending on sample size, but should not error
    assert isinstance(edges, list)
