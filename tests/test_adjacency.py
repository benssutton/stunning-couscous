"""Tests for POST /adjacency_matrix endpoint."""

from contextlib import asynccontextmanager
from unittest.mock import MagicMock

import httpx

from main import app
from services.adjacency_service import AdjacencyResult, AdjacencyService
from services.dependencies import get_adjacency_service, get_clickhouse_service, get_redis_service
from services.inference import Edge
from services.redis_service import RedisService


@asynccontextmanager
async def _noop_lifespan(app):
    yield


def _make_mock_adjacency(edges: list[Edge], root_events: list[str]) -> MagicMock:
    mock = MagicMock(spec=AdjacencyService)
    mock.compute.return_value = AdjacencyResult(
        run_id="test-run-id",
        method="pearson",
        max_pval=0.05,
        edges=edges,
        root_events=root_events,
    )
    return mock


async def test_adjacency_matrix_returns_edges():
    """POST /adjacency_matrix returns expected edges and root events."""
    edges = [
        Edge(source="A", target="B", correlation=0.99, p_value=0.001,
             mean_delta_ms=100.0, std_delta_ms=5.0, max_delta_ms=110.0,
             min_delta_ms=90.0, sample_count=50),
    ]
    mock_adj = _make_mock_adjacency(edges, root_events=["A"])
    mock_redis = MagicMock(spec=RedisService)

    original_lifespan = app.router.lifespan_context
    app.router.lifespan_context = _noop_lifespan
    app.dependency_overrides[get_adjacency_service] = lambda: mock_adj
    app.dependency_overrides[get_redis_service] = lambda: mock_redis
    app.dependency_overrides[get_clickhouse_service] = lambda: MagicMock()

    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.put("/adjacency_matrix", json={"method": "pearson", "max_pval": 0.05})

        assert resp.status_code == 200
        data = resp.json()
        assert data["run_id"] == "test-run-id"
        assert data["method"] == "pearson"
        assert data["edge_count"] == 1
        assert data["root_events"] == ["A"]
        assert data["edges"][0]["source"] == "A"
        assert data["edges"][0]["target"] == "B"
        assert data["edges"][0]["correlation"] == 0.99
        assert data["edges"][0]["p_value"] == 0.001
    finally:
        app.dependency_overrides.clear()
        app.router.lifespan_context = original_lifespan


async def test_adjacency_matrix_unknown_method_returns_400():
    """Unknown inference method should return 400."""
    mock_adj = MagicMock(spec=AdjacencyService)
    mock_adj.compute.side_effect = KeyError("bogus")
    mock_redis = MagicMock(spec=RedisService)

    original_lifespan = app.router.lifespan_context
    app.router.lifespan_context = _noop_lifespan
    app.dependency_overrides[get_adjacency_service] = lambda: mock_adj
    app.dependency_overrides[get_redis_service] = lambda: mock_redis
    app.dependency_overrides[get_clickhouse_service] = lambda: MagicMock()

    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.put("/adjacency_matrix", json={"method": "bogus"})

        assert resp.status_code == 400
        assert "bogus" in resp.json()["detail"]
    finally:
        app.dependency_overrides.clear()
        app.router.lifespan_context = original_lifespan


async def test_adjacency_matrix_empty_result():
    """No edges should return empty list and all events as roots."""
    mock_adj = _make_mock_adjacency(edges=[], root_events=["X", "Y"])

    original_lifespan = app.router.lifespan_context
    app.router.lifespan_context = _noop_lifespan
    app.dependency_overrides[get_adjacency_service] = lambda: mock_adj
    app.dependency_overrides[get_redis_service] = lambda: MagicMock(spec=RedisService)
    app.dependency_overrides[get_clickhouse_service] = lambda: MagicMock()

    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.put("/adjacency_matrix", json={})

        assert resp.status_code == 200
        data = resp.json()
        assert data["edge_count"] == 0
        assert data["edges"] == []
        assert data["root_events"] == ["X", "Y"]
    finally:
        app.dependency_overrides.clear()
        app.router.lifespan_context = original_lifespan
