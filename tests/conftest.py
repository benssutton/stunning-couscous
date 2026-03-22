import random
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from unittest.mock import MagicMock

import httpx
import pytest
import redis

from main import app
from services.adjacency_service import AdjacencyService
from services.clickhouse_service import ClickHouseService
from services.dependencies import get_adjacency_service, get_clickhouse_service, get_redis_service
from services.redis_service import RedisService


# ---------------------------------------------------------------------------
# Redis availability check
# ---------------------------------------------------------------------------

def _redis_available() -> bool:
    try:
        r = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
        r.ping()
        r.close()
        return True
    except redis.ConnectionError:
        return False


_REDIS_UP = _redis_available()
requires_redis = pytest.mark.skipif(not _REDIS_UP, reason="Redis not available on localhost:6379")


# ---------------------------------------------------------------------------
# Service fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def redis_pool():
    """Redis connection pool on DB 0 (RediSearch requires db=0)."""
    if not _REDIS_UP:
        pytest.skip("Redis not available")
    pool = redis.ConnectionPool(
        host="localhost", port=6379, db=0, decode_responses=True
    )
    yield pool
    pool.disconnect()


@pytest.fixture(scope="session")
def redis_service(redis_pool):
    """Real RedisService backed by DB 0, with a clean slate."""
    r = redis.Redis(connection_pool=redis_pool)
    r.flushdb()
    svc = RedisService(redis_pool)
    svc.ensure_index()
    yield svc
    r.flushdb()


@pytest.fixture(scope="session")
def mock_redis():
    """Mock RedisService for tests that don't need real Redis."""
    mock = MagicMock(spec=RedisService)
    mock.add_or_merge_event.return_value = "argus:ec:mock-chain-id"
    return mock


@pytest.fixture(scope="session")
def mock_clickhouse():
    """Mock ClickHouseService — verifies calls without a real ClickHouse."""
    return MagicMock(spec=ClickHouseService)


@pytest.fixture(scope="session")
def mock_adjacency():
    """Mock AdjacencyService for tests that don't need real inference."""
    return MagicMock(spec=AdjacencyService)


@pytest.fixture(autouse=True)
def _flush_redis_between_tests(request):
    """Flush Redis DB 0 before each test for isolation.

    Only activates for tests that use the real Redis `client` fixture.
    """
    if not _REDIS_UP or "client" not in request.fixturenames:
        yield
        return
    pool = request.getfixturevalue("redis_pool")
    r = redis.Redis(connection_pool=pool)
    r.flushdb()
    svc = RedisService(pool)
    svc.ensure_index()
    yield


@pytest.fixture(scope="session")
async def client(redis_service, mock_clickhouse):
    """Async httpx client with real Redis + mock ClickHouse."""
    app.dependency_overrides[get_redis_service] = lambda: redis_service
    app.dependency_overrides[get_clickhouse_service] = lambda: mock_clickhouse
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()


@asynccontextmanager
async def _noop_lifespan(app):
    yield


@pytest.fixture(scope="session")
async def client_no_redis(mock_redis, mock_clickhouse, mock_adjacency):
    """Async httpx client with mocked Redis + mock ClickHouse (no Redis required)."""
    original_lifespan = app.router.lifespan_context
    app.router.lifespan_context = _noop_lifespan
    app.dependency_overrides[get_redis_service] = lambda: mock_redis
    app.dependency_overrides[get_clickhouse_service] = lambda: mock_clickhouse
    app.dependency_overrides[get_adjacency_service] = lambda: mock_adjacency
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()
    app.router.lifespan_context = original_lifespan


# ---------------------------------------------------------------------------
# Event data fixtures
# ---------------------------------------------------------------------------

def _make_ref(ref_type: str, index: int) -> dict:
    return {"type": ref_type, "id": f"{ref_type}{index}", "ver": 1}


def _make_event(
    name: str,
    refs: list[dict],
    timestamp: str,
    context: dict | None = None,
) -> dict:
    return {
        "EventName": name,
        "Timestamp": timestamp,
        "Refs": refs,
        "Context": context or {},
    }


@pytest.fixture
def sample_event_a():
    """Single-ref event from process A."""
    return _make_event(
        name="A",
        refs=[_make_ref("A", 0)],
        timestamp="2025-03-19T12:00:10.024",
        context={"tea": 0},
    )


@pytest.fixture
def sample_event_d():
    """Multi-ref event from process D (depends on B)."""
    return _make_event(
        name="D",
        refs=[_make_ref("D", 0), _make_ref("B", 0)],
        timestamp="2025-03-19T12:00:14.038",
    )


@pytest.fixture
def sample_event_f():
    """Triple-ref event from process F (depends on C and E) with context."""
    return _make_event(
        name="F",
        refs=[_make_ref("F", 0), _make_ref("C", 0), _make_ref("E", 0)],
        timestamp="2025-03-19T12:00:18.268",
        context={"milkshake": 0},
    )


@pytest.fixture
def event_chain_batch():
    """Complete set of events for one transaction through the processing tree.

    Tree: A->B->D, A->C->E->F, A->C->E->G, A->H
    Timestamps are ordered so that dependencies always precede dependents.
    """
    i = 42  # transaction index
    base = datetime(2025, 3, 19, 12, 0, 0)

    # Offsets in ms — parents always before children
    offsets = {"A": 0, "B": 1, "C": 2, "H": 3, "D": 5, "E": 8, "F": 12, "G": 14}

    def ts(name: str) -> str:
        return (base + timedelta(milliseconds=offsets[name])).isoformat(
            timespec="milliseconds"
        )

    # Pick random ref for G's third ref (A or C, matching notebook behavior)
    g_third_ref = random.choice(["A", "C"])

    return [
        _make_event("A", [_make_ref("A", i)], ts("A"), {"tea": i}),
        _make_event("B", [_make_ref("A", i), _make_ref("B", i)], ts("B")),
        _make_event("C", [_make_ref("C", i), _make_ref("A", i)], ts("C"), {"tea": i, "coffee": i}),
        _make_event("H", [_make_ref("A", i)], ts("H")),
        _make_event("D", [_make_ref("D", i), _make_ref("B", i)], ts("D")),
        _make_event("E", [_make_ref("E", i), _make_ref("C", i)], ts("E")),
        _make_event("F", [_make_ref("F", i), _make_ref("C", i), _make_ref("E", i)], ts("F"), {"milkshake": i}),
        _make_event("G", [_make_ref("G", i), _make_ref("E", i), _make_ref(g_third_ref, i)], ts("G")),
    ]
