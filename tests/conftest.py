import os
import random
from datetime import datetime, timedelta

# Disable Ryuk (testcontainers resource reaper) — its port 8080 is not
# mapped on Windows/Docker Desktop, causing a ConnectionError on startup.
os.environ.setdefault("TESTCONTAINERS_RYUK_DISABLED", "true")

import clickhouse_connect
import httpx
import pytest
import redis
import redis.asyncio as aioredis
from redis.commands.search.field import TagField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from testcontainers.clickhouse import ClickHouseContainer
from testcontainers.redis import RedisContainer

from main import app
from services.adjacency_service import AdjacencyService
from services.clickhouse_service import ClickHouseBatchWriter, ClickHouseService
from services.dependencies import (
    get_adjacency_service,
    get_batch_writer,
    get_clickhouse_service,
    get_redis_service,
)
from services.redis_service import RedisService


def _ensure_index_sync(sync_r):
    """Create the RediSearch index using the sync Redis client."""
    try:
        sync_r.ft(RedisService.INDEX_NAME).info()
    except redis.ResponseError:
        sync_r.ft(RedisService.INDEX_NAME).create_index(
            [
                TagField(
                    "$.concatenatedrefs[*]",
                    as_name="concatenatedrefs",
                ),
                TagField("$.complete", as_name="complete"),
                TagField("$.terminated", as_name="terminated"),
            ],
            definition=IndexDefinition(
                index_type=IndexType.JSON,
                prefix=[f"{RedisService.KEY_BASE}:"],
            ),
        )


# ---------------------------------------------------------------------------
# Container fixtures (session-scoped — started once for the whole test run)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def redis_container():
    """Redis Stack container — required for RediSearch (FT.CREATE)."""
    with RedisContainer(
        image="redis/redis-stack:latest"
    ) as container:
        yield container


@pytest.fixture(scope="session")
def clickhouse_container():
    """ClickHouse container."""
    with ClickHouseContainer(
        image="clickhouse/clickhouse-server:latest"
    ) as container:
        yield container


# ---------------------------------------------------------------------------
# Service fixtures (all sync to avoid event-loop scope issues)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def redis_host_port(redis_container):
    """Extract host and port from the Redis container for reuse."""
    host = redis_container.get_container_host_ip()
    port = int(redis_container.get_exposed_port(6379))
    return host, port


@pytest.fixture(scope="session")
def redis_pool(redis_host_port):
    """Async Redis connection pool pointed at the testcontainer."""
    host, port = redis_host_port
    pool = aioredis.ConnectionPool(
        host=host, port=port, db=0, decode_responses=True,
    )
    yield pool


@pytest.fixture(scope="session")
def sync_redis(redis_host_port):
    """Sync Redis client for test setup/teardown operations."""
    host, port = redis_host_port
    r = redis.Redis(
        host=host, port=port, db=0, decode_responses=True,
    )
    yield r
    r.close()


@pytest.fixture(scope="session")
def redis_service(redis_pool, sync_redis):
    """RedisService backed by containerized Redis Stack."""
    sync_redis.flushdb()
    svc = RedisService(redis_pool)
    _ensure_index_sync(sync_redis)
    yield svc
    sync_redis.flushdb()


@pytest.fixture(scope="session")
def clickhouse_service(clickhouse_container):
    """Real ClickHouseService backed by a containerized ClickHouse."""
    host = clickhouse_container.get_container_host_ip()
    port = int(clickhouse_container.get_exposed_port(8123))
    ch_client = clickhouse_connect.get_client(
        host=host,
        port=port,
        username=clickhouse_container.username,
        password=clickhouse_container.password,
    )
    svc = ClickHouseService(ch_client, database="arestor")
    svc.ensure_table()
    svc.ensure_adjacency_table()
    svc.ensure_profiles_table()
    svc.ensure_classifier_model_table()
    yield svc
    ch_client.close()


@pytest.fixture(scope="session")
def batch_writer(clickhouse_service):
    """ClickHouse batch writer for tests — flushes every append (batch=1)."""
    writer = ClickHouseBatchWriter(
        clickhouse_service.client,
        f"{clickhouse_service.database}.events",
        max_batch_size=1,
        flush_interval_s=600,
    )
    # No start() needed: with batch_size=1 every append() auto-flushes
    yield writer


@pytest.fixture(autouse=True)
def _flush_redis_between_tests(request):
    """Flush Redis and truncate ClickHouse tables before each test."""
    if "client" not in request.fixturenames:
        yield
        return
    sync_r = request.getfixturevalue("sync_redis")
    sync_r.flushdb()
    _ensure_index_sync(sync_r)
    ch_svc = request.getfixturevalue("clickhouse_service")
    ch_svc.client.command(
        "TRUNCATE TABLE IF EXISTS arestor.events"
    )
    ch_svc.client.command(
        "TRUNCATE TABLE IF EXISTS arestor.adjacency_edges"
    )
    yield


@pytest.fixture(scope="session")
def client(redis_service, clickhouse_service, batch_writer):
    """Async httpx client with real Redis Stack + real ClickHouse."""
    adjacency_service = AdjacencyService(clickhouse_service)
    app.dependency_overrides[get_redis_service] = (
        lambda: redis_service
    )
    app.dependency_overrides[get_clickhouse_service] = (
        lambda: clickhouse_service
    )
    app.dependency_overrides[get_adjacency_service] = (
        lambda: adjacency_service
    )
    app.dependency_overrides[get_batch_writer] = (
        lambda: batch_writer
    )
    transport = httpx.ASGITransport(app=app)
    c = httpx.AsyncClient(
        transport=transport, base_url="http://test",
    )
    yield c
    app.dependency_overrides.clear()


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
    """Complete set of events for one transaction.

    Tree: A->B->D, A->C->E->F, A->C->E->G, A->H
    """
    i = 42
    base = datetime(2025, 3, 19, 12, 0, 0)
    offsets = {
        "A": 0, "B": 1, "C": 2, "H": 3,
        "D": 5, "E": 8, "F": 12, "G": 14,
    }

    def ts(name: str) -> str:
        return (
            base + timedelta(milliseconds=offsets[name])
        ).isoformat(timespec="milliseconds")

    g_third_ref = random.choice(["A", "C"])

    return [
        _make_event(
            "A", [_make_ref("A", i)], ts("A"), {"tea": i},
        ),
        _make_event(
            "B",
            [_make_ref("A", i), _make_ref("B", i)],
            ts("B"),
        ),
        _make_event(
            "C",
            [_make_ref("C", i), _make_ref("A", i)],
            ts("C"),
            {"tea": i, "coffee": i},
        ),
        _make_event("H", [_make_ref("A", i)], ts("H")),
        _make_event(
            "D",
            [_make_ref("D", i), _make_ref("B", i)],
            ts("D"),
        ),
        _make_event(
            "E",
            [_make_ref("E", i), _make_ref("C", i)],
            ts("E"),
        ),
        _make_event(
            "F",
            [_make_ref("F", i), _make_ref("C", i), _make_ref("E", i)],
            ts("F"),
            {"milkshake": i},
        ),
        _make_event(
            "G",
            [
                _make_ref("G", i),
                _make_ref("E", i),
                _make_ref(g_third_ref, i),
            ],
            ts("G"),
        ),
    ]
