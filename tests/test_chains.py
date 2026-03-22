"""Tests for POST /events endpoint — contract validation and chain assembly."""

import numpy as np
import redis as redis_lib
from sklearn.tree import DecisionTreeClassifier as SklearnDT

from services.chain_classifier import ChainProfilePredictor
from services.models import PathProfile
from tests.conftest import requires_redis


# ---------------------------------------------------------------------------
# Contract / validation tests (no Redis required)
# ---------------------------------------------------------------------------


async def test_post_single_event(client_no_redis, sample_event_a):
    """POST a single well-formed event returns 200 with acknowledgement."""
    resp = await client_no_redis.post("/events", json=sample_event_a)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "received"
    assert body["event_name"] == "A"
    assert "chain_id" in body


async def test_post_multi_ref_event(client_no_redis, sample_event_d):
    """POST an event with multiple refs returns 200."""
    resp = await client_no_redis.post("/events", json=sample_event_d)
    assert resp.status_code == 200
    assert resp.json()["event_name"] == "D"


async def test_post_triple_ref_event_with_context(client_no_redis, sample_event_f):
    """POST an event with three refs and context returns 200."""
    resp = await client_no_redis.post("/events", json=sample_event_f)
    assert resp.status_code == 200
    assert resp.json()["event_name"] == "F"


async def test_post_event_missing_fields(client_no_redis):
    """POST with missing required fields returns 422 validation error."""
    resp = await client_no_redis.post("/events", json={"EventName": "X"})
    assert resp.status_code == 422


async def test_post_event_bad_ref_schema(client_no_redis):
    """POST with malformed Refs returns 422."""
    resp = await client_no_redis.post("/events", json={
        "EventName": "X",
        "Timestamp": "2025-01-01T00:00:00.000",
        "Refs": [{"wrong_key": "value"}],
    })
    assert resp.status_code == 422


async def test_post_event_empty_body(client_no_redis):
    """POST with empty body returns 422."""
    resp = await client_no_redis.post("/events", json={})
    assert resp.status_code == 422


async def test_clickhouse_insert_called(client_no_redis, mock_clickhouse, sample_event_a):
    """Verify ClickHouse insert_event is called for each POST."""
    mock_clickhouse.reset_mock()
    await client_no_redis.post("/events", json=sample_event_a)
    mock_clickhouse.insert_event.assert_called_once()


# ---------------------------------------------------------------------------
# Redis chain assembly tests (require Redis)
# ---------------------------------------------------------------------------


def _get_all_chains(redis_pool) -> list[dict]:
    """Helper: fetch all event chain JSON docs from Redis DB 1."""
    r = redis_lib.Redis(connection_pool=redis_pool)
    keys = r.keys("argus:ec:*")
    # Filter out stream and index keys
    keys = [k for k in keys if not k.endswith(":ecstream") and ":idx" not in k]
    if not keys:
        return []
    docs = r.json().mget(keys, "$")
    return [doc[0] for doc in docs if doc]


@requires_redis
async def test_post_creates_redis_chain(client, redis_pool, sample_event_a):
    """POST event A creates a new chain in Redis with correct structure."""
    resp = await client.post("/events", json=sample_event_a)
    assert resp.status_code == 200

    chains = _get_all_chains(redis_pool)
    assert len(chains) == 1

    chain = chains[0]
    assert chain["timestamps"]["A"] == sample_event_a["Timestamp"]
    assert "A_A0_1" in chain["concatenatedrefs"]
    assert chain["context"] == {"tea": 0}
    assert chain["complete"] is False
    assert chain["terminated"] is False


@requires_redis
async def test_post_merges_into_existing_chain(client, redis_pool):
    """POST event A then event B (sharing ref A) merges into one chain."""
    event_a = {
        "EventName": "A",
        "Timestamp": "2025-03-19T12:00:00.000",
        "Refs": [{"type": "A", "id": "A1", "ver": 1}],
        "Context": {"tea": 1},
    }
    event_b = {
        "EventName": "B",
        "Timestamp": "2025-03-19T12:00:01.000",
        "Refs": [{"type": "A", "id": "A1", "ver": 1}, {"type": "B", "id": "B1", "ver": 1}],
        "Context": {},
    }

    await client.post("/events", json=event_a)
    await client.post("/events", json=event_b)

    chains = _get_all_chains(redis_pool)
    assert len(chains) == 1

    chain = chains[0]
    assert "A" in chain["timestamps"]
    assert "B" in chain["timestamps"]
    assert "A_A1_1" in chain["concatenatedrefs"]
    assert "B_B1_1" in chain["concatenatedrefs"]


@requires_redis
async def test_event_chain_batch_assembles(client, redis_pool, event_chain_batch):
    """POST all 8 events for one transaction — they assemble into chains."""
    for event in event_chain_batch:
        resp = await client.post("/events", json=event)
        assert resp.status_code == 200

    chains = _get_all_chains(redis_pool)

    # All 8 event names should appear across the chain(s)
    all_timestamps = {}
    for chain in chains:
        all_timestamps.update(chain["timestamps"])

    expected_names = {"A", "B", "C", "D", "E", "F", "G", "H"}
    assert set(all_timestamps.keys()) == expected_names


@requires_redis
async def test_terminated_chain_deleted_from_redis(client, redis_pool, redis_service):
    """When all terminal nodes arrive, the chain is deleted from Redis."""
    # Build a minimal fitted predictor that maps to a single profile
    profile = PathProfile(
        profile_id=0,
        node_set=frozenset({"A", "B", "D"}),
        terminal_nodes=frozenset({"D"}),
        chain_count=1,
        fraction=1.0,
    )
    feature_names = ["event:A", "event:B", "event:D"]
    # Train a trivial tree on synthetic data matching this profile
    X = np.array([[1, 1, 1], [1, 1, 0], [1, 0, 0]], dtype=np.int8)
    y = np.array([0, 0, 0])
    tree = SklearnDT(max_depth=1)
    tree.fit(X, y)
    predictor = ChainProfilePredictor(
        model=tree,
        feature_names=feature_names,
        profiles={0: profile},
    )
    redis_service.set_predictor(predictor)

    event_a = {
        "EventName": "A",
        "Timestamp": "2025-03-19T12:00:00.000",
        "Refs": [{"type": "A", "id": "A99", "ver": 1}],
    }
    event_b = {
        "EventName": "B",
        "Timestamp": "2025-03-19T12:00:01.000",
        "Refs": [{"type": "A", "id": "A99", "ver": 1}, {"type": "B", "id": "B99", "ver": 1}],
    }
    event_d = {
        "EventName": "D",
        "Timestamp": "2025-03-19T12:00:02.000",
        "Refs": [{"type": "D", "id": "D99", "ver": 1}, {"type": "B", "id": "B99", "ver": 1}],
    }

    await client.post("/events", json=event_a)
    await client.post("/events", json=event_b)

    # Chain should still exist before terminal node arrives
    chains = _get_all_chains(redis_pool)
    assert len(chains) == 1

    await client.post("/events", json=event_d)

    # After terminal node D, chain should be deleted
    chains = _get_all_chains(redis_pool)
    assert len(chains) == 0

    # Clean up predictor for other tests
    redis_service.set_predictor(None)
    redis_service.set_path_profiles([])
