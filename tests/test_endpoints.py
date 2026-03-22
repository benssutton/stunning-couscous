"""Tests for all REST endpoints — contract validation, chain assembly, and adjacency pipeline."""

import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier as SklearnDT

from services.chain_classifier import ChainProfilePredictor
from schemas.models import PathProfile
from tests.data_simulation import DataSimulator

pytestmark = pytest.mark.asyncio(loop_scope="session")


def _get_all_chains(sync_redis) -> list[dict]:
    """Fetch all event chain JSON docs from Redis using the sync client."""
    keys = sync_redis.keys("argus:ec:*")
    keys = [k for k in keys if not k.endswith(":ecstream") and ":idx" not in k]
    if not keys:
        return []
    docs = sync_redis.json().mget(keys, "$")
    return [doc[0] for doc in docs if doc]


# ---------------------------------------------------------------------------
# Contract / validation tests
# ---------------------------------------------------------------------------


async def test_post_single_event(client, sample_event_a):
    """POST a single well-formed event returns 201 with acknowledgement."""
    resp = await client.post("/events", json=sample_event_a)
    assert resp.status_code == 201
    body = resp.json()
    assert body["status"] == "received"
    assert body["event_name"] == "A"
    assert "chain_id" in body


async def test_post_multi_ref_event(client, sample_event_d):
    """POST an event with multiple refs returns 201."""
    resp = await client.post("/events", json=sample_event_d)
    assert resp.status_code == 201
    assert resp.json()["event_name"] == "D"


async def test_post_triple_ref_event_with_context(client, sample_event_f):
    """POST an event with three refs and context returns 201."""
    resp = await client.post("/events", json=sample_event_f)
    assert resp.status_code == 201
    assert resp.json()["event_name"] == "F"


async def test_post_event_missing_fields(client):
    """POST with missing required fields returns 422 validation error."""
    resp = await client.post("/events", json={"EventName": "X"})
    assert resp.status_code == 422


async def test_post_event_bad_ref_schema(client):
    """POST with malformed Refs returns 422."""
    resp = await client.post("/events", json={
        "EventName": "X",
        "Timestamp": "2025-01-01T00:00:00.000",
        "Refs": [{"wrong_key": "value"}],
    })
    assert resp.status_code == 422


async def test_post_event_empty_body(client):
    """POST with empty body returns 422."""
    resp = await client.post("/events", json={})
    assert resp.status_code == 422


async def test_clickhouse_insert_persists_event(client, clickhouse_service, sample_event_a):
    """POST an event persists exactly one row to ClickHouse."""
    await client.post("/events", json=sample_event_a)
    result = clickhouse_service.client.query("SELECT count() FROM argus.events")
    assert result.result_rows[0][0] == 1


# ---------------------------------------------------------------------------
# Redis chain assembly tests
# ---------------------------------------------------------------------------


async def test_post_creates_redis_chain(client, sync_redis, sample_event_a):
    """POST event A creates a new chain in Redis with correct structure."""
    resp = await client.post("/events", json=sample_event_a)
    assert resp.status_code == 201

    chains = _get_all_chains(sync_redis)
    assert len(chains) == 1

    chain = chains[0]
    assert chain["timestamps"]["A"] == sample_event_a["Timestamp"]
    assert "A_A0_1" in chain["concatenatedrefs"]
    assert chain["context"] == {"tea": 0}
    assert chain["complete"] is False
    assert chain["terminated"] is False


async def test_post_merges_into_existing_chain(client, sync_redis):
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

    chains = _get_all_chains(sync_redis)
    assert len(chains) == 1

    chain = chains[0]
    assert "A" in chain["timestamps"]
    assert "B" in chain["timestamps"]
    assert "A_A1_1" in chain["concatenatedrefs"]
    assert "B_B1_1" in chain["concatenatedrefs"]


async def test_event_chain_batch_assembles(client, sync_redis, event_chain_batch):
    """POST all 8 events for one transaction — they assemble into chains."""
    for event in event_chain_batch:
        resp = await client.post("/events", json=event)
        assert resp.status_code == 201

    chains = _get_all_chains(sync_redis)

    all_timestamps = {}
    for chain in chains:
        all_timestamps.update(chain["timestamps"])

    expected_names = {"A", "B", "C", "D", "E", "F", "G", "H"}
    assert set(all_timestamps.keys()) == expected_names


async def test_terminated_chain_deleted_from_redis(client, sync_redis, redis_service):
    """When all terminal nodes arrive, the chain is deleted from Redis."""
    profile = PathProfile(
        profile_id=0,
        node_set=frozenset({"A", "B", "D"}),
        terminal_nodes=frozenset({"D"}),
        chain_count=1,
        fraction=1.0,
    )
    feature_names = ["event:A", "event:B", "event:D"]
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

    chains = _get_all_chains(sync_redis)
    assert len(chains) == 1

    await client.post("/events", json=event_d)

    chains = _get_all_chains(sync_redis)
    assert len(chains) == 0

    redis_service.set_predictor(None)
    redis_service.set_path_profiles([])


# ---------------------------------------------------------------------------
# Adjacency matrix pipeline
# ---------------------------------------------------------------------------


async def test_adjacency_pipeline(client):
    """End-to-end: ingest simulated events then compute adjacency matrix.

    1. POST simulated events — assert 201 for each.
    2. GET /adjacency_matrix — assert empty (no matrix computed yet).
    3. PUT /adjacency_matrix — assert 200 (compute from event data).
    4. GET /adjacency_matrix — assert 200 and edges are returned.
    """
    simulator = DataSimulator(num_intervals=1, seed=42)
    events = simulator.generate(prefix="adj_")

    for event in events:
        resp = await client.post("/events", json=event)
        assert resp.status_code == 201

    resp = await client.get("/adjacency_matrix")
    assert resp.status_code == 200
    assert resp.json()["edge_count"] == 0

    resp = await client.put("/adjacency_matrix", json={})
    assert resp.status_code == 200

    resp = await client.get("/adjacency_matrix")
    assert resp.status_code == 200
    data = resp.json()
    assert data["edge_count"] > 0


# ---------------------------------------------------------------------------
# Cache operations
# ---------------------------------------------------------------------------


async def test_cache_operations(client):
    """End-to-end: verify Redis cache key lifecycle.

    1.  GET /cache/event_chain_keys  — assert 200 and empty.
    2.  POST simulated events        — assert 201 for each.
    3.  GET /cache/event_chain_keys  — assert 200 and keys present.
    4.  DELETE /cache/event_chain_keys — assert 200.
    5.  GET /cache/event_chain_keys  — assert 200 and empty again.
    6.  PUT /cache                   — reload chains from ClickHouse.
    7.  GET /cache/event_chain_keys  — assert 200 and keys restored.
    """
    resp = await client.get("/cache/event_chain_keys")
    assert resp.status_code == 200
    assert resp.json()["count"] == 0

    simulator = DataSimulator(num_intervals=1, seed=7)
    events = simulator.generate(prefix="cache_")
    for event in events:
        resp = await client.post("/events", json=event)
        assert resp.status_code == 201

    resp = await client.get("/cache/event_chain_keys")
    assert resp.status_code == 200
    populated_count = resp.json()["count"]
    assert populated_count > 0

    resp = await client.delete("/cache/event_chain_keys")
    assert resp.status_code == 200

    resp = await client.get("/cache/event_chain_keys")
    assert resp.status_code == 200
    assert resp.json()["count"] == 0

    resp = await client.put("/cache")
    assert resp.status_code == 200

    resp = await client.get("/cache/event_chain_keys")
    assert resp.status_code == 200
    assert resp.json()["count"] > 0


# ---------------------------------------------------------------------------
# Full termination pipeline
# ---------------------------------------------------------------------------


async def test_full_termination_pipeline(client, redis_service):
    """End-to-end: classifier-driven chain termination works for all profiles.

    1. POST events with no predictor loaded — chains stay in cache.
    2. Train adjacency + classifier from the ingested data.
    3. Re-POST the same events grouped by chain — all chains should terminate.

    Step 3 posts events grouped by transaction so that each chain receives
    all its events (including J where present) before the next chain starts.
    This avoids a cross-chain interleaving race where a chain momentarily
    matches the smaller profile before its final events arrive.
    """
    prefix = "term_"

    # 1. Generate a small batch of events
    simulator = DataSimulator(num_intervals=1, seed=99)
    events = simulator.generate(prefix=prefix)

    # 2. POST all events — no predictor, so no termination
    for event in events:
        resp = await client.post("/events", json=event)
        assert resp.status_code == 201

    # 3. Assert chains remain in cache
    resp = await client.get("/cache/event_chain_keys")
    assert resp.status_code == 200
    chains_before = resp.json()["count"]
    assert chains_before > 0, "Events should have created chains in the cache"

    # 4. Clear the Redis cache
    resp = await client.delete("/cache/event_chain_keys")
    assert resp.status_code == 200

    # 5. Train: compute adjacency matrix then run classifier
    resp = await client.put("/adjacency_matrix", json={})
    assert resp.status_code == 200

    resp = await client.put("/classifier", json={})
    assert resp.status_code == 200
    profiles = resp.json()["profiles"]
    assert len(profiles) >= 2, "Expect at least 2 profiles (with/without J)"

    # 6. Re-POST events grouped by chain so each chain completes atomically.
    #    Extract the transaction index from the first ref id ({prefix}{letter}{index}).
    def _chain_key(event):
        ref_id = event["Refs"][0]["id"]
        return int(ref_id[len(prefix) + 1:])

    grouped_events = sorted(events, key=lambda e: (_chain_key(e), e["Timestamp"]))

    for event in grouped_events:
        resp = await client.post("/events", json=event)
        assert resp.status_code == 201

    # 7. Assert all chains have been terminated (cache empty)
    resp = await client.get("/cache/event_chain_keys")
    assert resp.status_code == 200
    assert resp.json()["count"] == 0, (
        f"All chains should have terminated, but {resp.json()['count']} remain"
    )

    # 8. Cleanup — reset predictor to avoid polluting other tests
    redis_service.set_predictor(None)
    redis_service.set_path_profiles([])
