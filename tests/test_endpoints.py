import asyncio

import pytest

from services.redis_service import RedisService
from tests.data_simulation import DataSimulator

pytestmark = pytest.mark.asyncio(loop_scope="session")

async def test_full_pipeline(client, sync_redis):

    # 1. Generate simulated events
    simulator = DataSimulator(num_intervals=2, seed=42)
    num_chains, events = simulator.generate(prefix="full_")

    # 2. POST all events — no predictor, so no termination
    resp = await client.post("/events", json=events)
    assert resp.status_code == 201

    # 3. If not trained, then no events/chains will have been evicted from the cache immediately
    resp = await client.get("/cache/event_chain_keys")
    assert resp.status_code == 200
    resp_body = resp.json()
    assert resp_body["count"] == num_chains, (
        f"Expected {num_chains} chains from ClickHouse, got {resp_body['count']}"
    )

    # 4. All chains should be present on the chains endpoint
    resp = await client.get("/chains")
    assert resp.status_code == 200
    resp_body = resp.json()
    assert resp_body["count"] == num_chains, (
        f"Expected {num_chains} chains from ClickHouse, got {resp_body['count']}"
    )

    # 5. Clear the Redis cache
    resp = await client.delete("/cache/event_chain_keys")
    assert resp.status_code == 200

    # 5. Train: compute adjacency matrix then run classifier
    resp = await client.get("/adjacency_matrix")
    assert resp.status_code == 200
    assert len(resp.json()["edges"]) == 0

    resp = await client.put("/adjacency_matrix", json={})
    assert resp.status_code == 200

    resp = await client.get("/adjacency_matrix")
    assert resp.status_code == 200
    assert len(resp.json()["edges"]) == 8

    resp = await client.put("/classifier", json={})
    assert resp.status_code == 200
    profiles = resp.json()["profiles"]
    assert len(profiles) >= 2, "Expect at least 2 profiles (with/without J)"

    # 6. Re-POST events
    resp = await client.post("/events", json=events)
    assert resp.status_code == 201

    # 7. Chains are in draining state: any still present must have TTL <= TERMINATED_TTL_SECONDS
    resp = await client.get("/cache/event_chain_keys")
    assert resp.status_code == 200
    remaining_keys = resp.json()["keys"]
    for key in remaining_keys:
        ttl = sync_redis.ttl(key)
        assert ttl <= RedisService.TERMINATED_TTL_SECONDS, (
            f"Chain {key} should be draining (TTL <= {RedisService.TERMINATED_TTL_SECONDS}s) but TTL={ttl}"
        )

    # Wait for draining TTL to expire, then assert cache is empty
    await asyncio.sleep(RedisService.TERMINATED_TTL_SECONDS + 1)
    resp = await client.get("/cache/event_chain_keys")
    assert resp.status_code == 200
    assert resp.json()["count"] == 0, (
        f"All chains should have expired, but {resp.json()['count']} remain: {resp.json()['keys']}"
    )

    # 8. All chains should be present on the chains endpoint x 2
    resp = await client.get("/chains")
    assert resp.status_code == 200
    resp_body = resp.json()
    assert resp_body["count"] == num_chains * 2, (
        f"Expected {num_chains * 2} chains from ClickHouse, got {resp_body['count']}"
    )

async def test_cache_operations(client):

    resp = await client.get("/cache/event_chain_keys")
    assert resp.status_code == 200
    assert resp.json()["count"] == 0

    simulator = DataSimulator(num_intervals=1, seed=42)
    _, events = simulator.generate(prefix="cache_")
    resp = await client.post("/events", json=events)
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
    assert resp.json()["total_chains"] != 0

    resp = await client.get("/chains")
    assert resp.status_code == 200
    assert resp.json()["count"] != 0

async def test_adjacency_operations(client):

    resp = await client.get("/adjacency_matrix")
    assert resp.status_code == 200
    assert resp.json()["run_id"] == ""

    simulator = DataSimulator(num_intervals=1, seed=42)
    _, events = simulator.generate(prefix="cache_")
    resp = await client.post("/events", json=events)
    assert resp.status_code == 201

    req = {
        "method": "pearson",
        "max_pval": 0.5
        }
    resp = await client.put("/adjacency_matrix", json=req)
    assert resp.status_code == 200
    assert resp.json()["run_id"] != ""

    resp = await client.get("/adjacency_matrix")
    assert resp.status_code == 200
    assert resp.json()["run_id"] != ""

    resp = await client.delete("/adjacency_matrix")
    assert resp.status_code == 200
    assert resp.json()["run_id"] != ""

    resp = await client.get("/adjacency_matrix")
    assert resp.status_code == 200
    assert resp.json()["run_id"] == ""

async def test_simulator(client):

    resp = await client.post("/events/simulation?num_intervals=1")
    assert resp.status_code == 201
    assert resp.json()["event_count"] != 0
