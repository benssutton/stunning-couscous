import pytest

from tests.data_simulation import DataSimulator

pytestmark = pytest.mark.asyncio(loop_scope="session")

async def test_full_pipeline(client):

    # 1. Generate simulated events
    simulator = DataSimulator(num_intervals=10, seed=42)
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

    # 7. Assert all chains have been terminated (cache empty)
    resp = await client.get("/cache/event_chain_keys")
    assert resp.status_code == 200
    assert resp.json()["count"] == 0, (
        f"All chains should have terminated, but {resp.json()['count']} remain"
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

    resp = await client.get("/cache/event_chain_keys")
    assert resp.status_code == 200
    assert resp.json()["count"] > 0

