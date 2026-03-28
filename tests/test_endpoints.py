import asyncio
from datetime import datetime, timedelta, timezone

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

async def test_latency_operations(client):

    # 422 when no params provided
    resp = await client.get("/latencies")
    assert resp.status_code == 422

    # 404 for nonexistent chain
    resp = await client.get("/latencies", params={"chain_id": "nonexistent"})
    assert resp.status_code == 404

    # Ingest events and compute adjacency
    simulator = DataSimulator(num_intervals=1, seed=42)
    _, events = simulator.generate(prefix="lat_")
    resp = await client.post("/events", json=events)
    assert resp.status_code == 201

    resp = await client.put("/adjacency_matrix", json={})
    assert resp.status_code == 200
    assert len(resp.json()["edges"]) > 0

    # Get a chain_id from ClickHouse
    resp = await client.get("/chains")
    assert resp.status_code == 200
    chains = resp.json()["chains"]
    assert len(chains) > 0
    chain_id = chains[0]["chain_id"]

    # Lookup latencies by chain_id
    resp = await client.get("/latencies", params={"chain_id": chain_id})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 1
    assert body[0]["chain_id"] == chain_id
    assert len(body[0]["latencies"]) > 0
    for lat in body[0]["latencies"]:
        assert lat["delta_ms"] >= 0

    # Lookup latencies by ref (use first ref from the chain)
    ref = chains[0]["concatenatedrefs"][0]
    resp = await client.get("/latencies", params={"ref": ref})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) >= 1
    assert len(body[0]["latencies"]) > 0

    # 404 for nonexistent ref
    resp = await client.get("/latencies", params={"ref": "nonexistent_0_1"})
    assert resp.status_code == 404


async def test_average_latency_operations(client):

    # 422 when neither chain_id nor ref provided
    resp = await client.get("/latencies/averages", params={"start": "2020-01-01T00:00:00"})
    assert resp.status_code == 422

    # 422 when start is missing
    resp = await client.get("/latencies/averages", params={"chain_id": "x"})
    assert resp.status_code == 422

    # 404 for nonexistent chain
    resp = await client.get(
        "/latencies/averages",
        params={"chain_id": "nonexistent", "start": "2020-01-01T00:00:00"},
    )
    assert resp.status_code == 404

    # Ingest events, compute adjacency, train classifier
    simulator = DataSimulator(num_intervals=2, seed=42)
    _, events = simulator.generate(prefix="avg_")
    start_time = datetime.now(timezone.utc) - timedelta(minutes=1)

    resp = await client.post("/events", json=events)
    assert resp.status_code == 201

    resp = await client.put("/adjacency_matrix", json={})
    assert resp.status_code == 200
    assert len(resp.json()["edges"]) > 0

    resp = await client.put("/classifier", json={})
    assert resp.status_code == 200

    # Get a chain_id
    resp = await client.get("/chains")
    assert resp.status_code == 200
    chains = resp.json()["chains"]
    assert len(chains) > 0
    chain_id = chains[0]["chain_id"]

    # GET /latencies/averages by chain_id (open-ended window)
    resp = await client.get(
        "/latencies/averages",
        params={"chain_id": chain_id, "start": start_time.isoformat()},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["chain_id"] == chain_id
    assert body["matching_chains"] >= 1
    assert len(body["edges"]) > 0
    assert len(body["node_set"]) > 0
    for edge in body["edges"]:
        assert edge["min_ms"] <= edge["p5_ms"] <= edge["p50_ms"]
        assert edge["p50_ms"] <= edge["p95_ms"] <= edge["max_ms"]
        assert edge["stddev_ms"] >= 0
        assert edge["sample_count"] >= 1

    # GET /latencies/averages by ref
    ref = chains[0]["concatenatedrefs"][0]
    resp = await client.get(
        "/latencies/averages",
        params={"ref": ref, "start": start_time.isoformat()},
    )
    assert resp.status_code == 200
    assert len(resp.json()["edges"]) > 0

    # 404 for time window that excludes all data
    resp = await client.get(
        "/latencies/averages",
        params={"chain_id": chain_id, "start": "2099-01-01T00:00:00"},
    )
    assert resp.status_code == 404


async def test_state_detector_operations(client):

    # 404 when no model trained
    resp = await client.get("/state_detectors/latencies")
    assert resp.status_code == 404

    # Ingest events, compute adjacency, train classifier
    simulator = DataSimulator(num_intervals=2, seed=42)
    _, events = simulator.generate(prefix="hmm_")
    resp = await client.post("/events", json=events)
    assert resp.status_code == 201

    resp = await client.put("/adjacency_matrix", json={})
    assert resp.status_code == 200
    assert len(resp.json()["edges"]) > 0

    resp = await client.put("/classifier", json={})
    assert resp.status_code == 200

    # Train state detector
    start_time = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
    resp = await client.put(
        "/state_detectors/latencies",
        json={"start": start_time},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["method"] == "gaussian_hmm"
    assert len(body["profiles"]) >= 1
    for profile in body["profiles"]:
        assert profile["chain_count"] > 0
        assert len(profile["node_set"]) > 0
        for edge in profile["edges"]:
            assert len(edge["means"]) == 2
            assert len(edge["variances"]) == 2
            assert len(edge["transition_matrix"]) == 2
            assert all(len(row) == 2 for row in edge["transition_matrix"])
            assert edge["normal_state"] != edge["anomalous_state"]
            assert edge["sample_count"] > 0

    # GET returns persisted model
    resp = await client.get("/state_detectors/latencies")
    assert resp.status_code == 200
    get_body = resp.json()
    assert get_body["method"] == "gaussian_hmm"
    assert len(get_body["profiles"]) == len(body["profiles"])


async def test_search_operations(client):

    # 422 when no params provided to chain search
    resp = await client.get("/search/chains")
    assert resp.status_code == 422

    # Empty results for unknown prefix
    resp = await client.get("/search/refs", params={"q": "nonexistent"})
    assert resp.status_code == 200
    assert resp.json()["results"] == []

    # Ingest events
    resp = await client.post("/events/simulation?num_intervals=1")
    assert resp.status_code == 201
    assert resp.json()["event_count"] != 0

    # Autocomplete ref IDs by prefix — all generated refs start with "srch_"
    resp = await client.get("/search/refs", params={"q": "srch_"})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["results"]) > 0
    for ref_id in body["results"]:
        assert ref_id.startswith("srch_")

    # Search chains by exact ref ID
    ref_id = body["results"][0]
    resp = await client.get("/search/chains", params={"ref": ref_id})
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] > 0
    assert len(body["chain_ids"]) == body["count"]

    # Search chains by ref prefix
    resp = await client.get("/search/chains", params={"ref_prefix": "srch_"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] > 0

    # Validate min_length=2 constraint on prefix
    resp = await client.get("/search/refs", params={"q": "x"})
    assert resp.status_code == 422

    resp = await client.get("/search/chains", params={"ref_prefix": "x"})
    assert resp.status_code == 422


async def test_chain_detail(client):

    # 404 for nonexistent chain
    resp = await client.get("/chains/nonexistent")
    assert resp.status_code == 404

    # Ingest events
    simulator = DataSimulator(num_intervals=1, seed=42)
    _, events = simulator.generate(prefix="det_")
    resp = await client.post("/events", json=events)
    assert resp.status_code == 201

    # Get a chain_id from the chains list
    resp = await client.get("/chains")
    assert resp.status_code == 200
    chains = resp.json()["chains"]
    assert len(chains) > 0
    chain_id = chains[0]["chain_id"]

    # Fetch single chain detail
    resp = await client.get(f"/chains/{chain_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["chain_id"] == chain_id
    assert len(body["concatenatedrefs"]) > 0
    assert len(body["timestamps"]) > 0
    assert isinstance(body["context"], dict)
    assert "complete" in body
    assert "terminated" in body


async def test_simulator(client):

    resp = await client.post("/events/simulation?num_intervals=1")
    assert resp.status_code == 201
    assert resp.json()["event_count"] != 0
