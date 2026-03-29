"""Tests for Req 1 & 2: Produce and Compression request header support on all GET endpoints."""
import io

import polars as pl
import pytest

from tests.data_simulation import DataSimulator

pytestmark = pytest.mark.asyncio(loop_scope="session")


# ---------------------------------------------------------------------------
# Session-level data setup
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
async def _setup_data(client):
    """Simulate events, compute adjacency, train classifier so GET endpoints have data."""
    simulator = DataSimulator(num_intervals=3, seed=99)
    _, events = simulator.generate(prefix="produce_")
    resp = await client.post("/events", json=events)
    assert resp.status_code == 201

    resp = await client.put("/adjacency_matrix", json={})
    assert resp.status_code == 200

    resp = await client.put("/classifier", json={})
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_ipc_stream(content: bytes) -> pl.DataFrame:
    return pl.read_ipc_stream(io.BytesIO(content))


def _load_ipc_file(content: bytes) -> pl.DataFrame:
    return pl.read_ipc(io.BytesIO(content))


def _load_json(content: bytes) -> pl.DataFrame:
    return pl.read_json(io.BytesIO(content))


PRODUCE_CASES = [
    ("application/json", _load_json),
    ("application/vnd.apache.arrow.stream", _load_ipc_stream),
    ("application/vnd.apache.arrow.file", _load_ipc_file),
]

IPC_COMPRESSIONS = ["uncompressed", "LZ4", "ZSTD"]


async def _assert_produce(client, method: str, url: str, produce: str, loader, **kwargs):
    """Call an endpoint with the given Produce header and assert 200 + Polars load."""
    headers = {"Produce": produce, **kwargs.pop("extra_headers", {})}
    if method == "get":
        resp = await client.get(url, headers=headers, **kwargs)
    else:
        resp = await client.post(url, headers=headers, **kwargs)
    assert resp.status_code == 200, f"{url} with Produce={produce}: {resp.text}"
    df = loader(resp.content)
    assert isinstance(df, pl.DataFrame)


async def _assert_compression(client, url: str, compression: str, **kwargs):
    """IPC Stream with each compression codec — assert 200 + Polars load."""
    headers = {
        "Produce": "application/vnd.apache.arrow.stream",
        "Compression": compression,
    }
    resp = await client.get(url, headers=headers, **kwargs)
    assert resp.status_code == 200, f"{url} Compression={compression}: {resp.text}"
    df = _load_ipc_stream(resp.content)
    assert isinstance(df, pl.DataFrame)


# ---------------------------------------------------------------------------
# GET /adjacency_matrix
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("produce,loader", PRODUCE_CASES)
async def test_adjacency_matrix_produce(client, produce, loader):
    await _assert_produce(client, "get", "/adjacency_matrix", produce, loader)


@pytest.mark.parametrize("compression", IPC_COMPRESSIONS)
async def test_adjacency_matrix_compression(client, compression):
    await _assert_compression(client, "/adjacency_matrix", compression)


# ---------------------------------------------------------------------------
# GET /cache/event_chain_keys
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("produce,loader", PRODUCE_CASES)
async def test_cache_keys_produce(client, produce, loader):
    await _assert_produce(client, "get", "/cache/event_chain_keys", produce, loader)


@pytest.mark.parametrize("compression", IPC_COMPRESSIONS)
async def test_cache_keys_compression(client, compression):
    await _assert_compression(client, "/cache/event_chain_keys", compression)


# ---------------------------------------------------------------------------
# GET /chains
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("produce,loader", PRODUCE_CASES)
async def test_chains_produce(client, produce, loader):
    await _assert_produce(client, "get", "/chains", produce, loader)


@pytest.mark.parametrize("compression", IPC_COMPRESSIONS)
async def test_chains_compression(client, compression):
    await _assert_compression(client, "/chains", compression)


# ---------------------------------------------------------------------------
# GET /classifier
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("produce,loader", PRODUCE_CASES)
async def test_classifier_produce(client, produce, loader):
    await _assert_produce(client, "get", "/classifier", produce, loader)


@pytest.mark.parametrize("compression", IPC_COMPRESSIONS)
async def test_classifier_compression(client, compression):
    await _assert_compression(client, "/classifier", compression)


# ---------------------------------------------------------------------------
# GET /events/names
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("produce,loader", PRODUCE_CASES)
async def test_event_names_produce(client, produce, loader):
    await _assert_produce(client, "get", "/events/names", produce, loader)


@pytest.mark.parametrize("compression", IPC_COMPRESSIONS)
async def test_event_names_compression(client, compression):
    await _assert_compression(client, "/events/names", compression)


# ---------------------------------------------------------------------------
# GET /search/refs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("produce,loader", PRODUCE_CASES)
async def test_search_refs_produce(client, produce, loader):
    await _assert_produce(client, "get", "/search/refs", produce, loader, params={"q": "pro"})


@pytest.mark.parametrize("compression", IPC_COMPRESSIONS)
async def test_search_refs_compression(client, compression):
    await _assert_compression(client, "/search/refs", compression, params={"q": "pro"})


# ---------------------------------------------------------------------------
# GET /search/chains
# ---------------------------------------------------------------------------

async def test_search_chains_produce(client):
    # Get a real ref prefix to search for
    resp = await client.get("/search/refs", params={"q": "pro"})
    assert resp.status_code == 200
    results = resp.json().get("results", [])
    if not results:
        pytest.skip("No refs available for search/chains test")
    ref = results[0]
    for produce, loader in PRODUCE_CASES:
        await _assert_produce(client, "get", "/search/chains", produce, loader, params={"ref": ref})


# ---------------------------------------------------------------------------
# 415 on unsupported mime type
# ---------------------------------------------------------------------------

GET_ENDPOINTS = [
    "/adjacency_matrix",
    "/cache/event_chain_keys",
    "/chains",
    "/classifier",
    "/events/names",
]


@pytest.mark.parametrize("url", GET_ENDPOINTS)
async def test_unsupported_produce_returns_415(client, url):
    resp = await client.get(url, headers={"Produce": "text/csv"})
    assert resp.status_code == 415, f"{url} should return 415, got {resp.status_code}"


# ---------------------------------------------------------------------------
# 400 on unsupported compression
# ---------------------------------------------------------------------------

async def test_unsupported_compression_returns_400(client):
    resp = await client.get(
        "/events/names",
        headers={
            "Produce": "application/vnd.apache.arrow.stream",
            "Compression": "brotli",
        },
    )
    assert resp.status_code == 400
