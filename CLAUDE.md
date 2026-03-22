# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Telemetry event chain inference system. Captures events emitted by transactions passing through non-linear sequences of noisy processes, infers the dependency graph from noisy event data, and detects statistically significant latency anomalies.

Key challenge: the collector knows almost nothing in advance — not the processing tree, not the latencies, not the arrival rates. Everything is inferred from observed events correlated by shared transaction references.

The problem domain, simulation, and prototype algorithms are detailed in `notebooks/Event Chains Revisited.ipynb`.

## Development Environment

- **Python**: `C:\Users\Ben\.conda\envs\p312` (conda, Python 3.12)
- **Package manager**: conda (configured in `.vscode/settings.json`)
- **Activate env**: `conda activate p312`
- **Install deps**: `pip install -r requirements.txt` (requirements.txt in root folder)
- **Run API**: `python main.py` or `uvicorn main:app --reload` (FastAPI app in `main.py`, port 8000)
- **Run notebook**: `jupyter lab notebooks/`
- **Redis Stack 7.4+** required locally on `localhost:6379` (Redis Insight at `http://localhost:8001`)
- **ClickHouse** required locally on `localhost:8123` (HTTP) / `localhost:9000` (native), database `argus`
- **Reset Redis** (clears all data): `redis-cli FLUSHDB`
- **Reset all data** (Redis + ClickHouse): `python scripts/cleanup_events.py`

## Coding Standards
- Use zero memory copy when moving large datasets.  Make use of the Arrow format and Polars for processing.
- Use Pydantic for datamodeling.  Prefer Pydantic over dataclasses.

## Architecture

```
REST endpoint (FastAPI, main.py)
  → Redis (event cache + event chain assembly via JSON + Search)
  → ClickHouse (persisted events, adjacency matrix, path profiles, classifier model)
```

### Processing Pipeline
1. Receive events as JSON via `POST /events`
2. Correlate events into event chains atomically via a Lua script that performs FT.SEARCH + chain create/merge in a single Redis round-trip
3. Persist every event to ClickHouse via an async in-process batch writer (`ClickHouseBatchWriter`) — the hot path never waits for ClickHouse
4. Compute adjacency matrix on demand via `PUT /adjacency_matrix` using Pearson correlation of timestamp series from ClickHouse
5. Classify event chain path profiles via `PUT /classifier` — discovers which node-sets occur, identifies terminal nodes, and finds discriminating features using pluggable classification methods
6. Trained classifier model (sklearn DecisionTree) is serialized via joblib, persisted to ClickHouse, and loaded on startup for runtime profile prediction
7. Determine chain termination using inferred path profiles: when all terminal nodes for the predicted profile are received, the chain is deleted from Redis. Chain keys have a 10-minute TTL as a safety net against stale chains

### API Endpoints
- `POST /events` — ingest an event, merge into Redis event chain, persist to ClickHouse
- `GET /adjacency_matrix` — return latest adjacency edges from ClickHouse
- `POST /adjacency_matrix` — overwrite adjacency edges in ClickHouse with caller-provided data
- `PUT /adjacency_matrix` — compute dependency graph from ClickHouse event data
- `GET /classifier` — return persisted path profiles from ClickHouse
- `POST /classifier` — overwrite path profiles in ClickHouse with caller-provided data
- `PUT /classifier` — run classifier pipeline: discover profiles, terminal nodes, train model (supports `method` param: `"ratio"`, `"decision_tree"`, or `null` for all)
- `GET /chains?unterminated=true` — return all unterminated chain documents from Redis
- `GET /cache/event_chain_keys` — list all event chain keys in Redis
- `DELETE /cache/event_chain_keys` — delete all event chain keys from Redis
- `PUT /cache` — load non-terminated chains from ClickHouse into Redis

### Key Data Structures
- **Event**: `{EventName, Timestamp, Refs: [{type, id, ver}], Context: {}}`
- **Event Chain** (Redis JSON): `{concatenatedrefs: [str], timestamps: {process: timestamp}, context: {}, complete: bool, terminated: bool}`
- Redis index `argus:ec:idx` on `$.concatenatedrefs[*]`, `$.complete`, `$.terminated` (TagFields)
- Redis Stream `argus:ecstream` tracks created/updated chains
- **PathProfile**: `{profile_id, node_set, terminal_nodes, chain_count, fraction}` — discovered from observed data

### Graph Inference
Adjacency matrix derived from Pearson correlation of timestamp series between event sets, filtering by temporal ordering (events must always follow their dependencies) and p-value threshold.

## Key Dependencies

Core: numpy, pandas, polars, polars-ds, scipy, redis, ulid, pydantic, pydantic-settings, fastapi, uvicorn, clickhouse-connect, httpx, scikit-learn, joblib

Visualization (notebook only): plotly, ipycytoscape

Test: pytest, pytest-asyncio, pytest-cov, testcontainers

## Scripts

- `scripts/cleanup_events.py` — truncate ClickHouse tables and flush Redis keys. Supports `--clickhouse` or `--redis` flags to clean selectively. **Drops the Redis index** `argus:ec:idx`; FastAPI must be restarted after cleanup to recreate it.
- `scripts/generate_events.py` — simulate and POST telemetry events to the API. Usage: `python scripts/generate_events.py --intervals 60 --base-url http://localhost:8000 --seed 42`

## Performance Target

3,000 events/second (200M events/day). Performance is the top priority at every layer.

## Current Status

Core functionality to ingest events, train classifiers, and employ classifiers has been completed. `main.py` is a FastAPI app with routers in `routers/` (events, adjacency, classifier, cache, chains). Data models are in `schemas/models.py`. Services in `services/` include: `redis_service.py` (async, uses redis.asyncio with Lua scripting), `clickhouse_service.py` (includes `ClickHouseBatchWriter`), `adjacency_service.py`, `inference.py`, `chain_classifier.py`, `dependencies.py`. Adjacency matrix computation and chain path classification work end-to-end. The classifier supports pluggable methods (`RatioClassifier` and `TreeClassifier`) and persists a fitted sklearn model via joblib to ClickHouse for runtime use. At startup, the model and path profiles are loaded from ClickHouse and used to predict the single matching profile for each chain, deleting terminated chains from Redis. All REST endpoints support full CRUD: GET reads from ClickHouse, POST overwrites with caller data, PUT computes/trains from source data.

## Testing

- **Run tests**: `python -m pytest` (or `python -m pytest -v` for verbose)
- **Run single test**: `python -m pytest tests/test_endpoints.py::test_post_single_event`
- **Always write async tests** — use `async def test_*` with `httpx.AsyncClient`. `asyncio_mode = auto` is set in `pytest.ini` so no `@pytest.mark.asyncio` decorator needed.
- Tests use `httpx.AsyncClient` with `httpx.ASGITransport(app=app)` — not `fastapi.testclient.TestClient`
- Tests use **testcontainers** (Docker) for Redis Stack and ClickHouse — Docker must be running
- Session-scoped event loop: test files that use session-scoped fixtures (client, redis, clickhouse) must declare `pytestmark = pytest.mark.asyncio(loop_scope="session")`. `asyncio_default_fixture_loop_scope = session` is set in `pytest.ini`.
- Session-scoped fixtures in `conftest.py` are **sync** (not async) to avoid event-loop-scope conflicts. Use the `sync_redis` fixture for test assertions against Redis data.
- Coverage is configured in `pytest.ini` (`--cov=services --cov=main`)

## Gotchas

- No `.gitignore` yet — avoid committing `.vscode/`, `__pycache__/`, `*.pyc`, `.env`, `notebooks/.ipynb_checkpoints/`
- The notebook prototype uses `redis_om` in one cell but it's not installed and that cell fails — the working implementation uses `redis.asyncio` commands instead
- Redis index `argus:ec:idx` is created by FastAPI on startup via `ensure_index()` — but `scripts/cleanup_events.py` drops it, so FastAPI must be restarted after cleanup
- ClickHouse events table has a 90-day TTL — the generator uses current timestamps to avoid expiry
- ClickHouse UUID columns cannot be exported via Arrow format (`query_arrow`) — use `toString(col)` in the SELECT or avoid `query_arrow` for tables with UUID columns
- Redis chain keys have a 10-minute TTL (`CHAIN_TTL_SECONDS = 600`) — refreshed on each merge, but truly stale chains will auto-evict
