# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Telemetry event chain inference system. Captures events emitted by transactions passing through non-linear sequences of noisy processes, infers the dependency graph from noisy event data, and detects statistically significant latency anomalies.

Key challenge: the collector knows almost nothing in advance — not the processing tree, not the latencies, not the arrival rates. Everything is inferred from observed events correlated by shared transaction references.

The problem domain, simulation, and prototype algorithms are detailed in `notebooks/Event Chains Revisited.ipynb`.

## Development Environment

- **Python**: `C:\Users\benss\.conda\envs\p312` (conda, Python 3.12)
- **Package manager**: conda (configured in `.vscode/settings.json`)
- **Activate env**: `conda activate p312`
- **Install deps**: `pip install fastapi uvicorn redis numpy pandas polars scipy plotly ipycytoscape ulid-py pydantic pydantic-settings clickhouse-connect httpx scikit-learn joblib pytest pytest-asyncio pytest-cov`
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
2. Correlate events into event chains by matching shared transaction references (O(1) via Redis Search index on concatenated refs)
3. Persist every event to ClickHouse (`argus.events` table)
4. Compute adjacency matrix on demand via `PUT /adjacency_matrix` using Pearson correlation of timestamp series from ClickHouse
5. Classify event chain path profiles via `PUT /classifier` — discovers which node-sets occur, identifies terminal nodes, and finds discriminating features using pluggable classification methods
6. Trained classifier model (sklearn DecisionTree) is serialized via joblib, persisted to ClickHouse, and loaded on startup for runtime profile prediction
7. Determine chain termination using inferred path profiles: when all terminal nodes for the predicted profile are received, the chain is deleted from Redis

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

Core: numpy, pandas, polars, scipy, redis, ulid, pydantic, pydantic-settings, fastapi, uvicorn, clickhouse-connect, httpx, scikit-learn, joblib

Visualization (notebook only): plotly, ipycytoscape

Test: pytest, pytest-asyncio, pytest-cov

## Scripts

- `scripts/cleanup_events.py` — truncate ClickHouse tables and flush Redis keys. Supports `--clickhouse` or `--redis` flags to clean selectively. **Drops the Redis index** `argus:ec:idx`; FastAPI must be restarted after cleanup to recreate it.
- `scripts/generate_events.py` — simulate and POST telemetry events to the API. Usage: `python scripts/generate_events.py --intervals 60 --base-url http://localhost:8000 --seed 42`

## Performance Target

3,000 events/second (200M events/day). Performance is the top priority at every layer.

## Current Status

Core functionality to ingest events, train classifiers, and employ classifiers has been completed. `main.py` is a FastAPI app that receives events, assembles event chains in Redis, and persists events to ClickHouse. The `services/` directory contains: `redis_service.py`, `clickhouse_service.py`, `adjacency_service.py`, `inference.py`, `chain_classifier.py`, `models.py`, `dependencies.py`. Adjacency matrix computation and chain path classification work end-to-end. The classifier supports pluggable methods (`RatioClassifier` and `TreeClassifier`) and persists a fitted sklearn model via joblib to ClickHouse for runtime use. At startup, the model and path profiles are loaded from ClickHouse and used to predict the single matching profile for each chain, deleting terminated chains from Redis. All REST endpoints support full CRUD: GET reads from ClickHouse, POST overwrites with caller data, PUT computes/trains from source data.

**Next steps**: Build out more sophisticated test scenarios to ensure correct function of the correlation and classifiers.

## Testing

- **Run tests**: `python -m pytest` (or `python -m pytest -v` for verbose)
- **Run single test**: `python -m pytest tests/test_chains.py::test_post_single_event`
- **Always write async tests** — use `async def test_*` with `httpx.AsyncClient`. `asyncio_mode = auto` is set in `pytest.ini` so no `@pytest.mark.asyncio` decorator needed.
- Tests use `httpx.AsyncClient` with `httpx.ASGITransport(app=app)` — not `fastapi.testclient.TestClient`
- Redis-dependent tests are marked `@requires_redis` and skip gracefully when Redis is unavailable
- Coverage is configured in `pytest.ini` (`--cov=services --cov=main`)

## Gotchas

- No `.gitignore` yet — avoid committing `.vscode/`, `__pycache__/`, `*.pyc`, `.env`, `notebooks/.ipynb_checkpoints/`
- The notebook prototype uses `redis_om` in one cell but it's not installed and that cell fails — the working implementation uses raw `redis` commands instead
- Redis index `argus:ec:idx` is created by FastAPI on startup via `ensure_index()` — but `scripts/cleanup_events.py` drops it, so FastAPI must be restarted after cleanup
- ClickHouse events table has a 90-day TTL — the generator uses current timestamps to avoid expiry
