# CLAUDE.md
This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
Telemetry event chain inference system. Captures events emitted by transactions passing through non-linear sequences of noisy processes, infers the dependency graph from noisy event data, and detects statistically significant latency anomalies and queuing at nodes in the graph.

Key challenge: the collector knows almost nothing in advance — not the processing tree, not the latencies, not the arrival rates. Everything is inferred from observed events correlated by shared transaction references.

The problem domain, simulation, and prototype algorithms are detailed in `notebooks/Event Chains Revisited.ipynb`.

## Performance Target
3,000 events/second (200M events/day). Performance is the top priority at every layer.

## Coding Standards
- Use zero memory copy data including formats such as Arrow and packages such as Polars
- Use Pydantic for datamodeling
- Router methods should not implement any logic other than calling the appropriate service(s).  All logic should be implemented in the service classes.
- Services should not import other services.  Instead these should be passed via dependency injection.

## Development Environment
- **Python**: `C:\Users\Ben\.conda\envs\p312` (conda, Python 3.12)
- **Package manager**: conda (configured in `.vscode/settings.json`)
- **Activate env**: `conda activate p312`
- **Install deps**: `pip install -r requirements.txt` (requirements.txt in root folder)
- **Run API**: `python main.py` or `uvicorn main:app --reload` (FastAPI app in `main.py`, port 8000)
- **Run notebook**: `jupyter lab notebooks/`
- **Redis Stack 7.4+** required locally on `localhost:6379` (Redis Insight at `http://localhost:8001`)
- **ClickHouse** required locally on `localhost:8123` (HTTP) / `localhost:9000` (native), database `arestor`
- **Reset Redis** (clears all data): `redis-cli FLUSHDB`
- **Simulate events**: `python scripts/generate_events.py --intervals 60 --base-url http://localhost:8000 --seed 42`

## Architecture
```
REST endpoint (FastAPI, main.py)
  → Redis (event cache + event chain assembly via JSON + Search)
  → ClickHouse (persisted events, adjacency matrix, path profiles, classifier model)
```

### Project Structure
```
main.py                          # FastAPI app, includes routers
core/
  dependencies.py                # Lifespan, Settings, dependency-injection getters
routers/
  events.py                      # POST/DELETE /events, POST /events/simulation
  adjacency.py                   # GET/PUT/DELETE /adjacency_matrix
  classifier.py                  # GET/PUT /classifier
  cache.py                       # GET/PUT/DELETE /cache
  chains.py                      # GET /chains
services/
  redis_service.py               # Async Redis ops, Lua chain merge script
  clickhouse_service.py          # ClickHouse queries, ClickHouseBatchWriter
  adjacency_service.py           # Adjacency matrix computation
  chain_classifier_service.py    # Profile discovery, TreeClassifier, ChainClassifier
  cache_service.py               # Bulk Redis chain CRUD (load/delete)
  data_simulator.py              # In-process event simulation
schemas/
  models.py                      # Pydantic models (Event, PathProfile, etc.)
scripts/
  generate_events.py             # Simulate and POST events to the API
tests/
  conftest.py                    # Session-scoped fixtures (testcontainers)
  data_simulation.py             # DataSimulator for test event generation
  test_endpoints.py              # Integration tests (test_full_pipeline, test_cache_operations)
  test_classifier.py             # Classifier unit tests
  test_correlation.py            # Correlation/adjacency tests
```

### Processing Pipeline
1. Receive events as JSON via `POST /events`
2. Correlate events into event chains atomically via a Lua script that performs FT.SEARCH + chain create/merge in a single Redis round-trip
3. Persist every event to ClickHouse via an async in-process batch writer (`ClickHouseBatchWriter`) — the hot path never waits for ClickHouse
4. Compute adjacency matrix on demand via `PUT /adjacency_matrix` by passing timestamp series from ClickHouse to pluggable correlation methods
5. Classify event chain path profiles via `PUT /classifier` — discovers which node-sets occur, identifies terminal nodes, and finds discriminating features using `TreeClassifier` (sklearn DecisionTree)
6. Trained classifier model is serialized via joblib, persisted to ClickHouse (along with feature importances, model parameters, and accuracy), and loaded on startup for runtime profile prediction
7. Determine chain termination using inferred path profiles: when all terminal nodes for the predicted profile are received, the chain is given a 30-second draining TTL (`TERMINATED_TTL_SECONDS`) to allow late-arriving out-of-order events to merge before expiry. Chain keys also have a 10-minute TTL (`CHAIN_TTL_SECONDS`) as a safety net against stale chains — refreshed on each merge.

### Lua Chain Merge Script
The Lua script in `redis_service.py` (lines 32–143) is the critical hot-path code. It performs an atomic FT.SEARCH + chain create/merge in a single Redis round-trip.

**Return format:** `[status, chain_id, events_json, chain_json]`

**Status codes:**
- `CREATED` — new chain created (no prior match)
- `MERGED` — event merged into an existing chain (common path)
- `CONFLICT` — ref-type conflict detected; timestamps merged but Python must handle duplicate-chain creation
- `MULTI_MATCH` — more than one chain matched; Python fallback pipeline handles

### Key Data Structures
- **Event**: `{EventName, Timestamp, Refs: [{type, id, ver}], Context: {}}`
- **Event Chain** (Redis JSON): `{concatenatedrefs: [str], timestamps: {process: timestamp}, context: {}, complete: bool, terminated: bool}`
- Redis index `arestor:ec:idx` on `$.concatenatedrefs[*]`, `$.complete`, `$.terminated` (TagFields)
- Redis Stream `arestor:ecstream` tracks created/updated chains
- **PathProfile**: `{profile_id, node_set, terminal_nodes, chain_count, fraction}` — discovered from observed data

## Dependencies
See `requirements.txt`

## Testing
- **Run tests**: `python -m pytest` (or `python -m pytest -v` for verbose)
- **Run single test**: `python -m pytest tests/test_endpoints.py::test_full_pipeline`
- **Always write async tests** — use `async def test_*` with `httpx.AsyncClient`. `asyncio_mode = auto` is set in `pytest.ini` so no `@pytest.mark.asyncio` decorator needed.
- Tests use `httpx.AsyncClient` with `httpx.ASGITransport(app=app)` — not `fastapi.testclient.TestClient`
- Tests use **testcontainers** (Docker) for Redis Stack and ClickHouse — Docker must be running
- Session-scoped event loop: test files that use session-scoped fixtures (client, redis, clickhouse) must declare `pytestmark = pytest.mark.asyncio(loop_scope="session")`. `asyncio_default_fixture_loop_scope = session` is set in `pytest.ini`.
- Session-scoped fixtures in `conftest.py` are **sync** (not async) to avoid event-loop-scope conflicts. Use the `sync_redis` fixture for test assertions against Redis data.
- Coverage is configured in `pytest.ini` (`--cov=services --cov=main`)

## Gotchas
- ClickHouse events table has a 90-day TTL — the generator uses current timestamps to avoid expiry
- Redis chain keys have a 10-minute TTL (`CHAIN_TTL_SECONDS = 600`) — refreshed on each merge, but truly stale chains will auto-evict
- Terminated chains get a 30-second draining TTL (`TERMINATED_TTL_SECONDS = 30`) — if a late event arrives within this window, the chain's TTL resets to 10 minutes and the termination check re-runs after the next terminal event
- ClickHouse UUID columns cannot be exported via Arrow format (`query_arrow`) — use `toString(col)` in the SELECT or avoid `query_arrow` for tables with UUID columns
- The ClickHouse client uses `async_insert=1, wait_for_async_insert=0` globally for throughput; `insert_classifier_model` overrides this to synchronous to ensure immediate read-after-write consistency
- Redis index `arestor:ec:idx` is created by FastAPI on startup via `ensure_index()` — if the index is dropped manually, FastAPI must be restarted to recreate it

## Current Status
Core functionality to ingest events, infer the processing tree, group events into sub-trees and train classifiers to identify the expected sub-tree and terminal events has been completed.

We are now working on calculating the expected latency between nodes in the tree and:
a. we have just added the ability train a state model (such as a hidden markov model) to infer when latency between 2 nodes in the tree probabilistically falls outside of expected bounds which now needs testing thoroughly
b. calculate arrival/service rates at each node and identify when transations are queuing at a particular node in the processing tree based on arival/service rates at each node.
c. in real-time identify if the latencies between nodes in all incoming chains are either anomalous according to the state model (per a above), or if some queing was likely (per b above) and persist these assessments and stream these over websockets

Anomalous events chains identified by steps c and d above should be held on the cache for a short period, say 5 minutes, and should be streamed over websockets.