---
name: refresh-data
description: "Refreshes simulated telemetry data by cleaning ClickHouse and Redis, ensuring the FastAPI service is running, and regenerating events. Use this skill whenever the user says 'refresh data', 'regenerate events', 'reset data', 'reload test data', 'clean and regenerate', or anything about resetting/refreshing the simulated event data. Also use when the user needs fresh data after schema changes or code updates."
---

# Refresh Simulated Data

This skill cleans existing event data from ClickHouse and Redis, ensures the FastAPI service is running, and regenerates simulated telemetry events.

## Configuration

- **Python**: `C:/Users/benss/.conda/envs/p312/python.exe`
- **Working directory**: `c:\dev\stunning-couscous`
- **FastAPI port**: 8000
- **Base URL**: `http://localhost:8000`

## Steps

Execute these steps in order, reporting progress to the user after each one.

### Step 1: Clean existing data

Run the cleanup script to truncate ClickHouse tables and flush Redis keys:

```bash
C:/Users/benss/.conda/envs/p312/python.exe scripts/cleanup_events.py
```

Report the output (row counts deleted, keys removed).

### Step 2: Restart FastAPI

**Important**: The cleanup script drops the Redis index `argus:ec:idx`. FastAPI recreates it on startup via `ensure_index()`. Therefore you must **always restart** FastAPI after cleanup, even if it was already running.

1. Kill any existing uvicorn process:

```bash
pkill -f "uvicorn main:app" 2>/dev/null
```

2. Start the service in the background:

```bash
C:/Users/benss/.conda/envs/p312/python.exe -m uvicorn main:app --port 8000 &
```

3. Wait a few seconds then verify it's ready:

```bash
curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/docs
```

If the response is `200`, move on. If it's still not responding after ~10 seconds, tell the user and stop.

### Step 3: Generate events

Run the event generator, passing through any extra arguments the user provided (e.g., `--intervals 60`, `--seed 42`):

```bash
C:/Users/benss/.conda/envs/p312/python.exe scripts/generate_events.py --base-url http://localhost:8000 [extra args]
```

If the user didn't specify `--intervals` or `--seed`, use the defaults (the script defaults to 10 intervals, no seed).

Report the final output: number of transactions, events generated, success/error counts, and throughput.

### Step 4: Verify ingestion in ClickHouse

The generator script buffers stdout, so it may take a while before output appears. While waiting, monitor ingestion progress by polling ClickHouse:

```bash
curl -s "http://localhost:8123/" --data-binary "SELECT count() FROM argus.events"
```

Poll every 30 seconds until the count stabilises or the generator completes. Report the running count to the user so they can see progress.

Once the generator finishes, run a final summary query:

```bash
curl -s "http://localhost:8123/" --data-binary "SELECT EventName, count() AS cnt FROM argus.events GROUP BY EventName ORDER BY EventName"
```

Report the per-event breakdown to the user and confirm the total matches the generator output.
