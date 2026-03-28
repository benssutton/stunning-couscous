# Event Counts Screen Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an Event Counts screen with time-series chart, date comparison, and server-side T-test, plus a persistent nav bar across all screens.

**Architecture:** New `POST /events/counts` and `GET /events/names` endpoints on the existing events router; a new generic `POST /stats/ttest` endpoint in its own router backed by scipy; a stateless `EventCountsService` for metric transformation and `StatsService` wrapping scipy. On the frontend, a new `NavComponent` is added to the app shell, and a new `EventCountsComponent` renders a Plotly.js multi-line time-series chart with a T-test result panel.

**Tech Stack:** Python/FastAPI, Polars, ClickHouse, scipy, Angular 21 (standalone), Plotly.js (`plotly.js-dist-min`)

---

## File Map

**Create (backend):**
- `services/stats_service.py` — Welch's T-test via scipy
- `services/event_counts_service.py` — metric transformation (rolling avg, cumsum) on a Polars DataFrame
- `routers/stats.py` — `POST /stats/ttest`

**Modify (backend):**
- `schemas/models.py` — add `TTestRequest`, `TTestResult`, `EventCountsRequest`, `BucketPoint`, `DateSeries`, `EventCountsResponse`, `EventNamesResponse`
- `services/clickhouse_service.py` — add `get_event_counts()` and `get_distinct_event_names()`
- `routers/events.py` — add `POST /events/counts` and `GET /events/names`
- `core/dependencies.py` — add `get_stats_service()` and `get_event_counts_service()`
- `main.py` — include `stats.router`

**Create (tests):**
- `tests/test_stats.py` — unit + integration tests for StatsService and `/stats/ttest`
- `tests/test_event_counts.py` — integration tests for `/events/counts` and `/events/names`

**Modify (frontend):**
- `ui/chain-search/package.json` — add `plotly.js-dist-min` and `@types/plotly.js-dist-min`
- `ui/chain-search/src/app/models/api.models.ts` — add new interfaces
- `ui/chain-search/src/app/services/api.service.ts` — add `getEventNames()`, `getEventCounts()`, `runTTest()`
- `ui/chain-search/src/app/app.ts` — import `NavComponent`
- `ui/chain-search/src/app/app.html` — add `<app-nav>` above `<router-outlet>`
- `ui/chain-search/src/app/app.routes.ts` — add `/counts` and placeholder routes

**Create (frontend):**
- `ui/chain-search/src/app/components/nav/nav.component.ts`
- `ui/chain-search/src/app/components/nav/nav.component.html`
- `ui/chain-search/src/app/components/nav/nav.component.scss`
- `ui/chain-search/src/app/components/event-counts/event-counts.component.ts`
- `ui/chain-search/src/app/components/event-counts/event-counts.component.html`
- `ui/chain-search/src/app/components/event-counts/event-counts.component.scss`
- `ui/chain-search/src/app/components/placeholder/placeholder.component.ts`

---

## Task 1: Pydantic models

**Files:**
- Modify: `schemas/models.py`

- [ ] **Step 1: Add the new models at the bottom of `schemas/models.py`**

```python
# ---------------------------------------------------------------------------
# Event counts models
# ---------------------------------------------------------------------------

from typing import Literal

class EventCountsRequest(BaseModel):
    event_name: str
    dates: list[str]          # YYYY-MM-DD
    bucket_seconds: int       # 1–60
    metric: Literal["count", "rolling_avg", "cumulative_sum"]

class BucketPoint(BaseModel):
    time: str                 # HH:MM:SS
    value: float

class DateSeries(BaseModel):
    date: str
    buckets: list[BucketPoint]

class EventCountsResponse(BaseModel):
    series: list[DateSeries]

class EventNamesResponse(BaseModel):
    names: list[str]


# ---------------------------------------------------------------------------
# Stats models
# ---------------------------------------------------------------------------

class TTestRequest(BaseModel):
    series_a: list[float]
    series_b: list[float]
    alpha: float = 0.05

class TTestResult(BaseModel):
    t_statistic: float
    p_value: float
    degrees_of_freedom: int
    significant: bool
    alpha: float
```

Note: `from typing import Literal` must be added to the existing imports at the top of `schemas/models.py`.

- [ ] **Step 2: Commit**

```bash
git add schemas/models.py
git commit -m "feat: add Pydantic models for event counts and T-test"
```

---

## Task 2: StatsService

**Files:**
- Create: `services/stats_service.py`
- Create: `tests/test_stats.py`

- [ ] **Step 1: Write the failing unit test**

Create `tests/test_stats.py`:

```python
import pytest
from services.stats_service import StatsService

pytestmark = pytest.mark.asyncio(loop_scope="session")


def test_ttest_significant():
    svc = StatsService()
    # Two clearly different distributions
    a = [10.0, 11.0, 10.5, 10.2, 11.1, 10.8, 10.3, 11.2, 10.6, 10.9]
    b = [20.0, 21.0, 19.5, 20.8, 21.3, 19.9, 20.5, 21.1, 20.2, 19.7]
    result = svc.run_ttest(a, b, alpha=0.05)
    assert result.significant is True
    assert result.p_value < 0.05
    assert result.degrees_of_freedom > 0
    assert result.alpha == 0.05


def test_ttest_not_significant():
    svc = StatsService()
    # Two samples from the same distribution
    a = [10.0, 10.1, 9.9, 10.2, 10.0, 9.8, 10.1, 10.3, 9.7, 10.0]
    b = [10.1, 10.0, 10.2, 9.9, 10.1, 10.0, 9.8, 10.2, 10.1, 9.9]
    result = svc.run_ttest(a, b, alpha=0.05)
    assert result.significant is False
    assert result.p_value >= 0.05


def test_ttest_custom_alpha():
    svc = StatsService()
    a = [10.0, 11.0, 10.5, 10.2, 11.1]
    b = [10.3, 10.4, 10.1, 10.5, 10.2]
    result = svc.run_ttest(a, b, alpha=0.5)
    assert result.alpha == 0.5
    assert result.significant == (result.p_value < 0.5)
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
python -m pytest tests/test_stats.py -v
```

Expected: `ImportError: cannot import name 'StatsService'`

- [ ] **Step 3: Implement `services/stats_service.py`**

```python
from scipy import stats as scipy_stats
from schemas.models import TTestResult


class StatsService:
    def run_ttest(
        self,
        series_a: list[float],
        series_b: list[float],
        alpha: float = 0.05,
    ) -> TTestResult:
        result = scipy_stats.ttest_ind(series_a, series_b, equal_var=False)
        return TTestResult(
            t_statistic=float(result.statistic),
            p_value=float(result.pvalue),
            degrees_of_freedom=int(result.df),
            significant=bool(result.pvalue < alpha),
            alpha=alpha,
        )
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
python -m pytest tests/test_stats.py -v
```

Expected: 3 tests PASSED

- [ ] **Step 5: Commit**

```bash
git add services/stats_service.py tests/test_stats.py
git commit -m "feat: add StatsService with Welch T-test"
```

---

## Task 3: POST /stats/ttest router

**Files:**
- Create: `routers/stats.py`
- Modify: `core/dependencies.py`
- Modify: `main.py`
- Modify: `tests/test_stats.py`

- [ ] **Step 1: Add the integration test to `tests/test_stats.py`**

Append to the existing file:

```python
async def test_ttest_endpoint_significant(client):
    response = await client.post("/stats/ttest", json={
        "series_a": [10.0, 11.0, 10.5, 10.2, 11.1, 10.8, 10.3, 11.2, 10.6, 10.9],
        "series_b": [20.0, 21.0, 19.5, 20.8, 21.3, 19.9, 20.5, 21.1, 20.2, 19.7],
        "alpha": 0.05,
    })
    assert response.status_code == 200
    body = response.json()
    assert body["significant"] is True
    assert body["p_value"] < 0.05
    assert "t_statistic" in body
    assert "degrees_of_freedom" in body


async def test_ttest_endpoint_default_alpha(client):
    response = await client.post("/stats/ttest", json={
        "series_a": [1.0, 2.0, 1.5],
        "series_b": [100.0, 200.0, 150.0],
    })
    assert response.status_code == 200
    body = response.json()
    assert body["alpha"] == 0.05
    assert body["significant"] is True
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest tests/test_stats.py::test_ttest_endpoint_significant -v
```

Expected: FAIL — 404 Not Found

- [ ] **Step 3: Add `get_stats_service` to `core/dependencies.py`**

Add import at the top of `core/dependencies.py` (with the other service imports):

```python
from services.stats_service import StatsService
from services.event_counts_service import EventCountsService
```

Add these two functions at the bottom of `core/dependencies.py`:

```python
def get_stats_service() -> StatsService:
    return StatsService()


def get_event_counts_service() -> EventCountsService:
    return EventCountsService()
```

- [ ] **Step 4: Create `routers/stats.py`**

```python
from fastapi import APIRouter, Depends

from schemas.models import TTestRequest, TTestResult
from services.stats_service import StatsService
from core.dependencies import get_stats_service

router = APIRouter()


@router.post("/stats/ttest",
             summary="Run a two-sample Welch's T-test on two series of observations")
async def run_ttest(
    request: TTestRequest,
    stats_svc: StatsService = Depends(get_stats_service),
) -> TTestResult:
    return stats_svc.run_ttest(request.series_a, request.series_b, request.alpha)
```

- [ ] **Step 5: Register the stats router in `main.py`**

Add `stats` to the import line:

```python
from routers import adjacency, cache, chains, classifier, events, latencies, search, state_detectors, stats
```

Add after the existing `app.include_router` calls:

```python
app.include_router(stats.router)
```

- [ ] **Step 6: Run integration tests to confirm they pass**

```bash
python -m pytest tests/test_stats.py -v
```

Expected: 5 tests PASSED

- [ ] **Step 7: Commit**

```bash
git add routers/stats.py core/dependencies.py main.py tests/test_stats.py
git commit -m "feat: add POST /stats/ttest endpoint"
```

---

## Task 4: ClickHouseService query methods

**Files:**
- Modify: `services/clickhouse_service.py`

- [ ] **Step 1: Add `get_distinct_event_names` to `ClickHouseService`**

Find the `ClickHouseService` class in `services/clickhouse_service.py` and add this method:

```python
def get_distinct_event_names(self) -> list[str]:
    result = self._client.query(
        f"SELECT DISTINCT event_name FROM {self.database}.events ORDER BY event_name"
    )
    return [row[0] for row in result.result_rows]
```

- [ ] **Step 2: Add `get_event_counts` to `ClickHouseService`**

Add this method to the same class, after `get_distinct_event_names`:

```python
def get_event_counts(
    self,
    event_name: str,
    dates: list[str],
    bucket_seconds: int,
) -> pl.DataFrame:
    """Return per-bucket event counts for the given dates.

    Returns a Polars DataFrame with columns:
        date (str, YYYY-MM-DD), bucket_time (datetime), count (int)
    """
    result = self._client.query(
        f"""
        SELECT
            toString(toDate(timestamp))           AS date,
            toStartOfInterval(timestamp, INTERVAL {{bucket_seconds:UInt32}} SECOND) AS bucket_time,
            COUNT(*)                              AS count
        FROM {self.database}.events
        WHERE event_name = {{event_name:String}}
          AND toDate(timestamp) IN {{dates:Array(String)}}
        GROUP BY date, bucket_time
        ORDER BY date, bucket_time
        """,
        parameters={"event_name": event_name, "dates": dates, "bucket_seconds": bucket_seconds},
    )
    rows = result.result_rows
    if not rows:
        return pl.DataFrame(schema={"date": pl.Utf8, "bucket_time": pl.Datetime, "count": pl.Int64})
    return pl.DataFrame(
        {
            "date": [r[0] for r in rows],
            "bucket_time": [r[1] for r in rows],
            "count": [r[2] for r in rows],
        }
    )
```

Also ensure `polars` is imported at the top of the file — it already is (`import polars as pl`).

- [ ] **Step 3: Commit**

```bash
git add services/clickhouse_service.py
git commit -m "feat: add get_event_counts and get_distinct_event_names to ClickHouseService"
```

---

## Task 5: EventCountsService

**Files:**
- Create: `services/event_counts_service.py`
- Create: `tests/test_event_counts.py` (unit portion)

- [ ] **Step 1: Write the failing unit tests**

Create `tests/test_event_counts.py`:

```python
from datetime import datetime

import polars as pl
import pytest

from services.event_counts_service import EventCountsService

pytestmark = pytest.mark.asyncio(loop_scope="session")


def _make_df(rows: list[tuple]) -> pl.DataFrame:
    """rows: (date_str, bucket_datetime, count)"""
    return pl.DataFrame({
        "date": [r[0] for r in rows],
        "bucket_time": [r[1] for r in rows],
        "count": [r[2] for r in rows],
    })


def test_count_metric():
    svc = EventCountsService()
    df = _make_df([
        ("2026-03-27", datetime(2026, 3, 27, 0, 0, 0), 10),
        ("2026-03-27", datetime(2026, 3, 27, 0, 0, 30), 20),
    ])
    response = svc.build_response(df, "count")
    assert len(response.series) == 1
    assert response.series[0].date == "2026-03-27"
    assert response.series[0].buckets[0].value == 10.0
    assert response.series[0].buckets[1].value == 20.0
    assert response.series[0].buckets[0].time == "00:00:00"
    assert response.series[0].buckets[1].time == "00:00:30"


def test_cumulative_sum_metric():
    svc = EventCountsService()
    df = _make_df([
        ("2026-03-27", datetime(2026, 3, 27, 0, 0, 0), 5),
        ("2026-03-27", datetime(2026, 3, 27, 0, 0, 30), 10),
        ("2026-03-27", datetime(2026, 3, 27, 0, 1, 0), 3),
    ])
    response = svc.build_response(df, "cumulative_sum")
    values = [b.value for b in response.series[0].buckets]
    assert values == [5.0, 15.0, 18.0]


def test_rolling_avg_metric():
    svc = EventCountsService()
    # Build 10 buckets with known values
    rows = [
        ("2026-03-27", datetime(2026, 3, 27, 0, 0, i * 30), float(i + 1))
        for i in range(10)
    ]
    df = _make_df(rows)
    response = svc.build_response(df, "rolling_avg")
    buckets = response.series[0].buckets
    # First 6 buckets have fewer than 7 observations — values will be null → 0.0
    # Bucket 7 (index 6) is the first full window: mean of [1,2,3,4,5,6,7] = 4.0
    assert buckets[6].value == pytest.approx(4.0)


def test_multiple_dates():
    svc = EventCountsService()
    df = _make_df([
        ("2026-03-27", datetime(2026, 3, 27, 0, 0, 0), 10),
        ("2026-03-26", datetime(2026, 3, 26, 0, 0, 0), 5),
    ])
    response = svc.build_response(df, "count")
    dates = {s.date for s in response.series}
    assert dates == {"2026-03-27", "2026-03-26"}
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest tests/test_event_counts.py -v
```

Expected: `ImportError: cannot import name 'EventCountsService'`

- [ ] **Step 3: Implement `services/event_counts_service.py`**

```python
from datetime import datetime

import polars as pl

from schemas.models import BucketPoint, DateSeries, EventCountsResponse


class EventCountsService:
    def build_response(
        self,
        df: pl.DataFrame,
        metric: str,
    ) -> EventCountsResponse:
        if df.is_empty():
            return EventCountsResponse(series=[])

        if metric == "cumulative_sum":
            df = df.with_columns(
                pl.col("count").cast(pl.Float64).cum_sum().over("date").alias("value")
            )
        elif metric == "rolling_avg":
            df = df.with_columns(
                pl.col("count")
                .cast(pl.Float64)
                .rolling_mean(window_size=7)
                .over("date")
                .fill_null(0.0)
                .alias("value")
            )
        else:  # count
            df = df.with_columns(pl.col("count").cast(pl.Float64).alias("value"))

        series = []
        for date_str in df["date"].unique().sort().to_list():
            date_df = df.filter(pl.col("date") == date_str).sort("bucket_time")
            buckets = [
                BucketPoint(
                    time=row["bucket_time"].strftime("%H:%M:%S"),
                    value=row["value"],
                )
                for row in date_df.iter_rows(named=True)
            ]
            series.append(DateSeries(date=date_str, buckets=buckets))

        return EventCountsResponse(series=series)
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
python -m pytest tests/test_event_counts.py -v
```

Expected: 4 tests PASSED

- [ ] **Step 5: Commit**

```bash
git add services/event_counts_service.py tests/test_event_counts.py
git commit -m "feat: add EventCountsService with metric transformation"
```

---

## Task 6: POST /events/counts and GET /events/names endpoints

**Files:**
- Modify: `routers/events.py`
- Modify: `tests/test_event_counts.py`

- [ ] **Step 1: Add integration tests to `tests/test_event_counts.py`**

Append to the existing file:

```python
from datetime import datetime, timezone, timedelta


def _iso(dt: datetime) -> str:
    return dt.isoformat(timespec="milliseconds")


async def test_get_event_names_empty(client):
    response = await client.get("/events/names")
    assert response.status_code == 200
    assert response.json() == {"names": []}


async def test_get_event_names_returns_distinct(client, batch_writer):
    now = datetime.now(timezone.utc)
    events = [
        {"EventName": "A", "Timestamp": _iso(now), "Refs": [{"type": "A", "id": "1", "ver": 1}], "Context": {}},
        {"EventName": "A", "Timestamp": _iso(now + timedelta(seconds=1)), "Refs": [{"type": "A", "id": "2", "ver": 1}], "Context": {}},
        {"EventName": "B", "Timestamp": _iso(now + timedelta(seconds=2)), "Refs": [{"type": "B", "id": "1", "ver": 1}], "Context": {}},
    ]
    for e in events:
        await batch_writer.append(["", e["EventName"], datetime.fromisoformat(e["Timestamp"]), [], [], []])
    import asyncio; await asyncio.sleep(0.2)

    response = await client.get("/events/names")
    assert response.status_code == 200
    names = response.json()["names"]
    assert set(names) == {"A", "B"}


async def test_post_event_counts(client, batch_writer):
    today = datetime.now(timezone.utc).date().isoformat()
    now = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    for i in range(5):
        ts = now + timedelta(seconds=i * 30)
        await batch_writer.append(["", "myevent", ts, [], [], []])
    import asyncio; await asyncio.sleep(0.2)

    response = await client.post("/events/counts", json={
        "event_name": "myevent",
        "dates": [today],
        "bucket_seconds": 30,
        "metric": "count",
    })
    assert response.status_code == 200
    body = response.json()
    assert len(body["series"]) == 1
    assert body["series"][0]["date"] == today
    assert len(body["series"][0]["buckets"]) > 0


async def test_post_event_counts_cumulative_sum(client, batch_writer):
    today = datetime.now(timezone.utc).date().isoformat()
    now = datetime.now(timezone.utc).replace(hour=1, minute=0, second=0, microsecond=0)
    for i in range(3):
        ts = now + timedelta(seconds=i * 30)
        await batch_writer.append(["", "csevent", ts, [], [], []])
    import asyncio; await asyncio.sleep(0.2)

    response = await client.post("/events/counts", json={
        "event_name": "csevent",
        "dates": [today],
        "bucket_seconds": 30,
        "metric": "cumulative_sum",
    })
    assert response.status_code == 200
    buckets = response.json()["series"][0]["buckets"]
    values = [b["value"] for b in buckets]
    # Each bucket has 1 event; cumsum should be [1, 2, 3]
    assert values == [1.0, 2.0, 3.0]
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest tests/test_event_counts.py::test_get_event_names_empty -v
```

Expected: FAIL — 404 Not Found

- [ ] **Step 3: Add the new endpoints to `routers/events.py`**

Add these imports at the top of `routers/events.py` (merge with existing imports):

```python
from schemas.models import Event, EventCountsRequest, EventCountsResponse, EventNamesResponse
from services.event_counts_service import EventCountsService
from core.dependencies import get_batch_writer, get_clickhouse_service, get_redis_service, get_event_counts_service
```

Add these two endpoints at the bottom of `routers/events.py`:

```python
@router.get("/events/names",
            summary="List distinct event names")
async def get_event_names(
    ch_svc: ClickHouseService = Depends(get_clickhouse_service),
) -> EventNamesResponse:
    names = ch_svc.get_distinct_event_names()
    return EventNamesResponse(names=names)


@router.post("/events/counts", status_code=200,
             summary="Create a time-series count of events")
async def get_event_counts(
    request: EventCountsRequest,
    ch_svc: ClickHouseService = Depends(get_clickhouse_service),
    counts_svc: EventCountsService = Depends(get_event_counts_service),
) -> EventCountsResponse:
    df = ch_svc.get_event_counts(request.event_name, request.dates, request.bucket_seconds)
    return counts_svc.build_response(df, request.metric)
```

- [ ] **Step 4: Run integration tests to confirm they pass**

```bash
python -m pytest tests/test_event_counts.py -v
```

Expected: All tests PASSED

- [ ] **Step 5: Commit**

```bash
git add routers/events.py tests/test_event_counts.py
git commit -m "feat: add POST /events/counts and GET /events/names endpoints"
```

---

## Task 7: Angular — install Plotly and add TypeScript models

**Files:**
- Modify: `ui/chain-search/package.json`
- Modify: `ui/chain-search/src/app/models/api.models.ts`

- [ ] **Step 1: Install Plotly**

```bash
cd ui/chain-search
npm install plotly.js-dist-min @types/plotly.js-dist-min
```

Expected: packages added to `node_modules` and `package.json` updated.

- [ ] **Step 2: Add new interfaces to `api.models.ts`**

Append to `ui/chain-search/src/app/models/api.models.ts`:

```typescript
// Event counts
export interface EventNamesResponse {
  names: string[];
}

export interface BucketPoint {
  time: string;   // HH:MM:SS
  value: number;
}

export interface DateSeries {
  date: string;   // YYYY-MM-DD
  buckets: BucketPoint[];
}

export interface EventCountsRequest {
  event_name: string;
  dates: string[];
  bucket_seconds: number;
  metric: 'count' | 'rolling_avg' | 'cumulative_sum';
}

export interface EventCountsResponse {
  series: DateSeries[];
}

// T-test
export interface TTestRequest {
  series_a: number[];
  series_b: number[];
  alpha?: number;
}

export interface TTestResult {
  t_statistic: number;
  p_value: number;
  degrees_of_freedom: number;
  significant: boolean;
  alpha: number;
}
```

- [ ] **Step 3: Commit**

```bash
cd ui/chain-search
git add package.json package-lock.json src/app/models/api.models.ts
git commit -m "feat: add Plotly dependency and event counts TypeScript models"
```

---

## Task 8: NavComponent

**Files:**
- Create: `ui/chain-search/src/app/components/nav/nav.component.ts`
- Create: `ui/chain-search/src/app/components/nav/nav.component.html`
- Create: `ui/chain-search/src/app/components/nav/nav.component.scss`

- [ ] **Step 1: Create `nav.component.ts`**

```typescript
import { Component } from '@angular/core';
import { RouterLink, RouterLinkActive } from '@angular/router';

@Component({
  selector: 'app-nav',
  standalone: true,
  imports: [RouterLink, RouterLinkActive],
  templateUrl: './nav.component.html',
  styleUrl: './nav.component.scss',
})
export class NavComponent {
  readonly navItems = [
    { label: 'Search', route: '/search' },
    { label: 'Counts', route: '/counts' },
    { label: 'Latencies', route: '/latencies' },
    { label: 'Real-time Dashboard', route: '/realtime' },
    { label: 'Models', route: '/models' },
  ];
}
```

- [ ] **Step 2: Create `nav.component.html`**

```html
<nav class="top-nav">
  <span class="brand">Arestor</span>
  @for (item of navItems; track item.route) {
    <a
      [routerLink]="item.route"
      routerLinkActive="active"
      class="nav-link"
    >{{ item.label }}</a>
  }
</nav>
```

- [ ] **Step 3: Create `nav.component.scss`**

```scss
.top-nav {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  z-index: 100;
  display: flex;
  align-items: center;
  gap: 4px;
  height: 44px;
  padding: 0 20px;
  background: #161b22;
  border-bottom: 1px solid #30363d;
}

.brand {
  color: #58a6ff;
  font-weight: 700;
  font-size: 14px;
  margin-right: 12px;
  user-select: none;
}

.nav-link {
  color: #8b949e;
  font-size: 13px;
  padding: 6px 12px;
  border-radius: 4px;
  text-decoration: none;
  transition: color 0.15s, background 0.15s;

  &:hover {
    color: #e6edf3;
    background: #21262d;
  }

  &.active {
    color: #e6edf3;
    background: #21262d;
    font-weight: 500;
  }
}
```

- [ ] **Step 4: Commit**

```bash
git add ui/chain-search/src/app/components/nav/
git commit -m "feat: add NavComponent with fixed top nav bar"
```

---

## Task 9: App shell, routes, and placeholder

**Files:**
- Modify: `ui/chain-search/src/app/app.ts`
- Modify: `ui/chain-search/src/app/app.html`
- Modify: `ui/chain-search/src/app/app.routes.ts`
- Create: `ui/chain-search/src/app/components/placeholder/placeholder.component.ts`

- [ ] **Step 1: Create `placeholder.component.ts`**

```typescript
import { Component, input } from '@angular/core';

@Component({
  selector: 'app-placeholder',
  standalone: true,
  template: `
    <div class="placeholder-page">
      <h2>{{ title() }}</h2>
      <p>Coming soon.</p>
    </div>
  `,
  styles: [`
    .placeholder-page {
      padding: 80px 32px 32px;
      color: #8b949e;
      h2 { color: #e6edf3; margin-bottom: 8px; }
    }
  `],
})
export class PlaceholderComponent {
  title = input<string>('This screen');
}
```

- [ ] **Step 2: Update `app.routes.ts`**

Replace the entire file contents:

```typescript
import { Routes } from '@angular/router';
import { SearchComponent } from './components/search/search.component';
import { EventCountsComponent } from './components/event-counts/event-counts.component';
import { PlaceholderComponent } from './components/placeholder/placeholder.component';

export const routes: Routes = [
  { path: 'search', component: SearchComponent },
  { path: 'counts', component: EventCountsComponent },
  { path: 'latencies', component: PlaceholderComponent, data: { title: 'Latencies' } },
  { path: 'realtime', component: PlaceholderComponent, data: { title: 'Real-time Dashboard' } },
  { path: 'models', component: PlaceholderComponent, data: { title: 'Models' } },
  { path: '', redirectTo: 'search', pathMatch: 'full' },
];
```

- [ ] **Step 3: Update `app.ts`**

Replace the entire file contents:

```typescript
import { Component } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { NavComponent } from './components/nav/nav.component';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [RouterOutlet, NavComponent],
  templateUrl: './app.html',
  styleUrl: './app.scss',
})
export class App {}
```

- [ ] **Step 4: Update `app.html`**

Replace the entire file contents:

```html
<app-nav />
<div class="page-body">
  <router-outlet />
</div>
```

- [ ] **Step 5: Update `app.scss` to add page-body offset**

Replace the entire file contents (it's currently empty):

```scss
.page-body {
  margin-top: 44px; // height of the fixed nav
}
```

- [ ] **Step 6: Commit**

```bash
git add ui/chain-search/src/app/app.ts ui/chain-search/src/app/app.html ui/chain-search/src/app/app.scss ui/chain-search/src/app/app.routes.ts ui/chain-search/src/app/components/placeholder/
git commit -m "feat: add nav bar to app shell and placeholder routes"
```

---

## Task 10: ApiService — new methods

**Files:**
- Modify: `ui/chain-search/src/app/services/api.service.ts`

- [ ] **Step 1: Add imports and three new methods to `api.service.ts`**

Add to the existing imports at the top:

```typescript
import {
  RefAutocompleteResponse,
  ChainSearchResponse,
  ChainDetail,
  ChainLatencyResponse,
  AverageLatencyResponse,
  EventNamesResponse,
  EventCountsRequest,
  EventCountsResponse,
  TTestRequest,
  TTestResult,
} from '../models/api.models';
```

Add these three methods to the `ApiService` class (after the existing methods):

```typescript
  getEventNames(): Observable<EventNamesResponse> {
    return this.http.get<EventNamesResponse>('/events/names');
  }

  getEventCounts(request: EventCountsRequest): Observable<EventCountsResponse> {
    return this.http.post<EventCountsResponse>('/events/counts', request);
  }

  runTTest(request: TTestRequest): Observable<TTestResult> {
    return this.http.post<TTestResult>('/stats/ttest', request);
  }
```

- [ ] **Step 2: Commit**

```bash
git add ui/chain-search/src/app/services/api.service.ts
git commit -m "feat: add getEventNames, getEventCounts, runTTest to ApiService"
```

---

## Task 11: EventCountsComponent

**Files:**
- Create: `ui/chain-search/src/app/components/event-counts/event-counts.component.ts`
- Create: `ui/chain-search/src/app/components/event-counts/event-counts.component.html`
- Create: `ui/chain-search/src/app/components/event-counts/event-counts.component.scss`

- [ ] **Step 1: Create `event-counts.component.ts`**

```typescript
import { Component, OnInit, OnDestroy, ElementRef, ViewChild, AfterViewInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Subscription, forkJoin, of } from 'rxjs';
import { catchError } from 'rxjs/operators';
import Plotly from 'plotly.js-dist-min';

import { ApiService } from '../../services/api.service';
import { EventCountsResponse, TTestResult } from '../../models/api.models';

type Metric = 'count' | 'rolling_avg' | 'cumulative_sum';

const PLOTLY_COLORS = [
  '#58a6ff', '#3fb950', '#f78166', '#d2a8ff', '#ffa657',
  '#79c0ff', '#56d364', '#ff7b72', '#bc8cff', '#ffb86c',
];

const BUCKET_OPTIONS = [
  { label: '1 second',  value: 1 },
  { label: '5 seconds', value: 5 },
  { label: '10 seconds', value: 10 },
  { label: '15 seconds', value: 15 },
  { label: '30 seconds', value: 30 },
  { label: '1 minute',  value: 60 },
];

@Component({
  selector: 'app-event-counts',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './event-counts.component.html',
  styleUrl: './event-counts.component.scss',
})
export class EventCountsComponent implements OnInit, OnDestroy, AfterViewInit {
  @ViewChild('chartEl') chartEl!: ElementRef<HTMLDivElement>;

  eventNames: string[] = [];
  selectedEvent = '';
  selectedDates: string[] = [];
  newDate = '';
  bucketSeconds = 30;
  metric: Metric = 'count';

  bucketOptions = BUCKET_OPTIONS;
  metricOptions: { label: string; value: Metric }[] = [
    { label: 'Count', value: 'count' },
    { label: 'Rolling Average', value: 'rolling_avg' },
    { label: 'Cumulative Sum', value: 'cumulative_sum' },
  ];

  loading = false;
  error = '';
  ttestResult: TTestResult | null = null;
  ttestDates: string[] = [];

  private chartInitialised = false;
  private subs = new Subscription();

  constructor(private api: ApiService) {}

  ngOnInit(): void {
    this.subs.add(
      this.api.getEventNames().subscribe({
        next: (res) => {
          this.eventNames = res.names;
          if (res.names.length) this.selectedEvent = res.names[0];
        },
        error: () => { this.error = 'Failed to load event names.'; },
      })
    );
  }

  ngAfterViewInit(): void {
    Plotly.newPlot(this.chartEl.nativeElement, [], this.plotlyLayout(), { responsive: true });
    this.chartInitialised = true;
  }

  ngOnDestroy(): void {
    this.subs.unsubscribe();
    if (this.chartInitialised) {
      Plotly.purge(this.chartEl.nativeElement);
    }
  }

  addDate(): void {
    if (this.newDate && !this.selectedDates.includes(this.newDate)) {
      this.selectedDates = [...this.selectedDates, this.newDate];
    }
    this.newDate = '';
  }

  removeDate(date: string): void {
    this.selectedDates = this.selectedDates.filter(d => d !== date);
    if (this.selectedDates.length < 2) this.ttestResult = null;
  }

  dateColor(index: number): string {
    return PLOTLY_COLORS[index % PLOTLY_COLORS.length];
  }

  refresh(): void {
    if (!this.selectedEvent || this.selectedDates.length === 0) {
      this.error = 'Select an event name and at least one date.';
      return;
    }
    this.loading = true;
    this.error = '';
    this.ttestResult = null;

    this.subs.add(
      this.api.getEventCounts({
        event_name: this.selectedEvent,
        dates: this.selectedDates,
        bucket_seconds: this.bucketSeconds,
        metric: this.metric,
      }).pipe(catchError(err => { this.error = 'Failed to load event counts.'; this.loading = false; throw err; }))
      .subscribe(response => {
        this.renderChart(response);

        if (this.selectedDates.length >= 2) {
          this.runTTest(response);
        } else {
          this.loading = false;
        }
      })
    );
  }

  private renderChart(response: EventCountsResponse): void {
    const traces: Plotly.Data[] = response.series.map((series, i) => ({
      x: series.buckets.map(b => b.time),
      y: series.buckets.map(b => b.value),
      type: 'scatter' as const,
      mode: 'lines' as const,
      name: series.date,
      line: { color: PLOTLY_COLORS[i % PLOTLY_COLORS.length], width: 2 },
    }));

    Plotly.react(this.chartEl.nativeElement, traces, this.plotlyLayout(), { responsive: true });
  }

  private runTTest(response: EventCountsResponse): void {
    const [a, b] = this.selectedDates.slice(0, 2);
    const seriesA = response.series.find(s => s.date === a);
    const seriesB = response.series.find(s => s.date === b);

    if (!seriesA || !seriesB) { this.loading = false; return; }

    this.subs.add(
      this.api.runTTest({
        series_a: seriesA.buckets.map(p => p.value),
        series_b: seriesB.buckets.map(p => p.value),
        alpha: 0.05,
      }).subscribe({
        next: (result) => {
          this.ttestResult = result;
          this.ttestDates = [a, b];
          this.loading = false;
        },
        error: () => {
          this.error = 'Failed to run T-test.';
          this.loading = false;
        },
      })
    );
  }

  private plotlyLayout(): Partial<Plotly.Layout> {
    return {
      paper_bgcolor: '#161b22',
      plot_bgcolor: '#161b22',
      font: { color: '#e6edf3', size: 12 },
      xaxis: {
        title: 'Time of day',
        gridcolor: '#21262d',
        color: '#8b949e',
        type: 'category',
      },
      yaxis: {
        title: this.metricOptions.find(m => m.value === this.metric)?.label ?? 'Value',
        gridcolor: '#21262d',
        color: '#8b949e',
      },
      legend: { bgcolor: '#161b22', bordercolor: '#30363d', borderwidth: 1 },
      margin: { t: 20, r: 20, b: 60, l: 60 },
    };
  }
}
```

- [ ] **Step 2: Create `event-counts.component.html`**

```html
<div class="counts-layout">
  <!-- Left control panel -->
  <aside class="control-panel">

    <div class="control-group">
      <label class="control-label">Event Name</label>
      <select class="control-select" [(ngModel)]="selectedEvent">
        @for (name of eventNames; track name) {
          <option [value]="name">{{ name }}</option>
        }
      </select>
    </div>

    <div class="control-group">
      <label class="control-label">Dates</label>
      <div class="date-chips">
        @for (date of selectedDates; track date; let i = $index) {
          <div class="date-chip" [style.border-color]="dateColor(i)">
            <span [style.color]="dateColor(i)">{{ date }}</span>
            <button class="remove-btn" (click)="removeDate(date)">✕</button>
          </div>
        }
      </div>
      <div class="add-date-row">
        <input type="date" class="control-input" [(ngModel)]="newDate" />
        <button class="btn-secondary" (click)="addDate()">Add</button>
      </div>
    </div>

    <div class="control-group">
      <label class="control-label">Bucket Size</label>
      <select class="control-select" [(ngModel)]="bucketSeconds">
        @for (opt of bucketOptions; track opt.value) {
          <option [value]="opt.value">{{ opt.label }}</option>
        }
      </select>
    </div>

    <div class="control-group">
      <label class="control-label">Metric</label>
      @for (opt of metricOptions; track opt.value) {
        <label class="radio-label">
          <input type="radio" name="metric" [value]="opt.value" [(ngModel)]="metric" />
          {{ opt.label }}
        </label>
      }
    </div>

    <button class="btn-primary" (click)="refresh()" [disabled]="loading">
      {{ loading ? 'Loading…' : 'Refresh' }}
    </button>

    @if (error) {
      <p class="error-msg">{{ error }}</p>
    }
  </aside>

  <!-- Chart area -->
  <main class="chart-area">
    <div #chartEl class="plotly-chart"></div>

    @if (ttestResult && ttestDates.length === 2) {
      <div class="ttest-panel">
        <div class="ttest-title">Two-Sample Student's T-Test</div>
        <div class="ttest-stats">
          <span><span class="stat-label">p-value:</span>
            <strong [class.significant]="ttestResult.significant"
                    [class.not-significant]="!ttestResult.significant">
              {{ ttestResult.p_value | number:'1.4-4' }}
            </strong>
          </span>
          <span><span class="stat-label">t-statistic:</span> {{ ttestResult.t_statistic | number:'1.2-2' }}</span>
          <span><span class="stat-label">df:</span> {{ ttestResult.degrees_of_freedom }}</span>
          <span class="badge" [class.badge-significant]="ttestResult.significant"
                              [class.badge-not-significant]="!ttestResult.significant">
            {{ ttestResult.significant ? 'SIGNIFICANT (α = 0.05)' : 'NOT SIGNIFICANT' }}
          </span>
        </div>
        <p class="ttest-summary">
          The distributions on {{ ttestDates[0] }} and {{ ttestDates[1] }} are
          <strong>{{ ttestResult.significant ? 'statistically different' : 'not statistically different' }}</strong>.
        </p>
      </div>
    }
  </main>
</div>
```

- [ ] **Step 3: Create `event-counts.component.scss`**

```scss
.counts-layout {
  display: flex;
  height: calc(100vh - 44px);
  overflow: hidden;
}

.control-panel {
  width: 240px;
  min-width: 240px;
  background: #161b22;
  border-right: 1px solid #30363d;
  padding: 20px 16px;
  display: flex;
  flex-direction: column;
  gap: 20px;
  overflow-y: auto;
}

.control-label {
  display: block;
  font-size: 11px;
  font-weight: 600;
  color: #8b949e;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  margin-bottom: 6px;
}

.control-group {
  display: flex;
  flex-direction: column;
}

.control-select,
.control-input {
  background: #21262d;
  border: 1px solid #30363d;
  border-radius: 6px;
  padding: 6px 10px;
  font-size: 13px;
  color: #e6edf3;
  width: 100%;

  &:focus {
    outline: none;
    border-color: #58a6ff;
  }
}

.date-chips {
  display: flex;
  flex-direction: column;
  gap: 4px;
  margin-bottom: 6px;
}

.date-chip {
  display: flex;
  justify-content: space-between;
  align-items: center;
  background: #21262d;
  border: 1px solid;
  border-radius: 6px;
  padding: 4px 10px;
  font-size: 12px;
}

.remove-btn {
  background: none;
  border: none;
  color: #8b949e;
  cursor: pointer;
  font-size: 11px;
  padding: 0 0 0 8px;
  &:hover { color: #f85149; }
}

.add-date-row {
  display: flex;
  gap: 6px;
  align-items: center;

  .control-input { flex: 1; }
}

.radio-label {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 13px;
  color: #e6edf3;
  cursor: pointer;
  padding: 3px 0;
  input { accent-color: #58a6ff; }
}

.btn-primary {
  margin-top: auto;
  background: #238636;
  border: 1px solid #2ea043;
  border-radius: 6px;
  padding: 8px;
  font-size: 13px;
  color: #fff;
  cursor: pointer;
  font-weight: 500;
  width: 100%;

  &:hover { background: #2ea043; }
  &:disabled { opacity: 0.5; cursor: not-allowed; }
}

.btn-secondary {
  background: #21262d;
  border: 1px solid #30363d;
  border-radius: 6px;
  padding: 6px 10px;
  font-size: 13px;
  color: #e6edf3;
  cursor: pointer;
  white-space: nowrap;
  &:hover { background: #30363d; }
}

.error-msg {
  font-size: 12px;
  color: #f85149;
  margin: 0;
}

.chart-area {
  flex: 1;
  display: flex;
  flex-direction: column;
  padding: 16px;
  gap: 12px;
  overflow-y: auto;
  background: #0d1117;
}

.plotly-chart {
  flex: 1;
  min-height: 300px;
  background: #161b22;
  border: 1px solid #30363d;
  border-radius: 8px;
}

.ttest-panel {
  background: #161b22;
  border: 1px solid #30363d;
  border-radius: 8px;
  padding: 14px 16px;
}

.ttest-title {
  font-size: 11px;
  font-weight: 600;
  color: #8b949e;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  margin-bottom: 10px;
}

.ttest-stats {
  display: flex;
  flex-wrap: wrap;
  gap: 20px;
  align-items: center;
  font-size: 13px;
  color: #e6edf3;
  margin-bottom: 8px;
}

.stat-label {
  color: #8b949e;
  margin-right: 4px;
}

.significant { color: #f85149; font-weight: 700; }
.not-significant { color: #3fb950; font-weight: 700; }

.badge {
  display: inline-block;
  border-radius: 4px;
  padding: 2px 10px;
  font-size: 11px;
  font-weight: 600;
  border: 1px solid;
}

.badge-significant {
  background: #3d1f1f;
  border-color: #f85149;
  color: #f85149;
}

.badge-not-significant {
  background: #1a2d1f;
  border-color: #3fb950;
  color: #3fb950;
}

.ttest-summary {
  font-size: 12px;
  color: #8b949e;
  margin: 0;
}
```

- [ ] **Step 4: Build the Angular app to verify no compile errors**

```bash
cd ui/chain-search
npm run build
```

Expected: Build succeeds with no TypeScript errors.

- [ ] **Step 5: Commit**

```bash
git add ui/chain-search/src/app/components/event-counts/
git commit -m "feat: add EventCountsComponent with Plotly chart and T-test panel"
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Covered by |
|---|---|
| Event name dropdown | Task 11 component + Task 6 GET /events/names |
| Multi-date selector | Task 11 component |
| Bucket granularity 1s–1m | Task 11 component + Task 6 request |
| Count / rolling avg / cumulative sum | Task 5 EventCountsService + Task 11 |
| One line per date, different colour | Task 11 renderChart |
| Midnight-to-midnight time axis | Handled: bucket_time is time-of-day string |
| T-test server-side | Task 2 StatsService + Task 3 endpoint |
| T-test shown for 2+ dates | Task 11 runTTest + template condition |
| T-test p-value, t-stat, df, badge | Task 11 ttest-panel |
| Fixed nav bar | Task 8 NavComponent + Task 9 app shell |
| All 5 nav items | Task 8 navItems array |
| Placeholder routes don't 404 | Task 9 routes + PlaceholderComponent |
| POST /events/counts endpoint | Task 6 |
| GET /events/names endpoint | Task 6 |
| POST /stats/ttest endpoint | Task 3 |
| Pydantic models | Task 1 |

All requirements covered. ✓
