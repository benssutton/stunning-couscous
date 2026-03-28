# Event Counts Screen Design

**Date:** 2026-03-28
**Status:** Approved

---

## Overview

A new "Counts" screen that plots a time-series of event counts queried from ClickHouse. Users can compare event volumes across dates, switch between count/rolling-average/cumulative-sum views, and run a two-sample Student's T-test to assess whether two date distributions differ significantly.

This feature introduces a shared top navigation bar across all screens, a new `POST /events/counts` backend endpoint, and a generic `POST /stats/ttest` endpoint for server-side T-test computation.

---

## Navigation Bar

A fixed horizontal nav bar is added to the Angular app shell (`app.component`). It sits above the `<router-outlet>` and is always visible, even when scrolling.

**Nav items (in order):**

| Label | Route |
|---|---|
| Search | `/search` |
| Counts | `/counts` |
| Latencies | `/latencies` (placeholder — not implemented in this task) |
| Real-time Dashboard | `/realtime` (placeholder) |
| Models | `/models` (placeholder) |

The active route is highlighted. Placeholder routes render an empty or "coming soon" view — they must not 404.

---

## Counts Screen Layout

Route: `/counts`
Component: `EventCountsComponent`

Two-column layout:
- **Left panel** (fixed ~230px width): all controls
- **Right area** (flex-grow): chart + T-test result

### Left Panel Controls

| Control | Type | Detail |
|---|---|---|
| Event Name | Dropdown (single select) | Populated by `GET /events/names` — a new lightweight endpoint that returns distinct event names from ClickHouse |
| Dates | Multi-date selector | Each selected date shown as a colour-coded chip with a remove button. An "+ Add date" button opens a date picker. Minimum 1 date, no maximum enforced. |
| Bucket Size | Slider + dropdown | Range 1 second to 60 seconds. Snap points: 1s, 5s, 10s, 15s, 30s, 60s. Display label shows selected value (e.g. "30 seconds"). |
| Metric | Radio group | Options: **Count**, **Rolling Average**, **Cumulative Sum** |
| Refresh | Button | Triggers the API call and re-renders the chart. No auto-refresh. |

### Chart Area

Built with **Plotly.js** (`plotly.js-dist-min` package). One trace per selected date. Each date gets a distinct colour assigned in order from a fixed palette (Plotly's default categorical palette is acceptable).

- X-axis: time of day, always spanning **00:00 – 24:00** for the selected date (i.e. timestamps are normalised to time-of-day so dates overlay on the same axis)
- Y-axis: the selected metric value
- Legend: one entry per date, labelled with the date string (YYYY-MM-DD)
- Hover tooltip: shows time, date, and metric value

### T-Test Panel

Shown only when **2 or more dates** are selected. Positioned beneath the chart.

Runs a **two-sample (independent) Student's T-test** on the per-bucket metric values across the two most recently selected dates. If more than two dates are selected, the test compares the first two dates in the selection.

Displays:
- p-value (4 decimal places)
- t-statistic (2 decimal places)
- Degrees of freedom (integer)
- A badge: **SIGNIFICANT (α = 0.05)** if p < 0.05, otherwise **NOT SIGNIFICANT**
- One-line plain-English summary: "The distributions on {date1} and {date2} are [statistically different / not statistically different]."

The T-test is computed **server-side** via `POST /stats/ttest`. The frontend sends the two bucket value series from the already-fetched counts response and displays the result.

---

## Backend

### `POST /events/counts`

**Summary:** "Create a time-series count of events"

**Request body:**

```json
{
  "event_name": "transaction.received",
  "dates": ["2026-03-27", "2026-03-26"],
  "bucket_seconds": 30,
  "metric": "count"
}
```

| Field | Type | Constraints |
|---|---|---|
| `event_name` | string | Required |
| `dates` | list[string] | ISO date strings (YYYY-MM-DD), 1–N items |
| `bucket_seconds` | int | 1–60 |
| `metric` | enum | `count`, `rolling_avg`, `cumulative_sum` |

**Response body:**

```json
{
  "series": [
    {
      "date": "2026-03-27",
      "buckets": [
        { "time": "00:00:00", "value": 42 },
        { "time": "00:00:30", "value": 37 }
      ]
    }
  ]
}
```

Each bucket's `time` is a time-of-day string (HH:MM:SS) relative to midnight of the date, so the frontend can plot all dates on the same x-axis.

**ClickHouse query strategy:**

- Query the `events` table, filtering by `event_name` and a date range covering all requested dates
- Use `toStartOfInterval(timestamp, INTERVAL {bucket_seconds} SECOND)` for bucketing
- For `count`: `COUNT(*) per bucket`
- For `rolling_avg`: compute in Python/Polars after fetching count series (7-bucket rolling mean)
- For `cumulative_sum`: compute in Python/Polars after fetching count series (`cumsum`)
- Return one series per date

**Router/Service split:**

- Router method (`routers/events.py`): validates request, calls service, returns response
- New method on `ClickHouseService`: `get_event_counts(event_name, dates, bucket_seconds) -> pl.DataFrame` — returns raw counts only
- `EventCountsService` (new): applies the rolling average or cumulative sum transformation and formats the response

### `POST /stats/ttest`

**Summary:** "Run a two-sample Welch's T-test on two series of observations"

Lives in a new router `routers/stats.py` and a new service `services/stats_service.py`. Generic — not coupled to events.

**Request body:**

```json
{
  "series_a": [42, 37, 55, 61, 48],
  "series_b": [38, 31, 50, 44, 39],
  "alpha": 0.05
}
```

| Field | Type | Constraints |
|---|---|---|
| `series_a` | list[float] | ≥ 2 observations |
| `series_b` | list[float] | ≥ 2 observations |
| `alpha` | float | Default `0.05`, range 0–1 exclusive |

**Response body:**

```json
{
  "t_statistic": 3.41,
  "p_value": 0.0032,
  "degrees_of_freedom": 5758,
  "significant": true,
  "alpha": 0.05
}
```

**Implementation:** Uses `scipy.stats.ttest_ind` with `equal_var=False` (Welch's T-test). `scipy` is already available in the conda environment.

**Router/Service split:**

- `routers/stats.py`: validates request, calls service, returns response
- `services/stats_service.py`: wraps `scipy.stats.ttest_ind`, returns a `TTestResult`

---

### `GET /events/names`

Returns distinct event names present in ClickHouse, used to populate the dropdown.

```json
{ "names": ["transaction.received", "transaction.processed", ...] }
```

Implemented as a simple `SELECT DISTINCT event_name FROM events` query on `ClickHouseService`.

---

## Angular Structure

New files:

```
routers/
  events.py                          # add POST /events/counts, GET /events/names
  stats.py                           # new: POST /stats/ttest
services/
  event_counts_service.py            # new: metric transformation logic
  stats_service.py                   # new: Welch's T-test via scipy
ui/chain-search/src/app/
  components/
    nav/
      nav.component.ts               # new: fixed top nav bar
      nav.component.html
      nav.component.scss
    event-counts/
      event-counts.component.ts      # new: Counts screen
      event-counts.component.html
      event-counts.component.scss
  app.routes.ts                      # add /counts route, placeholder routes
  app.html                           # add <app-nav> + <router-outlet>
  services/
    api.service.ts                   # add getEventCounts(), getEventNames(), runTTest()
  models/
    api.models.ts                    # add EventCountsRequest, EventCountsResponse, TTestRequest, TTestResponse
```

**Plotly installation:** `npm install plotly.js-dist-min @types/plotly.js-dist-min` inside `ui/chain-search`.

**Backend wiring:** `main.py` — include `stats.router` with prefix `/stats`.

---

## Pydantic Models

```python
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

---

## Out of Scope

- Auto-refresh / live polling
- Exporting the chart as an image
- Comparing more than two dates in the T-test (always tests first two selected)
- Implementing the Latencies, Real-time Dashboard, or Models screens (nav links rendered but inactive)
- Client-side statistical computation (all stats go through `POST /stats/ttest`)
