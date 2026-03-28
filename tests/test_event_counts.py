from datetime import datetime, timedelta

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
    base = datetime(2026, 3, 27, 0, 0, 0)
    rows = [
        ("2026-03-27", base + timedelta(minutes=i), float(i + 1))
        for i in range(10)
    ]
    df = _make_df(rows)
    response = svc.build_response(df, "rolling_avg")
    buckets = response.series[0].buckets
    # First 6 buckets have fewer than 7 observations — fill_null → 0.0
    for i in range(6):
        assert buckets[i].value == 0.0, f"bucket {i} should be 0.0 (fill_null), got {buckets[i].value}"
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
