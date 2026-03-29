from datetime import datetime

from schemas.models import (
    AverageLatencyResponse,
    ChainLatency,
    ChainLatencyResponse,
    EdgeLatencyStats,
    LatencyBucket,
    LatencyDateSeries,
    LatencyTimeseriesResponse,
)
from services.clickhouse_service import ClickHouseService


class LatencyService:
    def __init__(self, ch_svc: ClickHouseService):
        self.ch_svc = ch_svc

    def get(
        self,
        chain_id: str | None = None,
        ref: str | None = None,
    ) -> list[ChainLatencyResponse]:
        """Return per-edge observed latencies for chains matching the query.

        At least one of *chain_id* or *ref* must be provided.  When
        *chain_id* is given the lookup uses the bloom-filter skip-index
        for a fast point query.  When *ref* is given a subquery scans
        the refs array to find matching chain_ids.
        """
        if chain_id is not None:
            rows = self.ch_svc.query_chain_latencies(chain_id)
        else:
            # ref must be set (router validates at least one param)
            rows = self.ch_svc.query_chain_latencies_by_ref(ref)  # type: ignore[arg-type]

        if not rows:
            return []

        # Group rows by chain_id (single-chain query still uses grouping
        # for a uniform code path).
        by_chain: dict[str, list[ChainLatency]] = {}
        for r in rows:
            cid = r.get("chain_id", chain_id or "")
            by_chain.setdefault(cid, []).append(
                ChainLatency(source=r["source"], target=r["target"], delta_ms=r["delta_ms"])
            )
        return [
            ChainLatencyResponse(chain_id=cid, latencies=lats)
            for cid, lats in by_chain.items()
        ]

    def get_averages(
        self,
        start: datetime,
        end: datetime | None = None,
        chain_id: str | None = None,
        ref: str | None = None,
    ) -> AverageLatencyResponse | None:
        """Aggregate latency stats for chains sharing the same profile path."""
        # Resolve ref → chain_id if needed
        if chain_id is None:
            chain_id = self.ch_svc.query_chain_id_by_ref(ref)  # type: ignore[arg-type]
            if chain_id is None:
                return None

        # Get the chain's node set
        node_set = self.ch_svc.query_chain_node_set(chain_id)
        if not node_set:
            return None

        # Match node_set to a persisted profile_id
        node_set_frozen = frozenset(node_set)
        profile_id = -1
        profiles = self.ch_svc.query_path_profiles()
        for p in profiles:
            if p.node_set == node_set_frozen:
                profile_id = p.profile_id
                break

        # Compute aggregate latencies
        rows, matching_count = self.ch_svc.query_average_latencies(
            chain_id, start, end
        )
        if not rows:
            return None

        edges = [
            EdgeLatencyStats(
                source=r["source"],
                target=r["target"],
                avg_ms=r["avg_ms"],
                stddev_ms=r["stddev_ms"],
                min_ms=r["min_ms"],
                max_ms=r["max_ms"],
                p5_ms=r["p5_ms"],
                p50_ms=r["p50_ms"],
                p95_ms=r["p95_ms"],
                sample_count=int(r["sample_count"]),
            )
            for r in rows
        ]

        return AverageLatencyResponse(
            chain_id=chain_id,
            profile_id=profile_id,
            node_set=node_set,
            matching_chains=matching_count,
            start=start.isoformat(timespec="milliseconds"),
            end=end.isoformat(timespec="milliseconds") if end else "",
            edges=edges,
        )

    def get_timeseries(
        self,
        source_event: str,
        target_event: str,
        dates: list[str],
        bucket_seconds: int,
    ) -> LatencyTimeseriesResponse:
        """Return bucketed latency stats + raw latencies per date for the given event pair."""
        bucketed_df = self.ch_svc.query_latency_timeseries(
            source_event, target_event, dates, bucket_seconds
        )
        raw_rows = self.ch_svc.query_latency_raw(source_event, target_event, dates)

        # Group raw latencies by date
        raw_by_date: dict[str, list[float]] = {}
        for row in raw_rows:
            raw_by_date.setdefault(row["date"], []).append(row["delta_ms"])

        series: list[LatencyDateSeries] = []
        for date_str in sorted({r[0] for r in bucketed_df.iter_rows()} | set(raw_by_date.keys())):
            date_df = bucketed_df.filter(bucketed_df["date"] == date_str).sort("bucket_time")
            buckets = [
                LatencyBucket(
                    time=row["bucket_time"].strftime("%H:%M:%S"),
                    mean_ms=row["mean_ms"],
                    min_ms=row["min_ms"],
                    max_ms=row["max_ms"],
                    p5_ms=row["p5_ms"],
                    p50_ms=row["p50_ms"],
                    p95_ms=row["p95_ms"],
                    event_count=row["event_count"],
                )
                for row in date_df.iter_rows(named=True)
            ]
            series.append(LatencyDateSeries(
                date=date_str,
                buckets=buckets,
                raw_latencies=raw_by_date.get(date_str, []),
            ))

        return LatencyTimeseriesResponse(
            source_event=source_event,
            target_event=target_event,
            series=series,
        )
