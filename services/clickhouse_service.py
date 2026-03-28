import asyncio
import logging
from datetime import datetime
from uuid import UUID

import polars as pl
from clickhouse_connect.driver.client import Client

from schemas.models import Edge, Event, FeatureImportance, MethodResult

logger = logging.getLogger(__name__)


class ClickHouseBatchWriter:
    """Async buffer that accumulates event rows and flushes to ClickHouse in batches.

    Flush triggers: buffer reaches ``max_batch_size`` rows **or**
    ``flush_interval_s`` seconds elapse — whichever comes first.
    All ClickHouse I/O happens in a thread via ``asyncio.to_thread`` so
    the caller never blocks the event loop.
    """

    COLUMN_NAMES = [
        "chain_id",
        "event_name",
        "timestamp",
        "refs",
        "context_keys",
        "context_values",
    ]

    def __init__(
        self,
        client: Client,
        table: str,
        *,
        max_batch_size: int = 500,
        flush_interval_s: float = 0.1,
    ):
        self._client = client
        self._table = table
        self._max_batch_size = max_batch_size
        self._flush_interval_s = flush_interval_s
        self._buffer: list[list] = []
        self._lock = asyncio.Lock()
        self._flush_task: asyncio.Task | None = None

    async def start(self) -> None:
        """Start the periodic flush background task."""
        self._flush_task = asyncio.create_task(self._periodic_flush())

    async def stop(self) -> None:
        """Cancel the background task and flush remaining rows."""
        if self._flush_task is not None:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        async with self._lock:
            if self._buffer:
                await self._do_flush()

    async def append(self, row: list) -> None:
        """Append a single row; triggers immediate flush if buffer is full."""
        async with self._lock:
            self._buffer.append(row)
            if len(self._buffer) >= self._max_batch_size:
                await self._do_flush()

    async def _periodic_flush(self) -> None:
        """Background loop that flushes on a timer."""
        while True:
            await asyncio.sleep(self._flush_interval_s)
            async with self._lock:
                if self._buffer:
                    await self._do_flush()

    async def _do_flush(self) -> None:
        """Send buffered rows to ClickHouse in a thread (must hold ``_lock``)."""
        batch = self._buffer
        self._buffer = []
        try:
            await asyncio.to_thread(
                self._client.insert,
                self._table,
                batch,
                column_names=self.COLUMN_NAMES,
            )
        except Exception:
            logger.exception(
                "ClickHouse batch insert failed — %d rows dropped", len(batch)
            )

    @property
    def pending(self) -> int:
        """Number of rows waiting to be flushed (non-locking, approximate)."""
        return len(self._buffer)

CREATE_EVENTS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS {database}.events (
    event_id        UUID DEFAULT generateUUIDv4(),
    chain_id        String DEFAULT '',
    event_name      LowCardinality(String),
    timestamp       DateTime64(3),
    refs            Array(Tuple(
                        type LowCardinality(String),
                        id   String,
                        ver  UInt16
                    )),
    context_keys    Array(LowCardinality(String)),
    context_values  Array(String),
    ingested_at     DateTime64(3) DEFAULT now64(3)
) ENGINE = MergeTree()
PARTITION BY toYYYYMMDD(timestamp)
ORDER BY (event_name, timestamp, event_id)
TTL toDateTime(timestamp) + INTERVAL 90 DAY
SETTINGS index_granularity = 8192
"""

CREATE_PROFILES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS {database}.path_profiles (
    computed_at     DateTime64(3) DEFAULT now64(3),
    profile_id      Int64,
    node_set        Array(LowCardinality(String)),
    terminal_nodes  Array(LowCardinality(String)),
    chain_count     UInt64,
    fraction        Float64
) ENGINE = MergeTree()
ORDER BY (profile_id)
"""

CREATE_CLASSIFIER_MODEL_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS {database}.classifier_model (
    computed_at          DateTime64(3) DEFAULT now64(3),
    model_bytes          String,
    model_name           LowCardinality(String) DEFAULT '',
    model_params         String DEFAULT '{{}}',
    feature_importances  Array(Tuple(String, Float64)) DEFAULT [],
    method_name          LowCardinality(String) DEFAULT '',
    accuracy             Float64 DEFAULT 0
) ENGINE = MergeTree()
ORDER BY (computed_at)
"""

CREATE_ADJACENCY_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS {database}.adjacency_edges (
    run_id          UUID,
    computed_at     DateTime64(3) DEFAULT now64(3),
    method          LowCardinality(String),
    max_pval        Float64,
    source          LowCardinality(String),
    target          LowCardinality(String),
    correlation     Float64,
    p_value         Float64,
    mean_delta_ms   Nullable(Float64),
    std_delta_ms    Nullable(Float64),
    max_delta_ms    Nullable(Float64),
    min_delta_ms    Nullable(Float64),
    sample_count    UInt64
) ENGINE = MergeTree()
ORDER BY (run_id, source, target)
"""


CREATE_EVENT_REFS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS {database}.event_refs (
    ref_type      LowCardinality(String),
    ref_id        String,
    ref_ver       UInt16,
    chain_id      String,
    event_name    LowCardinality(String),
    timestamp     DateTime64(3)
) ENGINE = MergeTree()
PARTITION BY toYYYYMMDD(timestamp)
ORDER BY (ref_id, timestamp, chain_id)
TTL toDateTime(timestamp) + INTERVAL 90 DAY
SETTINGS index_granularity = 8192
"""

CREATE_EVENT_REFS_MV_SQL = """
CREATE MATERIALIZED VIEW IF NOT EXISTS {database}.event_refs_mv
TO {database}.event_refs AS
SELECT
    ref.1 AS ref_type,
    ref.2 AS ref_id,
    ref.3 AS ref_ver,
    chain_id,
    event_name,
    timestamp
FROM {database}.events
ARRAY JOIN refs AS ref
"""

CREATE_STATE_DETECTOR_MODEL_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS {database}.state_detector_model (
    computed_at    DateTime64(3) DEFAULT now64(3),
    model_bytes    String,
    model_name     LowCardinality(String) DEFAULT '',
    model_params   String DEFAULT '{{}}',
    method_name    LowCardinality(String) DEFAULT '',
    start_time     String DEFAULT '',
    end_time       String DEFAULT ''
) ENGINE = MergeTree()
ORDER BY (computed_at)
"""


class ClickHouseService:
    def __init__(self, client: Client, database: str = "arestor"):
        self.client = client
        self.database = database

    def ensure_table(self) -> None:
        """Create the arestor database and events table if they don't exist."""
        self.client.command(f"CREATE DATABASE IF NOT EXISTS {self.database}")
        self.client.command(CREATE_EVENTS_TABLE_SQL.format(database=self.database))
        # Add chain_id column to existing tables that lack it
        self.client.command(
            f"ALTER TABLE {self.database}.events "
            "ADD COLUMN IF NOT EXISTS chain_id String DEFAULT '' AFTER event_id"
        )
        # Bloom filter skip-index on chain_id for faster analytical queries
        try:
            self.client.command(
                f"ALTER TABLE {self.database}.events "
                "ADD INDEX IF NOT EXISTS idx_chain_id chain_id "
                "TYPE bloom_filter GRANULARITY 4"
            )
        except Exception:
            pass  # index may already exist or ALTER not supported on this version

    def ensure_adjacency_table(self) -> None:
        """Create the adjacency_edges table if it doesn't exist."""
        self.client.command(CREATE_ADJACENCY_TABLE_SQL.format(database=self.database))

    def ensure_profiles_table(self) -> None:
        """Create the path_profiles table if it doesn't exist."""
        self.client.command(CREATE_PROFILES_TABLE_SQL.format(database=self.database))

    def ensure_classifier_model_table(self) -> None:
        """Create the classifier_model table if it doesn't exist."""
        self.client.command(CREATE_CLASSIFIER_MODEL_TABLE_SQL.format(database=self.database))
        for col_ddl in (
            f"ALTER TABLE {self.database}.classifier_model"
            f" ADD COLUMN IF NOT EXISTS model_name LowCardinality(String) DEFAULT ''",
            f"ALTER TABLE {self.database}.classifier_model"
            f" ADD COLUMN IF NOT EXISTS model_params String DEFAULT '{{}}'",
            f"ALTER TABLE {self.database}.classifier_model"
            f" ADD COLUMN IF NOT EXISTS feature_importances"
            f" Array(Tuple(String, Float64)) DEFAULT []",
            f"ALTER TABLE {self.database}.classifier_model"
            f" ADD COLUMN IF NOT EXISTS method_name LowCardinality(String) DEFAULT ''",
            f"ALTER TABLE {self.database}.classifier_model"
            f" ADD COLUMN IF NOT EXISTS accuracy Float64 DEFAULT 0",
        ):
            self.client.command(col_ddl)

    def ensure_state_detector_model_table(self) -> None:
        """Create the state_detector_model table if it doesn't exist."""
        self.client.command(CREATE_STATE_DETECTOR_MODEL_TABLE_SQL.format(database=self.database))

    def ensure_event_refs_table(self) -> None:
        """Create the event_refs table and materialized view if they don't exist."""
        self.client.command(CREATE_EVENT_REFS_TABLE_SQL.format(database=self.database))
        self.client.command(CREATE_EVENT_REFS_MV_SQL.format(database=self.database))

    def backfill_event_refs(self) -> int:
        """One-time backfill of event_refs from existing events. Returns rows inserted."""
        query = (
            f"INSERT INTO {self.database}.event_refs "
            f"SELECT ref.1, ref.2, ref.3, chain_id, event_name, timestamp "
            f"FROM {self.database}.events ARRAY JOIN refs AS ref"
        )
        self.client.command(query)
        count = self.client.query(
            f"SELECT count() FROM {self.database}.event_refs"
        ).result_rows[0][0]
        return count

    @staticmethod
    def parse_concat_ref(ref: str) -> tuple[str, str, int]:
        """Parse a concatenated ref 'type_id_ver' into (type, id, ver).

        The ref type is always a single segment before the first underscore,
        and the version is the last segment after the final underscore.
        Everything in between is the id (which may contain underscores).
        """
        first_sep = ref.index("_")
        last_sep = ref.rindex("_")
        return ref[:first_sep], ref[first_sep + 1:last_sep], int(ref[last_sep + 1:])

    def search_ref_ids(self, prefix: str, limit: int = 20) -> list[str]:
        """Return distinct ref IDs matching a prefix (autocomplete)."""
        query = (
            f"SELECT DISTINCT ref_id FROM {self.database}.event_refs "
            f"WHERE startsWith(ref_id, {{prefix:String}}) "
            f"ORDER BY ref_id LIMIT {{limit:UInt32}}"
        )
        result = self.client.query(
            query, parameters={"prefix": prefix, "limit": limit}
        )
        return [row[0] for row in result.result_rows]

    def search_chains_by_ref_id(self, ref_id: str, limit: int = 100) -> list[str]:
        """Return chain IDs containing an exact ref ID."""
        query = (
            f"SELECT DISTINCT chain_id FROM {self.database}.event_refs "
            f"WHERE ref_id = {{ref_id:String}} "
            f"ORDER BY chain_id LIMIT {{limit:UInt32}}"
        )
        result = self.client.query(
            query, parameters={"ref_id": ref_id, "limit": limit}
        )
        return [row[0] for row in result.result_rows]

    def search_chains_by_ref_prefix(self, prefix: str, limit: int = 100) -> list[str]:
        """Return chain IDs containing refs whose ID matches a prefix."""
        query = (
            f"SELECT DISTINCT chain_id FROM {self.database}.event_refs "
            f"WHERE startsWith(ref_id, {{prefix:String}}) "
            f"ORDER BY chain_id LIMIT {{limit:UInt32}}"
        )
        result = self.client.query(
            query, parameters={"prefix": prefix, "limit": limit}
        )
        return [row[0] for row in result.result_rows]

    def truncate_events(self) -> int:
        """Truncate the events table. Returns the row count before truncation."""
        count = self.client.query(f"SELECT count() FROM {self.database}.events").result_rows[0][0]
        self.client.command(f"TRUNCATE TABLE {self.database}.events")
        return count

    def insert_event(self, event: Event, chain_id: str = "") -> None:
        """Insert a single event into ClickHouse."""
        ts = datetime.fromisoformat(event.Timestamp)
        refs = [(ref.type, ref.id, ref.ver) for ref in event.Refs]
        context_keys = list(event.Context.keys())
        context_values = [str(v) for v in event.Context.values()]

        self.client.insert(
            f"{self.database}.events",
            [[chain_id, event.EventName, ts, refs, context_keys, context_values]],
            column_names=[
                "chain_id",
                "event_name",
                "timestamp",
                "refs",
                "context_keys",
                "context_values",
            ],
        )

    def query_timestamp_matrix(self) -> pl.DataFrame:
        """Query events from ClickHouse and return a wide Polars DataFrame.

        Returns a pivoted DataFrame: rows = chains, columns = event names,
        values = timestamps as float64 (ms since epoch).
        """
        query = (
            f"SELECT chain_id, event_name, timestamp "
            f"FROM {self.database}.events "
            f"WHERE chain_id != ''"
        )
        arrow_table = self.client.query_arrow(query, use_strings=True)
        df = pl.from_arrow(arrow_table)

        if df.is_empty():
            return df

        # Convert timestamp to float64 ms since epoch for numeric correlation
        df = df.with_columns(
            pl.col("timestamp").cast(pl.Datetime("ms")).dt.epoch("ms").cast(pl.Float64)
        )

        # Pivot: rows=chain_id, columns=event_name, values=timestamp
        matrix = df.pivot(on="event_name", index="chain_id", values="timestamp")
        return matrix

    def query_chain_features(self) -> pl.DataFrame:
        """Query per-chain feature summary: event names and context keys.

        Returns a DataFrame with columns: chain_id, events (list[str]),
        ctx_keys (list[str]).
        """
        query = (
            f"SELECT chain_id, "
            f"groupArray(DISTINCT event_name) AS events, "
            f"arrayDistinct(groupArrayArray(context_keys)) AS ctx_keys "
            f"FROM {self.database}.events "
            f"WHERE chain_id != '' "
            f"GROUP BY chain_id"
        )
        arrow_table = self.client.query_arrow(query, use_strings=True)
        return pl.from_arrow(arrow_table)

    def query_adjacency(self) -> dict | None:
        """Fetch all adjacency edges from ClickHouse.

        The table is truncated before each insert, so the full table
        content represents the single latest result.
        """
        query = (
            f"SELECT toString(run_id) AS run_id, method, max_pval, "
            f"source, target, correlation, p_value, "
            f"mean_delta_ms, std_delta_ms, max_delta_ms, min_delta_ms, sample_count "
            f"FROM {self.database}.adjacency_edges"
        )
        arrow_table = self.client.query_arrow(query, use_strings=True)
        df = pl.from_arrow(arrow_table)
        if df.is_empty():
            return None
        meta = df.row(0, named=True)
        edges = [
            Edge(
                source=row["source"],
                target=row["target"],
                correlation=row["correlation"],
                p_value=row["p_value"],
                mean_delta_ms=row["mean_delta_ms"],
                std_delta_ms=row["std_delta_ms"],
                max_delta_ms=row["max_delta_ms"],
                min_delta_ms=row["min_delta_ms"],
                sample_count=row["sample_count"],
            )
            for row in df.iter_rows(named=True)
        ]
        return {
            "run_id": meta["run_id"],
            "method": meta["method"],
            "max_pval": meta["max_pval"],
            "edges": edges,
        }

    def query_chain_latencies(self, chain_id: str) -> list[dict]:
        """Return per-edge observed latencies for a single chain.

        Joins the chain's events with adjacency_edges to compute
        target_timestamp - source_timestamp for each inferred edge.
        """
        query = (
            f"SELECT ae.source, ae.target, "
            f"toFloat64(date_diff('millisecond', e1.timestamp, e2.timestamp)) AS delta_ms "
            f"FROM {self.database}.events AS e1 "
            f"INNER JOIN {self.database}.adjacency_edges AS ae "
            f"  ON e1.event_name = ae.source "
            f"INNER JOIN {self.database}.events AS e2 "
            f"  ON e2.chain_id = e1.chain_id AND e2.event_name = ae.target "
            f"WHERE e1.chain_id = {{chain_id:String}}"
        )
        arrow_table = self.client.query_arrow(
            query, parameters={"chain_id": chain_id}, use_strings=True
        )
        return pl.from_arrow(arrow_table).to_dicts()

    def query_chain_latencies_by_ref(self, ref: str) -> list[dict]:
        """Return per-edge observed latencies for all chains containing a ref.

        Uses event_refs for fast chain_id lookup via primary key, then
        computes latencies via the same self-join on events.
        """
        ref_type, ref_id, ref_ver = self.parse_concat_ref(ref)
        query = (
            f"SELECT e1.chain_id, ae.source, ae.target, "
            f"toFloat64(date_diff('millisecond', e1.timestamp, e2.timestamp)) AS delta_ms "
            f"FROM {self.database}.events AS e1 "
            f"INNER JOIN {self.database}.adjacency_edges AS ae "
            f"  ON e1.event_name = ae.source "
            f"INNER JOIN {self.database}.events AS e2 "
            f"  ON e2.chain_id = e1.chain_id AND e2.event_name = ae.target "
            f"WHERE e1.chain_id IN ("
            f"  SELECT DISTINCT chain_id FROM {self.database}.event_refs "
            f"  WHERE ref_id = {{ref_id:String}}"
            f"    AND ref_type = {{ref_type:String}}"
            f"    AND ref_ver = {{ref_ver:UInt16}}"
            f")"
        )
        arrow_table = self.client.query_arrow(
            query,
            parameters={"ref_id": ref_id, "ref_type": ref_type, "ref_ver": ref_ver},
            use_strings=True,
        )
        return pl.from_arrow(arrow_table).to_dicts()

    def query_chain_id_by_ref(self, ref: str) -> str | None:
        """Resolve a concatenated ref string to the first matching chain_id."""
        ref_type, ref_id, ref_ver = self.parse_concat_ref(ref)
        query = (
            f"SELECT DISTINCT chain_id FROM {self.database}.event_refs "
            f"WHERE ref_id = {{ref_id:String}}"
            f"  AND ref_type = {{ref_type:String}}"
            f"  AND ref_ver = {{ref_ver:UInt16}}"
            f" LIMIT 1"
        )
        result = self.client.query(
            query,
            parameters={"ref_id": ref_id, "ref_type": ref_type, "ref_ver": ref_ver},
        )
        if result.result_rows:
            return result.result_rows[0][0]
        return None

    def query_chain_node_set(self, chain_id: str) -> list[str]:
        """Return the sorted unique event names for a chain."""
        query = (
            f"SELECT arraySort(groupUniqArray(event_name)) AS node_set "
            f"FROM {self.database}.events "
            f"WHERE chain_id = {{chain_id:String}}"
        )
        result = self.client.query(query, parameters={"chain_id": chain_id})
        if result.result_rows and result.result_rows[0][0]:
            return result.result_rows[0][0]
        return []

    def query_average_latencies(
        self,
        chain_id: str,
        start: datetime,
        end: datetime | None = None,
    ) -> tuple[list[dict], int]:
        """Aggregate latency stats across chains sharing the same profile.

        Returns (edge_stats_rows, matching_chain_count).
        When *end* is None the time window is open-ended (no upper bound).
        """
        time_filter = f"WHERE timestamp >= {{start:DateTime64(3)}}"
        if end is not None:
            time_filter += f" AND timestamp <= {{end:DateTime64(3)}}"

        cte = (
            f"WITH target_node_set AS ("
            f"  SELECT arraySort(groupUniqArray(event_name)) AS node_set"
            f"  FROM {self.database}.events"
            f"  WHERE chain_id = {{chain_id:String}}"
            f"), "
            f"matching_chains AS ("
            f"  SELECT chain_id"
            f"  FROM {self.database}.events"
            f"  {time_filter}"
            f"  GROUP BY chain_id"
            f"  HAVING arraySort(groupUniqArray(event_name)) = "
            f"    (SELECT node_set FROM target_node_set)"
            f") "
        )

        stats_query = (
            f"{cte}"
            f"SELECT source, target, "
            f"  avg(delta_ms) AS avg_ms, "
            f"  stddevPop(delta_ms) AS stddev_ms, "
            f"  min(delta_ms) AS min_ms, "
            f"  max(delta_ms) AS max_ms, "
            f"  quantile(0.05)(delta_ms) AS p5_ms, "
            f"  quantile(0.50)(delta_ms) AS p50_ms, "
            f"  quantile(0.95)(delta_ms) AS p95_ms, "
            f"  count() AS sample_count, "
            f"  (SELECT count() FROM matching_chains) AS matching_chains "
            f"FROM ("
            f"  SELECT ae.source, ae.target, "
            f"    toFloat64(date_diff('millisecond', e1.timestamp, e2.timestamp)) AS delta_ms "
            f"  FROM {self.database}.events AS e1 "
            f"  INNER JOIN {self.database}.adjacency_edges AS ae "
            f"    ON e1.event_name = ae.source "
            f"  INNER JOIN {self.database}.events AS e2 "
            f"    ON e2.chain_id = e1.chain_id AND e2.event_name = ae.target "
            f"  WHERE e1.chain_id IN (SELECT chain_id FROM matching_chains)"
            f") "
            f"GROUP BY source, target "
            f"ORDER BY source, target"
        )

        params: dict = {"chain_id": chain_id, "start": start}
        if end is not None:
            params["end"] = end

        arrow_table = self.client.query_arrow(
            stats_query, parameters=params, use_strings=True
        )
        df = pl.from_arrow(arrow_table)
        if df.is_empty():
            return [], 0

        matching_count = int(df["matching_chains"][0])
        rows = df.drop("matching_chains").to_dicts()
        return rows, matching_count

    def truncate_adjacency(self) -> int:
        """Truncate the adjacency_edges table. Returns the row count before truncation."""
        count = self.client.query(f"SELECT count() FROM {self.database}.adjacency_edges").result_rows[0][0]
        self.client.command(f"TRUNCATE TABLE IF EXISTS {self.database}.adjacency_edges")
        return count

    def insert_adjacency_result(
        self,
        run_id: UUID,
        edges: list[Edge],
        method: str,
        max_pval: float,
    ) -> None:
        """Truncate and insert adjacency edges into ClickHouse."""
        self.client.command(
            f"TRUNCATE TABLE IF EXISTS {self.database}.adjacency_edges"
        )
        if not edges:
            return

        rows = [
            [
                str(run_id),
                method,
                max_pval,
                e.source,
                e.target,
                e.correlation,
                e.p_value,
                e.mean_delta_ms,
                e.std_delta_ms,
                e.max_delta_ms,
                e.min_delta_ms,
                e.sample_count,
            ]
            for e in edges
        ]
        self.client.insert(
            f"{self.database}.adjacency_edges",
            rows,
            column_names=[
                "run_id",
                "method",
                "max_pval",
                "source",
                "target",
                "correlation",
                "p_value",
                "mean_delta_ms",
                "std_delta_ms",
                "max_delta_ms",
                "min_delta_ms",
                "sample_count",
            ],
        )

    def insert_classification_result(self, profiles: list) -> None:
        """Truncate and insert path profiles into ClickHouse."""
        self.client.command(
            f"TRUNCATE TABLE IF EXISTS {self.database}.path_profiles"
        )
        if not profiles:
            return

        rows = [
            [
                p.profile_id,
                sorted(p.node_set),
                sorted(p.terminal_nodes),
                p.chain_count,
                p.fraction,
            ]
            for p in profiles
        ]
        self.client.insert(
            f"{self.database}.path_profiles",
            rows,
            column_names=[
                "profile_id",
                "node_set",
                "terminal_nodes",
                "chain_count",
                "fraction",
            ],
        )

    def query_path_profiles(self) -> list:
        """Load persisted path profiles from ClickHouse."""
        from schemas.models import PathProfile

        query = (
            f"SELECT profile_id, node_set, terminal_nodes, chain_count, fraction "
            f"FROM {self.database}.path_profiles"
        )
        try:
            arrow_table = self.client.query_arrow(query, use_strings=True)
        except Exception:
            return []
        df = pl.from_arrow(arrow_table)
        if df.is_empty():
            return []
        return [
            PathProfile(
                profile_id=row["profile_id"],
                node_set=frozenset(row["node_set"]),
                terminal_nodes=frozenset(row["terminal_nodes"]),
                chain_count=row["chain_count"],
                fraction=row["fraction"],
            )
            for row in df.iter_rows(named=True)
        ]

    def insert_classifier_model(
        self,
        model_bytes: bytes,
        model_name: str = "",
        model_params: str = "{}",
        feature_importances: list[tuple[str, float]] | None = None,
        method_name: str = "",
        accuracy: float = 0.0,
    ) -> None:
        """Truncate and insert the serialized classifier model with metadata."""
        import base64

        self.client.command(
            f"TRUNCATE TABLE IF EXISTS {self.database}.classifier_model"
        )
        encoded = base64.b64encode(model_bytes).decode("ascii")
        self.client.insert(
            f"{self.database}.classifier_model",
            [[encoded, model_name, model_params, feature_importances or [], method_name, accuracy]],
            column_names=["model_bytes", "model_name", "model_params", "feature_importances", "method_name", "accuracy"],
            settings={"async_insert": 0, "wait_for_async_insert": 1},
        )

    def query_chains_for_cache(self) -> list[dict]:
        """Reconstruct event chain documents from ClickHouse for Redis cache loading.

        Groups events by chain_id and rebuilds the chain JSON structure.
        Returns all chains — termination filtering is done by the caller.
        """
        query = (
            f"SELECT chain_id, "
            f"groupArray(event_name) AS event_names, "
            f"groupArray(toString(timestamp)) AS timestamps, "
            f"arrayMap(x -> concat(x.1, '_', x.2, '_', toString(x.3)), "
            f"  arrayDistinct(groupArrayArray(refs))) AS concat_refs, "
            f"arrayDistinct(groupArrayArray(context_keys)) AS ctx_keys, "
            f"arrayDistinct(groupArrayArray(context_values)) AS ctx_values "
            f"FROM {self.database}.events "
            f"WHERE chain_id != '' "
            f"GROUP BY chain_id"
        )
        arrow_table = self.client.query_arrow(query, use_strings=True)
        df = pl.from_arrow(arrow_table)
        if df.is_empty():
            return []

        chains: list[dict] = []
        for row in df.iter_rows(named=True):
            timestamps = dict(zip(row["event_names"], row["timestamps"]))
            ctx_keys = row["ctx_keys"] or []
            ctx_values = row["ctx_values"] or []
            context = dict(zip(ctx_keys, ctx_values)) if ctx_keys else {}
            chains.append({
                "chain_id": row["chain_id"],
                "concatenatedrefs": list(row["concat_refs"]),
                "timestamps": timestamps,
                "context": context,
                "complete": False,
                "terminated": False,
            })
        return chains

    def query_chain_by_id(self, chain_id: str) -> dict | None:
        """Fetch a single event chain by chain_id from ClickHouse."""
        query = (
            f"SELECT chain_id, "
            f"groupArray(event_name) AS event_names, "
            f"groupArray(toString(timestamp)) AS timestamps, "
            f"arrayMap(x -> concat(x.1, '_', x.2, '_', toString(x.3)), "
            f"  arrayDistinct(groupArrayArray(refs))) AS concat_refs, "
            f"arrayDistinct(groupArrayArray(context_keys)) AS ctx_keys, "
            f"arrayDistinct(groupArrayArray(context_values)) AS ctx_values "
            f"FROM {self.database}.events "
            f"WHERE chain_id = {{chain_id:String}} "
            f"GROUP BY chain_id"
        )
        arrow_table = self.client.query_arrow(
            query, parameters={"chain_id": chain_id}, use_strings=True,
        )
        df = pl.from_arrow(arrow_table)
        if df.is_empty():
            return None
        row = df.row(0, named=True)
        ctx_keys = row["ctx_keys"] or []
        ctx_values = row["ctx_values"] or []
        return {
            "chain_id": row["chain_id"],
            "concatenatedrefs": list(row["concat_refs"]),
            "timestamps": dict(zip(row["event_names"], row["timestamps"])),
            "context": dict(zip(ctx_keys, ctx_values)) if ctx_keys else {},
            "complete": False,
            "terminated": False,
        }

    def query_classifier_model(self) -> bytes | None:
        """Load the most recent serialized classifier model."""
        import base64

        query = (
            f"SELECT model_bytes FROM {self.database}.classifier_model "
            f"ORDER BY computed_at DESC LIMIT 1"
        )
        try:
            arrow_table = self.client.query_arrow(query, use_strings=True)
        except Exception:
            return None
        df = pl.from_arrow(arrow_table)
        if df.is_empty():
            return None
        encoded = df.row(0)[0]
        return base64.b64decode(encoded)

    def query_classifier_model_metadata(self) -> MethodResult | None:
        """Load the most recent classifier model metadata (no binary blob)."""
        query = (
            f"SELECT method_name, accuracy, feature_importances "
            f"FROM {self.database}.classifier_model "
            f"ORDER BY computed_at DESC LIMIT 1"
        )
        try:
            result = self.client.query(query)
        except Exception:
            return None
        if not result.result_rows:
            return None
        method_name, accuracy, raw_importances = result.result_rows[0]
        feature_importances = [
            FeatureImportance(feature_name=name, importance=importance)
            for name, importance in raw_importances
        ]
        return MethodResult(
            method=method_name,
            accuracy=accuracy,
            feature_importances=feature_importances,
        )

    # ------------------------------------------------------------------
    # State detector model persistence
    # ------------------------------------------------------------------

    def query_per_chain_edge_latencies(
        self,
        start: datetime,
        end: datetime | None = None,
    ) -> pl.DataFrame:
        """Return per-chain per-edge raw latencies for all chains in the time window.

        Columns: [chain_id, source, target, delta_ms].
        """
        time_filter = "WHERE e1.timestamp >= {start:DateTime64(3)}"
        if end is not None:
            time_filter += " AND e1.timestamp <= {end:DateTime64(3)}"

        query = (
            f"SELECT e1.chain_id AS chain_id, ae.source, ae.target, "
            f"  toFloat64(date_diff('millisecond', e1.timestamp, e2.timestamp)) AS delta_ms "
            f"FROM {self.database}.events AS e1 "
            f"INNER JOIN {self.database}.adjacency_edges AS ae "
            f"  ON e1.event_name = ae.source "
            f"INNER JOIN {self.database}.events AS e2 "
            f"  ON e2.chain_id = e1.chain_id AND e2.event_name = ae.target "
            f"{time_filter}"
        )
        params: dict = {"start": start}
        if end is not None:
            params["end"] = end

        arrow_table = self.client.query_arrow(query, parameters=params, use_strings=True)
        return pl.from_arrow(arrow_table)

    def query_chain_node_sets(
        self,
        start: datetime,
        end: datetime | None = None,
    ) -> pl.DataFrame:
        """Return [chain_id, node_set] for all chains in the time window."""
        time_filter = "WHERE timestamp >= {start:DateTime64(3)}"
        if end is not None:
            time_filter += " AND timestamp <= {end:DateTime64(3)}"

        query = (
            f"SELECT chain_id, arraySort(groupUniqArray(event_name)) AS node_set "
            f"FROM {self.database}.events "
            f"{time_filter} "
            f"GROUP BY chain_id"
        )
        params: dict = {"start": start}
        if end is not None:
            params["end"] = end

        arrow_table = self.client.query_arrow(query, parameters=params, use_strings=True)
        return pl.from_arrow(arrow_table)

    def insert_state_detector_model(
        self,
        model_bytes: bytes,
        model_name: str = "",
        model_params: str = "{}",
        method_name: str = "",
        start_time: str = "",
        end_time: str = "",
    ) -> None:
        """Truncate and insert the serialized state detector model."""
        import base64

        self.client.command(
            f"TRUNCATE TABLE IF EXISTS {self.database}.state_detector_model"
        )
        encoded = base64.b64encode(model_bytes).decode("ascii")
        self.client.insert(
            f"{self.database}.state_detector_model",
            [[encoded, model_name, model_params, method_name, start_time, end_time]],
            column_names=["model_bytes", "model_name", "model_params", "method_name", "start_time", "end_time"],
            settings={"async_insert": 0, "wait_for_async_insert": 1},
        )

    def query_state_detector_model(self) -> bytes | None:
        """Load the most recent serialized state detector model."""
        import base64

        query = (
            f"SELECT model_bytes FROM {self.database}.state_detector_model "
            f"ORDER BY computed_at DESC LIMIT 1"
        )
        try:
            arrow_table = self.client.query_arrow(query, use_strings=True)
        except Exception:
            return None
        df = pl.from_arrow(arrow_table)
        if df.is_empty():
            return None
        encoded = df.row(0)[0]
        return base64.b64decode(encoded)

    def query_state_detector_model_metadata(self) -> dict | None:
        """Load the most recent state detector model metadata (no binary blob)."""
        query = (
            f"SELECT method_name, start_time, end_time "
            f"FROM {self.database}.state_detector_model "
            f"ORDER BY computed_at DESC LIMIT 1"
        )
        try:
            result = self.client.query(query)
        except Exception:
            return None
        if not result.result_rows:
            return None
        method_name, start_time, end_time = result.result_rows[0]
        return {
            "method_name": method_name,
            "start_time": start_time,
            "end_time": end_time,
        }

    def get_distinct_event_names(self) -> list[str]:
        result = self.client.query(
            f"SELECT DISTINCT event_name FROM {self.database}.events ORDER BY event_name"
        )
        return [row[0] for row in result.result_rows]

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
        result = self.client.query(
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
