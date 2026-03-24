import asyncio
import logging
from datetime import datetime
from uuid import UUID

import polars as pl
from clickhouse_connect.driver.client import Client

from schemas.models import Edge
from schemas.models import Event

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
    computed_at     DateTime64(3) DEFAULT now64(3),
    model_bytes     String
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

    def insert_classifier_model(self, model_bytes: bytes) -> None:
        """Truncate and insert the serialized classifier model."""
        import base64

        self.client.command(
            f"TRUNCATE TABLE IF EXISTS {self.database}.classifier_model"
        )
        encoded = base64.b64encode(model_bytes).decode("ascii")
        self.client.insert(
            f"{self.database}.classifier_model",
            [[encoded]],
            column_names=["model_bytes"],
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
