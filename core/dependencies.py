import logging
from contextlib import asynccontextmanager

import clickhouse_connect
import redis.asyncio as aioredis
from pydantic_settings import BaseSettings

from services.adjacency_service import AdjacencyService
from services.cache_service import CacheService
from services.chain_classifier_service import ChainClassifier
from services.clickhouse_service import ClickHouseBatchWriter, ClickHouseService
from services.event_counts_service import EventCountsService
from services.latency_service import LatencyService
from services.redis_service import RedisService
from services.search_service import SearchService
from services.state_detector_service import StateDetectorService
from services.stats_service import StatsService

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    redis_host: str = "localhost"
    redis_port: int = 6379
    clickhouse_host: str = "localhost"
    clickhouse_port: int = 8123
    clickhouse_database: str = "arestor"
    clickhouse_password: str = "clickhouse"


settings = Settings()

# Module-level service instances (populated during lifespan)
_redis_service: RedisService | None = None
_clickhouse_service: ClickHouseService | None = None
_adjacency_service: AdjacencyService | None = None
_latency_service: LatencyService | None = None
_state_detector_service: StateDetectorService | None = None
_cache_service: CacheService | None = None
_search_service: SearchService | None = None
_batch_writer: ClickHouseBatchWriter | None = None
_redis_pool: aioredis.ConnectionPool | None = None
_clickhouse_client = None


@asynccontextmanager
async def lifespan(app):
    """FastAPI lifespan: create connections on startup, close on shutdown."""
    global _redis_service, _clickhouse_service, _adjacency_service, _latency_service, _state_detector_service, _cache_service, _search_service, _batch_writer, _redis_pool, _clickhouse_client

    # Redis
    _redis_pool = aioredis.BlockingConnectionPool(
        host=settings.redis_host,
        port=settings.redis_port,
        decode_responses=True,
        max_connections=50,
        timeout=5,
    )
    _redis_service = RedisService(_redis_pool)
    await _redis_service.ensure_index()
    _cache_service = CacheService(_redis_service)
    logger.info("Redis connected and index ensured")

    # ClickHouse
    _clickhouse_client = clickhouse_connect.get_client(
        host=settings.clickhouse_host,
        port=settings.clickhouse_port,
        password=settings.clickhouse_password,
        settings={
            "async_insert": 1,
            "wait_for_async_insert": 0,
        },
    )
    _clickhouse_service = ClickHouseService(
        _clickhouse_client, settings.clickhouse_database
    )
    _clickhouse_service.ensure_table()
    _clickhouse_service.ensure_adjacency_table()
    _clickhouse_service.ensure_profiles_table()
    _clickhouse_service.ensure_classifier_model_table()
    _clickhouse_service.ensure_state_detector_model_table()
    _clickhouse_service.ensure_event_refs_table()
    logger.info("ClickHouse connected and tables ensured")

    # Batch writer for event inserts
    _batch_writer = ClickHouseBatchWriter(
        _clickhouse_client,
        f"{settings.clickhouse_database}.events",
        max_batch_size=500,
        flush_interval_s=0.1,
    )
    await _batch_writer.start()
    logger.info("ClickHouse batch writer started")

    # Load persisted path profiles into RedisService
    profiles = _clickhouse_service.query_path_profiles()
    if profiles:
        _redis_service.set_path_profiles(profiles)
        logger.info("Loaded %d path profiles from ClickHouse", len(profiles))

    # Load persisted classifier model
    model_bytes = _clickhouse_service.query_classifier_model()
    if model_bytes:
        predictor = ChainClassifier.deserialize(model_bytes)
        _redis_service.set_predictor(predictor)
        logger.info("Loaded classifier model with %d profiles", len(predictor.profiles))

    # Adjacency
    _adjacency_service = AdjacencyService(_clickhouse_service)

    # Latency
    _latency_service = LatencyService(_clickhouse_service)

    # State detector
    _state_detector_service = StateDetectorService(_clickhouse_service)

    # Search
    _search_service = SearchService(_clickhouse_service, _redis_service)

    yield

    # Shutdown
    if _batch_writer:
        await _batch_writer.stop()
    if _redis_pool:
        await _redis_pool.aclose()
    if _clickhouse_client:
        _clickhouse_client.close()


def get_redis_service() -> RedisService:
    assert _redis_service is not None, "Redis service not initialized"
    return _redis_service


def get_clickhouse_service() -> ClickHouseService:
    assert _clickhouse_service is not None, "ClickHouse service not initialized"
    return _clickhouse_service


def get_batch_writer() -> ClickHouseBatchWriter:
    assert _batch_writer is not None, "Batch writer not initialized"
    return _batch_writer


def get_adjacency_service() -> AdjacencyService:
    assert _adjacency_service is not None, "Adjacency service not initialized"
    return _adjacency_service


def get_latency_service() -> LatencyService:
    assert _latency_service is not None, "Latency service not initialized"
    return _latency_service


def get_state_detector_service() -> StateDetectorService:
    assert _state_detector_service is not None, "State detector service not initialized"
    return _state_detector_service


def get_cache_service() -> CacheService:
    assert _cache_service is not None, "Cache service not initialized"
    return _cache_service


def get_search_service() -> SearchService:
    assert _search_service is not None, "Search service not initialized"
    return _search_service


def get_stats_service() -> StatsService:
    return StatsService()


def get_event_counts_service() -> EventCountsService:
    return EventCountsService()
