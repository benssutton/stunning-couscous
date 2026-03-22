import logging
from contextlib import asynccontextmanager

import clickhouse_connect
import redis
from pydantic_settings import BaseSettings

from services.adjacency_service import AdjacencyService
from services.chain_classifier import ChainProfilePredictor
from services.clickhouse_service import ClickHouseService
from services.redis_service import RedisService

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    redis_host: str = "localhost"
    redis_port: int = 6379
    clickhouse_host: str = "localhost"
    clickhouse_port: int = 8123
    clickhouse_database: str = "argus"


settings = Settings()

# Module-level service instances (populated during lifespan)
_redis_service: RedisService | None = None
_clickhouse_service: ClickHouseService | None = None
_adjacency_service: AdjacencyService | None = None
_redis_pool: redis.ConnectionPool | None = None
_clickhouse_client = None


@asynccontextmanager
async def lifespan(app):
    """FastAPI lifespan: create connections on startup, close on shutdown."""
    global _redis_service, _clickhouse_service, _adjacency_service, _redis_pool, _clickhouse_client

    # Redis
    _redis_pool = redis.ConnectionPool(
        host=settings.redis_host,
        port=settings.redis_port,
        decode_responses=True,
    )
    _redis_service = RedisService(_redis_pool)
    _redis_service.ensure_index()
    logger.info("Redis connected and index ensured")

    # ClickHouse
    _clickhouse_client = clickhouse_connect.get_client(
        host=settings.clickhouse_host,
        port=settings.clickhouse_port,
    )
    _clickhouse_service = ClickHouseService(
        _clickhouse_client, settings.clickhouse_database
    )
    _clickhouse_service.ensure_table()
    _clickhouse_service.ensure_adjacency_table()
    _clickhouse_service.ensure_profiles_table()
    _clickhouse_service.ensure_classifier_model_table()
    logger.info("ClickHouse connected and tables ensured")

    # Load persisted path profiles into RedisService
    profiles = _clickhouse_service.query_path_profiles()
    if profiles:
        _redis_service.set_path_profiles(profiles)
        logger.info("Loaded %d path profiles from ClickHouse", len(profiles))

    # Load persisted classifier model
    model_bytes = _clickhouse_service.query_classifier_model()
    if model_bytes:
        predictor = ChainProfilePredictor.deserialize(model_bytes)
        _redis_service.set_predictor(predictor)
        logger.info("Loaded classifier model with %d profiles", len(predictor.profiles))

    # Adjacency
    _adjacency_service = AdjacencyService(_clickhouse_service)

    yield

    # Shutdown
    if _redis_pool:
        _redis_pool.disconnect()
    if _clickhouse_client:
        _clickhouse_client.close()


def get_redis_service() -> RedisService:
    assert _redis_service is not None, "Redis service not initialized"
    return _redis_service


def get_clickhouse_service() -> ClickHouseService:
    assert _clickhouse_service is not None, "ClickHouse service not initialized"
    return _clickhouse_service


def get_adjacency_service() -> AdjacencyService:
    assert _adjacency_service is not None, "Adjacency service not initialized"
    return _adjacency_service
