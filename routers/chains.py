from fastapi import APIRouter, Depends

from services.clickhouse_service import ClickHouseService
from services.dependencies import get_clickhouse_service, get_redis_service
from services.redis_service import RedisService

router = APIRouter()


@router.get("/chains")
async def get_chains(
    unterminated: bool = False,
    redis_svc: RedisService = Depends(get_redis_service),
    clickhouse_svc: ClickHouseService = Depends(get_clickhouse_service),
):
    if unterminated:
        chains = await redis_svc.get_all_chains()
    else:
        chains = clickhouse_svc.query_chains_for_cache()
    return {"count": len(chains), "chains": chains}
