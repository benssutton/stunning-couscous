from fastapi import APIRouter, Depends

from services.cache_service import CacheService
from services.clickhouse_service import ClickHouseService
from core.dependencies import get_cache_service, get_clickhouse_service

router = APIRouter()


@router.get("/chains",
            summary="Event chain query with query parameters",
            description="Returns event chains according to the supplied criteria")
async def get_chains(
    unterminated: bool = False,
    cache_svc: CacheService = Depends(get_cache_service),
    clickhouse_svc: ClickHouseService = Depends(get_clickhouse_service),
):
    if unterminated:
        chains = await cache_svc.get_all_chains()
    else:
        chains = clickhouse_svc.query_chains_for_cache()
    return {"count": len(chains), "chains": chains}
