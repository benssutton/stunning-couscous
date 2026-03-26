from fastapi import APIRouter, Depends

from services.cache_service import CacheService
from services.clickhouse_service import ClickHouseService
from core.dependencies import get_cache_service, get_clickhouse_service

router = APIRouter()


@router.get("/cache/event_chain_keys",
            summary="Get all event chain keys in the cache")
async def get_event_chain_keys(
    cache_svc: CacheService = Depends(get_cache_service),
):
    keys = await cache_svc.get_all_chain_keys()
    return {"count": len(keys), "keys": keys}


@router.delete("/cache/event_chain_keys", 
               summary="Delete all event chain keys in the cache")
async def delete_event_chain_keys(
    cache_svc: CacheService = Depends(get_cache_service),
):
    deleted = await cache_svc.delete_all_chains()
    return {"deleted": deleted}


@router.put("/cache",
            summary="Reload any non-terminated event chains from the persistent store into the cache ")
async def load_cache(
    cache_svc: CacheService = Depends(get_cache_service),
    ch_svc: ClickHouseService = Depends(get_clickhouse_service),
):
    all_chains = ch_svc.query_chains_for_cache()
    return await cache_svc.load_unterminated_chains(all_chains)
