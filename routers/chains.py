from fastapi import APIRouter, Depends, HTTPException

from services.cache_service import CacheService
from services.clickhouse_service import ClickHouseService
from core.dependencies import get_cache_service, get_clickhouse_service
from core.arrow_serializer import ProduceParams, get_produce_params, produce_response

router = APIRouter()


@router.get("/chains",
            summary="Event chain query with query parameters",
            description="Returns event chains according to the supplied criteria")
async def get_chains(
    unterminated: bool = False,
    cache_svc: CacheService = Depends(get_cache_service),
    clickhouse_svc: ClickHouseService = Depends(get_clickhouse_service),
    produce: ProduceParams = Depends(get_produce_params),
):
    if unterminated:
        chains = await cache_svc.get_all_chains()
    else:
        chains = clickhouse_svc.query_chains_for_cache()
    return produce_response({"count": len(chains), "chains": chains}, produce)


@router.get("/chains/{chain_id}",
            summary="Get a single event chain by ID",
            description="Returns the event chain with the given chain_id")
async def get_chain(
    chain_id: str,
    clickhouse_svc: ClickHouseService = Depends(get_clickhouse_service),
    produce: ProduceParams = Depends(get_produce_params),
):
    result = clickhouse_svc.query_chain_by_id(chain_id)
    if not result:
        raise HTTPException(status_code=404, detail="Chain not found")
    return produce_response(result, produce)
