from fastapi import APIRouter, Depends

from services.clickhouse_service import ClickHouseService
from services.dependencies import get_clickhouse_service, get_redis_service
from services.redis_service import RedisService

router = APIRouter()


@router.get("/cache/event_chain_keys")
async def get_event_chain_keys(
    redis_svc: RedisService = Depends(get_redis_service),
):
    """Get all event chain keys in the cache."""
    keys = await redis_svc.get_all_chain_keys()
    return {"count": len(keys), "keys": keys}


@router.delete("/cache/event_chain_keys")
async def delete_event_chain_keys(
    redis_svc: RedisService = Depends(get_redis_service),
):
    """Delete all event chain keys in the cache."""
    deleted = await redis_svc.delete_all_chains()
    return {"deleted": deleted}


@router.put("/cache")
async def load_cache(
    redis_svc: RedisService = Depends(get_redis_service),
    ch_svc: ClickHouseService = Depends(get_clickhouse_service),
):
    """Re-load non-terminated event chains from the persistent store into the cache."""
    all_chains = ch_svc.query_chains_for_cache()

    # Filter to non-terminated chains using the predictor or profiles
    chains_to_load: list[dict] = []
    for chain in all_chains:
        events = set(chain["timestamps"].keys())
        context_keys = set(chain["context"].keys())

        terminated = False
        if redis_svc._predictor is not None:
            profile = redis_svc._predictor.predict(events, context_keys)
            if profile and profile.terminal_nodes and profile.terminal_nodes <= events:
                terminated = True
        elif redis_svc.path_profiles:
            for profile in redis_svc.path_profiles:
                if events <= profile.node_set:
                    if profile.terminal_nodes and profile.terminal_nodes <= events:
                        terminated = True
                    break

        if not terminated:
            chains_to_load.append(chain)

    loaded = await redis_svc.load_chains(chains_to_load)
    return {
        "total_chains": len(all_chains),
        "terminated": len(all_chains) - len(chains_to_load),
        "loaded": loaded,
    }
