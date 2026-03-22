from fastapi import APIRouter, Depends

from services.dependencies import get_redis_service
from services.redis_service import RedisService

router = APIRouter()


@router.get("/chains")
async def get_chains(
    unterminated: bool = False,
    redis_svc: RedisService = Depends(get_redis_service),
):
    if unterminated:
        chains = await redis_svc.get_all_chains()
        return {"count": len(chains), "chains": chains}
    return {"count": 0, "chains": []}
