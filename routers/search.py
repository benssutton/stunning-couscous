from fastapi import APIRouter, Depends, HTTPException, Query

from core.dependencies import get_search_service
from services.search_service import SearchService

router = APIRouter(prefix="/search", tags=["search"])


@router.get(
    "/refs",
    summary="Autocomplete reference IDs",
    description="Returns distinct ref IDs matching a prefix string",
)
async def autocomplete_refs(
    q: str = Query(..., min_length=2, description="Prefix to search"),
    limit: int = Query(20, ge=1, le=100),
    search_svc: SearchService = Depends(get_search_service),
):
    results = search_svc.autocomplete_ref_ids(q, limit)
    return {"prefix": q, "results": results}


@router.get(
    "/chains",
    summary="Search event chains by reference ID",
    description="Returns chain IDs containing events with matching ref IDs",
)
async def search_chains(
    ref: str | None = Query(None, description="Exact ref ID"),
    ref_prefix: str | None = Query(None, min_length=2, description="Ref ID prefix"),
    limit: int = Query(100, ge=1, le=1000),
    search_svc: SearchService = Depends(get_search_service),
):
    if ref:
        chains = search_svc.search_chains_by_ref(ref, limit)
    elif ref_prefix:
        chains = search_svc.search_chains_by_ref_prefix(ref_prefix, limit)
    else:
        raise HTTPException(
            status_code=422, detail="Provide 'ref' or 'ref_prefix' query parameter"
        )
    return {"count": len(chains), "chain_ids": chains}
