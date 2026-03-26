from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query

from schemas.models import AverageLatencyResponse, ChainLatencyResponse
from services.latency_service import LatencyService
from core.dependencies import get_latency_service

router = APIRouter()


@router.get(
    "/latencies",
    response_model=list[ChainLatencyResponse],
    summary="Observed latencies between adjacent nodes for matching chains",
)
async def get_latencies(
    chain_id: str | None = Query(None, description="Exact chain ID to look up"),
    ref: str | None = Query(None, description="Concatenated ref (type_id_ver) to search for"),
    latency_svc: LatencyService = Depends(get_latency_service),
):
    if chain_id is None and ref is None:
        raise HTTPException(
            status_code=422,
            detail="At least one of 'chain_id' or 'ref' must be provided",
        )
    results = latency_svc.get(chain_id=chain_id, ref=ref)
    if not results:
        raise HTTPException(status_code=404, detail="No latencies found")
    return results


@router.get(
    "/latencies/averages",
    response_model=AverageLatencyResponse,
    summary="Average latency stats per edge for chains matching the same profile",
)
async def get_average_latencies(
    chain_id: str | None = Query(None, description="Exact chain ID to look up"),
    ref: str | None = Query(None, description="Concatenated ref (type_id_ver) to search for"),
    start: datetime = Query(..., description="Start of time window"),
    end: datetime | None = Query(None, description="End of time window (omit for open-ended)"),
    latency_svc: LatencyService = Depends(get_latency_service),
):
    if chain_id is None and ref is None:
        raise HTTPException(
            status_code=422,
            detail="At least one of 'chain_id' or 'ref' must be provided",
        )
    result = latency_svc.get_averages(
        start=start, end=end, chain_id=chain_id, ref=ref,
    )
    if result is None:
        raise HTTPException(status_code=404, detail="No matching chains found")
    return result
