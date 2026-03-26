from fastapi import APIRouter, Depends, HTTPException

from schemas.models import StateDetectorRequest, StateDetectorResponse
from services.state_detector_service import StateDetectorService
from core.dependencies import get_state_detector_service

router = APIRouter()


@router.put(
    "/state_detectors/latencies",
    response_model=StateDetectorResponse,
    summary="Train a state detector on per-edge latencies",
)
async def train_state_detector(
    request: StateDetectorRequest,
    svc: StateDetectorService = Depends(get_state_detector_service),
):
    try:
        return svc.train(start=request.start, end=request.end, method=request.method)
    except KeyError:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown state detection method: {request.method}",
        )


@router.get(
    "/state_detectors/latencies",
    response_model=StateDetectorResponse,
    summary="Retrieve the most recently trained latency state detector",
)
async def get_state_detector(
    svc: StateDetectorService = Depends(get_state_detector_service),
):
    result = svc.get()
    if result is None:
        raise HTTPException(status_code=404, detail="No state detector model found")
    return result
