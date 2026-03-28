from fastapi import APIRouter, Depends

from schemas.models import TTestRequest, TTestResult
from services.stats_service import StatsService
from core.dependencies import get_stats_service

router = APIRouter()


@router.post("/stats/ttest",
             response_model=TTestResult,
             summary="Run a two-sample Welch's T-test on two series of observations")
async def run_ttest(
    request: TTestRequest,
    stats_svc: StatsService = Depends(get_stats_service),
) -> TTestResult:
    return stats_svc.run_ttest(request.series_a, request.series_b, request.alpha)
