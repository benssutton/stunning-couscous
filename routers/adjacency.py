from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException

from schemas.models import (
    AdjacencyMatrixRequest,
    AdjacencyMatrixResponse,
    EdgeResponse,
)
from services.adjacency_service import AdjacencyService
from services.clickhouse_service import ClickHouseService
from services.dependencies import get_adjacency_service, get_clickhouse_service
from services.inference import Edge

router = APIRouter()


@router.get("/adjacency_matrix")
async def get_adjacency_matrix(
    ch_svc: ClickHouseService = Depends(get_clickhouse_service),
):
    result = ch_svc.query_adjacency()
    if result is None:
        return AdjacencyMatrixResponse(
            run_id="", method="", max_pval=0.0,
            edge_count=0, edges=[], root_events=[],
        )
    edges = result["edges"]
    # Root events: sources that are never targets
    all_targets = {e.target for e in edges}
    root_events = sorted({e.source for e in edges} - all_targets)
    return AdjacencyMatrixResponse(
        run_id=result["run_id"],
        method=result["method"],
        max_pval=result["max_pval"],
        edge_count=len(edges),
        edges=[
            EdgeResponse(
                source=e.source, target=e.target,
                correlation=e.correlation, p_value=e.p_value,
                mean_delta_ms=e.mean_delta_ms, std_delta_ms=e.std_delta_ms,
                max_delta_ms=e.max_delta_ms, min_delta_ms=e.min_delta_ms,
                sample_count=e.sample_count,
            )
            for e in edges
        ],
        root_events=root_events,
    )


@router.post("/adjacency_matrix")
async def post_adjacency_matrix(
    request: AdjacencyMatrixResponse,
    ch_svc: ClickHouseService = Depends(get_clickhouse_service),
):
    """Overwrite adjacency matrix data in ClickHouse with caller-provided data."""
    edges = [
        Edge(
            source=e.source, target=e.target,
            correlation=e.correlation, p_value=e.p_value,
            mean_delta_ms=e.mean_delta_ms, std_delta_ms=e.std_delta_ms,
            max_delta_ms=e.max_delta_ms, min_delta_ms=e.min_delta_ms,
            sample_count=e.sample_count,
        )
        for e in request.edges
    ]
    ch_svc.insert_adjacency_result(
        run_id=UUID(request.run_id) if request.run_id else UUID(int=0),
        edges=edges,
        method=request.method,
        max_pval=request.max_pval,
    )
    return {"status": "ok", "edge_count": len(edges)}


@router.put("/adjacency_matrix")
async def compute_adjacency_matrix(
    request: AdjacencyMatrixRequest,
    adj_svc: AdjacencyService = Depends(get_adjacency_service),
):
    try:
        result = adj_svc.compute(method=request.method, max_pval=request.max_pval)
    except KeyError:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown inference method: {request.method}",
        )
    return AdjacencyMatrixResponse(
        run_id=result.run_id,
        method=result.method,
        max_pval=result.max_pval,
        edge_count=len(result.edges),
        edges=[
            EdgeResponse(
                source=e.source,
                target=e.target,
                correlation=e.correlation,
                p_value=e.p_value,
                mean_delta_ms=e.mean_delta_ms,
                std_delta_ms=e.std_delta_ms,
                max_delta_ms=e.max_delta_ms,
                min_delta_ms=e.min_delta_ms,
                sample_count=e.sample_count,
            )
            for e in result.edges
        ],
        root_events=result.root_events,
    )
