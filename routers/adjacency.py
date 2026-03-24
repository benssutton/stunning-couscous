from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException

from schemas.models import AdjacencyMatrixRequest, AdjacencyMatrixResponse
from services.adjacency_service import AdjacencyService
from services.dependencies import get_adjacency_service

router = APIRouter()

@router.delete("/adjacency_matrix")
async def delete_adjacency_matrix(
    adj_svc: AdjacencyService = Depends(get_adjacency_service),
):
    """Delete all adjacency matrices from Arestor."""
    return adj_svc.delete()


@router.get("/adjacency_matrix")
async def get_adjacency_matrix(
    adj_svc: AdjacencyService = Depends(get_adjacency_service),
):
    """Get all adjacency matrices from Arestor."""
    return adj_svc.get()

@router.post("/adjacency_matrix")
async def post_adjacency_matrix(
    request: AdjacencyMatrixResponse,
    adj_svc: AdjacencyService = Depends(get_adjacency_service),
):
    """Overwrite the adjacency matrix with caller-provided data."""
    run_id = UUID(request.run_id) if request.run_id else UUID(int=0)
    return adj_svc.post(run_id=run_id, edges=request.edges, method=request.method, max_pval=request.max_pval)

@router.put("/adjacency_matrix")
async def compute_adjacency_matrix(
    request: AdjacencyMatrixRequest,
    adj_svc: AdjacencyService = Depends(get_adjacency_service),
):
    """Infer the adjacency matrices from the event chains in Arestor and persist these."""
    try:
        return adj_svc.compute(method=request.method, max_pval=request.max_pval)
    except KeyError:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown inference method: {request.method}",
        )
