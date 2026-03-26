from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException

from schemas.models import AdjacencyMatrixRequest, AdjacencyMatrixResponse
from services.adjacency_service import AdjacencyService
from core.dependencies import get_adjacency_service

router = APIRouter()

@router.delete("/adjacency_matrix",
               summary="Delete all stored adjacency matrices")
async def delete_adjacency_matrix(
    adj_svc: AdjacencyService = Depends(get_adjacency_service),
):
    return adj_svc.delete()

@router.get("/adjacency_matrix",
            summary="Return all adjacency matrices")
async def get_adjacency_matrix(
    adj_svc: AdjacencyService = Depends(get_adjacency_service),
):
    return adj_svc.get()

@router.post("/adjacency_matrix",
             summary="Overwrite all adjacency matrices with those supplied")
async def post_adjacency_matrix(
    request: AdjacencyMatrixResponse,
    adj_svc: AdjacencyService = Depends(get_adjacency_service),
):
    run_id = UUID(request.run_id) if request.run_id else UUID(int=0)
    return adj_svc.post(run_id=run_id, edges=request.edges, method=request.method, max_pval=request.max_pval)

@router.put("/adjacency_matrix",
             summary="Infer and update adjacency matrices",
             description="Creates adjacency matrices from scratch and replaces any that "\
                "were previously stored using the correlation method supplied")
async def compute_adjacency_matrix(
    request: AdjacencyMatrixRequest,
    adj_svc: AdjacencyService = Depends(get_adjacency_service),
):
    try:
        return adj_svc.compute(method=request.method, max_pval=request.max_pval)
    except KeyError:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown correlation method: {request.method}",
        )
