from uuid import UUID

from fastapi import Depends, FastAPI, HTTPException

from services.adjacency_service import AdjacencyService
from services.chain_classifier import ChainClassifier
from services.clickhouse_service import ClickHouseService
from services.inference import Edge
from services.dependencies import (
    get_adjacency_service,
    get_clickhouse_service,
    get_redis_service,
    lifespan,
)
from services.models import (
    AdjacencyMatrixRequest,
    AdjacencyMatrixResponse,
    ClassifierRequest,
    ClassifierResponse,
    EdgeResponse,
    Event,
    Ref,  # noqa: F401 — re-export for consumers
)
from services.redis_service import RedisService

app = FastAPI(lifespan=lifespan)


@app.post("/events")
async def receive_event(
    event: Event,
    redis_svc: RedisService = Depends(get_redis_service),
    ch_svc: ClickHouseService = Depends(get_clickhouse_service),
):
    chain_id = redis_svc.add_or_merge_event(event)
    ch_svc.insert_event(event, chain_id=chain_id)
    return {"status": "received", "event_name": event.EventName, "chain_id": chain_id}


@app.get("/adjacency_matrix")
async def get_adjacency_matrix(
    ch_svc: ClickHouseService = Depends(get_clickhouse_service),
):
    result = ch_svc.query_latest_adjacency()
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


@app.post("/adjacency_matrix")
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


@app.put("/adjacency_matrix")
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


@app.get("/classifier")
async def get_classifier(
    ch_svc: ClickHouseService = Depends(get_clickhouse_service),
):
    profiles = ch_svc.query_path_profiles()
    return ClassifierResponse(
        profiles=profiles,
        method_results={},
    )


@app.post("/classifier")
async def post_classifier(
    request: ClassifierResponse,
    ch_svc: ClickHouseService = Depends(get_clickhouse_service),
    redis_svc: RedisService = Depends(get_redis_service),
):
    """Overwrite classifier profiles in ClickHouse with caller-provided data."""
    ch_svc.insert_classification_result(request.profiles)
    redis_svc.set_path_profiles(request.profiles)
    return {"status": "ok", "profile_count": len(request.profiles)}


@app.put("/classifier")
async def classify_paths(
    request: ClassifierRequest = ClassifierRequest(),
    ch_svc: ClickHouseService = Depends(get_clickhouse_service),
    redis_svc: RedisService = Depends(get_redis_service),
):
    classifier = ChainClassifier(ch_svc)
    edges = ch_svc.query_latest_edges()
    result = classifier.analyze(edges, method=request.method)

    # Persist profiles and fitted model, propagate to RedisService
    if result.profiles:
        ch_svc.insert_classification_result(result.profiles)
        redis_svc.set_path_profiles(result.profiles)

        # Build and persist the runtime predictor (requires decision_tree)
        method_for_predictor = request.method or "decision_tree"
        if method_for_predictor in classifier.methods:
            predictor = classifier.build_predictor(
                result.profiles, method=method_for_predictor
            )
            ch_svc.insert_classifier_model(predictor.serialize())
            redis_svc.set_predictor(predictor)

    return ClassifierResponse(
        profiles=result.profiles,
        method_results=result.method_results,
    )


@app.get("/cache/event_chain_keys")
async def get_event_chain_keys(
    redis_svc: RedisService = Depends(get_redis_service),
):
    keys = redis_svc.get_all_chain_keys()
    return {"count": len(keys), "keys": keys}


@app.delete("/cache/event_chain_keys")
async def delete_event_chain_keys(
    redis_svc: RedisService = Depends(get_redis_service),
):
    deleted = redis_svc.delete_all_chains()
    return {"deleted": deleted}


@app.put("/cache")
async def load_cache(
    redis_svc: RedisService = Depends(get_redis_service),
    ch_svc: ClickHouseService = Depends(get_clickhouse_service),
):
    """Load non-terminated event chains from ClickHouse into Redis."""
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

    loaded = redis_svc.load_chains(chains_to_load)
    return {
        "total_chains": len(all_chains),
        "terminated": len(all_chains) - len(chains_to_load),
        "loaded": loaded,
    }


@app.get("/chains")
async def get_chains(
    unterminated: bool = False,
    redis_svc: RedisService = Depends(get_redis_service),
):
    if unterminated:
        chains = redis_svc.get_all_chains()
        return {"count": len(chains), "chains": chains}
    return {"count": 0, "chains": []}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
