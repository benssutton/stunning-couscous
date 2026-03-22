from fastapi import APIRouter, Depends

from schemas.models import ClassifierRequest, ClassifierResponse
from services.chain_classifier import ChainClassifier
from services.clickhouse_service import ClickHouseService
from services.dependencies import get_clickhouse_service, get_redis_service
from services.redis_service import RedisService

router = APIRouter()


@router.get("/classifier")
async def get_classifier(
    ch_svc: ClickHouseService = Depends(get_clickhouse_service),
):
    profiles = ch_svc.query_path_profiles()
    return ClassifierResponse(
        profiles=profiles,
        method_results={},
    )


@router.post("/classifier")
async def post_classifier(
    request: ClassifierResponse,
    ch_svc: ClickHouseService = Depends(get_clickhouse_service),
    redis_svc: RedisService = Depends(get_redis_service),
):
    """Overwrite classifier profiles in ClickHouse with caller-provided data."""
    ch_svc.insert_classification_result(request.profiles)
    redis_svc.set_path_profiles(request.profiles)
    return {"status": "ok", "profile_count": len(request.profiles)}


@router.put("/classifier")
async def classify_paths(
    request: ClassifierRequest = ClassifierRequest(),
    ch_svc: ClickHouseService = Depends(get_clickhouse_service),
    redis_svc: RedisService = Depends(get_redis_service),
):
    classifier = ChainClassifier(ch_svc)
    adjacency = ch_svc.query_adjacency()
    edges = adjacency["edges"] if adjacency else []
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
