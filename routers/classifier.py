from fastapi import APIRouter, Depends

from schemas.models import ClassifierRequest, ClassifierResponse
from services.chain_classifier_service import ChainClassifierService
from services.clickhouse_service import ClickHouseService
from core.dependencies import get_clickhouse_service, get_redis_service
from services.redis_service import RedisService

router = APIRouter()


@router.get("/classifier",
            summary="Returns the inferred event paths and details of the trained classifier")
async def get_classifier(
    ch_svc: ClickHouseService = Depends(get_clickhouse_service),
):
    profiles = ch_svc.query_path_profiles()
    method_result = ch_svc.query_classifier_model_metadata()
    method_results = {method_result.method: method_result} if method_result else {}
    return ClassifierResponse(
        profiles=profiles,
        method_results=method_results,
    )

@router.put("/classifier",
             summary="Re-trains the classifier to identify the expected event path for events received",
             description="")
async def classify_paths(
    request: ClassifierRequest = ClassifierRequest(),
    ch_svc: ClickHouseService = Depends(get_clickhouse_service),
    redis_svc: RedisService = Depends(get_redis_service),
):
    classifier = ChainClassifierService(ch_svc)
    adjacency = ch_svc.query_adjacency()
    edges = adjacency["edges"] if adjacency else []
    result = classifier.analyze(edges, method=request.method)

    # Persist profiles and fitted model, propagate to RedisService
    if result.profiles:
        ch_svc.insert_classification_result(result.profiles)
        redis_svc.set_path_profiles(result.profiles)

        # Build and persist the runtime predictor (requires decision_tree)
        method_for_predictor = request.method or "decision_tree"
        method_result = result.method_results.get(method_for_predictor)
        predictor = classifier.build_and_persist_predictor(
            result.profiles,
            method=method_for_predictor,
            accuracy=method_result.accuracy if method_result else 0.0,
        )
        if predictor:
            redis_svc.set_predictor(predictor)

    return ClassifierResponse(
        profiles=result.profiles,
        method_results=result.method_results,
    )
