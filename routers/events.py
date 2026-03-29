from datetime import datetime, timezone
import time

from fastapi import APIRouter, Depends, Query

from schemas.models import Event, EventCountsRequest, EventCountsResponse, EventNamesResponse
from services.clickhouse_service import ClickHouseBatchWriter, ClickHouseService
from services.data_simulator import DataSimulator
from services.event_counts_service import EventCountsService
from core.dependencies import get_batch_writer, get_clickhouse_service, get_redis_service, get_event_counts_service
from services.redis_service import RedisService
from core.arrow_serializer import ProduceParams, get_produce_params, produce_response

router = APIRouter()


@router.delete("/events",
               summary="Delete all events")
async def delete_events(
    ch_svc: ClickHouseService = Depends(get_clickhouse_service),
):
    deleted = ch_svc.truncate_events()
    return {"deleted": deleted}


@router.post("/events/simulation", status_code=201,
             summary="Generate and ingest simulated telemetry events",
             description="Generates simulated events for the number of seconds specified "\
                "and saves these")
async def simulate_events(
    num_intervals: int = Query(default=10, ge=1),
    seed: int | None = Query(default=None),
    redis_svc: RedisService = Depends(get_redis_service),
    batch_writer: ClickHouseBatchWriter = Depends(get_batch_writer),
):
    start_time = time.perf_counter_ns()
    run_prefix = datetime.now(timezone.utc).strftime("%y%m%d%H%M%S") + "_"
    num_chains, raw_events = DataSimulator(num_intervals=num_intervals, seed=seed).generate(prefix=run_prefix)

    for raw in raw_events:
        event = Event(**raw)
        chain_id = await redis_svc.add_or_merge_event(event)
        ts = datetime.fromisoformat(event.Timestamp)
        refs = [(ref.type, ref.id, ref.ver) for ref in event.Refs]
        context_keys = list(event.Context.keys())
        context_values = [str(v) for v in event.Context.values()]
        await batch_writer.append(
            [chain_id, event.EventName, ts, refs, context_keys, context_values]
        )

    total_time_ms = (time.perf_counter_ns() - start_time) / 1000 ** 2
    return {"event_count": len(raw_events), "chain_count": num_chains, "total_time_ms": total_time_ms}


@router.post("/events", status_code=201,
             summary="Save a list of telemetry events")
async def receive_event(
    events: list[Event],
    redis_svc: RedisService = Depends(get_redis_service),
    batch_writer: ClickHouseBatchWriter = Depends(get_batch_writer),
):
    results = []
    for event in events:
        chain_id = await redis_svc.add_or_merge_event(event)

        # Build the row and append to the async batch buffer (non-blocking)
        ts = datetime.fromisoformat(event.Timestamp)
        refs = [(ref.type, ref.id, ref.ver) for ref in event.Refs]
        context_keys = list(event.Context.keys())
        context_values = [str(v) for v in event.Context.values()]
        await batch_writer.append(
            [chain_id, event.EventName, ts, refs, context_keys, context_values]
        )
        results.append({"status": "received", "event_name": event.EventName, "chain_id": chain_id})
    return results


@router.get("/events/names",
            response_model=EventNamesResponse,
            summary="List distinct event names")
async def get_event_names(
    ch_svc: ClickHouseService = Depends(get_clickhouse_service),
    produce: ProduceParams = Depends(get_produce_params),
) -> EventNamesResponse:
    names = ch_svc.get_distinct_event_names()
    return produce_response(EventNamesResponse(names=names), produce)


@router.post("/events/counts",
             response_model=EventCountsResponse,
             summary="Create a time-series count of events")
async def get_event_counts(
    request: EventCountsRequest,
    ch_svc: ClickHouseService = Depends(get_clickhouse_service),
    counts_svc: EventCountsService = Depends(get_event_counts_service),
) -> EventCountsResponse:
    df = ch_svc.get_event_counts(request.event_name, request.dates, request.bucket_seconds)
    return counts_svc.build_response(df, request.metric, request.rolling_window)
