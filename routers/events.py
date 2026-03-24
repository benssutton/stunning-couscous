from datetime import datetime

from fastapi import APIRouter, Depends

from schemas.models import Event
from services.clickhouse_service import ClickHouseBatchWriter
from services.dependencies import get_batch_writer, get_redis_service
from services.redis_service import RedisService

router = APIRouter()


@router.post("/events", status_code=201)
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
