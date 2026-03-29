import io
import json
from dataclasses import dataclass

import polars as pl
from fastapi import Header, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, Response

SUPPORTED_MIME_TYPES = {
    "application/json",
    "application/vnd.apache.arrow.stream",
    "application/vnd.apache.arrow.file",
}

SUPPORTED_COMPRESSIONS = {"uncompressed", "lz4", "zstd"}


@dataclass
class ProduceParams:
    format: str
    compression: str


def get_produce_params(
    produce: str = Header(default="application/json", alias="Produce"),
    compression: str = Header(default="uncompressed", alias="Compression"),
) -> ProduceParams:
    if produce not in SUPPORTED_MIME_TYPES:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported media type '{produce}'. "
                f"Supported types: {sorted(SUPPORTED_MIME_TYPES)}"
            ),
        )
    compression_lower = compression.lower()
    if compression_lower not in SUPPORTED_COMPRESSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported compression '{compression}'. "
                f"Supported values: {sorted(SUPPORTED_COMPRESSIONS)}"
            ),
        )
    return ProduceParams(format=produce, compression=compression_lower)


def _to_polars(data) -> pl.DataFrame:
    encoded = jsonable_encoder(data)
    if isinstance(encoded, dict):
        rows = [encoded]
    elif isinstance(encoded, list):
        rows = encoded if encoded else [{}]
    else:
        rows = [{"value": encoded}]
    return pl.read_json(io.BytesIO(json.dumps(rows).encode()))


def produce_response(data, params: ProduceParams) -> Response:
    if params.format == "application/json":
        return JSONResponse(content=jsonable_encoder(data))

    df = _to_polars(data)
    buf = io.BytesIO()

    if params.format == "application/vnd.apache.arrow.stream":
        df.write_ipc_stream(buf, compression=params.compression)
        media_type = "application/vnd.apache.arrow.stream"
    else:
        df.write_ipc(buf, compression=params.compression)
        media_type = "application/vnd.apache.arrow.file"

    return Response(content=buf.getvalue(), media_type=media_type)
