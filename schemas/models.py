from datetime import datetime
from typing import Literal

from pydantic import BaseModel

# Events
class Ref(BaseModel):
    type: str
    id: str
    ver: int

class Event(BaseModel):
    EventName: str
    Timestamp: str
    Refs: list[Ref]
    Context: dict = {}


# Adjacency Matrix
class AdjacencyMatrixRequest(BaseModel):
    method: str = "pearson"
    max_pval: float = 0.05

class Edge(BaseModel):
    source: str
    target: str
    correlation: float
    p_value: float
    mean_delta_ms: float | None
    std_delta_ms: float | None
    max_delta_ms: float | None
    min_delta_ms: float | None
    sample_count: int

class AdjacencyMatrixResponse(BaseModel):
    run_id: str
    method: str
    max_pval: float
    edges: list[Edge]
    root_events: list[str]


# ---------------------------------------------------------------------------
# Chain classifier models
# ---------------------------------------------------------------------------

class PathProfile(BaseModel):
    profile_id: int
    node_set: frozenset[str]
    terminal_nodes: frozenset[str] = frozenset()
    chain_count: int
    fraction: float


class FeatureImportance(BaseModel):
    feature_name: str
    importance: float


class MethodResult(BaseModel):
    method: str
    accuracy: float
    feature_importances: list[FeatureImportance]


class ClassifierResult(BaseModel):
    profiles: list[PathProfile]
    method_results: dict[str, MethodResult]


class ClassifierRequest(BaseModel):
    method: str | None = "decision_tree"


class ClassifierResponse(BaseModel):
    profiles: list[PathProfile]
    method_results: dict[str, MethodResult]


# ---------------------------------------------------------------------------
# Latency models
# ---------------------------------------------------------------------------

class ChainLatency(BaseModel):
    source: str
    target: str
    delta_ms: float


class ChainLatencyResponse(BaseModel):
    chain_id: str
    latencies: list[ChainLatency]


class EdgeLatencyStats(BaseModel):
    source: str
    target: str
    avg_ms: float
    stddev_ms: float
    min_ms: float
    max_ms: float
    p5_ms: float
    p50_ms: float
    p95_ms: float
    sample_count: int


class AverageLatencyResponse(BaseModel):
    chain_id: str
    profile_id: int
    node_set: list[str]
    matching_chains: int
    start: str
    end: str
    edges: list[EdgeLatencyStats]


# ---------------------------------------------------------------------------
# State detector models
# ---------------------------------------------------------------------------

class EdgeStateResult(BaseModel):
    source: str
    target: str
    means: list[float]
    variances: list[float]
    transition_matrix: list[list[float]]
    start_probabilities: list[float]
    normal_state: int
    anomalous_state: int
    sample_count: int


class ProfileStateResult(BaseModel):
    profile_id: int
    node_set: list[str]
    chain_count: int
    edges: list[EdgeStateResult]


class StateDetectorRequest(BaseModel):
    start: datetime
    end: datetime | None = None
    method: str = "gaussian_hmm"


class StateDetectorResponse(BaseModel):
    method: str
    start: str
    end: str
    profiles: list[ProfileStateResult]


# ---------------------------------------------------------------------------
# Event counts models
# ---------------------------------------------------------------------------

class EventCountsRequest(BaseModel):
    event_name: str
    dates: list[str]          # YYYY-MM-DD
    bucket_seconds: int       # 1–60
    metric: Literal["count", "rolling_avg", "cumulative_sum"]

class BucketPoint(BaseModel):
    time: str                 # HH:MM:SS
    value: float

class DateSeries(BaseModel):
    date: str
    buckets: list[BucketPoint]

class EventCountsResponse(BaseModel):
    series: list[DateSeries]

class EventNamesResponse(BaseModel):
    names: list[str]


# ---------------------------------------------------------------------------
# Stats models
# ---------------------------------------------------------------------------

class TTestRequest(BaseModel):
    series_a: list[float]
    series_b: list[float]
    alpha: float = 0.05

class TTestResult(BaseModel):
    t_statistic: float
    p_value: float
    degrees_of_freedom: int
    significant: bool
    alpha: float
