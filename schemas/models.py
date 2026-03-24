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
