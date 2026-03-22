from pydantic import BaseModel


class Ref(BaseModel):
    type: str
    id: str
    ver: int


class Event(BaseModel):
    EventName: str
    Timestamp: str
    Refs: list[Ref]
    Context: dict = {}


# ---------------------------------------------------------------------------
# Adjacency matrix models
# ---------------------------------------------------------------------------

class AdjacencyMatrixRequest(BaseModel):
    method: str = "pearson"
    max_pval: float = 0.05


class EdgeResponse(BaseModel):
    source: str
    target: str
    correlation: float
    p_value: float
    mean_delta_ms: float
    std_delta_ms: float
    max_delta_ms: float
    min_delta_ms: float
    sample_count: int


class AdjacencyMatrixResponse(BaseModel):
    run_id: str
    method: str
    max_pval: float
    edge_count: int
    edges: list[EdgeResponse]
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
