from typing import Any, Tuple

from datetime import datetime, timezone

import numpy as np
import pandas as pd
from pydantic import BaseModel, model_validator
from scipy import stats

T_MU = int(np.log(15))
T_SIGMA = int(np.log(50))
T_ALPHA = 75


# ---------------------------------------------------------------------------
# Configuration models
# ---------------------------------------------------------------------------


class GraphNode(BaseModel):
    """A process node in the dependency graph."""
    mu: float
    sigma: float
    alpha: int
    dep: str
    refs: list[str | list[str]]
    context: list[str] = []


class ProfilePath(BaseModel):
    """A transaction routing path through a subset of graph nodes."""
    name: str
    nodes: frozenset[str]
    weight: float = 1.0
    context: list[str] = []

    @model_validator(mode="after")
    def _weight_positive(self) -> "ProfilePath":
        if self.weight <= 0:
            raise ValueError("weight must be positive")
        return self


class SimulatorConfig(BaseModel):
    """Complete simulator configuration: graph topology + profile paths."""
    graph: dict[str, GraphNode]
    profiles: list[ProfilePath]

    @model_validator(mode="after")
    def _validate_profile_nodes(self) -> "SimulatorConfig":
        graph_nodes = set(self.graph.keys())
        for profile in self.profiles:
            invalid = profile.nodes - graph_nodes
            if invalid:
                raise ValueError(
                    f"Profile '{profile.name}' references unknown nodes: {invalid}"
                )
        return self


# ---------------------------------------------------------------------------
# Default configuration (equivalent to prior PROC_PARAMS behaviour)
# ---------------------------------------------------------------------------

DEFAULT_GRAPH: dict[str, GraphNode] = {
    "A": GraphNode(mu=0.5, sigma=0.5, alpha=60, dep="T", refs=["A"], context=["tea"]),
    "B": GraphNode(mu=0.9, sigma=0.3, alpha=25, dep="A", refs=["A", "B"]),
    "C": GraphNode(mu=0.9, sigma=0.3, alpha=25, dep="A", refs=["C", "A"], context=["tea", "coffee"]),
    "D": GraphNode(mu=2.5, sigma=1.0, alpha=25, dep="B", refs=["D", "B"]),
    "E": GraphNode(mu=2.5, sigma=1.0, alpha=12, dep="C", refs=["E", "C"]),
    "F": GraphNode(mu=0.9, sigma=0.2, alpha=60, dep="E", refs=["F", "C", "E"], context=["milkshake"]),
    "G": GraphNode(mu=0.5, sigma=0.1, alpha=7, dep="E", refs=["G", "E", ["A", "C"]]),
    "H": GraphNode(mu=1.5, sigma=0.6, alpha=25, dep="A", refs=["A"]),
    "J": GraphNode(mu=1.0, sigma=0.4, alpha=25, dep="H", refs=["J", "A"]),
}

_ALL_NODES = frozenset(DEFAULT_GRAPH.keys())

DEFAULT_PROFILES: list[ProfilePath] = [
    ProfilePath(name="with_J", nodes=_ALL_NODES, weight=1.0, context=["juice"]),
    ProfilePath(name="without_J", nodes=_ALL_NODES - {"J"}, weight=1.0),
]

DEFAULT_CONFIG = SimulatorConfig(graph=DEFAULT_GRAPH, profiles=DEFAULT_PROFILES)


class DataSimulator:
    """Generates simulated telemetry events."""

    def __init__(
        self,
        num_intervals: int = 1,
        seed: int | None = None,
        config: SimulatorConfig | None = None,
    ) -> None:
        self.num_intervals = num_intervals
        self.config = config or DEFAULT_CONFIG
        self._start_time = np.datetime64(
            datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000")
        )
        if seed is not None:
            np.random.seed(seed)

    def generate(self, prefix: str = "") -> Tuple[int, list[dict]]:
        """Return simulated events as a list of JSON-serialisable dicts."""
        T = self._simulate_originating_events()
        num_obs = T.shape[0]
        profile_indices = self._assign_profiles(num_obs)
        timestamps = self._simulate_timestamps(T, profile_indices)
        trans_refs = self._simulate_refs(num_obs, prefix=prefix)
        context = self._simulate_context(num_obs, profile_indices)
        return len(T), self._build_event_list(timestamps, trans_refs, context)

    # ------------------------------------------------------------------
    # Private simulation steps
    # ------------------------------------------------------------------

    def _simulate_originating_events(self) -> np.ndarray:
        T = np.ndarray([0], dtype=np.datetime64)
        for i in range(self.num_intervals):
            count = min(
                int(stats.lognorm(s=T_SIGMA, scale=np.exp(T_MU)).rvs(1)[0]),
                T_ALPHA,
            )
            if count > 0:
                t_interval = (
                    np.sort(np.random.random(count).round(3) * 1000).astype(np.timedelta64)
                    + self._start_time
                    + np.timedelta64(i, "s")
                )
                T = np.concatenate((T, t_interval), axis=0)
        return T

    def _assign_profiles(self, num_obs: int) -> np.ndarray:
        """Assign each transaction to a profile based on weights."""
        weights = np.array([p.weight for p in self.config.profiles])
        probabilities = weights / weights.sum()
        return np.random.choice(len(self.config.profiles), size=num_obs, p=probabilities)

    def _simulate_timestamps(
        self, T: np.ndarray, profile_indices: np.ndarray
    ) -> dict[str, np.ndarray]:
        num_obs = T.shape[0]
        timestamps: dict[str, np.ndarray] = {"T": T}
        for proc, node in self.config.graph.items():
            latency = (
                np.absolute(
                    stats.norm(loc=node.mu, scale=node.sigma).rvs(num_obs).round(3)
                )
                * 1000
            ).astype(np.timedelta64)
            timestamps[proc] = (
                self._apply_service_rate(timestamps[node.dep], node.alpha) + latency
            )
        del timestamps["T"]

        # Apply profile masks: NaT for nodes not in the transaction's profile
        for proc in self.config.graph:
            mask = np.array([
                proc in self.config.profiles[pi].nodes
                for pi in profile_indices
            ])
            timestamps[proc][~mask] = np.datetime64("NaT")

        return timestamps

    @staticmethod
    def _apply_service_rate(timeseries: np.ndarray, rate_per_second: int) -> np.ndarray:
        df = pd.DataFrame(timeseries, columns=["timeseries"]).sort_values("timeseries")
        df["time_deltas"] = df["timeseries"].diff()
        min_delta = np.timedelta64(int(1 / rate_per_second * 1000), "ms")
        df["rate_limited_deltas"] = np.maximum(
            df["time_deltas"], np.full(df["time_deltas"].shape, min_delta)
        )
        df.loc[df["rate_limited_deltas"].isna(), "rate_limited_deltas"] = np.timedelta64(0, "ms")
        df["rate_limited_ts"] = df["rate_limited_deltas"].cumsum() + df.iloc[0]["timeseries"]
        return df.sort_index()["rate_limited_ts"].to_numpy()

    def _simulate_refs(self, num_obs: int, prefix: str = "") -> dict[str, list[list[dict]]]:
        trans_refs: dict[str, list[list[dict]]] = {}
        for proc, node in self.config.graph.items():
            refs_proc = []
            for i in range(num_obs):
                refs_event = []
                for ref in node.refs:
                    if isinstance(ref, str):
                        refs_event.append({"type": ref, "id": f"{prefix}{ref}{i}", "ver": 1})
                    elif isinstance(ref, list):
                        chosen = np.random.choice(ref)
                        refs_event.append({"type": str(chosen), "id": f"{prefix}{chosen}{i}", "ver": 1})
                refs_proc.append(refs_event)
            trans_refs[proc] = refs_proc
        return trans_refs

    def _simulate_context(
        self, num_obs: int, profile_indices: np.ndarray
    ) -> dict[str, list[dict]]:
        context: dict[str, list[dict]] = {}
        for proc, node in self.config.graph.items():
            ctx_proc = []
            for i in range(num_obs):
                # Node-level context
                ctx_event = {item: i for item in node.context}
                # Layer on profile-level context
                profile = self.config.profiles[profile_indices[i]]
                for item in profile.context:
                    ctx_event[item] = i
                ctx_proc.append(ctx_event)
            context[proc] = ctx_proc
        return context

    @staticmethod
    def _build_event_list(
        timestamps: dict[str, np.ndarray],
        trans_refs: dict[str, list[list[dict]]],
        context: dict[str, list[dict]],
    ) -> list[dict]:
        num_obs = len(timestamps["A"])
        rows = []
        for proc, ts_array in timestamps.items():
            for i, ts in enumerate(ts_array):
                if np.isnat(ts):
                    continue
                rows.append({
                    "EventName": proc,
                    "Timestamp": str(ts),
                    "Refs": trans_refs[proc][i],
                    "Context": context[proc][i],
                })
        df = pd.DataFrame(rows).sort_values("Timestamp").reset_index()
        df["index"] = df["index"] + np.random.randint(0, num_obs, df.shape[0])
        df = df.set_index("index").sort_index().reset_index(drop=True)
        return df.to_dict(orient="records")
