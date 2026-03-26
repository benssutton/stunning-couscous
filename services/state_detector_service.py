from __future__ import annotations

import io
import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime

import joblib
import numpy as np
import polars as pl

from schemas.models import (
    EdgeStateResult,
    ProfileStateResult,
    StateDetectorResponse,
)
from services.clickhouse_service import ClickHouseService

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base class for pluggable state detection methods
# ---------------------------------------------------------------------------

class StateDetectorMethod(ABC):
    """Interface for state detection algorithms (strategy pattern)."""

    @abstractmethod
    def fit_edge(self, deltas: np.ndarray) -> dict | None:
        """Train on a 1-D array of latency observations for a single edge.

        Returns a dict with keys: means, variances, transition_matrix,
        start_probabilities, normal_state, anomalous_state.
        Returns None if there is insufficient data.
        """
        ...

    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def params(self) -> dict: ...


# ---------------------------------------------------------------------------
# Concrete implementation: 2-state Gaussian HMM
# ---------------------------------------------------------------------------

class GaussianHMMDetector(StateDetectorMethod):
    """2-state Gaussian HMM via hmmlearn."""

    def __init__(
        self,
        n_states: int = 2,
        n_iter: int = 100,
        random_state: int = 42,
    ) -> None:
        self._n_states = n_states
        self._n_iter = n_iter
        self._random_state = random_state

    def name(self) -> str:
        return "GaussianHMM"

    def params(self) -> dict:
        return {
            "n_states": self._n_states,
            "n_iter": self._n_iter,
            "random_state": self._random_state,
        }

    def fit_edge(self, deltas: np.ndarray) -> dict | None:
        from hmmlearn.hmm import GaussianHMM

        if len(deltas) < self._n_states * 2:
            return None

        X = deltas.reshape(-1, 1)
        model = GaussianHMM(
            n_components=self._n_states,
            covariance_type="diag",
            n_iter=self._n_iter,
            random_state=self._random_state,
        )
        model.fit(X)

        means = model.means_.flatten().tolist()
        variances = model.covars_.flatten().tolist()
        transition_matrix = model.transmat_.tolist()
        start_probabilities = model.startprob_.tolist()

        normal_state = int(np.argmin(means))
        anomalous_state = 1 - normal_state

        return {
            "means": means,
            "variances": variances,
            "transition_matrix": transition_matrix,
            "start_probabilities": start_probabilities,
            "normal_state": normal_state,
            "anomalous_state": anomalous_state,
        }


# ---------------------------------------------------------------------------
# Public service (called by router)
# ---------------------------------------------------------------------------

class StateDetectorService:
    def __init__(self, ch_svc: ClickHouseService):
        self.ch_svc = ch_svc
        self.methods: dict[str, StateDetectorMethod] = {
            "gaussian_hmm": GaussianHMMDetector(),
        }

    def train(
        self,
        start: datetime,
        end: datetime | None = None,
        method: str = "gaussian_hmm",
    ) -> StateDetectorResponse:
        """Train state detectors per-profile per-edge and persist results.

        Raises KeyError if the method is not registered.
        """
        detector = self.methods[method]  # KeyError if unknown

        # 1. Get raw per-chain per-edge latencies and chain node_sets
        latency_df = self.ch_svc.query_per_chain_edge_latencies(start, end)
        if latency_df.is_empty():
            return self._empty_response(method, start, end)

        node_set_df = self.ch_svc.query_chain_node_sets(start, end)
        if node_set_df.is_empty():
            return self._empty_response(method, start, end)

        # 2. Match chains to profiles
        profiles = self.ch_svc.query_path_profiles()
        if not profiles:
            return self._empty_response(method, start, end)

        # Build lookup: frozenset(node_set) → PathProfile
        profile_lookup = {p.node_set: p for p in profiles}

        # Assign profile_id to each chain
        chain_profile_rows = []
        for row in node_set_df.iter_rows(named=True):
            ns = frozenset(row["node_set"])
            profile = profile_lookup.get(ns)
            if profile is not None:
                chain_profile_rows.append((row["chain_id"], profile.profile_id))

        if not chain_profile_rows:
            return self._empty_response(method, start, end)

        chain_profile_df = pl.DataFrame(
            chain_profile_rows,
            schema={"chain_id": pl.Utf8, "profile_id": pl.Int64},
            orient="row",
        )

        # 3. Join latency data with profile assignments
        tagged = latency_df.join(chain_profile_df, on="chain_id", how="inner")
        if tagged.is_empty():
            return self._empty_response(method, start, end)

        # 4. Group by (profile_id, source, target) and train HMM per group
        profile_results: list[ProfileStateResult] = []
        profile_id_to_profile = {p.profile_id: p for p in profiles}

        for profile_id, profile_group in tagged.group_by("profile_id"):
            pid = profile_id[0] if isinstance(profile_id, tuple) else profile_id
            profile = profile_id_to_profile.get(pid)
            if profile is None:
                continue

            chain_count = profile_group["chain_id"].n_unique()
            edge_results: list[EdgeStateResult] = []

            for edge_key, edge_group in profile_group.group_by(["source", "target"]):
                source, target = edge_key if isinstance(edge_key, tuple) else (edge_key[0], edge_key[1])
                deltas = edge_group["delta_ms"].to_numpy()
                result = detector.fit_edge(deltas)
                if result is None:
                    continue

                edge_results.append(EdgeStateResult(
                    source=source,
                    target=target,
                    means=result["means"],
                    variances=result["variances"],
                    transition_matrix=result["transition_matrix"],
                    start_probabilities=result["start_probabilities"],
                    normal_state=result["normal_state"],
                    anomalous_state=result["anomalous_state"],
                    sample_count=len(deltas),
                ))

            if edge_results:
                profile_results.append(ProfileStateResult(
                    profile_id=pid,
                    node_set=sorted(profile.node_set),
                    chain_count=chain_count,
                    edges=edge_results,
                ))

        # 5. Serialize and persist
        model_bytes = self._serialize(profile_results)
        self.ch_svc.insert_state_detector_model(
            model_bytes=model_bytes,
            model_name=detector.name(),
            model_params=json.dumps(detector.params()),
            method_name=method,
            start_time=start.isoformat(timespec="milliseconds"),
            end_time=end.isoformat(timespec="milliseconds") if end else "",
        )

        return StateDetectorResponse(
            method=method,
            start=start.isoformat(timespec="milliseconds"),
            end=end.isoformat(timespec="milliseconds") if end else "",
            profiles=profile_results,
        )

    def get(self) -> StateDetectorResponse | None:
        """Load and return the persisted state detector results."""
        model_bytes = self.ch_svc.query_state_detector_model()
        if model_bytes is None:
            return None

        metadata = self.ch_svc.query_state_detector_model_metadata()
        if metadata is None:
            return None

        profile_results = self._deserialize(model_bytes)

        return StateDetectorResponse(
            method=metadata["method_name"],
            start=metadata["start_time"],
            end=metadata["end_time"],
            profiles=profile_results,
        )

    @staticmethod
    def _empty_response(
        method: str, start: datetime, end: datetime | None,
    ) -> StateDetectorResponse:
        return StateDetectorResponse(
            method=method,
            start=start.isoformat(timespec="milliseconds"),
            end=end.isoformat(timespec="milliseconds") if end else "",
            profiles=[],
        )

    @staticmethod
    def _serialize(results: list[ProfileStateResult]) -> bytes:
        buf = io.BytesIO()
        joblib.dump([r.model_dump() for r in results], buf)
        return buf.getvalue()

    @staticmethod
    def _deserialize(data: bytes) -> list[ProfileStateResult]:
        raw = joblib.load(io.BytesIO(data))
        return [ProfileStateResult(**r) for r in raw]
