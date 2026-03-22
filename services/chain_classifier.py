"""Chain path classifier: discovers path profiles, terminal nodes, and
discriminating features from observed event chain data.

Uses pluggable classification methods (ClassificationMethod protocol) so
different algorithms can be compared side-by-side.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Protocol

import joblib
import numpy as np
import polars as pl
from sklearn.tree import DecisionTreeClassifier as SklearnDT

from services.clickhouse_service import ClickHouseService
from services.inference import Edge
from services.models import (
    ClassifierResult,
    FeatureImportance,
    MethodResult,
    PathProfile,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Runtime predictor — wraps a fitted sklearn model for chain profile lookup
# ---------------------------------------------------------------------------

class ChainProfilePredictor:
    """Lightweight wrapper around a fitted sklearn model for runtime prediction.

    Given a chain's current events and context keys, predicts which
    PathProfile the chain belongs to.  Serializable via joblib for
    persistence in ClickHouse.
    """

    def __init__(
        self,
        model: SklearnDT,
        feature_names: list[str],
        profiles: dict[int, PathProfile],
    ) -> None:
        self._model = model
        self._feature_names = feature_names
        self._profiles = profiles

    @property
    def profiles(self) -> dict[int, PathProfile]:
        return self._profiles

    def predict(self, events: set[str], context_keys: set[str]) -> PathProfile | None:
        """Predict the single best-matching profile for a chain.

        Builds a binary feature vector from the chain's current state
        and runs it through the fitted decision tree.
        """
        features = np.zeros(len(self._feature_names), dtype=np.int8)
        for i, name in enumerate(self._feature_names):
            if name.startswith("event:"):
                if name[6:] in events:
                    features[i] = 1
            elif name.startswith("ctx:"):
                if name[4:] in context_keys:
                    features[i] = 1
        pred = int(self._model.predict(features.reshape(1, -1))[0])
        return self._profiles.get(pred)

    def serialize(self) -> bytes:
        """Serialize the entire predictor to bytes via joblib."""
        import io

        buf = io.BytesIO()
        joblib.dump(self, buf)
        return buf.getvalue()

    @staticmethod
    def deserialize(data: bytes) -> ChainProfilePredictor:
        """Deserialize a predictor from joblib bytes."""
        import io

        return joblib.load(io.BytesIO(data))


# ---------------------------------------------------------------------------
# Classification method protocol + implementations
# ---------------------------------------------------------------------------

class ClassificationMethod(Protocol):
    def fit(self, features: pl.DataFrame, labels: pl.Series) -> None: ...
    def predict(self, features: pl.DataFrame) -> list[int]: ...
    def feature_importances(self) -> list[FeatureImportance]: ...


class RatioClassifier:
    """Threshold-based classifier using per-profile feature presence ratios.

    For each binary feature, computes the fraction of chains in each profile
    that have the feature set to 1.  A feature is discriminating when it has
    a large ratio gap between profiles.  At predict time, each row is scored
    against every profile by counting how many discriminating features match.
    """

    def __init__(self, min_gap: float = 0.5) -> None:
        self._min_gap = min_gap
        self._ratios: dict[int, dict[str, float]] = {}  # profile_id → {feat: ratio}
        self._profile_ids: list[int] = []
        self._feature_names: list[str] = []
        self._importances: list[FeatureImportance] = []

    def fit(self, features: pl.DataFrame, labels: pl.Series) -> None:
        self._feature_names = features.columns
        self._profile_ids = sorted(labels.unique().to_list())

        # Compute presence ratio per profile per feature
        combined = features.with_columns(labels.alias("_label"))
        self._ratios = {}
        for pid in self._profile_ids:
            subset = combined.filter(pl.col("_label") == pid)
            n = subset.height
            if n == 0:
                self._ratios[pid] = {f: 0.0 for f in self._feature_names}
                continue
            self._ratios[pid] = {
                f: float(subset[f].sum()) / n for f in self._feature_names
            }

        # Compute importances: max ratio gap across any pair of profiles
        importances = []
        for feat in self._feature_names:
            ratios = [self._ratios[pid][feat] for pid in self._profile_ids]
            gap = max(ratios) - min(ratios) if len(ratios) > 1 else 0.0
            if gap >= self._min_gap:
                importances.append(FeatureImportance(feature_name=feat, importance=gap))
        importances.sort(key=lambda x: x.importance, reverse=True)
        self._importances = importances

    def predict(self, features: pl.DataFrame) -> list[int]:
        discriminating = {fi.feature_name for fi in self._importances}
        predictions = []
        for row in features.iter_rows(named=True):
            best_pid = self._profile_ids[0]
            best_score = -1.0
            for pid in self._profile_ids:
                score = 0.0
                for feat in discriminating:
                    val = row[feat]
                    ratio = self._ratios[pid][feat]
                    # Score higher when feature presence matches profile ratio
                    if val == 1 and ratio > 0.5:
                        score += ratio
                    elif val == 0 and ratio < 0.5:
                        score += (1 - ratio)
                if score > best_score:
                    best_score = score
                    best_pid = pid
            predictions.append(best_pid)
        return predictions

    def feature_importances(self) -> list[FeatureImportance]:
        return self._importances


class TreeClassifier:
    """sklearn DecisionTreeClassifier wrapper."""

    def __init__(self, max_depth: int = 5, min_samples_leaf: int = 10) -> None:
        self._max_depth = max_depth
        self._min_samples_leaf = min_samples_leaf
        self._tree: SklearnDT | None = None
        self._feature_names: list[str] = []

    def fit(self, features: pl.DataFrame, labels: pl.Series) -> None:
        self._feature_names = features.columns
        self._tree = SklearnDT(
            max_depth=self._max_depth,
            min_samples_leaf=self._min_samples_leaf,
        )
        self._tree.fit(features.to_numpy(), labels.to_numpy())

    def predict(self, features: pl.DataFrame) -> list[int]:
        assert self._tree is not None, "TreeClassifier not fitted"
        return self._tree.predict(features.to_numpy()).tolist()

    def feature_importances(self) -> list[FeatureImportance]:
        assert self._tree is not None, "TreeClassifier not fitted"
        return sorted(
            [
                FeatureImportance(feature_name=name, importance=float(imp))
                for name, imp in zip(self._feature_names, self._tree.feature_importances_)
                if imp > 0
            ],
            key=lambda x: x.importance,
            reverse=True,
        )


# ---------------------------------------------------------------------------
# ChainClassifier orchestrator
# ---------------------------------------------------------------------------

class ChainClassifier:
    def __init__(self, ch_svc: ClickHouseService):
        self.ch_svc = ch_svc
        self.methods: dict[str, ClassificationMethod] = {
            "ratio": RatioClassifier(),
            "decision_tree": TreeClassifier(),
        }

    def analyze(
        self,
        edges: list[Edge],
        method: str | None = None,
    ) -> ClassifierResult:
        """Run the full classification pipeline.

        1. Discover path profiles from timestamp matrix
        2. Identify terminal nodes per profile using adjacency edges
        3. Build binary feature matrix
        4. Fit and evaluate classification method(s)

        If *method* is None, all registered methods run.
        """
        matrix = self.ch_svc.query_timestamp_matrix()
        if matrix.is_empty():
            return ClassifierResult(profiles=[], method_results={})

        profiles, chain_labels = self._discover_profiles(matrix)
        self._identify_terminals(profiles, edges)

        features, labels = self._build_feature_matrix(matrix, chain_labels)

        methods_to_run = (
            {method: self.methods[method]} if method else self.methods
        )
        method_results: dict[str, MethodResult] = {}
        for name, clf in methods_to_run.items():
            clf.fit(features, labels)
            preds = clf.predict(features)
            accuracy = sum(
                1 for p, a in zip(preds, labels.to_list()) if p == a
            ) / len(preds) if len(preds) > 0 else 0.0
            method_results[name] = MethodResult(
                method=name,
                accuracy=accuracy,
                feature_importances=clf.feature_importances(),
            )

        return ClassifierResult(profiles=profiles, method_results=method_results)

    def build_predictor(
        self,
        profiles: list[PathProfile],
        method: str = "decision_tree",
    ) -> ChainProfilePredictor:
        """Build a runtime predictor from a fitted classification method.

        Must be called after ``analyze()`` so the method is already fitted.
        """
        clf = self.methods[method]
        if not isinstance(clf, TreeClassifier):
            raise TypeError(
                f"build_predictor requires a TreeClassifier, got {type(clf).__name__}"
            )
        assert clf._tree is not None, "TreeClassifier not fitted — call analyze() first"
        return ChainProfilePredictor(
            model=clf._tree,
            feature_names=clf._feature_names,
            profiles={p.profile_id: p for p in profiles},
        )

    # ------------------------------------------------------------------
    # Profile discovery
    # ------------------------------------------------------------------

    @staticmethod
    def _discover_profiles(
        matrix: pl.DataFrame,
    ) -> tuple[list[PathProfile], pl.DataFrame]:
        """Group chains by their observed node-set (non-null event columns).

        Returns (profiles, chain_labels) where chain_labels is a DataFrame
        with columns [chain_id, profile_id].
        """
        event_cols = [c for c in matrix.columns if c != "chain_id"]

        # Build a signature string per row: concatenation of which columns are non-null
        sig_expr = pl.concat_str(
            [pl.col(c).is_not_null().cast(pl.Utf8) for c in event_cols],
            separator=",",
        ).alias("_sig")
        with_sig = matrix.select("chain_id", sig_expr)

        # Map each unique signature to a profile_id
        unique_sigs = with_sig.select("_sig").unique().sort("_sig")
        sig_to_id = {
            row["_sig"]: idx
            for idx, row in enumerate(unique_sigs.iter_rows(named=True))
        }

        # Assign profile_id per chain
        chain_labels = with_sig.with_columns(
            pl.col("_sig").replace_strict(sig_to_id).cast(pl.Int64).alias("profile_id")
        ).select("chain_id", "profile_id")

        # Build PathProfile objects
        total_chains = matrix.height
        profiles: list[PathProfile] = []
        for sig_str, pid in sig_to_id.items():
            parts = sig_str.split(",")
            node_set = frozenset(
                col for col, flag in zip(event_cols, parts) if flag == "true"
            )
            count = int(chain_labels.filter(pl.col("profile_id") == pid).height)
            profiles.append(PathProfile(
                profile_id=pid,
                node_set=node_set,
                chain_count=count,
                fraction=count / total_chains if total_chains > 0 else 0.0,
            ))

        return profiles, chain_labels

    # ------------------------------------------------------------------
    # Terminal node identification
    # ------------------------------------------------------------------

    @staticmethod
    def _identify_terminals(
        profiles: list[PathProfile],
        edges: list[Edge],
    ) -> None:
        """Compute terminal nodes for each profile in-place."""
        children: dict[str, set[str]] = defaultdict(set)
        for edge in edges:
            children[edge.source].add(edge.target)

        for profile in profiles:
            profile.terminal_nodes = frozenset(
                n for n in profile.node_set
                if not (children.get(n, set()) & profile.node_set)
            )

    # ------------------------------------------------------------------
    # Feature matrix construction
    # ------------------------------------------------------------------

    def _build_feature_matrix(
        self,
        matrix: pl.DataFrame,
        chain_labels: pl.DataFrame,
    ) -> tuple[pl.DataFrame, pl.Series]:
        """Build a binary feature matrix from event presence and chain features.

        Features:
        - event:{name} — 1 if event timestamp is not null
        - ctx:{key}    — 1 if context key present in chain

        Returns (features_df, labels_series).
        """
        event_cols = [c for c in matrix.columns if c != "chain_id"]

        # Event presence features
        event_features = matrix.select(
            "chain_id",
            *[pl.col(c).is_not_null().cast(pl.Int8).alias(f"event:{c}") for c in event_cols],
        )

        # Context key features from ClickHouse
        chain_feats = self.ch_svc.query_chain_features()

        if chain_feats.is_empty():
            features = event_features.drop("chain_id")
            labels = chain_labels.sort("chain_id")["profile_id"]
            return features, labels

        # Explode ctx_keys to find all unique context keys
        all_ctx_keys: list[str] = sorted(
            chain_feats.select(pl.col("ctx_keys").explode())
            .drop_nulls()
            .unique()
            .to_series()
            .to_list()
        )

        # Build binary columns for each context key
        ctx_exprs = []
        for key in all_ctx_keys:
            ctx_exprs.append(
                pl.col("ctx_keys").list.contains(key).cast(pl.Int8).alias(f"ctx:{key}")
            )

        ctx_features = chain_feats.select("chain_id", *ctx_exprs) if ctx_exprs else None

        # Join everything on chain_id
        combined = event_features
        if ctx_features is not None:
            combined = combined.join(ctx_features, on="chain_id", how="left")

        # Align with labels
        combined = combined.join(chain_labels, on="chain_id", how="inner")
        labels = combined["profile_id"]
        features = combined.drop("chain_id", "profile_id")

        # Fill any nulls from left join with 0
        features = features.fill_null(0)

        return features, labels
