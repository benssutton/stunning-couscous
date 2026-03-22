"""Tests for chain path classifier: profile discovery, terminal identification,
feature matrix construction, and classification methods."""

import polars as pl
import pytest

from services.chain_classifier import (
    ChainClassifier,
    RatioClassifier,
    TreeClassifier,
)
from services.inference import Edge
from services.models import PathProfile


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def edges() -> list[Edge]:
    """Adjacency edges representing A→B→D, A→C→E→F, A→C→E→G, A→H→J."""
    return [
        Edge(source="A", target="B", correlation=0.99, p_value=0.001,
             mean_delta_ms=100, std_delta_ms=10, max_delta_ms=120, min_delta_ms=80, sample_count=100),
        Edge(source="B", target="D", correlation=0.98, p_value=0.001,
             mean_delta_ms=200, std_delta_ms=20, max_delta_ms=250, min_delta_ms=150, sample_count=100),
        Edge(source="A", target="C", correlation=0.97, p_value=0.001,
             mean_delta_ms=150, std_delta_ms=15, max_delta_ms=180, min_delta_ms=120, sample_count=100),
        Edge(source="C", target="E", correlation=0.96, p_value=0.001,
             mean_delta_ms=250, std_delta_ms=25, max_delta_ms=300, min_delta_ms=200, sample_count=100),
        Edge(source="E", target="F", correlation=0.95, p_value=0.001,
             mean_delta_ms=100, std_delta_ms=10, max_delta_ms=120, min_delta_ms=80, sample_count=100),
        Edge(source="E", target="G", correlation=0.94, p_value=0.001,
             mean_delta_ms=80, std_delta_ms=8, max_delta_ms=100, min_delta_ms=60, sample_count=100),
        Edge(source="A", target="H", correlation=0.93, p_value=0.001,
             mean_delta_ms=150, std_delta_ms=15, max_delta_ms=180, min_delta_ms=120, sample_count=100),
        Edge(source="H", target="J", correlation=0.92, p_value=0.001,
             mean_delta_ms=100, std_delta_ms=10, max_delta_ms=120, min_delta_ms=80, sample_count=50),
    ]


@pytest.fixture
def timestamp_matrix() -> pl.DataFrame:
    """Simulated timestamp matrix with two path profiles:
    - Profile with J (chains 0-4): all 9 events
    - Profile without J (chains 5-9): 8 events, J is null
    """
    rows = []
    for i in range(10):
        base = 1000.0 + i * 1000
        row = {
            "chain_id": f"chain:{i}",
            "A": base,
            "B": base + 100,
            "C": base + 150,
            "D": base + 300,
            "E": base + 400,
            "F": base + 500,
            "G": base + 480,
            "H": base + 200,
        }
        if i < 5:
            row["J"] = base + 350
        else:
            row["J"] = None
        rows.append(row)
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# Profile discovery tests
# ---------------------------------------------------------------------------

class TestDiscoverProfiles:
    def test_discovers_two_profiles(self, timestamp_matrix):
        profiles, chain_labels = ChainClassifier._discover_profiles(timestamp_matrix)
        assert len(profiles) == 2

    def test_profile_node_sets(self, timestamp_matrix):
        profiles, _ = ChainClassifier._discover_profiles(timestamp_matrix)
        node_sets = {frozenset(p.node_set) for p in profiles}
        expected_full = frozenset("ABCDEFGHJ")
        expected_partial = frozenset("ABCDEFGH")
        assert expected_full in node_sets
        assert expected_partial in node_sets

    def test_profile_counts(self, timestamp_matrix):
        profiles, _ = ChainClassifier._discover_profiles(timestamp_matrix)
        by_count = {p.chain_count: p for p in profiles}
        assert 5 in by_count  # each profile has 5 chains

    def test_profile_fractions_sum_to_one(self, timestamp_matrix):
        profiles, _ = ChainClassifier._discover_profiles(timestamp_matrix)
        total = sum(p.fraction for p in profiles)
        assert abs(total - 1.0) < 1e-9

    def test_chain_labels_match(self, timestamp_matrix):
        profiles, chain_labels = ChainClassifier._discover_profiles(timestamp_matrix)
        assert chain_labels.height == timestamp_matrix.height
        assert set(chain_labels.columns) == {"chain_id", "profile_id"}


# ---------------------------------------------------------------------------
# Terminal node identification tests
# ---------------------------------------------------------------------------

class TestIdentifyTerminals:
    def test_terminals_with_j(self, edges):
        profile = PathProfile(
            profile_id=0,
            node_set=frozenset("ABCDEFGHJ"),
            chain_count=5,
            fraction=0.5,
        )
        ChainClassifier._identify_terminals([profile], edges)
        assert profile.terminal_nodes == frozenset({"D", "F", "G", "J"})

    def test_terminals_without_j(self, edges):
        profile = PathProfile(
            profile_id=1,
            node_set=frozenset("ABCDEFGH"),
            chain_count=5,
            fraction=0.5,
        )
        ChainClassifier._identify_terminals([profile], edges)
        # H has no children within {A-H} (J is excluded), so H is terminal
        assert profile.terminal_nodes == frozenset({"D", "F", "G", "H"})

    def test_root_is_not_terminal(self, edges):
        """A has children B, C, H — should never be terminal."""
        profile = PathProfile(
            profile_id=0,
            node_set=frozenset("ABCDEFGHJ"),
            chain_count=5,
            fraction=0.5,
        )
        ChainClassifier._identify_terminals([profile], edges)
        assert "A" not in profile.terminal_nodes

    def test_empty_edges(self):
        """With no edges, every node is terminal (no children)."""
        profile = PathProfile(
            profile_id=0,
            node_set=frozenset({"A", "B"}),
            chain_count=1,
            fraction=1.0,
        )
        ChainClassifier._identify_terminals([profile], [])
        assert profile.terminal_nodes == frozenset({"A", "B"})


# ---------------------------------------------------------------------------
# Classification method tests
# ---------------------------------------------------------------------------

def _make_features_and_labels() -> tuple[pl.DataFrame, pl.Series]:
    """Binary features + labels for two profiles.

    Profile 0 (rows 0-4): event:J=1, ctx:juice=1
    Profile 1 (rows 5-9): event:J=0, ctx:juice=0
    All rows have event:A=1, event:H=1.
    """
    rows = []
    for i in range(10):
        has_j = 1 if i < 5 else 0
        rows.append({
            "event:A": 1,
            "event:B": 1,
            "event:H": 1,
            "event:J": has_j,
            "ctx:tea": 1,
            "ctx:juice": has_j,
        })
    features = pl.DataFrame(rows)
    labels = pl.Series("profile_id", [0] * 5 + [1] * 5)
    return features, labels


class TestRatioClassifier:
    def test_fit_predict_perfect(self):
        features, labels = _make_features_and_labels()
        clf = RatioClassifier(min_gap=0.5)
        clf.fit(features, labels)
        preds = clf.predict(features)
        assert preds == labels.to_list()

    def test_feature_importances_identify_j_and_juice(self):
        features, labels = _make_features_and_labels()
        clf = RatioClassifier(min_gap=0.5)
        clf.fit(features, labels)
        important_names = {fi.feature_name for fi in clf.feature_importances()}
        assert "event:J" in important_names
        assert "ctx:juice" in important_names
        # Common features should NOT be important
        assert "event:A" not in important_names

    def test_importances_are_sorted_descending(self):
        features, labels = _make_features_and_labels()
        clf = RatioClassifier(min_gap=0.5)
        clf.fit(features, labels)
        imps = clf.feature_importances()
        for i in range(len(imps) - 1):
            assert imps[i].importance >= imps[i + 1].importance


class TestTreeClassifier:
    def test_fit_predict_perfect(self):
        features, labels = _make_features_and_labels()
        clf = TreeClassifier(max_depth=3, min_samples_leaf=1)
        clf.fit(features, labels)
        preds = clf.predict(features)
        assert preds == labels.to_list()

    def test_feature_importances_nonzero(self):
        features, labels = _make_features_and_labels()
        clf = TreeClassifier(max_depth=3, min_samples_leaf=1)
        clf.fit(features, labels)
        imps = clf.feature_importances()
        assert len(imps) > 0
        # At least one of event:J or ctx:juice should be important
        important_names = {fi.feature_name for fi in imps}
        assert important_names & {"event:J", "ctx:juice"}
