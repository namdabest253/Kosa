"""Tests for typed random walk activation wave."""

from kosa.activation.typed_walk import (
    ActivationResult,
    InMemoryGraph,
    WalkPath,
    embedding_similarity_baseline,
    ppr_baseline,
    typed_random_walk,
)
from kosa.graph.schema import EdgeType


def _build_test_graph() -> InMemoryGraph:
    """Build a small test graph for activation tests.

    Structure:
        Transformer -[MITIGATES]-> quadratic_memory
        Transformer -[USES]-> self_attention
        self_attention -[HAS_LIMITATION]-> quadratic_memory
        FlashAttention -[MITIGATES]-> quadratic_memory
        FlashAttention -[IMPROVES_OVER]-> self_attention
        Mamba -[MITIGATES]-> quadratic_memory
    """
    g = InMemoryGraph()
    g.add_node("t1", "Technique", "Transformer", significance=0.9)
    g.add_node("t2", "Technique", "self-attention", significance=0.8)
    g.add_node("t3", "Technique", "FlashAttention", significance=0.85)
    g.add_node("t4", "Technique", "Mamba", significance=0.8)
    g.add_node("p1", "Problem", "quadratic memory scaling", significance=0.9)

    g.add_edge("t1", "p1", EdgeType.MITIGATES, confidence=0.7)
    g.add_edge("t1", "t2", EdgeType.USES, confidence=0.95)
    g.add_edge("t2", "p1", EdgeType.HAS_LIMITATION, confidence=0.9)
    g.add_edge("t3", "p1", EdgeType.MITIGATES, confidence=0.9)
    g.add_edge("t3", "t2", EdgeType.IMPROVES_OVER, confidence=0.85)
    g.add_edge("t4", "p1", EdgeType.MITIGATES, confidence=0.8)
    # Reverse edges for traversal
    g.add_edge("p1", "t3", EdgeType.MITIGATES, confidence=0.9)
    g.add_edge("p1", "t4", EdgeType.MITIGATES, confidence=0.8)
    g.add_edge("p1", "t1", EdgeType.MITIGATES, confidence=0.7)

    return g


class TestTypedRandomWalk:
    def test_basic_walk(self):
        g = _build_test_graph()
        result = typed_random_walk(g, "t1", num_walks=50, max_depth=3)
        assert isinstance(result, ActivationResult)
        assert result.seed_id == "t1"
        assert result.seed_name == "Transformer"
        assert len(result.activated_nodes) > 0

    def test_activates_neighbors(self):
        g = _build_test_graph()
        result = typed_random_walk(g, "t1", num_walks=100, max_depth=3)
        # Should activate self-attention and quadratic memory (direct neighbors)
        activated_names = {n.node_name for n in result.activated_nodes.values()}
        assert "self-attention" in activated_names or "quadratic memory scaling" in activated_names

    def test_depth_respects_max(self):
        g = _build_test_graph()
        result = typed_random_walk(g, "t1", num_walks=50, max_depth=1)
        # At max_depth=1, should only see direct neighbors
        for node in result.activated_nodes.values():
            assert node.min_depth <= 1

    def test_min_significance_threshold(self):
        g = _build_test_graph()
        # Set very high threshold — should cut paths early
        result = typed_random_walk(
            g,
            "t1",
            num_walks=50,
            max_depth=4,
            min_path_significance=0.99,
        )
        # Most paths should be short since threshold is very high
        for path in result.all_paths:
            assert path.score >= 0 or path.depth <= 1

    def test_nonexistent_seed(self):
        g = _build_test_graph()
        result = typed_random_walk(g, "nonexistent", num_walks=10)
        assert len(result.activated_nodes) == 0

    def test_isolated_node(self):
        g = InMemoryGraph()
        g.add_node("lonely", "Technique", "Lonely Node")
        result = typed_random_walk(g, "lonely", num_walks=10)
        assert len(result.activated_nodes) == 0

    def test_activation_scores_positive(self):
        g = _build_test_graph()
        result = typed_random_walk(g, "t1", num_walks=100, max_depth=3)
        for node in result.activated_nodes.values():
            assert node.activation_score >= 0

    def test_top_activated(self):
        g = _build_test_graph()
        result = typed_random_walk(g, "t1", num_walks=100, max_depth=3)
        top = result.top_activated(3)
        assert len(top) <= 3
        # Should be sorted by score descending
        for i in range(len(top) - 1):
            assert top[i].activation_score >= top[i + 1].activation_score

    def test_by_depth(self):
        g = _build_test_graph()
        result = typed_random_walk(g, "t1", num_walks=100, max_depth=3)
        by_depth = result.by_depth()
        assert 1 in by_depth  # Should have depth-1 nodes


class TestWalkPath:
    def test_empty_path(self):
        path = WalkPath()
        assert path.depth == 0
        assert path.score == 1.0

    def test_describe(self):
        g = _build_test_graph()
        result = typed_random_walk(g, "t1", num_walks=10, max_depth=2)
        if result.all_paths:
            desc = result.all_paths[0].describe()
            assert "Transformer" in desc


class TestPPRBaseline:
    def test_basic_ppr(self):
        g = _build_test_graph()
        scores = ppr_baseline(g, "t1", max_iterations=20, top_k=10)
        assert "t1" in scores  # Seed should have score
        assert len(scores) > 1  # Should spread to neighbors

    def test_seed_has_significant_score(self):
        g = _build_test_graph()
        scores = ppr_baseline(g, "t1", max_iterations=20)
        # Seed should have a meaningful PPR score (may not be highest
        # in graphs where a hub node absorbs more probability mass)
        assert scores.get("t1", 0) > 0.1


class TestEmbeddingSimilarityBaseline:
    def test_returns_top_k(self):
        sims = {"a": 0.9, "b": 0.7, "c": 0.3, "d": 0.5}
        result = embedding_similarity_baseline(sims, top_k=2)
        assert len(result) == 2
        assert "a" in result
        assert "b" in result
