"""Tests for falsification layer."""

from kosa.activation.falsification import (
    FalsificationReason,
    check_already_tried,
    check_contradictions,
    check_implementation_constraints,
    check_prerequisites,
    falsify_path,
    falsify_paths,
)
from kosa.activation.typed_walk import (
    InMemoryGraph,
    PathEdge,
    PathNode,
    WalkPath,
)
from kosa.graph.schema import EdgeType


def _make_path(nodes: list[PathNode], edges: list[PathEdge]) -> WalkPath:
    """Helper to build a WalkPath."""
    return WalkPath(nodes=nodes, edges=edges)


def _make_technique(node_id: str, name: str) -> PathNode:
    return PathNode(node_id=node_id, node_type="Technique", node_name=name)


def _make_problem(node_id: str, name: str) -> PathNode:
    return PathNode(node_id=node_id, node_type="Problem", node_name=name)


class TestCheckPrerequisites:
    def test_passes_when_no_dependencies(self):
        g = InMemoryGraph()
        g.add_node("t1", "Technique", "Transformer")

        path = _make_path(
            [_make_technique("t1", "Transformer")],
            [],
        )
        result = check_prerequisites(path, g)
        assert result.passed

    def test_passes_when_deps_exist(self):
        g = InMemoryGraph()
        g.add_node("t1", "Technique", "Transformer")
        g.add_node("t2", "Technique", "self-attention")
        g.add_edge("t1", "t2", EdgeType.USES, confidence=0.9)

        path = _make_path(
            [_make_technique("t1", "Transformer")],
            [],
        )
        result = check_prerequisites(path, g)
        assert result.passed

    def test_fails_when_deps_missing(self):
        g = InMemoryGraph()
        g.add_node("t1", "Technique", "Transformer")
        # Add a USES edge to a node that doesn't exist in graph
        g.edges["t1"] = []
        from kosa.activation.typed_walk import Neighbor

        g.edges["t1"].append(
            Neighbor(
                node_id="missing",
                node_type="Technique",
                node_name="missing-dep",
                edge_type=EdgeType.USES,
                edge_confidence=0.9,
                edge_venue_weight=1.0,
                node_significance=0.5,
                node_degree=1,
            )
        )

        path = _make_path(
            [_make_technique("t1", "Transformer")],
            [],
        )
        result = check_prerequisites(path, g)
        assert not result.passed


class TestCheckContradictions:
    def test_passes_no_contradictions(self):
        g = InMemoryGraph()
        g.add_node("t1", "Technique", "FlashAttention")
        g.add_node("p1", "Problem", "memory scaling")

        path = _make_path(
            [_make_technique("t1", "FlashAttention"), _make_problem("p1", "memory scaling")],
            [PathEdge(edge_type=EdgeType.MITIGATES, confidence=0.9)],
        )
        result = check_contradictions(path, g)
        assert result.passed

    def test_detects_limitation_mitigation_conflict(self):
        g = InMemoryGraph()
        g.add_node("t1", "Technique", "Transformer")
        g.add_node("p1", "Problem", "quadratic memory")
        g.add_edge("t1", "p1", EdgeType.HAS_LIMITATION, confidence=0.9)

        path = _make_path(
            [_make_technique("t1", "Transformer"), _make_problem("p1", "quadratic memory")],
            [PathEdge(edge_type=EdgeType.MITIGATES, confidence=0.8)],
        )
        result = check_contradictions(path, g)
        assert not result.passed
        assert "limitation" in result.reason.lower()


class TestCheckAlreadyTried:
    def test_passes_no_shared_papers(self):
        g = InMemoryGraph()
        g.add_node("t1", "Technique", "Technique A")
        g.add_node("t2", "Technique", "Technique B")

        path = _make_path(
            [_make_technique("t1", "Technique A"), _make_technique("t2", "Technique B")],
            [PathEdge(edge_type=EdgeType.USES, confidence=0.8)],
        )
        result = check_already_tried(path, g)
        assert result.passed

    def test_passes_for_papers(self):
        g = InMemoryGraph()
        g.add_node("p1", "Paper", "Paper A")
        g.add_node("p2", "Paper", "Paper B")

        path = _make_path(
            [
                PathNode(node_id="p1", node_type="Paper", node_name="Paper A"),
                PathNode(node_id="p2", node_type="Paper", node_name="Paper B"),
            ],
            [PathEdge(edge_type=EdgeType.CITES, confidence=1.0)],
        )
        result = check_already_tried(path, g)
        assert result.passed

    def test_short_path_passes(self):
        g = InMemoryGraph()
        path = _make_path([_make_technique("t1", "Solo")], [])
        result = check_already_tried(path, g)
        assert result.passed


class TestCheckImplementationConstraints:
    def test_passes_no_constraints(self):
        g = InMemoryGraph()
        g.add_node("t1", "Technique", "LoRA")
        g.add_node("t2", "Technique", "Quantization")

        path = _make_path(
            [_make_technique("t1", "LoRA"), _make_technique("t2", "Quantization")],
            [PathEdge(edge_type=EdgeType.USES, confidence=0.8)],
        )
        result = check_implementation_constraints(path, g)
        assert result.passed


class TestFalsifyPath:
    def test_clean_path_survives(self):
        g = InMemoryGraph()
        g.add_node("t1", "Technique", "FlashAttention")
        g.add_node("p1", "Problem", "memory scaling")

        path = _make_path(
            [_make_technique("t1", "FlashAttention"), _make_problem("p1", "memory scaling")],
            [PathEdge(edge_type=EdgeType.MITIGATES, confidence=0.9)],
        )
        result = falsify_path(path, g)
        assert result.survived
        assert result.overall == FalsificationReason.PASS

    def test_contradicted_path_fails(self):
        g = InMemoryGraph()
        g.add_node("t1", "Technique", "Transformer")
        g.add_node("p1", "Problem", "quadratic memory")
        g.add_edge("t1", "p1", EdgeType.HAS_LIMITATION, confidence=0.9)

        path = _make_path(
            [_make_technique("t1", "Transformer"), _make_problem("p1", "quadratic memory")],
            [PathEdge(edge_type=EdgeType.MITIGATES, confidence=0.8)],
        )
        result = falsify_path(path, g)
        assert not result.survived
        assert result.overall == FalsificationReason.CONTRADICTION


class TestFalsifyPaths:
    def test_filters_bad_paths(self):
        g = InMemoryGraph()
        g.add_node("t1", "Technique", "FlashAttention")
        g.add_node("t2", "Technique", "Transformer")
        g.add_node("p1", "Problem", "memory scaling")
        g.add_node("p2", "Problem", "quadratic memory")
        g.add_edge("t2", "p2", EdgeType.HAS_LIMITATION, confidence=0.9)

        good_path = _make_path(
            [_make_technique("t1", "FlashAttention"), _make_problem("p1", "memory scaling")],
            [PathEdge(edge_type=EdgeType.MITIGATES, confidence=0.9)],
        )
        bad_path = _make_path(
            [_make_technique("t2", "Transformer"), _make_problem("p2", "quadratic memory")],
            [PathEdge(edge_type=EdgeType.MITIGATES, confidence=0.8)],
        )

        survivors, results = falsify_paths([good_path, bad_path], g)
        assert len(survivors) == 1
        assert len(results) == 2
