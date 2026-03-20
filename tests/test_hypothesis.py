"""Tests for hypothesis generation agent."""

from unittest.mock import MagicMock

from kosa.activation.typed_walk import (
    InMemoryGraph,
    PathEdge,
    PathNode,
    WalkPath,
)
from kosa.agents.hypothesis import (
    Hypothesis,
    check_novelty_kg,
    generate_hypothesis,
)
from kosa.graph.schema import EdgeType


def _make_path() -> WalkPath:
    """Build a test path: FlashAttention -[MITIGATES]-> memory scaling."""
    return WalkPath(
        nodes=[
            PathNode(
                node_id="t1",
                node_type="Technique",
                node_name="FlashAttention",
                significance=0.9,
            ),
            PathNode(
                node_id="p1",
                node_type="Problem",
                node_name="quadratic memory scaling",
                significance=0.9,
            ),
        ],
        edges=[PathEdge(edge_type=EdgeType.MITIGATES, confidence=0.9)],
        cumulative_confidence=0.9,
        cumulative_significance=0.81,
    )


class TestGenerateHypothesis:
    def test_returns_hypothesis_on_success(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = (
            '{"hypothesis": "Test hypothesis", "reasoning": "Because...", '
            '"experiment": "Run X", "confidence": 0.8, '
            '"source_techniques": ["FlashAttention"], '
            '"target_problem": "quadratic memory scaling"}'
        )
        mock_client.chat.completions.create.return_value = mock_response

        path = _make_path()
        hyp = generate_hypothesis(mock_client, path, model="gpt-4o")

        assert hyp is not None
        assert hyp.text == "Test hypothesis"
        assert hyp.reasoning == "Because..."
        assert hyp.confidence == 0.8
        assert "FlashAttention" in hyp.source_techniques
        assert hyp.model == "gpt-4o"
        assert hyp.generated_at != ""

    def test_returns_none_on_failure(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API error")

        path = _make_path()
        hyp = generate_hypothesis(mock_client, path)
        assert hyp is None

    def test_skips_path_without_techniques(self):
        mock_client = MagicMock()
        path = WalkPath(
            nodes=[PathNode(node_id="p1", node_type="Problem", node_name="some problem")],
            edges=[],
        )
        hyp = generate_hypothesis(mock_client, path)
        assert hyp is None
        mock_client.chat.completions.create.assert_not_called()


class TestCheckNoveltyKG:
    def test_novel_when_no_existing_edge(self):
        g = InMemoryGraph()
        g.add_node("t1", "Technique", "FlashAttention")
        g.add_node("p1", "Problem", "memory scaling")

        hyp = Hypothesis(
            id="test",
            text="Test",
            path=_make_path(),
            path_description="test path",
            confidence=0.8,
            source_techniques=["FlashAttention"],
            target_problem="memory scaling",
        )
        assert check_novelty_kg(hyp, g) is True

    def test_not_novel_when_edge_exists(self):
        g = InMemoryGraph()
        g.add_node("t1", "Technique", "FlashAttention")
        g.add_node("p1", "Problem", "quadratic memory scaling")
        g.add_edge("t1", "p1", EdgeType.MITIGATES, confidence=0.9)

        hyp = Hypothesis(
            id="test",
            text="Test",
            path=_make_path(),
            path_description="test path",
            confidence=0.8,
            source_techniques=["FlashAttention"],
            target_problem="quadratic memory scaling",
        )
        assert check_novelty_kg(hyp, g) is False

    def test_novel_when_no_techniques(self):
        g = InMemoryGraph()
        hyp = Hypothesis(
            id="test",
            text="Test",
            path=WalkPath(),
            path_description="",
            confidence=0.5,
            source_techniques=[],
            target_problem="",
        )
        assert check_novelty_kg(hyp, g) is True
