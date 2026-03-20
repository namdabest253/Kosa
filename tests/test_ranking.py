"""Tests for hypothesis ranking (Elo and structural feasibility)."""

from unittest.mock import MagicMock

from kosa.activation.typed_walk import (
    InMemoryGraph,
    PathEdge,
    PathNode,
    WalkPath,
)
from kosa.agents.hypothesis import Hypothesis
from kosa.graph.schema import EdgeType
from kosa.ranking.elo import (
    EloRating,
    FeasibilitySignals,
    TournamentResult,
    check_structural_feasibility,
    compare_pair,
    composite_score,
    expected_score,
    update_elo,
)


def _make_hypothesis(hyp_id: str, text: str, techniques: list[str] | None = None) -> Hypothesis:
    """Helper to create a test hypothesis."""
    path = WalkPath(
        nodes=[
            PathNode(node_id="t1", node_type="Technique", node_name="Tech A", significance=0.9),
            PathNode(node_id="p1", node_type="Problem", node_name="Problem X", significance=0.8),
        ],
        edges=[PathEdge(edge_type=EdgeType.MITIGATES, confidence=0.85)],
        cumulative_confidence=0.85,
        cumulative_significance=0.72,
    )
    return Hypothesis(
        id=hyp_id,
        text=text,
        path=path,
        path_description=path.describe(),
        confidence=0.8,
        source_techniques=techniques or ["Tech A"],
        target_problem="Problem X",
    )


class TestExpectedScore:
    def test_equal_ratings(self):
        assert expected_score(1500, 1500) == 0.5

    def test_higher_rated_favored(self):
        assert expected_score(1700, 1500) > 0.5

    def test_lower_rated_underdog(self):
        assert expected_score(1300, 1500) < 0.5

    def test_symmetric(self):
        ea = expected_score(1600, 1400)
        eb = expected_score(1400, 1600)
        assert abs(ea + eb - 1.0) < 0.001


class TestUpdateElo:
    def test_winner_gains(self):
        new_a, new_b = update_elo(1500, 1500, 1.0)
        assert new_a > 1500
        assert new_b < 1500

    def test_draw_no_change_for_equal(self):
        new_a, new_b = update_elo(1500, 1500, 0.5)
        assert abs(new_a - 1500) < 0.01
        assert abs(new_b - 1500) < 0.01

    def test_upset_win_bigger_change(self):
        # Underdog wins — bigger rating change
        new_a, _ = update_elo(1300, 1700, 1.0)
        normal_a, _ = update_elo(1500, 1500, 1.0)
        assert (new_a - 1300) > (normal_a - 1500)

    def test_ratings_conserved(self):
        """Total Elo in the system stays constant."""
        new_a, new_b = update_elo(1500, 1500, 1.0)
        assert abs((new_a + new_b) - 3000) < 0.01


class TestComparePair:
    def test_returns_winner(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"winner": "A", "reason": "More novel"}'
        mock_client.chat.completions.create.return_value = mock_response

        hyp_a = _make_hypothesis("a", "Hypothesis A")
        hyp_b = _make_hypothesis("b", "Hypothesis B")
        result = compare_pair(mock_client, hyp_a, hyp_b)

        assert result.winner_id == "a"
        assert result.loser_id == "b"
        assert not result.is_draw

    def test_draw_result(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"winner": "draw", "reason": "Equal"}'
        mock_client.chat.completions.create.return_value = mock_response

        hyp_a = _make_hypothesis("a", "Hypothesis A")
        hyp_b = _make_hypothesis("b", "Hypothesis B")
        result = compare_pair(mock_client, hyp_a, hyp_b)

        assert result.is_draw

    def test_handles_api_failure(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("fail")

        hyp_a = _make_hypothesis("a", "A")
        hyp_b = _make_hypothesis("b", "B")
        result = compare_pair(mock_client, hyp_a, hyp_b)

        assert result.is_draw  # Graceful fallback


class TestFeasibilitySignals:
    def test_all_signals_present(self):
        signals = FeasibilitySignals(
            prerequisites_exist=True,
            papers_cocited=True,
            datasets_available=True,
        )
        score = signals.compute_score()
        assert score == 1.0

    def test_no_signals(self):
        signals = FeasibilitySignals()
        score = signals.compute_score()
        assert score == 0.0

    def test_partial_signals(self):
        signals = FeasibilitySignals(prerequisites_exist=True)
        score = signals.compute_score()
        assert 0.0 < score < 1.0


class TestCheckStructuralFeasibility:
    def test_basic_feasibility(self):
        g = InMemoryGraph()
        g.add_node("t1", "Technique", "Tech A")
        g.add_node("p1", "Problem", "Problem X")

        hyp = _make_hypothesis("test", "Test hypothesis")
        signals = check_structural_feasibility(hyp, g)
        assert signals.prerequisites_exist


class TestCompositeScore:
    def test_novel_high_feasibility_high_path(self):
        hyp = _make_hypothesis("test", "Test")
        hyp.is_novel = True
        feasibility = FeasibilitySignals(
            prerequisites_exist=True, papers_cocited=True, datasets_available=True
        )
        feasibility.compute_score()
        score = composite_score(hyp, feasibility)
        assert score > 0.5

    def test_not_novel_penalized(self):
        hyp_novel = _make_hypothesis("a", "Novel")
        hyp_novel.is_novel = True
        hyp_known = _make_hypothesis("b", "Known")
        hyp_known.is_novel = False

        feasibility = FeasibilitySignals(
            prerequisites_exist=True, papers_cocited=True, datasets_available=True
        )
        feasibility.compute_score()

        assert composite_score(hyp_novel, feasibility) > composite_score(hyp_known, feasibility)


class TestTournamentResult:
    def test_ranked_by_elo(self):
        result = TournamentResult()
        result.ratings["a"] = EloRating(hypothesis_id="a", rating=1600)
        result.ratings["b"] = EloRating(hypothesis_id="b", rating=1400)
        result.ratings["c"] = EloRating(hypothesis_id="c", rating=1500)

        ranked = result.ranked()
        assert ranked[0].hypothesis_id == "a"
        assert ranked[1].hypothesis_id == "c"
        assert ranked[2].hypothesis_id == "b"

    def test_top_k(self):
        result = TournamentResult()
        result.ratings["a"] = EloRating(hypothesis_id="a", rating=1600)
        result.ratings["b"] = EloRating(hypothesis_id="b", rating=1400)
        result.ratings["c"] = EloRating(hypothesis_id="c", rating=1500)

        top = result.top_k(2)
        assert top == ["a", "c"]
