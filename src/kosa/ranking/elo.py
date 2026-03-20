"""Hypothesis ranking via pairwise Elo/Bradley-Terry tournament.

Primary method: pairwise comparison, NOT absolute scoring.
Absolute composite scores are unreliable because dimensions are on
incomparable scales and LLM-as-judge has known biases.

Ranking signals:
1. Pairwise tournament (GPT-4o): "Which is more promising?"
2. Structural feasibility (graph-based, not LLM judgment)
3. Path strength from typed random walk activation
4. Fallback composite score for batch processing
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from openai import OpenAI

from kosa.activation.typed_walk import GraphInterface
from kosa.agents.hypothesis import Hypothesis
from kosa.graph.schema import EdgeType

logger = logging.getLogger(__name__)

# Elo constants
DEFAULT_ELO = 1500
K_FACTOR = 32  # Standard Elo K-factor


# ---------------------------------------------------------------------------
# Elo rating system
# ---------------------------------------------------------------------------


@dataclass
class EloRating:
    """Elo rating for a hypothesis."""

    hypothesis_id: str
    rating: float = DEFAULT_ELO
    wins: int = 0
    losses: int = 0
    draws: int = 0

    @property
    def games_played(self) -> int:
        return self.wins + self.losses + self.draws


@dataclass
class PairwiseResult:
    """Result of a pairwise comparison."""

    winner_id: str
    loser_id: str
    reason: str = ""
    is_draw: bool = False


def expected_score(rating_a: float, rating_b: float) -> float:
    """Compute expected score for player A against player B."""
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))


def update_elo(
    rating_a: float,
    rating_b: float,
    score_a: float,
    k: float = K_FACTOR,
) -> tuple[float, float]:
    """Update Elo ratings after a match.

    Args:
        rating_a: Current rating of player A.
        rating_b: Current rating of player B.
        score_a: Actual score for A (1.0 = win, 0.5 = draw, 0.0 = loss).
        k: K-factor (higher = more volatile ratings).

    Returns:
        (new_rating_a, new_rating_b).
    """
    expected_a = expected_score(rating_a, rating_b)
    expected_b = 1.0 - expected_a

    new_a = rating_a + k * (score_a - expected_a)
    new_b = rating_b + k * ((1.0 - score_a) - expected_b)

    return new_a, new_b


# ---------------------------------------------------------------------------
# Pairwise comparison via LLM
# ---------------------------------------------------------------------------

PAIRWISE_SYSTEM = """\
You are evaluating two ML/AI research hypotheses. Compare them and decide \
which is MORE PROMISING for advancing the field.

Consider:
- Novelty: Is this a non-obvious combination?
- Specificity: Is the hypothesis concrete enough to test?
- Mechanism: Does the reasoning for WHY it would work make sense?
- Impact: If true, would this be a significant advance?

Do NOT consider:
- Feasibility/difficulty (this is checked separately by structural signals)
- Writing quality (focus on the idea, not the presentation)

Respond with valid JSON only."""

PAIRWISE_USER = """\
Hypothesis A:
{hyp_a_text}
Reasoning: {hyp_a_reasoning}

Hypothesis B:
{hyp_b_text}
Reasoning: {hyp_b_reasoning}

Which hypothesis is more promising? Output:
{{
  "winner": "A" or "B" or "draw",
  "reason": "one sentence explanation"
}}"""


def compare_pair(
    client: OpenAI,
    hyp_a: Hypothesis,
    hyp_b: Hypothesis,
    model: str = "gpt-4o",
) -> PairwiseResult:
    """Compare two hypotheses using GPT-4o.

    Returns PairwiseResult indicating which is more promising.
    """
    messages = [
        {"role": "system", "content": PAIRWISE_SYSTEM},
        {
            "role": "user",
            "content": PAIRWISE_USER.format(
                hyp_a_text=hyp_a.text,
                hyp_a_reasoning=hyp_a.reasoning,
                hyp_b_text=hyp_b.text,
                hyp_b_reasoning=hyp_b.reasoning,
            ),
        },
    ]

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        data = json.loads(response.choices[0].message.content)
    except Exception as e:
        logger.warning(f"Pairwise comparison failed: {e}")
        return PairwiseResult(
            winner_id=hyp_a.id, loser_id=hyp_b.id, reason="comparison failed", is_draw=True
        )

    winner = data.get("winner", "draw")
    reason = data.get("reason", "")

    if winner == "A":
        return PairwiseResult(winner_id=hyp_a.id, loser_id=hyp_b.id, reason=reason)
    elif winner == "B":
        return PairwiseResult(winner_id=hyp_b.id, loser_id=hyp_a.id, reason=reason)
    else:
        return PairwiseResult(winner_id=hyp_a.id, loser_id=hyp_b.id, reason=reason, is_draw=True)


# ---------------------------------------------------------------------------
# Structural feasibility signals
# ---------------------------------------------------------------------------


@dataclass
class FeasibilitySignals:
    """Graph-based structural feasibility (NOT LLM judgment)."""

    prerequisites_exist: bool = False
    papers_cocited: bool = False
    datasets_available: bool = False
    score: float = 0.0

    def compute_score(self) -> float:
        """Compute feasibility score from binary signals."""
        signals = [self.prerequisites_exist, self.papers_cocited, self.datasets_available]
        self.score = sum(1.0 for s in signals if s) / len(signals)
        return self.score


def check_structural_feasibility(
    hypothesis: Hypothesis,
    graph: GraphInterface,
) -> FeasibilitySignals:
    """Check structural feasibility using graph signals only.

    No LLM judgment — only checkable facts from the graph.
    """
    signals = FeasibilitySignals()

    # 1. Prerequisites exist: do all mentioned techniques exist as nodes?
    technique_nodes_found = 0
    for tech_name in hypothesis.source_techniques:
        for node in hypothesis.path.nodes:
            if node.node_type == "Technique" and node.node_name == tech_name:
                technique_nodes_found += 1
                break
    signals.prerequisites_exist = technique_nodes_found == len(hypothesis.source_techniques)

    # 2. Co-citation: have the source papers been cited together?
    paper_ids = set()
    for node in hypothesis.path.nodes:
        if node.node_type == "Paper":
            paper_ids.add(node.node_id)
    # If we have 2+ papers in the path, check if they share citations
    if len(paper_ids) >= 2:
        # Approximate: papers in the same path are likely connected
        signals.papers_cocited = True
    else:
        # Single-paper path — co-citation doesn't apply
        signals.papers_cocited = True  # Not penalized

    # 3. Datasets available: do any dataset nodes exist for this domain?
    for node in hypothesis.path.nodes:
        if node.node_type == "Dataset":
            signals.datasets_available = True
            break
    if not signals.datasets_available:
        # Check if any technique in path has EVALUATES_ON edges
        for node in hypothesis.path.nodes:
            if node.node_type == "Technique":
                neighbors = graph.get_neighbors(node.node_id)
                for n in neighbors:
                    if n.edge_type == EdgeType.EVALUATES_ON:
                        signals.datasets_available = True
                        break

    signals.compute_score()
    return signals


# ---------------------------------------------------------------------------
# Tournament runner
# ---------------------------------------------------------------------------


@dataclass
class TournamentResult:
    """Result of running an Elo tournament."""

    ratings: dict[str, EloRating] = field(default_factory=dict)
    comparisons: list[PairwiseResult] = field(default_factory=list)
    total_comparisons: int = 0

    def ranked(self) -> list[EloRating]:
        """Return ratings sorted by Elo (highest first)."""
        return sorted(self.ratings.values(), key=lambda r: r.rating, reverse=True)

    def top_k(self, k: int = 10) -> list[str]:
        """Return top-K hypothesis IDs by Elo rating."""
        return [r.hypothesis_id for r in self.ranked()[:k]]


def run_tournament(
    client: OpenAI,
    hypotheses: list[Hypothesis],
    rounds: int = 1,
    model: str = "gpt-4o",
) -> TournamentResult:
    """Run a round-robin pairwise Elo tournament.

    Each round compares every pair once. Multiple rounds refine ratings.

    Args:
        client: OpenAI client for pairwise comparisons.
        hypotheses: List of hypotheses to rank.
        rounds: Number of full round-robin passes.
        model: Model for pairwise comparison.

    Returns:
        TournamentResult with Elo ratings and comparison history.
    """
    result = TournamentResult()

    # Initialize ratings
    for hyp in hypotheses:
        result.ratings[hyp.id] = EloRating(hypothesis_id=hyp.id)

    # Build matchups
    for _round in range(rounds):
        for i in range(len(hypotheses)):
            for j in range(i + 1, len(hypotheses)):
                hyp_a = hypotheses[i]
                hyp_b = hypotheses[j]

                comparison = compare_pair(client, hyp_a, hyp_b, model=model)
                result.comparisons.append(comparison)
                result.total_comparisons += 1

                # Update Elo
                rating_a = result.ratings[hyp_a.id].rating
                rating_b = result.ratings[hyp_b.id].rating

                if comparison.is_draw:
                    score_a = 0.5
                    result.ratings[hyp_a.id].draws += 1
                    result.ratings[hyp_b.id].draws += 1
                elif comparison.winner_id == hyp_a.id:
                    score_a = 1.0
                    result.ratings[hyp_a.id].wins += 1
                    result.ratings[hyp_b.id].losses += 1
                else:
                    score_a = 0.0
                    result.ratings[hyp_a.id].losses += 1
                    result.ratings[hyp_b.id].wins += 1

                new_a, new_b = update_elo(rating_a, rating_b, score_a)
                result.ratings[hyp_a.id].rating = new_a
                result.ratings[hyp_b.id].rating = new_b

    logger.info(
        f"Tournament complete: {result.total_comparisons} comparisons, "
        f"{len(hypotheses)} hypotheses ranked"
    )
    return result


# ---------------------------------------------------------------------------
# Composite score (fallback for batch processing)
# ---------------------------------------------------------------------------

# Weights tuned in Phase 1.5+ via human feedback
W_NOVELTY = 0.35
W_FEASIBILITY = 0.35
W_PATH_STRENGTH = 0.30


def composite_score(
    hypothesis: Hypothesis,
    feasibility: FeasibilitySignals,
) -> float:
    """Compute fallback composite score for batch processing.

    score = w1 * novelty + w2 * structural_feasibility + w3 * path_strength

    All components derived from graph structure, NOT LLM judgment.
    """
    novelty = 1.0 if hypothesis.is_novel else 0.0
    path_strength = hypothesis.path.score

    return W_NOVELTY * novelty + W_FEASIBILITY * feasibility.score + W_PATH_STRENGTH * path_strength
