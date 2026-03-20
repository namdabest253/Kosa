"""Single hypothesis generation agent.

Receives activated + falsified paths and generates research hypotheses
using GPT-4o. Every hypothesis includes the reasoning chain (graph path).

Uses the expensive model (GPT-4o) — hypothesis generation is low-volume,
high-stakes. Never use GPT-4o-mini for this.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime

from openai import OpenAI

from kosa.activation.typed_walk import GraphInterface, WalkPath
from kosa.graph.schema import EdgeType

logger = logging.getLogger(__name__)


@dataclass
class Hypothesis:
    """A generated research hypothesis with full provenance."""

    id: str
    text: str
    path: WalkPath
    path_description: str
    confidence: float
    novelty_score: float = 0.0
    feasibility_signals: dict[str, bool] = field(default_factory=dict)
    reasoning: str = ""
    source_techniques: list[str] = field(default_factory=list)
    target_problem: str = ""
    generated_at: str = ""
    model: str = "gpt-4o"
    is_novel: bool = True  # KG-based novelty check result


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

HYPOTHESIS_SYSTEM = """\
You are a research hypothesis generator for ML/AI. Given a graph path \
connecting techniques, problems, and datasets, generate a specific, \
actionable research hypothesis.

A good hypothesis:
- Proposes a SPECIFIC combination or application of existing techniques
- Addresses a SPECIFIC problem or limitation
- Is falsifiable — you can design an experiment to test it
- Explains WHY the combination might work (the mechanism)
- Is non-obvious — not a straightforward application that experts would immediately try

A bad hypothesis:
- Is vague ("X could help with Y")
- Restates known results
- Proposes impossible combinations (check prerequisites)
- Ignores known limitations of the proposed techniques

Respond with valid JSON only."""

HYPOTHESIS_USER = """\
Graph path:
{path_description}

Path confidence: {path_confidence:.3f}

Techniques involved: {techniques}
Problems involved: {problems}

Generate a research hypothesis based on this graph path.

Output format:
{{
  "hypothesis": "One paragraph describing the specific hypothesis",
  "reasoning": "Why this combination might work — what structural/mathematical \
properties make these compatible?",
  "experiment": "How you would test this hypothesis",
  "confidence": 0.75,
  "source_techniques": ["technique1", "technique2"],
  "target_problem": "the problem being addressed"
}}"""


# ---------------------------------------------------------------------------
# Hypothesis generation
# ---------------------------------------------------------------------------


def generate_hypothesis(
    client: OpenAI,
    path: WalkPath,
    model: str = "gpt-4o",
) -> Hypothesis | None:
    """Generate a hypothesis from a single graph path.

    Args:
        client: OpenAI client.
        path: A WalkPath that survived falsification.
        model: Model to use (default: gpt-4o, the expensive model).

    Returns:
        Hypothesis or None on failure.
    """
    # Extract techniques and problems from path
    techniques = [n.node_name for n in path.nodes if n.node_type == "Technique"]
    problems = [n.node_name for n in path.nodes if n.node_type == "Problem"]

    if not techniques:
        logger.warning("Path has no techniques — skipping")
        return None

    path_desc = path.describe()
    messages = [
        {"role": "system", "content": HYPOTHESIS_SYSTEM},
        {
            "role": "user",
            "content": HYPOTHESIS_USER.format(
                path_description=path_desc,
                path_confidence=path.score,
                techniques=", ".join(techniques),
                problems=", ".join(problems) if problems else "none identified",
            ),
        },
    ]

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,  # Some creativity for hypothesis generation
            response_format={"type": "json_object"},
        )
        text = response.choices[0].message.content
        data = json.loads(text)
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        logger.warning(f"Failed to parse hypothesis response: {e}")
        return None
    except Exception as e:
        logger.error(f"Hypothesis generation failed: {e}")
        return None

    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    hypothesis_id = f"hyp_{ts}_{hash(path_desc) % 10000:04d}"

    return Hypothesis(
        id=hypothesis_id,
        text=data.get("hypothesis", ""),
        path=path,
        path_description=path_desc,
        confidence=float(data.get("confidence", 0.5)),
        reasoning=data.get("reasoning", ""),
        source_techniques=data.get("source_techniques", techniques),
        target_problem=data.get("target_problem", problems[0] if problems else ""),
        generated_at=datetime.now(UTC).isoformat(),
        model=model,
    )


def generate_hypotheses_batch(
    client: OpenAI,
    paths: list[WalkPath],
    model: str = "gpt-4o",
    max_hypotheses: int = 50,
) -> list[Hypothesis]:
    """Generate hypotheses from multiple paths.

    Processes paths in order of score (highest first), stops at max_hypotheses.
    """
    sorted_paths = sorted(paths, key=lambda p: p.score, reverse=True)
    hypotheses = []

    for path in sorted_paths[:max_hypotheses]:
        hyp = generate_hypothesis(client, path, model=model)
        if hyp is not None:
            hypotheses.append(hyp)
            logger.info(f"Generated: {hyp.text[:80]}...")

    logger.info(f"Generated {len(hypotheses)} hypotheses from {len(paths)} paths")
    return hypotheses


# ---------------------------------------------------------------------------
# KG-based novelty check
# ---------------------------------------------------------------------------


def check_novelty_kg(
    hypothesis: Hypothesis,
    graph: GraphInterface,
) -> bool:
    """Check if the proposed hypothesis edge already exists in the graph.

    If the hypothesis proposes "technique A mitigates problem B", check
    if there's already a MITIGATES edge between A and B.

    Returns True if the hypothesis is novel (edge doesn't exist).
    """
    if not hypothesis.source_techniques or not hypothesis.target_problem:
        return True  # Can't check — assume novel

    for technique in hypothesis.source_techniques:
        # Find technique node in graph
        # We check if any neighbor of this technique already addresses the target problem
        # This is approximate — we check by name matching
        for path_node in hypothesis.path.nodes:
            if path_node.node_name == technique and path_node.node_type == "Technique":
                neighbors = graph.get_neighbors(path_node.node_id)
                for n in neighbors:
                    if (
                        n.edge_type == EdgeType.MITIGATES
                        and n.node_name.lower() == hypothesis.target_problem.lower()
                    ):
                        logger.info(
                            f"Hypothesis not novel: {technique} already "
                            f"mitigates {hypothesis.target_problem}"
                        )
                        return False

    return True
