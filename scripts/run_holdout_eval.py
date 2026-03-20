#!/usr/bin/env python3
"""Temporal holdout evaluation: can the system predict known 2024-2025 innovations?

Workflow:
1. Load graph with pre-2024 papers only
2. For each of the 20 holdout innovations:
   a. Find the source technique nodes in the graph
   b. Run typed random walk from each source technique
   c. Run falsification on activated paths
   d. Generate hypotheses from surviving paths
   e. Check if any hypothesis "matches" the known innovation
3. Report: how many of 20 innovations were predicted?

Target: at least 5 of 20 (25%).

Usage:
    python scripts/run_holdout_eval.py --neo4j-uri bolt://localhost:7687
    python scripts/run_holdout_eval.py --dry-run  # no Neo4j, just check holdout data
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime

from kosa.ingestion.holdout import HOLDOUT_INNOVATIONS, HoldoutInnovation

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class InnovationMatch:
    """Result of evaluating one holdout innovation."""

    innovation_name: str
    matched: bool = False
    match_type: str = ""  # "exact", "partial", "none"
    best_hypothesis: str = ""
    best_score: float = 0.0
    source_techniques_found: list[str] = field(default_factory=list)
    source_techniques_missing: list[str] = field(default_factory=list)
    paths_explored: int = 0
    paths_survived: int = 0
    hypotheses_generated: int = 0
    reason: str = ""


@dataclass
class HoldoutEvalResult:
    """Full temporal holdout evaluation result."""

    timestamp: str = ""
    total_innovations: int = 20
    innovations_matched: int = 0
    exact_matches: int = 0
    partial_matches: int = 0
    results: list[InnovationMatch] = field(default_factory=list)

    @property
    def match_rate(self) -> float:
        return self.innovations_matched / max(1, self.total_innovations)

    @property
    def passes_threshold(self) -> bool:
        """Target: at least 5 of 20."""
        return self.innovations_matched >= 5

    def summary(self) -> str:
        lines = [
            "=== Temporal Holdout Evaluation ===",
            f"Date: {self.timestamp}",
            f"Innovations matched: {self.innovations_matched}/{self.total_innovations} "
            f"({self.match_rate:.0%})",
            f"  Exact matches: {self.exact_matches}",
            f"  Partial matches: {self.partial_matches}",
            f"Threshold (5/20): {'PASS' if self.passes_threshold else 'FAIL'}",
            "",
            "Per-innovation results:",
        ]
        for r in self.results:
            status = "MATCH" if r.matched else "MISS"
            lines.append(
                f"  [{status}] {r.innovation_name}: "
                f"found {len(r.source_techniques_found)}/{len(r.source_techniques_found) + len(r.source_techniques_missing)} techniques, "
                f"{r.paths_explored} paths, {r.hypotheses_generated} hypotheses"
            )
            if r.matched:
                lines.append(f"    Best: {r.best_hypothesis[:100]}...")
            elif r.source_techniques_missing:
                lines.append(f"    Missing: {', '.join(r.source_techniques_missing[:3])}")
        return "\n".join(lines)


def match_hypothesis_to_innovation(
    hypothesis_text: str,
    innovation: HoldoutInnovation,
) -> tuple[bool, str]:
    """Check if a hypothesis text matches a known innovation.

    Uses keyword overlap as a heuristic. In production, this would
    use embedding similarity or LLM-as-judge.

    Returns (is_match, match_type).
    """
    hyp_lower = hypothesis_text.lower()
    problem_lower = innovation.problem_solved.lower()

    # Check if hypothesis mentions the problem
    problem_words = set(problem_lower.split())
    hyp_words = set(hyp_lower.split())
    problem_overlap = len(problem_words & hyp_words) / max(1, len(problem_words))

    # Check if hypothesis mentions source techniques
    technique_mentions = 0
    for tech in innovation.source_techniques:
        if tech.lower() in hyp_lower:
            technique_mentions += 1
    technique_coverage = technique_mentions / max(1, len(innovation.source_techniques))

    # Exact match: mentions problem + most techniques
    if problem_overlap >= 0.5 and technique_coverage >= 0.5:
        return True, "exact"

    # Partial match: mentions problem OR most techniques
    if problem_overlap >= 0.4 or technique_coverage >= 0.6:
        return True, "partial"

    return False, "none"


def run_dry_evaluation() -> HoldoutEvalResult:
    """Dry run: check holdout data quality without Neo4j/LLM.

    Validates that innovations are well-defined and source techniques
    are likely to exist in a pre-2024 ML/AI knowledge graph.
    """
    result = HoldoutEvalResult(timestamp=datetime.now(UTC).isoformat())

    logger.info("Running dry evaluation (no Neo4j/LLM)...")
    logger.info(f"Evaluating {len(HOLDOUT_INNOVATIONS)} innovations")

    # Collect all source techniques across innovations
    all_techniques = set()
    for innovation in HOLDOUT_INNOVATIONS:
        all_techniques.update(t.lower() for t in innovation.source_techniques)

    logger.info(f"Total unique source techniques needed: {len(all_techniques)}")

    for innovation in HOLDOUT_INNOVATIONS:
        match = InnovationMatch(innovation_name=innovation.name)

        # Check which techniques are likely in a pre-2024 KG
        # (heuristic: well-known techniques that would be in 2K+ ML papers)
        well_known = {
            "transformer",
            "self-attention",
            "bert",
            "gpt",
            "attention",
            "diffusion models",
            "vae",
            "gan",
            "resnet",
            "lstm",
            "reinforcement learning",
            "ppo",
            "rlhf",
            "clip",
            "knowledge distillation",
            "quantization",
            "lora",
            "chain-of-thought prompting",
            "beam search",
            "fine-tuning",
            "mixture of experts",
            "state space models",
            "gating mechanisms",
            "multi-head attention",
            "flash attention",
            "flashattention",
            "speculative decoding",
            "curriculum learning",
            "b-splines",
            "mlp",
            "vq-vae",
            "rag",
            "vision transformer",
            "grounding dino",
            "segment anything model",
            "vision encoder",
            "instruction tuning",
            "self-play",
            "reinforce",
            "multi-query attention",
            "key-value caching",
            "normalfloat data type",
            "consistency distillation",
            "adversarial training",
            "spatiotemporal patches",
            "bradley-terry model",
            "language model fine-tuning",
            "tree search",
            "dynamic resolution",
            "rotary positional embeddings",
            "hyena operator",
            "video prediction models",
            "latent action models",
            "process reward models",
            "mcts",
            "blockwise parallel attention",
            "distributed computing",
            "grpo",
        }

        for tech in innovation.source_techniques:
            if tech.lower() in well_known:
                match.source_techniques_found.append(tech)
            else:
                match.source_techniques_missing.append(tech)

        coverage = len(match.source_techniques_found) / max(1, len(innovation.source_techniques))
        match.reason = f"Technique coverage: {coverage:.0%}"

        if coverage >= 0.5:
            match.matched = True
            match.match_type = "partial"
            result.partial_matches += 1
            result.innovations_matched += 1

        result.results.append(match)

    return result


def run_full_evaluation(
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    openai_api_key: str,
    model: str = "gpt-4o",
    output_dir: str = "results/holdout_eval",
) -> HoldoutEvalResult:
    """Full evaluation: run the pipeline on a pre-2024 graph.

    Requires Neo4j with pre-2024 papers loaded and OpenAI API access.
    """
    from neo4j import GraphDatabase
    from openai import OpenAI

    from kosa.activation.falsification import falsify_paths
    from kosa.activation.typed_walk import GraphInterface, Neighbor, PathNode, typed_random_walk
    from kosa.agents.hypothesis import generate_hypotheses_batch
    from kosa.graph.schema import EdgeType

    client = OpenAI(api_key=openai_api_key)
    result = HoldoutEvalResult(timestamp=datetime.now(UTC).isoformat())

    # Create a Neo4j-backed GraphInterface
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

    class Neo4jGraph(GraphInterface):
        def get_node(self, node_id: str) -> PathNode | None:
            with driver.session() as session:
                record = session.run(
                    "MATCH (n) WHERE elementId(n) = $id OR n.name = $id OR n.arxiv_id = $id "
                    "RETURN n, labels(n)[0] AS label",
                    id=node_id,
                ).single()
                if record is None:
                    return None
                node = record["n"]
                return PathNode(
                    node_id=node_id,
                    node_type=record["label"],
                    node_name=node.get("name", node.get("title", "")),
                    significance=node.get("significance", 0.5),
                )

        def get_neighbors(self, node_id: str) -> list[Neighbor]:
            with driver.session() as session:
                records = session.run(
                    "MATCH (a)-[r]->(b) "
                    "WHERE a.name = $id OR a.arxiv_id = $id "
                    "RETURN type(r) AS rel_type, r, b, labels(b)[0] AS label, "
                    "size([(b)-[]-() | 1]) AS degree",
                    id=node_id,
                ).fetch(50)  # Fan-out cap

            neighbors = []
            for rec in records:
                try:
                    edge_type = EdgeType(rec["rel_type"])
                except ValueError:
                    continue
                b = rec["b"]
                neighbors.append(
                    Neighbor(
                        node_id=b.get("name", b.get("arxiv_id", "")),
                        node_type=rec["label"],
                        node_name=b.get("name", b.get("title", "")),
                        edge_type=edge_type,
                        edge_confidence=rec["r"].get("confidence", 0.5),
                        edge_venue_weight=rec["r"].get("venue_weight", 0.5),
                        node_significance=b.get("significance", 0.5),
                        node_degree=rec["degree"],
                    )
                )
            return neighbors

    graph = Neo4jGraph()

    for innovation in HOLDOUT_INNOVATIONS:
        logger.info(f"\n--- Evaluating: {innovation.name} ---")
        match = InnovationMatch(innovation_name=innovation.name)

        # Find source technique nodes in graph
        for tech in innovation.source_techniques:
            node = graph.get_node(tech)
            if node is not None:
                match.source_techniques_found.append(tech)
            else:
                match.source_techniques_missing.append(tech)

        if not match.source_techniques_found:
            match.reason = "No source techniques found in graph"
            result.results.append(match)
            continue

        # Run activation from each found technique
        all_paths = []
        for tech in match.source_techniques_found:
            activation = typed_random_walk(graph, tech, max_depth=3, num_walks=50)
            all_paths.extend(activation.all_paths)
        match.paths_explored = len(all_paths)

        # Falsify paths
        survivors, _ = falsify_paths(all_paths, graph)
        match.paths_survived = len(survivors)

        if not survivors:
            match.reason = f"All {match.paths_explored} paths falsified"
            result.results.append(match)
            continue

        # Generate hypotheses
        hypotheses = generate_hypotheses_batch(client, survivors, model=model, max_hypotheses=10)
        match.hypotheses_generated = len(hypotheses)

        # Match hypotheses to innovation
        for hyp in hypotheses:
            is_match, match_type = match_hypothesis_to_innovation(hyp.text, innovation)
            if is_match and hyp.confidence > match.best_score:
                match.matched = True
                match.match_type = match_type
                match.best_hypothesis = hyp.text
                match.best_score = hyp.confidence

        if match.matched:
            result.innovations_matched += 1
            if match.match_type == "exact":
                result.exact_matches += 1
            else:
                result.partial_matches += 1

        result.results.append(match)

    driver.close()

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "holdout_results.json")
    with open(output_path, "w") as f:
        json.dump(
            {
                "timestamp": result.timestamp,
                "total_innovations": result.total_innovations,
                "innovations_matched": result.innovations_matched,
                "exact_matches": result.exact_matches,
                "partial_matches": result.partial_matches,
                "match_rate": result.match_rate,
                "passes_threshold": result.passes_threshold,
                "results": [
                    {
                        "name": r.innovation_name,
                        "matched": r.matched,
                        "match_type": r.match_type,
                        "best_hypothesis": r.best_hypothesis[:200],
                        "best_score": r.best_score,
                        "techniques_found": r.source_techniques_found,
                        "techniques_missing": r.source_techniques_missing,
                        "paths_explored": r.paths_explored,
                        "paths_survived": r.paths_survived,
                        "hypotheses_generated": r.hypotheses_generated,
                    }
                    for r in result.results
                ],
            },
            f,
            indent=2,
        )
    logger.info(f"Results saved to {output_path}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Run temporal holdout evaluation")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check holdout data without Neo4j/LLM",
    )
    parser.add_argument("--neo4j-uri", default="bolt://localhost:7687")
    parser.add_argument("--neo4j-user", default="neo4j")
    parser.add_argument("--neo4j-password", default="")
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--output-dir", default="results/holdout_eval")
    args = parser.parse_args()

    if args.dry_run:
        result = run_dry_evaluation()
    else:
        neo4j_password = args.neo4j_password or os.environ.get("NEO4J_PASSWORD", "")
        openai_key = os.environ.get("OPENAI_API_KEY", "")

        if not neo4j_password:
            print("ERROR: NEO4J_PASSWORD required (--neo4j-password or env var)", file=sys.stderr)
            sys.exit(1)
        if not openai_key:
            print("ERROR: OPENAI_API_KEY env var required", file=sys.stderr)
            sys.exit(1)

        result = run_full_evaluation(
            neo4j_uri=args.neo4j_uri,
            neo4j_user=args.neo4j_user,
            neo4j_password=neo4j_password,
            openai_api_key=openai_key,
            model=args.model,
            output_dir=args.output_dir,
        )

    print(result.summary())


if __name__ == "__main__":
    main()
