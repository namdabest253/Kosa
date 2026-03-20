#!/usr/bin/env python3
"""Ablation study: typed random walk vs PPR vs embedding-only.

Compares three activation strategies on the same graph:
1. Typed random walk (our approach)
2. Standard PPR (untyped baseline)
3. Embedding-only similarity (no graph structure)

Metrics:
- Number of useful paths found (paths that survive falsification)
- Path diversity (unique node types in activated set)
- Depth distribution (how far activation reaches)
- Hypothesis quality (if OpenAI API available)

Usage:
    python scripts/run_ablation.py --seed-nodes "Transformer,FlashAttention,Mamba"
    python scripts/run_ablation.py --dry-run  # in-memory test graph
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime

from kosa.activation.typed_walk import (
    InMemoryGraph,
    embedding_similarity_baseline,
    ppr_baseline,
    typed_random_walk,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class AblationRun:
    """Result of one ablation run (one method, one seed)."""

    method: str
    seed_node: str
    activated_nodes: int = 0
    unique_node_types: int = 0
    max_depth_reached: int = 0
    avg_activation_score: float = 0.0
    paths_explored: int = 0
    depth_distribution: dict[int, int] = field(default_factory=dict)


@dataclass
class AblationResult:
    """Full ablation study result."""

    timestamp: str = ""
    seed_nodes: list[str] = field(default_factory=list)
    runs: list[AblationRun] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "=== Ablation Study: Typed Walk vs PPR vs Embedding-Only ===",
            f"Date: {self.timestamp}",
            f"Seed nodes: {', '.join(self.seed_nodes)}",
            "",
        ]

        # Group by method
        by_method: dict[str, list[AblationRun]] = {}
        for run in self.runs:
            by_method.setdefault(run.method, []).append(run)

        for method, runs in by_method.items():
            avg_nodes = sum(r.activated_nodes for r in runs) / max(1, len(runs))
            avg_types = sum(r.unique_node_types for r in runs) / max(1, len(runs))
            avg_depth = sum(r.max_depth_reached for r in runs) / max(1, len(runs))
            avg_score = sum(r.avg_activation_score for r in runs) / max(1, len(runs))

            lines.append(f"  {method}:")
            lines.append(f"    Avg activated nodes: {avg_nodes:.1f}")
            lines.append(f"    Avg node type diversity: {avg_types:.1f}")
            lines.append(f"    Avg max depth: {avg_depth:.1f}")
            lines.append(f"    Avg activation score: {avg_score:.4f}")
            lines.append("")

        return "\n".join(lines)


def run_typed_walk_ablation(graph, seed_id: str) -> AblationRun:
    """Run typed random walk and collect metrics."""
    result = typed_random_walk(graph, seed_id, num_walks=100, max_depth=4)

    run = AblationRun(method="typed_walk", seed_node=seed_id)
    run.activated_nodes = len(result.activated_nodes)
    run.paths_explored = len(result.all_paths)

    if result.activated_nodes:
        types = {n.node_type for n in result.activated_nodes.values()}
        run.unique_node_types = len(types)
        run.avg_activation_score = sum(
            n.activation_score for n in result.activated_nodes.values()
        ) / len(result.activated_nodes)

        by_depth = result.by_depth()
        run.depth_distribution = {d: len(nodes) for d, nodes in by_depth.items()}
        run.max_depth_reached = max(by_depth.keys()) if by_depth else 0

    return run


def run_ppr_ablation(graph, seed_id: str) -> AblationRun:
    """Run PPR baseline and collect metrics."""
    scores = ppr_baseline(graph, seed_id, max_iterations=50, top_k=50)

    run = AblationRun(method="ppr", seed_node=seed_id)
    run.activated_nodes = len(scores) - 1  # exclude seed
    run.avg_activation_score = sum(v for k, v in scores.items() if k != seed_id) / max(
        1, run.activated_nodes
    )

    # PPR doesn't track depth or types — count unique types from graph
    types = set()
    for node_id in scores:
        node = graph.get_node(node_id)
        if node:
            types.add(node.node_type)
    run.unique_node_types = len(types)

    return run


def run_embedding_ablation(graph, seed_id: str) -> AblationRun:
    """Run embedding-only baseline (simulated with uniform similarities)."""
    # Without real embeddings, simulate with graph distance
    # In production, this uses pre-computed cosine similarities
    neighbors = graph.get_neighbors(seed_id)
    sims = {}
    for n in neighbors:
        sims[n.node_id] = 0.8  # Direct neighbors get high similarity
        # Second hop
        for n2 in graph.get_neighbors(n.node_id):
            if n2.node_id not in sims and n2.node_id != seed_id:
                sims[n2.node_id] = 0.5

    scores = embedding_similarity_baseline(sims, top_k=50)

    run = AblationRun(method="embedding_only", seed_node=seed_id)
    run.activated_nodes = len(scores)
    run.avg_activation_score = sum(scores.values()) / max(1, len(scores))

    types = set()
    for node_id in scores:
        node = graph.get_node(node_id)
        if node:
            types.add(node.node_type)
    run.unique_node_types = len(types)

    return run


def build_demo_graph() -> InMemoryGraph:
    """Build a demo graph for dry-run ablation."""
    from kosa.graph.schema import EdgeType

    g = InMemoryGraph()

    # Techniques
    g.add_node("transformer", "Technique", "Transformer", 0.95)
    g.add_node("self_attn", "Technique", "self-attention", 0.9)
    g.add_node("flash_attn", "Technique", "FlashAttention", 0.85)
    g.add_node("mamba", "Technique", "Mamba", 0.8)
    g.add_node("lora", "Technique", "LoRA", 0.8)
    g.add_node("quantization", "Technique", "quantization", 0.7)
    g.add_node("moe", "Technique", "Mixture of Experts", 0.75)
    g.add_node("dpo", "Technique", "DPO", 0.8)
    g.add_node("ppo", "Technique", "PPO", 0.7)

    # Problems
    g.add_node("quad_mem", "Problem", "quadratic memory scaling", 0.9)
    g.add_node("slow_inf", "Problem", "slow inference", 0.85)
    g.add_node("finetune_cost", "Problem", "fine-tuning cost", 0.8)
    g.add_node("rlhf_instab", "Problem", "RLHF training instability", 0.75)

    # Datasets
    g.add_node("imagenet", "Dataset", "ImageNet", 0.9)
    g.add_node("mmlu", "Dataset", "MMLU", 0.85)

    # Edges
    g.add_edge("transformer", "self_attn", EdgeType.USES, 0.95)
    g.add_edge("transformer", "quad_mem", EdgeType.HAS_LIMITATION, 0.9)
    g.add_edge("flash_attn", "quad_mem", EdgeType.MITIGATES, 0.9)
    g.add_edge("flash_attn", "self_attn", EdgeType.IMPROVES_OVER, 0.85)
    g.add_edge("mamba", "quad_mem", EdgeType.MITIGATES, 0.8)
    g.add_edge("lora", "finetune_cost", EdgeType.MITIGATES, 0.85)
    g.add_edge("quantization", "slow_inf", EdgeType.MITIGATES, 0.8)
    g.add_edge("moe", "slow_inf", EdgeType.MITIGATES, 0.7)
    g.add_edge("dpo", "rlhf_instab", EdgeType.MITIGATES, 0.85)
    g.add_edge("dpo", "ppo", EdgeType.IMPROVES_OVER, 0.8)

    # Reverse edges for traversal
    g.add_edge("quad_mem", "flash_attn", EdgeType.MITIGATES, 0.9)
    g.add_edge("quad_mem", "mamba", EdgeType.MITIGATES, 0.8)
    g.add_edge("quad_mem", "transformer", EdgeType.HAS_LIMITATION, 0.9)
    g.add_edge("slow_inf", "quantization", EdgeType.MITIGATES, 0.8)
    g.add_edge("slow_inf", "moe", EdgeType.MITIGATES, 0.7)
    g.add_edge("finetune_cost", "lora", EdgeType.MITIGATES, 0.85)
    g.add_edge("self_attn", "flash_attn", EdgeType.IMPROVES_OVER, 0.85)
    g.add_edge("rlhf_instab", "dpo", EdgeType.MITIGATES, 0.85)

    return g


def main():
    parser = argparse.ArgumentParser(description="Run ablation study")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use in-memory demo graph",
    )
    parser.add_argument(
        "--seed-nodes",
        default="transformer,flash_attn,mamba",
        help="Comma-separated seed node IDs",
    )
    parser.add_argument("--output", default="results/ablation_results.json")
    args = parser.parse_args()

    seeds = [s.strip() for s in args.seed_nodes.split(",")]

    if args.dry_run:
        graph = build_demo_graph()
    else:
        print("Full Neo4j-backed ablation not yet implemented.", file=sys.stderr)
        print("Use --dry-run for demo.", file=sys.stderr)
        sys.exit(1)

    result = AblationResult(
        timestamp=datetime.now(UTC).isoformat(),
        seed_nodes=seeds,
    )

    for seed in seeds:
        logger.info(f"\n--- Seed: {seed} ---")

        # Typed random walk
        run = run_typed_walk_ablation(graph, seed)
        result.runs.append(run)
        logger.info(
            f"  Typed walk: {run.activated_nodes} nodes, score={run.avg_activation_score:.4f}"
        )

        # PPR baseline
        run = run_ppr_ablation(graph, seed)
        result.runs.append(run)
        logger.info(f"  PPR: {run.activated_nodes} nodes, score={run.avg_activation_score:.4f}")

        # Embedding-only baseline
        run = run_embedding_ablation(graph, seed)
        result.runs.append(run)
        logger.info(
            f"  Embedding: {run.activated_nodes} nodes, score={run.avg_activation_score:.4f}"
        )

    print(result.summary())

    # Save results
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(
            {
                "timestamp": result.timestamp,
                "seed_nodes": result.seed_nodes,
                "runs": [
                    {
                        "method": r.method,
                        "seed_node": r.seed_node,
                        "activated_nodes": r.activated_nodes,
                        "unique_node_types": r.unique_node_types,
                        "max_depth_reached": r.max_depth_reached,
                        "avg_activation_score": r.avg_activation_score,
                        "depth_distribution": r.depth_distribution,
                    }
                    for r in result.runs
                ],
            },
            f,
            indent=2,
        )
    logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
