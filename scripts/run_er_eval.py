#!/usr/bin/env python3
"""Entity resolution evaluation: measure duplicate rate on extracted entities.

Workflow:
1. Load extraction results (Phase 0 output)
2. Run entity resolution on all extracted entities
3. Measure duplicate rate (target: <5%)
4. If ground truth pairs are provided, compute precision/recall

Usage:
    python scripts/run_er_eval.py --extraction-file results/full_50_v3/extraction.json
    python scripts/run_er_eval.py --extraction-file results/full_50_v3/extraction.json --ground-truth results/er_ground_truth.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime

from kosa.entity_resolution.resolver import (
    evaluate_resolution,
    resolve_entities,
)
from kosa.graph.schema import NodeLabel

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class EREvalResult:
    """Entity resolution evaluation result."""

    timestamp: str = ""
    total_entities: int = 0
    total_techniques: int = 0
    total_problems: int = 0
    total_datasets: int = 0
    pairs_evaluated: int = 0
    same_as_links: int = 0
    merges: int = 0
    duplicate_rate: float = 0.0
    passes_threshold: bool = False  # <5% target
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    top_duplicates: list[dict] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "=== Entity Resolution Evaluation ===",
            f"Date: {self.timestamp}",
            f"Total entities: {self.total_entities}",
            f"  Techniques: {self.total_techniques}",
            f"  Problems: {self.total_problems}",
            f"  Datasets: {self.total_datasets}",
            f"Pairs evaluated: {self.pairs_evaluated}",
            f"SAME_AS links: {self.same_as_links}",
            f"Merges: {self.merges}",
            f"Duplicate rate: {self.duplicate_rate:.1%}",
            f"Threshold (<5%): {'PASS' if self.passes_threshold else 'FAIL'}",
        ]

        if self.precision is not None:
            lines.extend(
                [
                    f"Precision: {self.precision:.3f}",
                    f"Recall: {self.recall:.3f}",
                    f"F1: {self.f1:.3f}",
                ]
            )

        if self.top_duplicates:
            lines.append("\nTop duplicate pairs:")
            for d in self.top_duplicates[:10]:
                lines.append(
                    f"  {d['name_a']} ↔ {d['name_b']} "
                    f"(conf={d['confidence']:.3f}, type={d['node_type']})"
                )

        return "\n".join(lines)


def load_entities_from_extraction(path: str) -> list[tuple[str, NodeLabel]]:
    """Load entities from extraction results JSON."""
    with open(path) as f:
        data = json.load(f)

    entities = []
    seen = set()

    for paper in data:
        for entity in paper.get("entities", []):
            name = entity.get("name", "")
            entity_type = entity.get("type", entity.get("entity_type", ""))

            if not name or not entity_type:
                continue

            try:
                node_type = NodeLabel(entity_type)
            except ValueError:
                continue

            key = (name.lower(), node_type)
            if key not in seen:
                seen.add(key)
                entities.append((name, node_type))

    return entities


def main():
    parser = argparse.ArgumentParser(description="Evaluate entity resolution quality")
    parser.add_argument(
        "--extraction-file",
        required=True,
        help="Path to extraction results JSON",
    )
    parser.add_argument(
        "--ground-truth",
        help="Path to ground truth pairs JSON (optional)",
    )
    parser.add_argument("--output", default="results/er_eval_results.json")
    args = parser.parse_args()

    if not os.path.exists(args.extraction_file):
        print(f"ERROR: {args.extraction_file} not found", file=sys.stderr)
        sys.exit(1)

    # Load entities
    entities = load_entities_from_extraction(args.extraction_file)
    logger.info(f"Loaded {len(entities)} unique entities")

    techniques = [e for e in entities if e[1] == NodeLabel.TECHNIQUE]
    problems = [e for e in entities if e[1] == NodeLabel.PROBLEM]
    datasets = [e for e in entities if e[1] == NodeLabel.DATASET]

    # Run entity resolution
    resolution = resolve_entities(entities)

    # Build result
    result = EREvalResult(
        timestamp=datetime.now(UTC).isoformat(),
        total_entities=len(entities),
        total_techniques=len(techniques),
        total_problems=len(problems),
        total_datasets=len(datasets),
        pairs_evaluated=resolution.pairs_evaluated,
        same_as_links=len(resolution.same_as_links),
        merges=len(resolution.merges),
        duplicate_rate=resolution.duplicate_rate,
        passes_threshold=resolution.duplicate_rate < 0.05,
    )

    # Top duplicates
    all_pairs = resolution.merges + resolution.same_as_links
    all_pairs.sort(key=lambda p: p.confidence, reverse=True)
    result.top_duplicates = [
        {
            "name_a": p.name_a,
            "name_b": p.name_b,
            "confidence": p.confidence,
            "node_type": p.node_type.value,
            "should_merge": p.should_merge,
        }
        for p in all_pairs[:20]
    ]

    # Evaluate against ground truth if provided
    if args.ground_truth and os.path.exists(args.ground_truth):
        with open(args.ground_truth) as f:
            gt_data = json.load(f)

        gt_pairs = {(p["name_a"], p["name_b"]) for p in gt_data}
        predicted_pairs = {(p.name_a, p.name_b) for p in all_pairs}

        metrics = evaluate_resolution(predicted_pairs, gt_pairs)
        result.precision = metrics.precision
        result.recall = metrics.recall
        result.f1 = metrics.f1

    print(result.summary())

    # Save results
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(
            {
                "timestamp": result.timestamp,
                "total_entities": result.total_entities,
                "total_techniques": result.total_techniques,
                "total_problems": result.total_problems,
                "total_datasets": result.total_datasets,
                "pairs_evaluated": result.pairs_evaluated,
                "same_as_links": result.same_as_links,
                "merges": result.merges,
                "duplicate_rate": result.duplicate_rate,
                "passes_threshold": result.passes_threshold,
                "precision": result.precision,
                "recall": result.recall,
                "f1": result.f1,
                "top_duplicates": result.top_duplicates,
            },
            f,
            indent=2,
        )
    logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
