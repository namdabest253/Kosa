#!/usr/bin/env python3
"""Run the Phase 0 extraction quality audit on the 50-paper corpus.

Usage:
    python scripts/run_phase0_audit.py [--model gpt-4o-mini] [--limit 10] [--output results/]

Fetches abstracts from arXiv, runs extraction pipeline, saves results as JSON.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from openai import OpenAI

from kosa.config import Settings
from kosa.ingestion.corpus import PHASE0_CORPUS
from kosa.ingestion.extract import PaperExtractionResult, extract_paper, fetch_arxiv_metadata

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def result_to_dict(r: PaperExtractionResult) -> dict:
    """Serialize extraction result to a JSON-compatible dict."""
    return {
        "arxiv_id": r.arxiv_id,
        "title": r.title,
        "authors": r.authors,
        "year": r.year,
        "venue": r.venue,
        "entities": [
            {
                "name": e.name,
                "type": e.entity_type.value,
                "description": e.description,
                "mathematical_structure": e.mathematical_structure or None,
                "bottleneck_class": e.bottleneck_class or None,
            }
            for e in r.entities
        ],
        "relations": [
            {
                "source": rel.source_name,
                "source_type": rel.source_type.value,
                "relation": rel.relation.value,
                "target": rel.target_name,
                "target_type": rel.target_type.value,
                "confidence": rel.confidence,
                "supporting_text": rel.supporting_text,
            }
            for rel in r.relations
        ],
        "novelty": (
            {
                "classification": r.novelty.classification,
                "confidence": r.novelty.confidence,
                "reasoning": r.novelty.reasoning,
            }
            if r.novelty
            else None
        ),
        "errors": r.errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 0 extraction audit")
    parser.add_argument("--model", default="gpt-4o-mini", help="Extraction model")
    parser.add_argument("--limit", type=int, default=0, help="Limit to N papers (0 = all 50)")
    parser.add_argument("--output", default="results", help="Output directory")
    args = parser.parse_args()

    settings = Settings()
    if not settings.openai_api_key:
        print("ERROR: OPENAI_API_KEY not set.", file=sys.stderr)
        sys.exit(1)

    corpus = PHASE0_CORPUS[: args.limit] if args.limit > 0 else PHASE0_CORPUS
    logger.info(f"Running extraction on {len(corpus)} papers with model {args.model}")

    # Fetch abstracts from arXiv
    arxiv_ids = [p.arxiv_id for p in corpus]
    logger.info("Fetching abstracts from arXiv API...")
    metadata = fetch_arxiv_metadata(arxiv_ids)
    logger.info(f"Fetched metadata for {len(metadata)}/{len(arxiv_ids)} papers")

    # Run extraction
    client = OpenAI(api_key=settings.openai_api_key)
    results = []

    for paper in corpus:
        meta = metadata.get(paper.arxiv_id)
        if meta is None:
            logger.warning(f"No arXiv metadata for {paper.arxiv_id}, skipping")
            continue

        logger.info(f"Extracting: {paper.arxiv_id} — {meta['title'][:60]}...")
        result = extract_paper(
            client=client,
            arxiv_id=paper.arxiv_id,
            title=meta["title"],
            abstract=meta["abstract"],
            authors=meta["authors"],
            year=paper.year,
            venue=paper.venue,
            model=args.model,
        )
        results.append(result)

        # Log summary
        n_ent = len(result.entities)
        n_rel = len(result.relations)
        novelty = result.novelty.classification if result.novelty else "N/A"
        logger.info(f"  → {n_ent} entities, {n_rel} relations, novelty={novelty}")

    # Save results
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "phase0_extraction.json"
    with open(out_file, "w") as f:
        json.dump([result_to_dict(r) for r in results], f, indent=2)
    logger.info(f"Results saved to {out_file}")

    # Print summary
    total_entities = sum(len(r.entities) for r in results)
    total_relations = sum(len(r.relations) for r in results)
    total_errors = sum(len(r.errors) for r in results)
    print(f"\n{'=' * 60}")
    print("Phase 0 Extraction Audit Summary")
    print(f"{'=' * 60}")
    print(f"Papers processed: {len(results)}/{len(corpus)}")
    print(f"Total entities:   {total_entities}")
    print(f"Total relations:  {total_relations}")
    print(f"Total errors:     {total_errors}")
    print(f"Avg entities/paper: {total_entities / max(len(results), 1):.1f}")
    print(f"Avg relations/paper: {total_relations / max(len(results), 1):.1f}")


if __name__ == "__main__":
    main()
