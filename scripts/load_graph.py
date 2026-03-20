#!/usr/bin/env python3
"""Load the knowledge graph into Neo4j: papers, citations, entities, relations.

Usage:
    python scripts/load_graph.py --extraction-file results/full_50_v3/extraction.json
    python scripts/load_graph.py --extraction-file results/full_50_v3/extraction.json --skip-citations
    python scripts/load_graph.py --fetch-citations-only

Requires NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD env vars (or .env file).
Optional: SEMANTIC_SCHOLAR_API_KEY for faster citation fetching.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

from kosa.config import Settings
from kosa.graph.loader import GraphLoader
from kosa.graph.schema import EdgeType, NodeLabel
from kosa.ingestion.citations import (
    citation_graph_stats,
    fetch_citations_batch,
    validate_citation_graph,
)
from kosa.ingestion.extract import (
    ExtractedEntity,
    ExtractedRelation,
    NoveltyResult,
    PaperExtractionResult,
)
from kosa.ingestion.pipeline import PaperMetadata, compute_recency_factor

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_extraction_results(path: str) -> list[PaperExtractionResult]:
    """Load extraction results from JSON file."""
    with open(path) as f:
        data = json.load(f)

    results = []
    for item in data:
        entities = []
        for e in item.get("entities", []):
            entity_type = e.get("type", e.get("entity_type", ""))
            entities.append(
                ExtractedEntity(
                    name=e["name"],
                    entity_type=NodeLabel(entity_type),
                    description=e.get("description", ""),
                    mathematical_structure=e.get("mathematical_structure", ""),
                    bottleneck_class=e.get("bottleneck_class", ""),
                    source_paper=e.get("source_paper", item.get("arxiv_id", "")),
                )
            )

        relations = []
        for r in item.get("relations", []):
            try:
                relations.append(
                    ExtractedRelation(
                        source_name=r.get("source", r.get("source_name", "")),
                        source_type=NodeLabel(r.get("source_type", "")),
                        relation=EdgeType(r.get("relation", "")),
                        target_name=r.get("target", r.get("target_name", "")),
                        target_type=NodeLabel(r.get("target_type", "")),
                        confidence=float(r.get("confidence", 0.5)),
                        supporting_text=r.get("supporting_text", ""),
                        source_paper=r.get("source_paper", item.get("arxiv_id", "")),
                    )
                )
            except (ValueError, KeyError) as e:
                logger.warning(f"Skipping invalid relation: {e}")

        novelty = None
        if item.get("novelty"):
            n = item["novelty"]
            novelty = NoveltyResult(
                classification=n.get("classification", ""),
                confidence=float(n.get("confidence", 0.0)),
                reasoning=n.get("reasoning", ""),
            )

        results.append(
            PaperExtractionResult(
                arxiv_id=item["arxiv_id"],
                title=item.get("title", ""),
                abstract=item.get("abstract", ""),
                authors=item.get("authors", []),
                year=item.get("year", 0),
                venue=item.get("venue"),
                entities=entities,
                relations=relations,
                novelty=novelty,
            )
        )

    return results


def results_to_paper_metadata(
    results: list[PaperExtractionResult],
) -> list[PaperMetadata]:
    """Convert extraction results to PaperMetadata for loading."""
    papers = []
    for r in results:
        from kosa.graph.schema import get_venue_weight
        from kosa.ingestion.pipeline import compute_novelty_signal

        tier, weight = get_venue_weight(r.venue)
        novelty_class = r.novelty.classification if r.novelty else None

        paper = PaperMetadata(
            arxiv_id=r.arxiv_id,
            title=r.title,
            abstract=r.abstract,
            authors=r.authors,
            year=r.year,
            venue=r.venue,
            venue_weight=weight,
            recency_factor=compute_recency_factor(r.year),
            citation_signal=0.5,  # default until we have real data
            novelty_signal=compute_novelty_signal(novelty_class),
        )
        paper.compute_significance()
        papers.append(paper)
    return papers


def main() -> None:
    parser = argparse.ArgumentParser(description="Load knowledge graph into Neo4j")
    parser.add_argument(
        "--extraction-file",
        help="Path to extraction results JSON",
    )
    parser.add_argument(
        "--citations-file",
        help="Path to pre-fetched citations JSON (skip S2 API)",
    )
    parser.add_argument(
        "--skip-citations",
        action="store_true",
        help="Skip citation fetching",
    )
    parser.add_argument(
        "--fetch-citations-only",
        action="store_true",
        help="Only fetch citations (no graph loading)",
    )
    parser.add_argument(
        "--output-citations",
        default="results/citations.json",
        help="Where to save fetched citations",
    )
    parser.add_argument(
        "--cocitation-boost",
        type=float,
        default=0.15,
        help="Citation co-occurrence confidence boost",
    )
    args = parser.parse_args()

    settings = Settings()

    # --- Citation fetching ---
    if args.fetch_citations_only or (args.extraction_file and not args.skip_citations):
        if args.citations_file:
            logger.info(f"Loading pre-fetched citations from {args.citations_file}")
            with open(args.citations_file) as f:
                citations = json.load(f)
        else:
            # Get arXiv IDs from extraction file
            if args.extraction_file:
                results = load_extraction_results(args.extraction_file)
                arxiv_ids = [r.arxiv_id for r in results]
            else:
                logger.error("Need --extraction-file to know which papers to fetch citations for")
                sys.exit(1)

            s2_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
            corpus_set = set(arxiv_ids)

            logger.info(f"Fetching citations for {len(arxiv_ids)} papers from Semantic Scholar...")
            citations = fetch_citations_batch(arxiv_ids, api_key=s2_key, corpus_ids=corpus_set)

            # Validate
            citations = validate_citation_graph(citations)

            # Save
            os.makedirs(os.path.dirname(args.output_citations) or ".", exist_ok=True)
            with open(args.output_citations, "w") as f:
                json.dump(citations, f, indent=2)
            logger.info(f"Citations saved to {args.output_citations}")

            # Stats
            stats = citation_graph_stats(citations)
            for k, v in stats.items():
                if k != "isolated_ids":
                    logger.info(f"  {k}: {v}")

        if args.fetch_citations_only:
            return

    # --- Graph loading ---
    if not args.extraction_file:
        logger.error("Need --extraction-file for graph loading")
        sys.exit(1)

    if not settings.neo4j_password:
        logger.error("NEO4J_PASSWORD not set. Configure .env or environment.")
        sys.exit(1)

    results = load_extraction_results(args.extraction_file)
    papers = results_to_paper_metadata(results)

    logger.info(f"Loading {len(papers)} papers into Neo4j at {settings.neo4j_uri}...")

    with GraphLoader(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password) as loader:
        # Layer 0: Paper nodes
        loader.load_papers(papers)

        # Layer 0: CITES edges
        if not args.skip_citations:
            citations_path = args.citations_file or args.output_citations
            if os.path.exists(citations_path):
                with open(citations_path) as f:
                    citations = json.load(f)
                loader.load_citations(citations)
            else:
                logger.warning(f"No citations file at {citations_path}, skipping CITES edges")

        # Layer 1-3: Entities and relations
        loader.load_extraction_results(results)

        # Citation co-occurrence boost
        if not args.skip_citations:
            boosted = loader.apply_citation_cooccurrence_boost(args.cocitation_boost)
            logger.info(f"Applied co-citation boost to {boosted} edges")

        # Stats
        stats = loader.get_graph_stats()
        print("\n=== Graph Statistics ===")
        print(f"Total nodes: {stats['total_nodes']}")
        for label, count in stats["node_counts"].items():
            print(f"  {label}: {count}")
        print(f"Total edges: {stats['total_edges']}")
        for etype, count in stats["edge_counts"].items():
            if count > 0:
                print(f"  {etype}: {count}")
        print(f"Isolated papers: {stats['isolated_papers']}")
        if stats["top_cited_papers"]:
            print("Top cited papers:")
            for p in stats["top_cited_papers"][:5]:
                print(f"  {p['id']}: in={p['in_degree']} out={p['out_degree']}")


if __name__ == "__main__":
    main()
