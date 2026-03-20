"""Neo4j graph loader: loads papers, citations, entities, and relations.

Provides batch loading operations for building the knowledge graph:
- Layer 0: Paper nodes + CITES edges (ground truth)
- Layer 1-3: Technique/Problem/Dataset nodes + extracted relation edges

Uses the sync Neo4j driver for batch operations (not the async FastAPI driver).
All operations are idempotent via MERGE.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from neo4j import GraphDatabase

from kosa.graph.schema import (
    EdgeType,
    NodeLabel,
    get_venue_weight,
)
from kosa.ingestion.extract import (
    ExtractedEntity,
    ExtractedRelation,
    PaperExtractionResult,
)
from kosa.ingestion.pipeline import PaperMetadata

logger = logging.getLogger(__name__)

# Batch sizes for Neo4j transactions
BATCH_SIZE = 100


class GraphLoader:
    """Loads data into Neo4j for knowledge graph construction."""

    def __init__(self, uri: str, user: str, password: str):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._driver.verify_connectivity()
        logger.info(f"Connected to Neo4j at {uri}")

    def close(self):
        self._driver.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # -----------------------------------------------------------------
    # Layer 0: Paper nodes + CITES edges
    # -----------------------------------------------------------------

    def load_papers(self, papers: list[PaperMetadata]) -> int:
        """Load Paper nodes into Neo4j. Returns count of papers loaded.

        Uses MERGE on arxiv_id to be idempotent.
        """
        loaded = 0
        for i in range(0, len(papers), BATCH_SIZE):
            batch = papers[i : i + BATCH_SIZE]
            params = [_paper_to_props(p) for p in batch]
            with self._driver.session() as session:
                result = session.run(
                    """
                    UNWIND $papers AS p
                    MERGE (paper:Paper {arxiv_id: p.arxiv_id})
                    SET paper += p
                    RETURN count(paper) AS cnt
                    """,
                    papers=params,
                )
                loaded += result.single()["cnt"]

        logger.info(f"Loaded {loaded} Paper nodes")
        return loaded

    def load_citations(self, citations: dict[str, list[str]]) -> int:
        """Load CITES edges between Paper nodes. Returns count of edges created.

        Only creates edges between papers that already exist in the graph.
        Uses MERGE to be idempotent.
        """
        # Flatten to list of (source, target) pairs
        edges = []
        for source_id, cited_ids in citations.items():
            for target_id in cited_ids:
                edges.append(
                    {
                        "source": source_id,
                        "target": target_id,
                        "created_at": datetime.now(UTC).isoformat(),
                    }
                )

        loaded = 0
        for i in range(0, len(edges), BATCH_SIZE):
            batch = edges[i : i + BATCH_SIZE]
            with self._driver.session() as session:
                result = session.run(
                    """
                    UNWIND $edges AS e
                    MATCH (a:Paper {arxiv_id: e.source})
                    MATCH (b:Paper {arxiv_id: e.target})
                    MERGE (a)-[r:CITES]->(b)
                    SET r.source_paper = e.source, r.created_at = e.created_at
                    RETURN count(r) AS cnt
                    """,
                    edges=batch,
                )
                loaded += result.single()["cnt"]

        logger.info(f"Loaded {loaded} CITES edges")
        return loaded

    # -----------------------------------------------------------------
    # Layer 1-3: Concept nodes + extracted relation edges
    # -----------------------------------------------------------------

    def load_extraction_results(self, results: list[PaperExtractionResult]) -> dict[str, int]:
        """Load all extracted entities and relations from a batch of papers.

        Returns dict with counts: {techniques, problems, datasets, relations}.
        """
        counts = {
            "techniques": 0,
            "problems": 0,
            "datasets": 0,
            "relations": 0,
        }

        for result in results:
            for entity in result.entities:
                if entity.entity_type == NodeLabel.TECHNIQUE:
                    counts["techniques"] += self._load_technique(entity)
                elif entity.entity_type == NodeLabel.PROBLEM:
                    counts["problems"] += self._load_problem(entity)
                elif entity.entity_type == NodeLabel.DATASET:
                    counts["datasets"] += self._load_dataset(entity)

            counts["relations"] += self._load_relations(result.relations, result.venue)

        logger.info(
            f"Loaded entities: {counts['techniques']} techniques, "
            f"{counts['problems']} problems, {counts['datasets']} datasets, "
            f"{counts['relations']} relations"
        )
        return counts

    def _load_technique(self, entity: ExtractedEntity) -> int:
        """Load a Technique node. Returns 1 if created/updated, 0 on error."""
        with self._driver.session() as session:
            result = session.run(
                """
                MERGE (t:Technique {name: $name})
                SET t.description = $description,
                    t.mathematical_structure = $math_structure,
                    t.properties = $properties
                RETURN count(t) AS cnt
                """,
                name=entity.name,
                description=entity.description,
                math_structure=entity.mathematical_structure,
                properties=[],
            )
            return result.single()["cnt"]

    def _load_problem(self, entity: ExtractedEntity) -> int:
        """Load a Problem node."""
        with self._driver.session() as session:
            result = session.run(
                """
                MERGE (p:Problem {name: $name})
                SET p.description = $description,
                    p.bottleneck_class = $bottleneck_class
                RETURN count(p) AS cnt
                """,
                name=entity.name,
                description=entity.description,
                bottleneck_class=entity.bottleneck_class,
            )
            return result.single()["cnt"]

    def _load_dataset(self, entity: ExtractedEntity) -> int:
        """Load a Dataset node."""
        with self._driver.session() as session:
            result = session.run(
                """
                MERGE (d:Dataset {name: $name})
                SET d.description = $description,
                    d.domain = '',
                    d.size = ''
                RETURN count(d) AS cnt
                """,
                name=entity.name,
                description=entity.description,
            )
            return result.single()["cnt"]

    def _load_relations(self, relations: list[ExtractedRelation], venue: str | None) -> int:
        """Load extracted relation edges. Returns count of edges created."""
        _, venue_weight = get_venue_weight(venue)
        now = datetime.now(UTC).isoformat()
        loaded = 0

        for rel in relations:
            props = {
                "confidence": rel.confidence,
                "source_paper": rel.source_paper,
                "source_venue": venue or "arXiv",
                "venue_weight": venue_weight,
                "supporting_text": rel.supporting_text,
                "extraction_method": "gpt-4o-mini",
                "validated": False,
                "created_at": now,
            }

            loaded += self._create_relation_edge(rel, props)

        return loaded

    def _create_relation_edge(self, rel: ExtractedRelation, props: dict) -> int:
        """Create a single relation edge in Neo4j. Returns 1 on success."""
        # Build the Cypher query dynamically based on source/target types
        src_label = rel.source_type.value
        tgt_label = rel.target_type.value
        edge_type = rel.relation.value

        # Determine match key based on node type
        src_key = "arxiv_id" if rel.source_type == NodeLabel.PAPER else "name"
        tgt_key = "arxiv_id" if rel.target_type == NodeLabel.PAPER else "name"

        # For Paper sources, use source_paper (arxiv_id) as the match value
        src_val = rel.source_paper if rel.source_type == NodeLabel.PAPER else rel.source_name
        tgt_val = rel.target_name

        query = f"""
            MATCH (a:{src_label} {{{src_key}: $src_val}})
            MATCH (b:{tgt_label} {{{tgt_key}: $tgt_val}})
            MERGE (a)-[r:{edge_type}]->(b)
            SET r += $props
            RETURN count(r) AS cnt
        """

        try:
            with self._driver.session() as session:
                result = session.run(
                    query,
                    src_val=src_val,
                    tgt_val=tgt_val,
                    props=props,
                )
                return result.single()["cnt"]
        except Exception as e:
            logger.warning(f"Failed to create edge {src_val} -[{edge_type}]-> {tgt_val}: {e}")
            return 0

    # -----------------------------------------------------------------
    # Citation co-occurrence boost
    # -----------------------------------------------------------------

    def apply_citation_cooccurrence_boost(self, boost: float = 0.15) -> int:
        """Boost confidence of edges whose source papers are co-cited.

        If paper A and paper B are both cited by some paper C, and there's
        an extracted edge between entities from A and B, boost that edge's
        confidence by `boost` (capped at 1.0).

        Returns count of edges boosted.
        """
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (p1:Paper)-[:CITES]->(common:Paper)<-[:CITES]-(p2:Paper)
                WHERE p1 <> p2
                WITH p1, p2
                MATCH (p1)-[:INTRODUCES|EVALUATES_ON]->(e1)
                MATCH (p2)-[:INTRODUCES|EVALUATES_ON]->(e2)
                MATCH (e1)-[r]->(e2)
                WHERE r.confidence IS NOT NULL
                  AND r.confidence < 1.0
                SET r.confidence = CASE
                    WHEN r.confidence + $boost > 1.0 THEN 1.0
                    ELSE r.confidence + $boost
                END,
                r.cocitation_boosted = true
                RETURN count(r) AS cnt
                """,
                boost=boost,
            )
            return result.single()["cnt"]

    # -----------------------------------------------------------------
    # Graph statistics
    # -----------------------------------------------------------------

    def get_graph_stats(self) -> dict[str, object]:
        """Compute basic graph statistics from Neo4j."""
        with self._driver.session() as session:
            # Node counts by label
            node_counts = {}
            for label in NodeLabel:
                result = session.run(f"MATCH (n:{label.value}) RETURN count(n) AS cnt")
                node_counts[label.value] = result.single()["cnt"]

            # Edge counts by type
            edge_counts = {}
            for etype in EdgeType:
                result = session.run(f"MATCH ()-[r:{etype.value}]->() RETURN count(r) AS cnt")
                edge_counts[etype.value] = result.single()["cnt"]

            # Citation graph degree distribution
            result = session.run("""
                MATCH (p:Paper)
                OPTIONAL MATCH (p)-[out:CITES]->()
                OPTIONAL MATCH ()-[inc:CITES]->(p)
                RETURN p.arxiv_id AS id,
                       count(DISTINCT out) AS out_degree,
                       count(DISTINCT inc) AS in_degree
                ORDER BY in_degree DESC
                LIMIT 10
            """)
            top_cited = [dict(r) for r in result]

            # Connected components (approximate via weakly connected)
            result = session.run("""
                MATCH (p:Paper)
                WHERE NOT EXISTS { MATCH (p)-[:CITES]-() }
                RETURN count(p) AS isolated
            """)
            isolated = result.single()["isolated"]

            return {
                "node_counts": node_counts,
                "edge_counts": edge_counts,
                "top_cited_papers": top_cited,
                "isolated_papers": isolated,
                "total_nodes": sum(node_counts.values()),
                "total_edges": sum(edge_counts.values()),
            }


def _paper_to_props(paper: PaperMetadata) -> dict:
    """Convert PaperMetadata to Neo4j property dict."""
    tier, weight = get_venue_weight(paper.venue)
    return {
        "arxiv_id": paper.arxiv_id,
        "title": paper.title,
        "abstract": paper.abstract,
        "authors": paper.authors,
        "year": paper.year,
        "venue": paper.venue or "arXiv",
        "citation_count": paper.citation_count,
        "significance": paper.significance,
        "venue_tier": tier.value,
        "venue_weight": weight,
    }
