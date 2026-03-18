"""Neo4j schema definitions for the Kosa knowledge graph.

Defines node labels, properties, edge types, constraints, and indexes.
Phase 1 schema: Paper, Technique, Problem, Dataset nodes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from typing import Any

# ---------------------------------------------------------------------------
# Node labels
# ---------------------------------------------------------------------------


class NodeLabel(StrEnum):
    PAPER = "Paper"
    TECHNIQUE = "Technique"
    PROBLEM = "Problem"
    DATASET = "Dataset"


# ---------------------------------------------------------------------------
# Edge types
# ---------------------------------------------------------------------------


class EdgeType(StrEnum):
    # Ground truth (no extraction)
    CITES = "CITES"  # Paper → Paper

    # High confidence
    INTRODUCES = "INTRODUCES"  # Paper → Technique
    EVALUATES_ON = "EVALUATES_ON"  # Paper → Dataset

    # Medium confidence
    HAS_LIMITATION = "HAS_LIMITATION"  # Technique → Problem
    MITIGATES = "MITIGATES"  # Technique → Problem
    IMPROVES_OVER = "IMPROVES_OVER"  # Technique → Technique
    IS_INSTANCE_OF = "IS_INSTANCE_OF"  # Problem → Problem, Technique → Technique
    CAUSED_BY = "CAUSED_BY"  # Problem → Problem
    TEMPORALLY_FOLLOWS = "TEMPORALLY_FOLLOWS"  # Technique → Technique

    # Entity resolution
    SAME_AS = "SAME_AS"  # Any → Any (same type)


# Valid (source_label, edge_type, target_label) triples.
EDGE_SCHEMA: list[tuple[NodeLabel, EdgeType, NodeLabel]] = [
    # Ground truth
    (NodeLabel.PAPER, EdgeType.CITES, NodeLabel.PAPER),
    # High confidence
    (NodeLabel.PAPER, EdgeType.INTRODUCES, NodeLabel.TECHNIQUE),
    (NodeLabel.PAPER, EdgeType.EVALUATES_ON, NodeLabel.DATASET),
    # Medium confidence
    (NodeLabel.TECHNIQUE, EdgeType.HAS_LIMITATION, NodeLabel.PROBLEM),
    (NodeLabel.TECHNIQUE, EdgeType.MITIGATES, NodeLabel.PROBLEM),
    (NodeLabel.TECHNIQUE, EdgeType.IMPROVES_OVER, NodeLabel.TECHNIQUE),
    (NodeLabel.PROBLEM, EdgeType.IS_INSTANCE_OF, NodeLabel.PROBLEM),
    (NodeLabel.TECHNIQUE, EdgeType.IS_INSTANCE_OF, NodeLabel.TECHNIQUE),
    (NodeLabel.PROBLEM, EdgeType.CAUSED_BY, NodeLabel.PROBLEM),
    (NodeLabel.TECHNIQUE, EdgeType.TEMPORALLY_FOLLOWS, NodeLabel.TECHNIQUE),
    # Entity resolution (same-type only)
    (NodeLabel.TECHNIQUE, EdgeType.SAME_AS, NodeLabel.TECHNIQUE),
    (NodeLabel.PROBLEM, EdgeType.SAME_AS, NodeLabel.PROBLEM),
    (NodeLabel.DATASET, EdgeType.SAME_AS, NodeLabel.DATASET),
]


# ---------------------------------------------------------------------------
# Venue tiers & weights
# ---------------------------------------------------------------------------


class VenueTier(IntEnum):
    TIER_1 = 1  # NeurIPS, ICML, ICLR, CVPR, ICCV, ACL, EMNLP, JMLR, TPAMI
    TIER_2 = 2  # AAAI, IJCAI, ECCV, KDD, COLT, NAACL, AISTATS
    TIER_3 = 3  # UAI, WSDM, CoRL, MLSys, Nature MI, TMLR, SIGIR
    TIER_4 = 4  # Cross-domain top venues (SOSP, S&P, CHI, SIGGRAPH)
    TIER_5 = 5  # Workshops at Tier 1 conferences, Findings papers
    TIER_6 = 6  # arXiv preprints


VENUE_WEIGHTS: dict[VenueTier, float] = {
    VenueTier.TIER_1: 1.0,
    VenueTier.TIER_2: 0.85,
    VenueTier.TIER_3: 0.7,
    VenueTier.TIER_4: 0.55,
    VenueTier.TIER_5: 0.35,
    VenueTier.TIER_6: 0.15,  # 0.25 for known labs — handled at ingestion time
}

# Lookup: venue name → tier. Lowercase keys for matching.
VENUE_TIER_LOOKUP: dict[str, VenueTier] = {
    # Tier 1
    "neurips": VenueTier.TIER_1,
    "icml": VenueTier.TIER_1,
    "iclr": VenueTier.TIER_1,
    "cvpr": VenueTier.TIER_1,
    "iccv": VenueTier.TIER_1,
    "acl": VenueTier.TIER_1,
    "emnlp": VenueTier.TIER_1,
    "jmlr": VenueTier.TIER_1,
    "tpami": VenueTier.TIER_1,
    # Tier 2
    "aaai": VenueTier.TIER_2,
    "ijcai": VenueTier.TIER_2,
    "eccv": VenueTier.TIER_2,
    "kdd": VenueTier.TIER_2,
    "colt": VenueTier.TIER_2,
    "naacl": VenueTier.TIER_2,
    "aistats": VenueTier.TIER_2,
    # Tier 3
    "uai": VenueTier.TIER_3,
    "wsdm": VenueTier.TIER_3,
    "corl": VenueTier.TIER_3,
    "mlsys": VenueTier.TIER_3,
    "nature machine intelligence": VenueTier.TIER_3,
    "tmlr": VenueTier.TIER_3,
    "sigir": VenueTier.TIER_3,
    # Tier 4
    "sosp": VenueTier.TIER_4,
    "s&p": VenueTier.TIER_4,
    "chi": VenueTier.TIER_4,
    "siggraph": VenueTier.TIER_4,
}


def get_venue_weight(venue: str | None, known_lab: bool = False) -> tuple[VenueTier, float]:
    """Return (tier, weight) for a venue name. Defaults to Tier 6 (arXiv)."""
    if venue is None:
        tier = VenueTier.TIER_6
        return tier, 0.25 if known_lab else VENUE_WEIGHTS[tier]

    tier = VENUE_TIER_LOOKUP.get(venue.lower().strip(), VenueTier.TIER_6)
    weight = VENUE_WEIGHTS[tier]
    if tier == VenueTier.TIER_6 and known_lab:
        weight = 0.25
    return tier, weight


# ---------------------------------------------------------------------------
# Required properties per node/edge type
# ---------------------------------------------------------------------------

PAPER_PROPERTIES = {
    "title": str,
    "authors": list,  # list[str]
    "venue": str,
    "year": int,
    "abstract": str,
    "arxiv_id": str,
    "citation_count": int,
    "significance": float,
    "venue_tier": int,
    "venue_weight": float,
}

TECHNIQUE_PROPERTIES = {
    "name": str,
    "description": str,
    "mathematical_structure": str,  # formal structure, e.g. "eigenvector of stochastic matrix"
    "properties": list,  # list[str] — key characteristics
}

PROBLEM_PROPERTIES = {
    "name": str,
    "description": str,
    "bottleneck_class": str,  # domain-agnostic structural description
}

DATASET_PROPERTIES = {
    "name": str,
    "description": str,
    "domain": str,
    "size": str,  # free-text, e.g. "1.2M images", "100K samples"
}

NODE_PROPERTIES: dict[NodeLabel, dict[str, type]] = {
    NodeLabel.PAPER: PAPER_PROPERTIES,
    NodeLabel.TECHNIQUE: TECHNIQUE_PROPERTIES,
    NodeLabel.PROBLEM: PROBLEM_PROPERTIES,
    NodeLabel.DATASET: DATASET_PROPERTIES,
}

# Every edge (except CITES) must carry these provenance fields.
EDGE_REQUIRED_PROPERTIES = {
    "confidence": float,
    "source_paper": str,  # arxiv_id of the paper this was extracted from
    "source_venue": str,
    "venue_weight": float,
    "supporting_text": str,  # excerpt from paper
    "extraction_method": str,  # e.g. "gpt-4o-mini", "citation-parse", "manual"
    "validated": bool,
    "created_at": str,  # ISO 8601
}

# CITES edges are ground truth — simpler provenance.
CITES_PROPERTIES = {
    "source_paper": str,
    "created_at": str,
}


# ---------------------------------------------------------------------------
# Cypher statements for schema setup
# ---------------------------------------------------------------------------

CONSTRAINTS: list[str] = [
    # Unique arxiv_id on Paper nodes
    "CREATE CONSTRAINT paper_arxiv_id IF NOT EXISTS " "FOR (p:Paper) REQUIRE p.arxiv_id IS UNIQUE",
    # Unique names within each concept type (prevents trivial duplicates)
    "CREATE CONSTRAINT technique_name IF NOT EXISTS " "FOR (t:Technique) REQUIRE t.name IS UNIQUE",
    "CREATE CONSTRAINT problem_name IF NOT EXISTS " "FOR (p:Problem) REQUIRE p.name IS UNIQUE",
    "CREATE CONSTRAINT dataset_name IF NOT EXISTS " "FOR (d:Dataset) REQUIRE d.name IS UNIQUE",
]

INDEXES: list[str] = [
    # Full-text indexes for search
    "CREATE FULLTEXT INDEX paper_search IF NOT EXISTS "
    "FOR (p:Paper) ON EACH [p.title, p.abstract]",
    "CREATE FULLTEXT INDEX technique_search IF NOT EXISTS "
    "FOR (t:Technique) ON EACH [t.name, t.description]",
    "CREATE FULLTEXT INDEX problem_search IF NOT EXISTS "
    "FOR (p:Problem) ON EACH [p.name, p.description, p.bottleneck_class]",
    "CREATE FULLTEXT INDEX dataset_search IF NOT EXISTS "
    "FOR (d:Dataset) ON EACH [d.name, d.description]",
    # B-tree indexes for common lookups
    "CREATE INDEX paper_year IF NOT EXISTS FOR (p:Paper) ON (p.year)",
    "CREATE INDEX paper_significance IF NOT EXISTS FOR (p:Paper) ON (p.significance)",
    "CREATE INDEX paper_venue IF NOT EXISTS FOR (p:Paper) ON (p.venue)",
]

# Vector index created separately (requires Neo4j 5.11+, dimensions depend on embedding model).
# Template — caller fills in dimensions.
VECTOR_INDEX_TEMPLATE = (
    "CREATE VECTOR INDEX {name} IF NOT EXISTS "
    "FOR (n:{label}) ON (n.embedding) "
    "OPTIONS {{indexConfig: {{`vector.dimensions`: {dimensions}, "
    "`vector.similarity_function`: 'cosine'}}}}"
)


@dataclass
class SchemaReport:
    """Result of applying schema to Neo4j."""

    constraints_created: int = 0
    indexes_created: int = 0
    errors: list[str] = field(default_factory=list)


def get_migration_statements() -> list[str]:
    """Return all Cypher statements needed to set up the schema."""
    return CONSTRAINTS + INDEXES


def get_vector_index_statement(label: str, name: str | None = None, dimensions: int = 1536) -> str:
    """Return Cypher for creating a vector index on a node label."""
    if name is None:
        name = f"{label.lower()}_embedding"
    return VECTOR_INDEX_TEMPLATE.format(name=name, label=label, dimensions=dimensions)


def validate_edge(source_label: NodeLabel, edge_type: EdgeType, target_label: NodeLabel) -> bool:
    """Check if a (source, edge, target) triple is valid per the schema."""
    return (source_label, edge_type, target_label) in EDGE_SCHEMA


def validate_node_properties(label: NodeLabel, props: dict[str, Any]) -> list[str]:
    """Return list of missing required property names for a node."""
    required = NODE_PROPERTIES.get(label, {})
    return [k for k in required if k not in props]


def validate_edge_properties(edge_type: EdgeType, props: dict[str, Any]) -> list[str]:
    """Return list of missing required property names for an edge."""
    required = CITES_PROPERTIES if edge_type == EdgeType.CITES else EDGE_REQUIRED_PROPERTIES
    return [k for k in required if k not in props]
