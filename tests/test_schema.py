"""Tests for the graph schema definitions."""

from kosa.graph.schema import (
    CONSTRAINTS,
    EDGE_SCHEMA,
    INDEXES,
    NODE_PROPERTIES,
    EdgeType,
    NodeLabel,
    VenueTier,
    get_migration_statements,
    get_vector_index_statement,
    get_venue_weight,
    validate_edge,
    validate_edge_properties,
    validate_node_properties,
)

# ---------------------------------------------------------------------------
# Node labels
# ---------------------------------------------------------------------------


class TestNodeLabels:
    def test_phase1_labels(self):
        assert set(NodeLabel) == {
            NodeLabel.PAPER,
            NodeLabel.TECHNIQUE,
            NodeLabel.PROBLEM,
            NodeLabel.DATASET,
        }

    def test_every_label_has_properties(self):
        for label in NodeLabel:
            assert label in NODE_PROPERTIES, f"Missing properties for {label}"


# ---------------------------------------------------------------------------
# Edge types
# ---------------------------------------------------------------------------


class TestEdgeTypes:
    def test_all_edge_types_present(self):
        expected = {
            "CITES",
            "INTRODUCES",
            "EVALUATES_ON",
            "HAS_LIMITATION",
            "MITIGATES",
            "IMPROVES_OVER",
            "USES",
            "IS_INSTANCE_OF",
            "CAUSED_BY",
            "TEMPORALLY_FOLLOWS",
            "SAME_AS",
        }
        assert {e.value for e in EdgeType} == expected

    def test_every_edge_type_in_schema(self):
        schema_edge_types = {triple[1] for triple in EDGE_SCHEMA}
        for et in EdgeType:
            assert et in schema_edge_types, f"{et} has no valid (source, target) in EDGE_SCHEMA"


# ---------------------------------------------------------------------------
# Edge validation
# ---------------------------------------------------------------------------


class TestEdgeValidation:
    def test_valid_cites(self):
        assert validate_edge(NodeLabel.PAPER, EdgeType.CITES, NodeLabel.PAPER)

    def test_valid_introduces(self):
        assert validate_edge(NodeLabel.PAPER, EdgeType.INTRODUCES, NodeLabel.TECHNIQUE)

    def test_valid_evaluates_on(self):
        assert validate_edge(NodeLabel.PAPER, EdgeType.EVALUATES_ON, NodeLabel.DATASET)

    def test_valid_mitigates(self):
        assert validate_edge(NodeLabel.TECHNIQUE, EdgeType.MITIGATES, NodeLabel.PROBLEM)

    def test_invalid_paper_mitigates_problem(self):
        assert not validate_edge(NodeLabel.PAPER, EdgeType.MITIGATES, NodeLabel.PROBLEM)

    def test_invalid_dataset_cites_dataset(self):
        assert not validate_edge(NodeLabel.DATASET, EdgeType.CITES, NodeLabel.DATASET)

    def test_same_as_only_same_type(self):
        # SAME_AS should not connect different types
        assert not validate_edge(NodeLabel.TECHNIQUE, EdgeType.SAME_AS, NodeLabel.PROBLEM)
        assert validate_edge(NodeLabel.TECHNIQUE, EdgeType.SAME_AS, NodeLabel.TECHNIQUE)


# ---------------------------------------------------------------------------
# Node property validation
# ---------------------------------------------------------------------------


class TestNodePropertyValidation:
    def test_paper_complete(self):
        props = {
            "title": "Test",
            "authors": ["A"],
            "venue": "NeurIPS",
            "year": 2024,
            "abstract": "...",
            "arxiv_id": "2401.00001",
            "citation_count": 10,
            "significance": 0.8,
            "venue_tier": 1,
            "venue_weight": 1.0,
        }
        assert validate_node_properties(NodeLabel.PAPER, props) == []

    def test_paper_missing_fields(self):
        missing = validate_node_properties(NodeLabel.PAPER, {"title": "Test"})
        assert "arxiv_id" in missing
        assert "significance" in missing
        assert "title" not in missing

    def test_technique_complete(self):
        props = {
            "name": "FlashAttention",
            "description": "...",
            "mathematical_structure": "block-sparse matrix multiplication",
            "properties": ["io-aware", "tiled"],
        }
        assert validate_node_properties(NodeLabel.TECHNIQUE, props) == []

    def test_problem_requires_bottleneck_class(self):
        missing = validate_node_properties(
            NodeLabel.PROBLEM, {"name": "vanishing gradients", "description": "..."}
        )
        assert "bottleneck_class" in missing

    def test_dataset_complete(self):
        props = {"name": "ImageNet", "description": "...", "domain": "CV", "size": "1.2M images"}
        assert validate_node_properties(NodeLabel.DATASET, props) == []


# ---------------------------------------------------------------------------
# Edge property validation
# ---------------------------------------------------------------------------


class TestEdgePropertyValidation:
    def test_cites_only_needs_source_and_timestamp(self):
        props = {"source_paper": "2401.00001", "created_at": "2024-01-01T00:00:00Z"}
        assert validate_edge_properties(EdgeType.CITES, props) == []

    def test_cites_missing_source(self):
        missing = validate_edge_properties(EdgeType.CITES, {"created_at": "2024-01-01"})
        assert "source_paper" in missing

    def test_regular_edge_requires_provenance(self):
        props = {
            "confidence": 0.85,
            "source_paper": "2401.00001",
            "source_venue": "NeurIPS",
            "venue_weight": 1.0,
            "supporting_text": "We introduce...",
            "extraction_method": "gpt-4o-mini",
            "validated": False,
            "created_at": "2024-01-01T00:00:00Z",
        }
        assert validate_edge_properties(EdgeType.INTRODUCES, props) == []

    def test_regular_edge_missing_confidence(self):
        missing = validate_edge_properties(EdgeType.INTRODUCES, {"source_paper": "x"})
        assert "confidence" in missing
        assert "supporting_text" in missing


# ---------------------------------------------------------------------------
# Venue weights
# ---------------------------------------------------------------------------


class TestVenueWeights:
    def test_tier1(self):
        tier, weight = get_venue_weight("NeurIPS")
        assert tier == VenueTier.TIER_1
        assert weight == 1.0

    def test_tier2(self):
        tier, weight = get_venue_weight("AAAI")
        assert tier == VenueTier.TIER_2
        assert weight == 0.85

    def test_unknown_defaults_to_tier6(self):
        tier, weight = get_venue_weight("some random workshop")
        assert tier == VenueTier.TIER_6
        assert weight == 0.15

    def test_none_venue_defaults_to_tier6(self):
        tier, weight = get_venue_weight(None)
        assert tier == VenueTier.TIER_6
        assert weight == 0.15

    def test_known_lab_boosts_tier6(self):
        tier, weight = get_venue_weight(None, known_lab=True)
        assert tier == VenueTier.TIER_6
        assert weight == 0.25

    def test_case_insensitive(self):
        tier, _ = get_venue_weight("neurips")
        assert tier == VenueTier.TIER_1
        tier, _ = get_venue_weight("NEURIPS")
        assert tier == VenueTier.TIER_1


# ---------------------------------------------------------------------------
# Migration statements
# ---------------------------------------------------------------------------


class TestMigration:
    def test_constraints_exist(self):
        assert len(CONSTRAINTS) >= 4  # at least one per node type

    def test_indexes_exist(self):
        assert len(INDEXES) >= 4  # full-text + b-tree

    def test_migration_includes_all(self):
        stmts = get_migration_statements()
        assert len(stmts) == len(CONSTRAINTS) + len(INDEXES)

    def test_vector_index_statement(self):
        stmt = get_vector_index_statement("Paper", dimensions=1536)
        assert "Paper" in stmt
        assert "1536" in stmt
        assert "cosine" in stmt

    def test_all_constraints_are_valid_cypher_prefix(self):
        for c in CONSTRAINTS:
            assert c.startswith("CREATE CONSTRAINT")

    def test_all_indexes_are_valid_cypher_prefix(self):
        for i in INDEXES:
            assert i.startswith("CREATE")
