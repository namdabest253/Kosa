"""Tests for graph loader (unit tests without Neo4j connection).

Tests the data preparation and conversion logic. Integration tests
with a real Neo4j instance are deferred — they'd go in tests/integration/.
"""

from kosa.graph.loader import _paper_to_props
from kosa.ingestion.pipeline import PaperMetadata


class TestPaperToProps:
    def test_basic_conversion(self):
        paper = PaperMetadata(
            arxiv_id="1706.03762",
            title="Attention Is All You Need",
            abstract="We propose a new architecture...",
            authors=["Vaswani", "Shazeer"],
            year=2017,
            venue="NeurIPS",
            citation_count=50000,
        )
        paper.significance = 0.85

        props = _paper_to_props(paper)
        assert props["arxiv_id"] == "1706.03762"
        assert props["title"] == "Attention Is All You Need"
        assert props["authors"] == ["Vaswani", "Shazeer"]
        assert props["year"] == 2017
        assert props["venue"] == "NeurIPS"
        assert props["citation_count"] == 50000
        assert props["significance"] == 0.85
        assert props["venue_tier"] == 1  # NeurIPS = Tier 1
        assert props["venue_weight"] == 1.0

    def test_arxiv_preprint(self):
        paper = PaperMetadata(
            arxiv_id="2401.00001",
            title="Test Paper",
            abstract="Test",
            authors=["Author"],
            year=2024,
            venue=None,
        )
        paper.significance = 0.1

        props = _paper_to_props(paper)
        assert props["venue"] == "arXiv"
        assert props["venue_tier"] == 6  # Tier 6
        assert props["venue_weight"] == 0.15

    def test_all_required_fields_present(self):
        paper = PaperMetadata(
            arxiv_id="test",
            title="Test",
            abstract="Abstract",
            authors=["A"],
            year=2024,
            venue="ICML",
        )
        paper.significance = 0.5

        props = _paper_to_props(paper)
        required_fields = [
            "arxiv_id",
            "title",
            "abstract",
            "authors",
            "year",
            "venue",
            "citation_count",
            "significance",
            "venue_tier",
            "venue_weight",
        ]
        for field in required_fields:
            assert field in props, f"Missing field: {field}"
