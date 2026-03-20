"""Tests for citation extraction and validation."""

from kosa.ingestion.citations import (
    citation_graph_stats,
    validate_citation_graph,
)


class TestValidateCitationGraph:
    def test_removes_self_citations(self):
        citations = {
            "1706.03762": ["1706.03762", "1409.0473"],
            "1810.04805": ["1706.03762"],
        }
        cleaned = validate_citation_graph(citations)
        assert "1706.03762" not in cleaned["1706.03762"]
        assert "1409.0473" in cleaned["1706.03762"]

    def test_removes_duplicates(self):
        citations = {
            "1706.03762": ["1409.0473", "1409.0473", "1512.03385"],
        }
        cleaned = validate_citation_graph(citations)
        assert cleaned["1706.03762"] == ["1409.0473", "1512.03385"]

    def test_empty_graph(self):
        cleaned = validate_citation_graph({})
        assert cleaned == {}

    def test_no_issues(self):
        citations = {
            "A": ["B", "C"],
            "B": ["C"],
        }
        cleaned = validate_citation_graph(citations)
        assert cleaned == citations


class TestCitationGraphStats:
    def test_basic_stats(self):
        citations = {
            "A": ["B", "C"],
            "B": ["C"],
            "C": [],
        }
        stats = citation_graph_stats(citations)
        assert stats["total_papers"] == 3
        assert stats["total_edges"] == 3
        assert stats["papers_with_refs"] == 2  # A and B have outgoing
        assert stats["papers_cited"] == 2  # B and C are cited
        assert stats["isolated_papers"] == 0

    def test_isolated_papers(self):
        citations = {
            "A": ["B"],
            "B": [],
            "C": [],  # isolated
        }
        stats = citation_graph_stats(citations)
        assert stats["isolated_papers"] == 1
        assert "C" in stats["isolated_ids"]

    def test_empty_graph(self):
        stats = citation_graph_stats({})
        assert stats["total_papers"] == 0
        assert stats["total_edges"] == 0

    def test_degree_stats(self):
        citations = {
            "A": ["B", "C", "D"],
            "B": ["C"],
            "C": [],
            "D": [],
        }
        stats = citation_graph_stats(citations)
        assert stats["avg_out_degree"] == 1.0  # 4 edges / 4 papers
        assert stats["max_out_degree"] == ("A", 3)
        # C is cited by A and B → in_degree 2
        assert stats["max_in_degree"] == ("C", 2)
