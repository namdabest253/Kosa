"""Tests for temporal holdout benchmark."""

from kosa.ingestion.holdout import (
    HOLDOUT_INNOVATIONS,
    get_problems_solved,
    get_source_techniques,
)


class TestHoldoutInnovations:
    def test_exactly_20_innovations(self):
        assert len(HOLDOUT_INNOVATIONS) == 20

    def test_all_2024_or_later(self):
        for i in HOLDOUT_INNOVATIONS:
            assert i.year >= 2024, f"{i.name} has year {i.year}"

    def test_unique_names(self):
        names = [i.name for i in HOLDOUT_INNOVATIONS]
        assert len(names) == len(set(names))

    def test_all_have_required_fields(self):
        for i in HOLDOUT_INNOVATIONS:
            assert i.name, "Missing name"
            assert i.papers, f"{i.name} has no papers"
            assert len(i.source_techniques) >= 2, f"{i.name} should combine at least 2 techniques"
            assert i.problem_solved, f"{i.name} has no problem_solved"
            assert i.solution_description, f"{i.name} has no solution_description"
            assert len(i.subfields_combined) >= 2, f"{i.name} should combine at least 2 subfields"
            assert i.graph_path, f"{i.name} has no graph_path"

    def test_cross_subfield_requirement(self):
        """Every innovation should combine techniques from different subfields."""
        for i in HOLDOUT_INNOVATIONS:
            assert (
                len(set(i.subfields_combined)) >= 2
            ), f"{i.name} combines {i.subfields_combined} — need at least 2 distinct subfields"

    def test_source_techniques_coverage(self):
        """Should have a good diversity of source techniques."""
        all_techniques = get_source_techniques()
        assert (
            len(all_techniques) >= 30
        ), f"Expected 30+ unique source techniques, got {len(all_techniques)}"

    def test_problems_solved_coverage(self):
        problems = get_problems_solved()
        assert len(problems) >= 15, f"Expected 15+ unique problems, got {len(problems)}"

    def test_graph_paths_contain_edge_types(self):
        """Graph paths should reference valid edge types."""
        valid_edges = {"USES", "MITIGATES", "HAS_LIMITATION", "IMPROVES_OVER", "INTRODUCES"}
        for i in HOLDOUT_INNOVATIONS:
            found_edge = any(e in i.graph_path for e in valid_edges)
            assert found_edge, f"{i.name} graph_path has no valid edge type: {i.graph_path}"
