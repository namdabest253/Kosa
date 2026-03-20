"""Tests for paper ingestion pipeline."""

from kosa.graph.schema import VenueTier
from kosa.ingestion.gate import GateDecision, GateResult
from kosa.ingestion.pipeline import (
    PaperMetadata,
    _extract_year_from_id,
    compute_citation_signal,
    compute_novelty_signal,
    compute_recency_factor,
    enrich_metadata,
)


class TestRecencyFactor:
    def test_current_year(self):
        assert compute_recency_factor(2025, current_year=2025) == 1.0

    def test_one_year_old(self):
        assert compute_recency_factor(2024, current_year=2025) == 0.9

    def test_five_years_old(self):
        assert compute_recency_factor(2020, current_year=2025) == 0.5

    def test_floor_at_0_3(self):
        assert compute_recency_factor(2010, current_year=2025) == 0.3

    def test_very_old_paper(self):
        assert compute_recency_factor(2000, current_year=2025) == 0.3

    def test_future_year_clamps(self):
        # Future year should give 1.0
        assert compute_recency_factor(2026, current_year=2025) == 1.0


class TestCitationSignal:
    def test_zero_citations(self):
        signal = compute_citation_signal(0, 2024, None)
        assert 0.1 <= signal <= 0.3

    def test_highly_cited(self):
        signal = compute_citation_signal(5000, 2020, None)
        assert signal >= 0.9

    def test_moderately_cited(self):
        signal = compute_citation_signal(100, 2023, None)
        assert 0.5 <= signal <= 0.9

    def test_with_cohort_stats(self):
        cohort = {"neurips": {"2023": 50.0}}
        signal = compute_citation_signal(100, 2023, "NeurIPS", cohort)
        assert signal == 1.0  # 100 / (2 * 50) = 1.0

    def test_cohort_below_median(self):
        cohort = {"neurips": {"2023": 50.0}}
        signal = compute_citation_signal(10, 2023, "NeurIPS", cohort)
        assert signal == 0.1  # 10 / (2 * 50) = 0.1


class TestNoveltySignal:
    def test_novel_technique(self):
        assert compute_novelty_signal("novel_technique") == 1.0

    def test_significant_improvement(self):
        assert compute_novelty_signal("significant_improvement") == 0.8

    def test_survey(self):
        assert compute_novelty_signal("survey") == 0.3

    def test_incremental(self):
        assert compute_novelty_signal("incremental") == 0.2

    def test_none_default(self):
        assert compute_novelty_signal(None) == 0.5

    def test_unknown_class(self):
        assert compute_novelty_signal("unknown") == 0.5


class TestExtractYearFromId:
    def test_old_format(self):
        assert _extract_year_from_id("1706.03762") == 2017

    def test_new_format(self):
        assert _extract_year_from_id("2401.04088") == 2024

    def test_2025_format(self):
        assert _extract_year_from_id("2501.12948") == 2025

    def test_invalid_id(self):
        # Should return current year as fallback
        assert _extract_year_from_id("invalid") == 2025


class TestPaperMetadata:
    def test_significance_computation(self):
        paper = PaperMetadata(
            arxiv_id="2401.00001",
            title="Test Paper",
            abstract="Test abstract",
            authors=["Author"],
            year=2024,
            venue="NeurIPS",
        )
        paper.venue_weight = 1.0
        paper.recency_factor = 0.9
        paper.citation_signal = 0.8
        paper.novelty_signal = 1.0
        sig = paper.compute_significance()
        assert sig == 1.0 * 0.9 * 0.8 * 1.0
        assert paper.significance == sig

    def test_low_significance(self):
        paper = PaperMetadata(
            arxiv_id="2401.00001",
            title="Test",
            abstract="Test",
            authors=["Author"],
            year=2015,
            venue=None,
        )
        paper.venue_weight = 0.15
        paper.recency_factor = 0.3
        paper.citation_signal = 0.2
        paper.novelty_signal = 0.2
        sig = paper.compute_significance()
        assert sig < 0.01  # very low


class TestEnrichMetadata:
    def test_enriches_all_fields(self):
        paper = PaperMetadata(
            arxiv_id="2401.00001",
            title="Test Paper",
            abstract="Test abstract",
            authors=["Author"],
            year=2024,
            venue="NeurIPS",
            citation_count=100,
        )
        gate_result = GateResult(
            decision=GateDecision.ADMIT,
            reason="Auto-admit",
            venue_tier=VenueTier.TIER_1,
            venue_weight=1.0,
            novelty_class="novel_technique",
        )
        enriched = enrich_metadata(paper, gate_result)
        assert enriched.venue_tier == VenueTier.TIER_1
        assert enriched.venue_weight == 1.0
        assert enriched.recency_factor > 0
        assert enriched.citation_signal > 0
        assert enriched.novelty_signal == 1.0
        assert enriched.significance > 0
        assert enriched.ingested_at != ""
