"""Tests for ingestion gate module."""

from kosa.graph.schema import VenueTier
from kosa.ingestion.gate import (
    CITATION_THRESHOLD_TIER6,
    CITATION_THRESHOLD_TIER456,
    GateDecision,
    gate_paper,
    gate_with_novelty,
    is_known_lab,
)


class TestIsKnownLab:
    def test_known_lab_in_authors(self):
        assert is_known_lab(["John Smith, Google DeepMind"])

    def test_known_lab_in_affiliations(self):
        assert is_known_lab(["John Smith"], affiliations=["OpenAI"])

    def test_unknown_lab(self):
        assert not is_known_lab(["John Smith"], affiliations=["Random University"])

    def test_case_insensitive(self):
        assert is_known_lab(["researcher at NVIDIA"])

    def test_known_lab_substring(self):
        assert is_known_lab(["Jane from Stanford University"])

    def test_empty_authors(self):
        assert not is_known_lab([])


class TestGatePaper:
    def test_tier1_auto_admit(self):
        result = gate_paper(venue="NeurIPS", citation_count=0, authors=["Unknown"])
        assert result.decision == GateDecision.ADMIT
        assert result.venue_tier == VenueTier.TIER_1

    def test_tier2_auto_admit(self):
        result = gate_paper(venue="AAAI", citation_count=0, authors=["Unknown"])
        assert result.decision == GateDecision.ADMIT
        assert result.venue_tier == VenueTier.TIER_2

    def test_tier3_auto_admit(self):
        result = gate_paper(venue="UAI", citation_count=0, authors=["Unknown"])
        assert result.decision == GateDecision.ADMIT
        assert result.venue_tier == VenueTier.TIER_3

    def test_tier4_high_citations_admit(self):
        result = gate_paper(
            venue="SOSP",
            citation_count=CITATION_THRESHOLD_TIER456,
            authors=["Unknown"],
        )
        assert result.decision == GateDecision.ADMIT

    def test_tier4_low_citations_needs_novelty(self):
        result = gate_paper(venue="SOSP", citation_count=0, authors=["Unknown"])
        assert result.decision == GateDecision.NEEDS_NOVELTY_CHECK

    def test_tier5_needs_novelty(self):
        result = gate_paper(
            venue="NeurIPS Workshop",
            citation_count=0,
            authors=["Unknown"],
        )
        # NeurIPS Workshop isn't in VENUE_TIER_LOOKUP, falls to Tier 6
        # so it needs novelty check
        assert result.decision in {GateDecision.NEEDS_NOVELTY_CHECK, GateDecision.ADMIT}

    def test_tier6_known_lab_admit(self):
        result = gate_paper(
            venue=None,
            citation_count=0,
            authors=["Researcher from Google DeepMind"],
        )
        assert result.decision == GateDecision.ADMIT
        assert result.known_lab is True

    def test_tier6_high_citations_admit(self):
        result = gate_paper(
            venue=None,
            citation_count=CITATION_THRESHOLD_TIER6,
            authors=["Unknown Author"],
        )
        assert result.decision == GateDecision.ADMIT

    def test_tier6_referenced_admit(self):
        result = gate_paper(
            venue=None,
            citation_count=0,
            authors=["Unknown Author"],
            referenced_by_admitted=True,
        )
        assert result.decision == GateDecision.ADMIT

    def test_tier6_no_signals_needs_novelty(self):
        result = gate_paper(
            venue=None,
            citation_count=0,
            authors=["Unknown Author"],
        )
        assert result.decision == GateDecision.NEEDS_NOVELTY_CHECK

    def test_venue_weight_known_lab(self):
        result = gate_paper(
            venue=None,
            citation_count=0,
            authors=["Researcher at OpenAI"],
        )
        assert result.venue_weight == 0.25  # known lab arXiv boost


class TestGateWithNovelty:
    def test_novel_technique_admits(self):
        gate_result = gate_paper(venue=None, citation_count=0, authors=["Nobody"])
        assert gate_result.decision == GateDecision.NEEDS_NOVELTY_CHECK
        final = gate_with_novelty(gate_result, "novel_technique")
        assert final.decision == GateDecision.ADMIT
        assert final.novelty_class == "novel_technique"

    def test_significant_improvement_admits(self):
        gate_result = gate_paper(venue=None, citation_count=0, authors=["Nobody"])
        final = gate_with_novelty(gate_result, "significant_improvement")
        assert final.decision == GateDecision.ADMIT

    def test_incremental_rejects(self):
        gate_result = gate_paper(venue=None, citation_count=0, authors=["Nobody"])
        final = gate_with_novelty(gate_result, "incremental")
        assert final.decision == GateDecision.REJECT

    def test_survey_rejects(self):
        gate_result = gate_paper(venue=None, citation_count=0, authors=["Nobody"])
        final = gate_with_novelty(gate_result, "survey")
        assert final.decision == GateDecision.REJECT

    def test_already_admitted_unchanged(self):
        gate_result = gate_paper(venue="NeurIPS", citation_count=0, authors=["Author"])
        assert gate_result.decision == GateDecision.ADMIT
        final = gate_with_novelty(gate_result, "incremental")
        assert final.decision == GateDecision.ADMIT  # unchanged
