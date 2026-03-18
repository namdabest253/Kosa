"""Tests for the Phase 0 corpus selection."""

from kosa.ingestion.corpus import PHASE0_CORPUS


class TestCorpus:
    def test_corpus_has_50_papers(self):
        assert len(PHASE0_CORPUS) == 50

    def test_no_duplicate_arxiv_ids(self):
        ids = [p.arxiv_id for p in PHASE0_CORPUS]
        assert len(ids) == len(set(ids))

    def test_all_papers_have_required_fields(self):
        for p in PHASE0_CORPUS:
            assert p.arxiv_id, "Missing arxiv_id"
            assert p.title, f"Missing title for {p.arxiv_id}"
            assert p.year >= 2013, f"Year too old for {p.arxiv_id}: {p.year}"
            assert p.subfield, f"Missing subfield for {p.arxiv_id}"

    def test_subfield_diversity(self):
        subfields = {p.subfield for p in PHASE0_CORPUS}
        # Must cover at least NLP, CV, generative, RL, optimization
        for required in ["NLP", "CV", "generative", "RL", "optimization"]:
            assert required in subfields, f"Missing subfield: {required}"

    def test_venue_tier_diversity(self):
        venues = {p.venue for p in PHASE0_CORPUS}
        # Should have some Tier 1 venues and some arXiv (None)
        assert None in venues, "No arXiv preprints in corpus"
        tier1 = {"NeurIPS", "ICML", "ICLR", "CVPR", "ICCV", "ACL", "EMNLP"}
        assert tier1 & venues, "No Tier 1 venues in corpus"

    def test_year_range(self):
        years = [p.year for p in PHASE0_CORPUS]
        assert min(years) <= 2015, "No seminal/older papers"
        assert max(years) >= 2024, "No recent papers"

    def test_mix_of_seminal_and_recent(self):
        old = [p for p in PHASE0_CORPUS if p.year <= 2020]
        recent = [p for p in PHASE0_CORPUS if p.year >= 2023]
        assert len(old) >= 10, f"Too few seminal papers: {len(old)}"
        assert len(recent) >= 20, f"Too few recent papers: {len(recent)}"
