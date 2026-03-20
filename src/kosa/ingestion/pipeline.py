"""Paper ingestion pipeline: arXiv bulk fetch, significance scoring, metadata storage.

Orchestrates the full ingestion workflow:
1. Fetch paper metadata from arXiv (bulk, with rate limiting)
2. Run ingestion gate (filter before extraction)
3. Compute significance scores
4. Run entity/relation extraction on admitted papers
5. Return structured results ready for Neo4j loading
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime

from openai import OpenAI

from kosa.graph.schema import VenueTier
from kosa.ingestion.extract import (
    PaperExtractionResult,
    extract_paper,
    fetch_arxiv_metadata,
)
from kosa.ingestion.gate import GateDecision, GateResult, run_gate

logger = logging.getLogger(__name__)

# Current year for recency calculation
CURRENT_YEAR = 2025


@dataclass
class PaperMetadata:
    """Metadata for a paper, enriched with significance scoring."""

    arxiv_id: str
    title: str
    abstract: str
    authors: list[str]
    year: int
    venue: str | None
    citation_count: int = 0
    affiliations: list[str] | None = None

    # Computed fields
    venue_tier: VenueTier | None = None
    venue_weight: float = 0.0
    recency_factor: float = 0.0
    citation_signal: float = 0.0
    novelty_signal: float = 0.0
    significance: float = 0.0
    gate_result: GateResult | None = None
    ingested_at: str = ""

    def compute_significance(self) -> float:
        """Compute and store the significance score.

        significance = venue_weight * recency_factor * citation_signal * novelty_signal

        All factors are in [0, 1] range. The result is a multiplicative score
        where low values in any dimension drag down the total.
        """
        self.significance = (
            self.venue_weight * self.recency_factor * self.citation_signal * self.novelty_signal
        )
        return self.significance


# ---------------------------------------------------------------------------
# Significance score components
# ---------------------------------------------------------------------------

NOVELTY_SIGNAL_MAP: dict[str, float] = {
    "novel_technique": 1.0,
    "significant_improvement": 0.8,
    "survey": 0.3,
    "incremental": 0.2,
}


def compute_recency_factor(year: int, current_year: int = CURRENT_YEAR) -> float:
    """Compute recency factor: newer papers score higher.

    Formula: max(0.3, 1.0 - (years_since_publication * 0.1))
    - 2025 paper: 1.0
    - 2020 paper: 0.5
    - 2017 paper: 0.3 (floor)
    """
    years_since = max(0, current_year - year)
    return max(0.3, 1.0 - (years_since * 0.1))


def compute_citation_signal(
    citation_count: int,
    year: int,
    venue: str | None,
    cohort_stats: dict[str, dict[str, float]] | None = None,
) -> float:
    """Compute citation signal: percentile rank within venue+year cohort.

    Without cohort statistics (Phase 1), uses a heuristic based on
    age-normalized citation rate. With cohort stats, computes true percentile.

    Args:
        citation_count: Raw citation count.
        year: Publication year.
        venue: Venue name (for cohort lookup).
        cohort_stats: Optional dict of {venue: {year: median_citations}}.
            When available, computes percentile rank. Otherwise uses heuristic.

    Returns:
        Citation signal in [0.1, 1.0] range.
    """
    if cohort_stats and venue:
        venue_data = cohort_stats.get(venue.lower(), {})
        median = venue_data.get(str(year), 0)
        if median > 0:
            # Percentile approximation: ratio to median, capped at 1.0
            return min(1.0, max(0.1, citation_count / (2 * median)))

    # Heuristic: age-normalized citation rate
    age = max(1, CURRENT_YEAR - year)
    citations_per_year = citation_count / age

    # Rough percentile mapping (calibrated for ML/AI papers)
    if citations_per_year >= 500:
        return 1.0  # top 1%
    if citations_per_year >= 100:
        return 0.9  # top 5%
    if citations_per_year >= 50:
        return 0.8  # top 10%
    if citations_per_year >= 20:
        return 0.7  # top 20%
    if citations_per_year >= 10:
        return 0.6  # top 30%
    if citations_per_year >= 5:
        return 0.5  # top 40%
    if citations_per_year >= 2:
        return 0.4  # top 50%
    if citations_per_year >= 1:
        return 0.3  # top 60%
    return max(0.1, 0.2)  # long tail


def compute_novelty_signal(novelty_class: str | None) -> float:
    """Map novelty classification to signal value."""
    if novelty_class is None:
        return 0.5  # default when not classified
    return NOVELTY_SIGNAL_MAP.get(novelty_class, 0.5)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


@dataclass
class IngestionBatch:
    """Results of ingesting a batch of papers."""

    admitted: list[PaperMetadata] = field(default_factory=list)
    rejected: list[PaperMetadata] = field(default_factory=list)
    extraction_results: list[PaperExtractionResult] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    total_papers: int = 0
    total_admitted: int = 0
    total_rejected: int = 0
    total_extracted: int = 0

    def summary(self) -> str:
        """Human-readable summary of the ingestion batch."""
        lines = [
            f"Ingestion batch: {self.total_papers} papers",
            f"  Admitted: {self.total_admitted}",
            f"  Rejected: {self.total_rejected}",
            f"  Extracted: {self.total_extracted}",
            f"  Errors: {len(self.errors)}",
        ]
        if self.admitted:
            avg_sig = sum(p.significance for p in self.admitted) / len(self.admitted)
            lines.append(f"  Avg significance (admitted): {avg_sig:.3f}")
        return "\n".join(lines)


def enrich_metadata(
    paper: PaperMetadata,
    gate_result: GateResult,
    novelty_class: str | None = None,
) -> PaperMetadata:
    """Enrich paper metadata with venue tier, significance components, and gate result."""
    paper.gate_result = gate_result
    paper.venue_tier = gate_result.venue_tier
    paper.venue_weight = gate_result.venue_weight
    paper.recency_factor = compute_recency_factor(paper.year)
    paper.citation_signal = compute_citation_signal(paper.citation_count, paper.year, paper.venue)
    paper.novelty_signal = compute_novelty_signal(novelty_class or gate_result.novelty_class)
    paper.compute_significance()
    paper.ingested_at = datetime.now(UTC).isoformat()
    return paper


def ingest_papers(
    client: OpenAI,
    arxiv_ids: list[str],
    venues: dict[str, str | None] | None = None,
    years: dict[str, int] | None = None,
    citation_counts: dict[str, int] | None = None,
    affiliations: dict[str, list[str]] | None = None,
    admitted_paper_ids: set[str] | None = None,
    extraction_model: str = "gpt-4o-mini",
    skip_extraction: bool = False,
) -> IngestionBatch:
    """Run the full ingestion pipeline on a batch of arXiv papers.

    1. Fetch metadata from arXiv API
    2. Run ingestion gate on each paper
    3. Compute significance scores for admitted papers
    4. Run extraction on admitted papers (unless skip_extraction=True)

    Args:
        client: OpenAI client for LLM calls.
        arxiv_ids: List of arXiv IDs to ingest.
        venues: Optional pre-known venue mapping {arxiv_id: venue_name}.
        years: Optional pre-known year mapping {arxiv_id: year}.
        citation_counts: Optional citation counts {arxiv_id: count}.
        affiliations: Optional author affiliations {arxiv_id: [affiliations]}.
        admitted_paper_ids: Set of already-admitted paper IDs (for reference check).
        extraction_model: Model to use for extraction (default: gpt-4o-mini).
        skip_extraction: If True, only run gate + significance scoring, no extraction.

    Returns:
        IngestionBatch with all results.
    """
    batch = IngestionBatch(total_papers=len(arxiv_ids))
    venues = venues or {}
    years = years or {}
    citation_counts = citation_counts or {}
    affiliations = affiliations or {}
    admitted_paper_ids = admitted_paper_ids or set()

    # Step 1: Fetch metadata from arXiv
    logger.info(f"Fetching metadata for {len(arxiv_ids)} papers from arXiv...")
    arxiv_data = fetch_arxiv_metadata(arxiv_ids)
    logger.info(f"Fetched {len(arxiv_data)} papers from arXiv")

    # Step 2: Gate + significance scoring
    for arxiv_id in arxiv_ids:
        meta = arxiv_data.get(arxiv_id)
        if meta is None:
            batch.errors.append(f"arXiv fetch failed for {arxiv_id}")
            continue

        title = meta["title"]
        abstract = meta["abstract"]
        authors = meta["authors"]
        venue = venues.get(arxiv_id)
        year = years.get(arxiv_id, _extract_year_from_id(arxiv_id))
        cites = citation_counts.get(arxiv_id, 0)
        affs = affiliations.get(arxiv_id)
        referenced = arxiv_id in admitted_paper_ids

        paper = PaperMetadata(
            arxiv_id=arxiv_id,
            title=title,
            abstract=abstract,
            authors=authors,
            year=year,
            venue=venue,
            citation_count=cites,
            affiliations=affs,
        )

        # Run gate
        gate_result = run_gate(
            client=client,
            title=title,
            abstract=abstract,
            venue=venue,
            citation_count=cites,
            authors=authors,
            affiliations=affs,
            referenced_by_admitted=referenced,
            model=extraction_model,
        )

        enrich_metadata(paper, gate_result)

        if gate_result.decision == GateDecision.ADMIT:
            batch.admitted.append(paper)
            batch.total_admitted += 1
        else:
            batch.rejected.append(paper)
            batch.total_rejected += 1

    # Step 3: Extract admitted papers
    if not skip_extraction:
        for paper in batch.admitted:
            try:
                logger.info(f"Extracting: {paper.title[:60]}...")
                result = extract_paper(
                    client=client,
                    arxiv_id=paper.arxiv_id,
                    title=paper.title,
                    abstract=paper.abstract,
                    authors=paper.authors,
                    year=paper.year,
                    venue=paper.venue,
                    model=extraction_model,
                )
                batch.extraction_results.append(result)
                batch.total_extracted += 1
            except Exception as e:
                batch.errors.append(f"Extraction failed for {paper.arxiv_id}: {e}")
                logger.error(f"Extraction failed for {paper.arxiv_id}: {e}")

    logger.info(batch.summary())
    return batch


def _extract_year_from_id(arxiv_id: str) -> int:
    """Extract approximate year from arXiv ID format.

    Old format: YYMM.NNNNN (e.g., 1706.03762 → 2017)
    New format: YYMM.NNNNN (e.g., 2401.04088 → 2024)
    """
    try:
        prefix = arxiv_id.split(".")[0]
        yy = int(prefix[:2])
        # arXiv IDs from 91+ are 1991+, those below are 2000+
        year = 1900 + yy if yy >= 91 else 2000 + yy
        return year
    except (ValueError, IndexError):
        return CURRENT_YEAR  # fallback
