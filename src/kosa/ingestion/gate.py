"""Ingestion gate: filters papers before expensive LLM extraction.

Gate logic (from DESIGN.md):
- Tier 1-3 venues: auto-admit
- Tier 4-5 venues: require secondary signal (high citations OR high novelty)
- Tier 6 (arXiv preprints): require known author/lab OR >N citations OR
  referenced by an already-admitted paper

The gate runs BEFORE extraction to avoid wasting LLM calls on papers that
won't contribute meaningful signal to the knowledge graph.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum

from openai import OpenAI

from kosa.graph.schema import VenueTier, get_venue_weight
from kosa.ingestion.extract import classify_novelty

logger = logging.getLogger(__name__)

# Thresholds
CITATION_THRESHOLD_TIER456 = 10  # min citations for Tier 4-5 secondary signal
CITATION_THRESHOLD_TIER6 = 50  # min citations for Tier 6 without other signals
NOVELTY_ADMIT_CLASSES = {"novel_technique", "significant_improvement"}

# Known labs whose arXiv preprints get priority (Tier 6 → 0.25 weight).
KNOWN_LABS: set[str] = {
    "google",
    "google deepmind",
    "deepmind",
    "openai",
    "meta",
    "meta ai",
    "fair",
    "microsoft",
    "microsoft research",
    "anthropic",
    "nvidia",
    "apple",
    "ibm research",
    "amazon",
    "amazon science",
    "stanford",
    "stanford university",
    "mit",
    "cmu",
    "carnegie mellon",
    "berkeley",
    "uc berkeley",
    "princeton",
    "mila",
    "inria",
    "tsinghua",
    "tsinghua university",
    "peking university",
    "shanghai ai laboratory",
    "baidu",
    "alibaba",
    "tencent",
    "bytedance",
    "hugging face",
    "huggingface",
    "eleutherai",
    "together ai",
    "cohere",
    "mistral",
    "mistral ai",
    "stability ai",
    "ai2",
    "allen institute",
}


class GateDecision(StrEnum):
    ADMIT = "admit"
    REJECT = "reject"
    NEEDS_NOVELTY_CHECK = "needs_novelty_check"


@dataclass
class GateResult:
    """Result of the ingestion gate for a single paper."""

    decision: GateDecision
    reason: str
    venue_tier: VenueTier
    venue_weight: float
    known_lab: bool = False
    novelty_class: str | None = None  # set if novelty check was performed


def is_known_lab(authors: list[str], affiliations: list[str] | None = None) -> bool:
    """Check if any author affiliation matches a known lab.

    Only checks affiliations (not author names, since substring matching on
    names produces false positives like "Smith" matching "mit").
    Falls back to checking if author strings contain lab names as whole words.
    """
    import re

    check_strings: list[str] = []
    if affiliations:
        check_strings.extend(a.lower() for a in affiliations)
    # Also check author strings — they sometimes contain affiliations
    check_strings.extend(a.lower() for a in authors)

    for s in check_strings:
        for lab in KNOWN_LABS:
            # Word boundary match to avoid "Smith" matching "mit"
            if re.search(r"\b" + re.escape(lab) + r"\b", s):
                return True
    return False


def gate_paper(
    venue: str | None,
    citation_count: int,
    authors: list[str],
    affiliations: list[str] | None = None,
    referenced_by_admitted: bool = False,
) -> GateResult:
    """Apply ingestion gate logic without novelty classification.

    For papers that return NEEDS_NOVELTY_CHECK, the caller should run
    novelty classification and then call gate_with_novelty().

    Args:
        venue: Venue name (None for arXiv preprints).
        citation_count: Number of citations at ingestion time.
        authors: List of author names.
        affiliations: Optional list of author affiliations.
        referenced_by_admitted: Whether an already-admitted paper cites this one.

    Returns:
        GateResult with decision and reason.
    """
    known = is_known_lab(authors, affiliations)
    tier, weight = get_venue_weight(venue, known_lab=known)

    # Tier 1-3: auto-admit
    if tier <= VenueTier.TIER_3:
        return GateResult(
            decision=GateDecision.ADMIT,
            reason=f"Auto-admit: Tier {tier.value} venue ({venue})",
            venue_tier=tier,
            venue_weight=weight,
            known_lab=known,
        )

    # Tier 4-5: require secondary signal
    if tier <= VenueTier.TIER_5:
        if citation_count >= CITATION_THRESHOLD_TIER456:
            return GateResult(
                decision=GateDecision.ADMIT,
                reason=(
                    f"Tier {tier.value} admitted: "
                    f"{citation_count} citations >= {CITATION_THRESHOLD_TIER456}"
                ),
                venue_tier=tier,
                venue_weight=weight,
                known_lab=known,
            )
        # Need novelty check as secondary signal
        return GateResult(
            decision=GateDecision.NEEDS_NOVELTY_CHECK,
            reason=f"Tier {tier.value} with {citation_count} citations: needs novelty check",
            venue_tier=tier,
            venue_weight=weight,
            known_lab=known,
        )

    # Tier 6 (arXiv preprints): require known lab OR high citations OR admitted reference
    if known:
        return GateResult(
            decision=GateDecision.ADMIT,
            reason="arXiv admitted: known lab/author",
            venue_tier=tier,
            venue_weight=weight,
            known_lab=known,
        )

    if citation_count >= CITATION_THRESHOLD_TIER6:
        return GateResult(
            decision=GateDecision.ADMIT,
            reason=f"arXiv admitted: {citation_count} citations >= {CITATION_THRESHOLD_TIER6}",
            venue_tier=tier,
            venue_weight=weight,
            known_lab=known,
        )

    if referenced_by_admitted:
        return GateResult(
            decision=GateDecision.ADMIT,
            reason="arXiv admitted: referenced by an already-admitted paper",
            venue_tier=tier,
            venue_weight=weight,
            known_lab=known,
        )

    # Tier 6 without any signal: need novelty check as last resort
    return GateResult(
        decision=GateDecision.NEEDS_NOVELTY_CHECK,
        reason=(
            "arXiv with no known lab, low citations, no admitted references: needs novelty check"
        ),
        venue_tier=tier,
        venue_weight=weight,
        known_lab=known,
    )


def gate_with_novelty(gate_result: GateResult, novelty_class: str) -> GateResult:
    """Resolve a NEEDS_NOVELTY_CHECK decision using the novelty classification.

    Args:
        gate_result: A GateResult with decision == NEEDS_NOVELTY_CHECK.
        novelty_class: The novelty classification from classify_novelty().

    Returns:
        Updated GateResult with final ADMIT or REJECT decision.
    """
    if gate_result.decision != GateDecision.NEEDS_NOVELTY_CHECK:
        return gate_result

    gate_result.novelty_class = novelty_class

    if novelty_class in NOVELTY_ADMIT_CLASSES:
        return GateResult(
            decision=GateDecision.ADMIT,
            reason=f"Admitted after novelty check: {novelty_class}",
            venue_tier=gate_result.venue_tier,
            venue_weight=gate_result.venue_weight,
            known_lab=gate_result.known_lab,
            novelty_class=novelty_class,
        )

    return GateResult(
        decision=GateDecision.REJECT,
        reason=f"Rejected: Tier {gate_result.venue_tier.value} with novelty={novelty_class}",
        venue_tier=gate_result.venue_tier,
        venue_weight=gate_result.venue_weight,
        known_lab=gate_result.known_lab,
        novelty_class=novelty_class,
    )


def run_gate(
    client: OpenAI,
    title: str,
    abstract: str,
    venue: str | None,
    citation_count: int,
    authors: list[str],
    affiliations: list[str] | None = None,
    referenced_by_admitted: bool = False,
    model: str = "gpt-4o-mini",
) -> GateResult:
    """Full gate check: venue tier + novelty classification if needed.

    This is the main entry point. It runs the cheap venue/citation check first,
    and only calls the LLM for novelty classification when necessary.
    """
    result = gate_paper(
        venue=venue,
        citation_count=citation_count,
        authors=authors,
        affiliations=affiliations,
        referenced_by_admitted=referenced_by_admitted,
    )

    if result.decision != GateDecision.NEEDS_NOVELTY_CHECK:
        logger.info(f"Gate: {result.decision.value} - {title[:60]}... ({result.reason})")
        return result

    # Run novelty classification (GPT-4o-mini call)
    logger.info(f"Gate: running novelty check for {title[:60]}...")
    novelty = classify_novelty(client, title, abstract, model)
    novelty_class = novelty.classification if novelty else "unknown"

    result = gate_with_novelty(result, novelty_class)
    logger.info(f"Gate: {result.decision.value} - {title[:60]}... ({result.reason})")
    return result
