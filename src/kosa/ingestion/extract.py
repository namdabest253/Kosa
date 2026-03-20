"""Extraction pipeline: fetch abstracts from arXiv, run entity/relation extraction.

Orchestrates the Phase 0 extraction audit workflow:
1. Fetch paper metadata from arXiv API
2. Run entity extraction (GPT-4o-mini)
3. Run relation extraction (GPT-4o-mini)
4. Run schema alignment (GPT-4o-mini)
5. Run novelty classification (GPT-4o-mini)
"""

from __future__ import annotations

import json
import logging
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import httpx
from openai import OpenAI

from kosa.graph.schema import EdgeType, NodeLabel, get_venue_weight
from kosa.ingestion.evaluate import RELATION_DIRECTION_RULES
from kosa.ingestion.prompts import (
    RELATION_TYPE_MAP,
    SOURCE_TYPE_MAP,
    VALID_NOVELTY_CLASSES,
    format_entity_extraction,
    format_novelty_classifier,
    format_relation_extraction,
    format_schema_alignment,
)

logger = logging.getLogger(__name__)

ARXIV_API_URL = "https://export.arxiv.org/api/query"
ARXIV_BATCH_SIZE = 20  # arXiv API allows up to ~20 IDs per request
ARXIV_RATE_LIMIT_SECONDS = 3  # be polite


# ---------------------------------------------------------------------------
# Data classes for extraction results
# ---------------------------------------------------------------------------


@dataclass
class ExtractedEntity:
    name: str
    entity_type: NodeLabel
    description: str = ""
    mathematical_structure: str = ""  # technique only
    bottleneck_class: str = ""  # problem only
    source_paper: str = ""  # arxiv_id


@dataclass
class ExtractedRelation:
    source_name: str
    source_type: NodeLabel
    relation: EdgeType
    target_name: str
    target_type: NodeLabel
    confidence: float
    supporting_text: str
    source_paper: str  # arxiv_id


@dataclass
class NoveltyResult:
    classification: str
    confidence: float
    reasoning: str


@dataclass
class PaperExtractionResult:
    arxiv_id: str
    title: str
    abstract: str
    authors: list[str]
    year: int
    venue: str | None
    entities: list[ExtractedEntity] = field(default_factory=list)
    relations: list[ExtractedRelation] = field(default_factory=list)
    novelty: NoveltyResult | None = None
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# arXiv fetching
# ---------------------------------------------------------------------------


def fetch_arxiv_metadata(arxiv_ids: list[str]) -> dict[str, dict]:
    """Fetch title and abstract for a list of arXiv IDs.

    Returns dict mapping arxiv_id → {"title": ..., "abstract": ..., "authors": [...]}.
    """
    results = {}
    for i in range(0, len(arxiv_ids), ARXIV_BATCH_SIZE):
        batch = arxiv_ids[i : i + ARXIV_BATCH_SIZE]
        id_list = ",".join(batch)
        url = f"{ARXIV_API_URL}?id_list={id_list}&max_results={len(batch)}"

        logger.info(f"Fetching arXiv batch {i // ARXIV_BATCH_SIZE + 1}: {len(batch)} papers")
        resp = httpx.get(url, timeout=30)
        resp.raise_for_status()

        root = ET.fromstring(resp.text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}

        for entry in root.findall("atom:entry", ns):
            title_el = entry.find("atom:title", ns)
            summary_el = entry.find("atom:summary", ns)
            id_el = entry.find("atom:id", ns)

            if id_el is None or title_el is None:
                continue

            # Extract arxiv_id from URL like http://arxiv.org/abs/1706.03762v5
            raw_id = id_el.text.strip().split("/abs/")[-1]
            # Remove version suffix
            clean_id = raw_id.split("v")[0]

            authors = []
            for author in entry.findall("atom:author", ns):
                name_el = author.find("atom:name", ns)
                if name_el is not None:
                    authors.append(name_el.text.strip())

            results[clean_id] = {
                "title": " ".join(title_el.text.strip().split()),
                "abstract": (
                    " ".join(summary_el.text.strip().split()) if summary_el is not None else ""
                ),
                "authors": authors,
            }

        if i + ARXIV_BATCH_SIZE < len(arxiv_ids):
            time.sleep(ARXIV_RATE_LIMIT_SECONDS)

    return results


# ---------------------------------------------------------------------------
# LLM extraction calls
# ---------------------------------------------------------------------------


def _call_llm(client: OpenAI, messages: list[dict], model: str) -> dict | None:
    """Call OpenAI API and parse JSON response. Returns None on failure."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        text = response.choices[0].message.content
        return json.loads(text)
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        logger.warning(f"Failed to parse LLM response: {e}")
        return None
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return None


def extract_entities(
    client: OpenAI, title: str, abstract: str, model: str
) -> tuple[list[dict], list[dict], list[dict]]:
    """Extract entities from a paper. Returns (techniques, problems, datasets)."""
    messages = format_entity_extraction(title, abstract)
    result = _call_llm(client, messages, model)
    if result is None:
        return [], [], []
    return (
        result.get("techniques", []),
        result.get("problems", []),
        result.get("datasets", []),
    )


def extract_relations(
    client: OpenAI,
    title: str,
    abstract: str,
    arxiv_id: str,
    techniques: list[str],
    problems: list[str],
    datasets: list[str],
    model: str,
) -> list[dict]:
    """Extract relations between entities. Returns list of relation dicts."""
    messages = format_relation_extraction(title, abstract, arxiv_id, techniques, problems, datasets)
    result = _call_llm(client, messages, model)
    if result is None:
        return []
    return result.get("relations", [])


def align_schema(
    client: OpenAI, techniques: list[dict], problems: list[dict], model: str
) -> tuple[list[dict], list[dict]]:
    """Run schema alignment to add mathematical_structure and bottleneck_class."""
    if not techniques and not problems:
        return [], []
    messages = format_schema_alignment(techniques, problems)
    result = _call_llm(client, messages, model)
    if result is None:
        return techniques, problems
    return result.get("techniques", techniques), result.get("problems", problems)


def classify_novelty(client: OpenAI, title: str, abstract: str, model: str) -> NoveltyResult | None:
    """Classify paper novelty. Returns NoveltyResult or None on failure."""
    messages = format_novelty_classifier(title, abstract)
    result = _call_llm(client, messages, model)
    if result is None:
        return None
    classification = result.get("classification", "")
    if classification not in VALID_NOVELTY_CLASSES:
        return None
    return NoveltyResult(
        classification=classification,
        confidence=float(result.get("confidence", 0.0)),
        reasoning=result.get("reasoning", ""),
    )


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def extract_paper(
    client: OpenAI,
    arxiv_id: str,
    title: str,
    abstract: str,
    authors: list[str],
    year: int,
    venue: str | None,
    model: str = "gpt-4o-mini",
) -> PaperExtractionResult:
    """Run full extraction pipeline on a single paper."""
    result = PaperExtractionResult(
        arxiv_id=arxiv_id,
        title=title,
        abstract=abstract,
        authors=authors,
        year=year,
        venue=venue,
    )

    # Step 1: Entity extraction
    techniques_raw, problems_raw, datasets_raw = extract_entities(client, title, abstract, model)

    # Step 2: Schema alignment (adds mathematical_structure, bottleneck_class)
    techniques_aligned, problems_aligned = align_schema(client, techniques_raw, problems_raw, model)

    # Build entity objects
    for t in techniques_aligned:
        result.entities.append(
            ExtractedEntity(
                name=t.get("name", t.get("original_name", "")),
                entity_type=NodeLabel.TECHNIQUE,
                description=t.get("description", ""),
                mathematical_structure=t.get("mathematical_structure", ""),
                source_paper=arxiv_id,
            )
        )
    for p in problems_aligned:
        result.entities.append(
            ExtractedEntity(
                name=p.get("name", p.get("original_name", "")),
                entity_type=NodeLabel.PROBLEM,
                description=p.get("description", ""),
                bottleneck_class=p.get("bottleneck_class", ""),
                source_paper=arxiv_id,
            )
        )
    for d in datasets_raw:
        result.entities.append(
            ExtractedEntity(
                name=d.get("name", ""),
                entity_type=NodeLabel.DATASET,
                description=d.get("description", ""),
                source_paper=arxiv_id,
            )
        )

    # Step 3: Relation extraction
    technique_names = [t.get("name", "") for t in techniques_aligned]
    problem_names = [p.get("name", "") for p in problems_aligned]
    dataset_names = [d.get("name", "") for d in datasets_raw]

    relations_raw = extract_relations(
        client,
        title,
        abstract,
        arxiv_id,
        technique_names,
        problem_names,
        dataset_names,
        model,
    )

    _, venue_weight = get_venue_weight(venue)

    for rel in relations_raw:
        # Drop self-referential relations
        if rel.get("source", "") == rel.get("target", "") and rel.get("source_type", "") == rel.get(
            "target_type", ""
        ):
            src = rel.get("source")
            tgt = rel.get("target")
            result.errors.append(
                f"Dropped self-referential relation: {src} → {rel.get('relation')} → {tgt}"
            )
            continue

        # Validate direction rules before accepting the relation
        rel_type_str = rel.get("relation", "")
        src_type_str = rel.get("source_type", "").lower()
        tgt_type_str = rel.get("target_type", "").lower()
        expected = RELATION_DIRECTION_RULES.get(rel_type_str)
        if expected is not None:
            expected_src, expected_tgt = expected
            src_ok = src_type_str in expected_src.split("|")
            tgt_ok = tgt_type_str in expected_tgt.split("|")
            if not (src_ok and tgt_ok):
                result.errors.append(
                    f"Dropped relation with wrong types: "
                    f"{src_type_str} → {rel_type_str} → {tgt_type_str} "
                    f"(expected {expected_src} → {expected_tgt})"
                )
                continue
        # Drop vague IMPROVES_OVER targets
        if rel_type_str == "IMPROVES_OVER":
            target = rel.get("target", "").lower()
            vague_targets = {
                "existing methods",
                "previous approaches",
                "prior work",
                "existing best results",
                "baseline",
                "baselines",
                "existing models",
                "previous methods",
                "state-of-the-art",
                "existing approaches",
                "traditional methods",
            }
            if target in vague_targets:
                result.errors.append(
                    f"Dropped vague IMPROVES_OVER: {rel.get('source')} → {rel.get('target')}"
                )
                continue

        source_type = SOURCE_TYPE_MAP.get(rel.get("source_type", ""))
        target_type = SOURCE_TYPE_MAP.get(rel.get("target_type", ""))
        relation_type = RELATION_TYPE_MAP.get(rel.get("relation", ""))

        if source_type is None or target_type is None or relation_type is None:
            result.errors.append(f"Invalid relation types in: {rel}")
            continue

        result.relations.append(
            ExtractedRelation(
                source_name=rel.get("source", ""),
                source_type=source_type,
                relation=relation_type,
                target_name=rel.get("target", ""),
                target_type=target_type,
                confidence=float(rel.get("confidence", 0.5)),
                supporting_text=rel.get("supporting_text", ""),
                source_paper=arxiv_id,
            )
        )

    # Step 4: Novelty classification
    result.novelty = classify_novelty(client, title, abstract, model)

    return result
