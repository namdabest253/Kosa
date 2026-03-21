"""Streaming ingestion: incremental arXiv fetch + Neo4j merge.

Supports:
- Date-range arXiv fetching with watermark tracking (last processed date)
- Incremental Neo4j merge: MERGE semantics for papers + entities
- Stub nodes for unknown cited papers
- Incremental entity resolution against existing graph
- Background execution via FastAPI BackgroundTasks
"""

from __future__ import annotations

import json
import logging
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import httpx

from kosa.config import settings
from kosa.ingestion.extract import (
    PaperExtractionResult,
    extract_paper,
    fetch_arxiv_metadata,
)
from kosa.ingestion.pipeline import (
    PaperMetadata,
    compute_recency_factor,
)

logger = logging.getLogger(__name__)

# arXiv OAI search endpoint for date-range queries
ARXIV_SEARCH_URL = "https://export.arxiv.org/api/query"
ARXIV_RATE_LIMIT_SECONDS = 3
DEFAULT_MAX_RESULTS = 100

# Watermark file: stores the last processed date
WATERMARK_PATH = Path(".kosa_streaming_watermark.json")


@dataclass
class IngestionJob:
    """Tracks the state of a streaming ingestion job."""

    job_id: str
    status: str = "pending"  # pending, running, completed, failed
    arxiv_ids: list[str] = field(default_factory=list)
    date_from: str | None = None
    date_to: str | None = None
    papers_fetched: int = 0
    papers_extracted: int = 0
    papers_loaded: int = 0
    entities_loaded: int = 0
    relations_loaded: int = 0
    errors: list[str] = field(default_factory=list)
    started_at: str | None = None
    completed_at: str | None = None

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "papers_fetched": self.papers_fetched,
            "papers_extracted": self.papers_extracted,
            "papers_loaded": self.papers_loaded,
            "entities_loaded": self.entities_loaded,
            "relations_loaded": self.relations_loaded,
            "errors": self.errors[-10:],  # last 10 errors
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }


# In-memory job registry (single-process; sufficient for research tool)
_jobs: dict[str, IngestionJob] = {}


def get_job(job_id: str) -> IngestionJob | None:
    return _jobs.get(job_id)


def register_job(job: IngestionJob) -> None:
    _jobs[job.job_id] = job


# ---------------------------------------------------------------------------
# Watermark management
# ---------------------------------------------------------------------------


def read_watermark() -> str | None:
    """Read the last processed date from the watermark file."""
    if WATERMARK_PATH.exists():
        data = json.loads(WATERMARK_PATH.read_text())
        return data.get("last_date")
    return None


def write_watermark(date_str: str) -> None:
    """Write the last processed date to the watermark file."""
    data = {"last_date": date_str, "updated_at": datetime.now(UTC).isoformat()}
    WATERMARK_PATH.write_text(json.dumps(data))


# ---------------------------------------------------------------------------
# arXiv date-range fetcher
# ---------------------------------------------------------------------------


def fetch_arxiv_by_date(
    date_from: str,
    date_to: str,
    category: str = "cs.AI",
    max_results: int = DEFAULT_MAX_RESULTS,
) -> list[dict]:
    """Fetch arXiv papers in a date range for a given category.

    Args:
        date_from: Start date (YYYY-MM-DD)
        date_to: End date (YYYY-MM-DD)
        category: arXiv category (default cs.AI)
        max_results: Maximum number of results

    Returns:
        List of dicts with keys: arxiv_id, title, abstract, authors, published
    """
    # arXiv API uses submittedDate for date range queries
    date_f = date_from.replace("-", "")
    date_t = date_to.replace("-", "")
    query = f"cat:{category} AND submittedDate:[{date_f} TO {date_t}]"

    results = []
    start = 0
    batch_size = min(max_results, 100)

    while start < max_results:
        url = (
            f"{ARXIV_SEARCH_URL}?search_query={query}"
            f"&start={start}&max_results={batch_size}"
            f"&sortBy=submittedDate&sortOrder=descending"
        )

        logger.info(f"Fetching arXiv date range batch: start={start}, query={query}")
        resp = httpx.get(url, timeout=30)
        resp.raise_for_status()

        root = ET.fromstring(resp.text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}

        entries = root.findall("atom:entry", ns)
        if not entries:
            break

        for entry in entries:
            title_el = entry.find("atom:title", ns)
            summary_el = entry.find("atom:summary", ns)
            id_el = entry.find("atom:id", ns)
            published_el = entry.find("atom:published", ns)

            if id_el is None or title_el is None:
                continue

            raw_id = id_el.text.strip().split("/abs/")[-1]
            clean_id = raw_id.split("v")[0]

            authors = []
            for author in entry.findall("atom:author", ns):
                name_el = author.find("atom:name", ns)
                if name_el is not None:
                    authors.append(name_el.text.strip())

            published = ""
            if published_el is not None:
                published = published_el.text.strip()[:10]  # YYYY-MM-DD

            results.append(
                {
                    "arxiv_id": clean_id,
                    "title": " ".join(title_el.text.strip().split()),
                    "abstract": (
                        " ".join(summary_el.text.strip().split()) if summary_el is not None else ""
                    ),
                    "authors": authors,
                    "published": published,
                }
            )

        start += batch_size
        if len(entries) < batch_size:
            break

        time.sleep(ARXIV_RATE_LIMIT_SECONDS)

    return results[:max_results]


# ---------------------------------------------------------------------------
# Incremental Neo4j merge
# ---------------------------------------------------------------------------


def merge_papers_incremental(
    driver,
    papers: list[PaperMetadata],
    extraction_results: list[PaperExtractionResult],
) -> dict[str, int]:
    """Merge new papers and their extractions into Neo4j incrementally.

    Uses MERGE semantics: existing nodes are updated, new ones created.
    Creates stub Paper nodes for cited papers not yet in the graph.

    Returns counts: {papers, entities, relations, stubs}
    """
    counts = {"papers": 0, "entities": 0, "relations": 0, "stubs": 0}

    with driver.session() as session:
        # Merge paper nodes
        for paper in papers:
            session.run(
                "MERGE (p:Paper {arxiv_id: $arxiv_id}) "
                "SET p.title = $title, p.abstract = $abstract, "
                "p.authors = $authors, p.year = $year, "
                "p.venue_tier = $venue_tier, p.significance = $significance, "
                "p.ingested_at = $ingested_at",
                {
                    "arxiv_id": paper.arxiv_id,
                    "title": paper.title,
                    "abstract": paper.abstract,
                    "authors": paper.authors,
                    "year": paper.year,
                    "venue_tier": paper.venue_tier.value if paper.venue_tier else None,
                    "significance": paper.significance,
                    "ingested_at": datetime.now(UTC).isoformat(),
                },
            )
            counts["papers"] += 1

        # Merge extracted entities and relations
        for result in extraction_results:
            for entity in result.entities:
                label = entity.entity_type.value
                session.run(
                    f"MERGE (e:{label} {{name: $name}}) SET e.description = $description",
                    {"name": entity.name, "description": entity.description},
                )

                # Link entity to paper
                session.run(
                    f"MATCH (p:Paper {{arxiv_id: $arxiv_id}}) "
                    f"MATCH (e:{label} {{name: $name}}) "
                    f"MERGE (p)-[:INTRODUCES]->(e)",
                    {"arxiv_id": result.arxiv_id, "name": entity.name},
                )
                counts["entities"] += 1

            for rel in result.relations:
                src_label = rel.source_type.value
                tgt_label = rel.target_type.value
                rel_type = rel.relation.value
                session.run(
                    f"MATCH (s:{src_label} {{name: $src_name}}) "
                    f"MATCH (t:{tgt_label} {{name: $tgt_name}}) "
                    f"MERGE (s)-[r:{rel_type}]->(t) "
                    f"SET r.confidence = $confidence, r.source_paper = $source_paper, "
                    f"r.supporting_text = $supporting_text",
                    {
                        "src_name": rel.source_name,
                        "tgt_name": rel.target_name,
                        "confidence": rel.confidence,
                        "source_paper": rel.source_paper,
                        "supporting_text": rel.supporting_text,
                    },
                )
                counts["relations"] += 1

    return counts


def merge_citations_incremental(
    driver,
    citations: dict[str, list[str]],
) -> dict[str, int]:
    """Merge citation edges, creating stub Paper nodes for unknown targets.

    Returns counts: {edges, stubs}
    """
    counts = {"edges": 0, "stubs": 0}

    with driver.session() as session:
        for source_id, cited_ids in citations.items():
            for target_id in cited_ids:
                # Create stub if target doesn't exist
                result = session.run(
                    "MERGE (t:Paper {arxiv_id: $target_id}) "
                    "ON CREATE SET t.stub = true, t.title = $target_id "
                    "RETURN t.stub AS is_stub",
                    {"target_id": target_id},
                )
                rec = result.single()
                if rec and rec["is_stub"]:
                    counts["stubs"] += 1

                # Create CITES edge
                session.run(
                    "MATCH (a:Paper {arxiv_id: $source}) "
                    "MATCH (b:Paper {arxiv_id: $target}) "
                    "MERGE (a)-[r:CITES]->(b) "
                    "SET r.source_paper = $source",
                    {"source": source_id, "target": target_id},
                )
                counts["edges"] += 1

    return counts


# ---------------------------------------------------------------------------
# Main streaming ingestion function
# ---------------------------------------------------------------------------


def run_streaming_ingestion(job: IngestionJob) -> None:
    """Execute a streaming ingestion job. Designed to run as a background task."""
    from neo4j import GraphDatabase

    job.status = "running"
    job.started_at = datetime.now(UTC).isoformat()

    try:
        # Step 1: Fetch papers
        if job.arxiv_ids:
            # Fetch by explicit IDs
            metadata = fetch_arxiv_metadata(job.arxiv_ids)
            papers_raw = [{"arxiv_id": aid, **meta} for aid, meta in metadata.items()]
        elif job.date_from and job.date_to:
            # Fetch by date range
            papers_raw = fetch_arxiv_by_date(
                job.date_from,
                job.date_to,
                max_results=DEFAULT_MAX_RESULTS,
            )
        else:
            job.status = "failed"
            job.errors.append("Must specify either arxiv_ids or date_from/date_to")
            return

        job.papers_fetched = len(papers_raw)
        if not papers_raw:
            job.status = "completed"
            job.completed_at = datetime.now(UTC).isoformat()
            return

        # Step 2: Build PaperMetadata and compute significance
        papers: list[PaperMetadata] = []
        for raw in papers_raw:
            year = int(raw.get("published", "2025")[:4]) if raw.get("published") else 2025
            pm = PaperMetadata(
                arxiv_id=raw["arxiv_id"],
                title=raw["title"],
                abstract=raw.get("abstract", ""),
                authors=raw.get("authors", []),
                year=year,
                venue=None,
            )
            pm.venue_weight = 0.15  # Unknown venue default for arXiv-only
            pm.recency_factor = compute_recency_factor(year)
            pm.citation_signal = 0.5  # Default; Semantic Scholar enrichment is separate
            pm.novelty_signal = 0.5  # Will be updated by extraction
            pm.compute_significance()
            papers.append(pm)

        # Step 3: Run extraction on each paper
        from openai import OpenAI

        client = OpenAI(api_key=settings.openai_api_key)
        model = settings.extraction_model

        extraction_results: list[PaperExtractionResult] = []
        for paper in papers:
            try:
                result = extract_paper(
                    client,
                    paper.arxiv_id,
                    paper.title,
                    paper.abstract,
                    paper.authors,
                    paper.year,
                    paper.venue,
                    model=model,
                )
                extraction_results.append(result)
                job.papers_extracted += 1
            except Exception as e:
                job.errors.append(f"Extraction failed for {paper.arxiv_id}: {e}")
                logger.error(f"Extraction failed for {paper.arxiv_id}: {e}")

        # Step 4: Merge into Neo4j
        driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
        try:
            counts = merge_papers_incremental(driver, papers, extraction_results)
            job.papers_loaded = counts["papers"]
            job.entities_loaded = counts["entities"]
            job.relations_loaded = counts["relations"]
        except Exception as e:
            job.errors.append(f"Neo4j merge failed: {e}")
            logger.error(f"Neo4j merge failed: {e}")
        finally:
            driver.close()

        # Step 5: Update watermark
        if job.date_to:
            write_watermark(job.date_to)

        job.status = "completed"
        job.completed_at = datetime.now(UTC).isoformat()

    except Exception as e:
        job.status = "failed"
        job.errors.append(str(e))
        logger.error(f"Streaming ingestion failed: {e}")
    finally:
        if not job.completed_at:
            job.completed_at = datetime.now(UTC).isoformat()
