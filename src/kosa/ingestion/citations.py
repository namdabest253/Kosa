"""Citation extraction via Semantic Scholar API.

Fetches reference lists for papers to build the Layer 0 citation graph.
arXiv doesn't expose citation data, so we use Semantic Scholar's free API.

Rate limit: 1 request/second without API key, 10 req/sec with key.
For 5,000 papers at 1 req/sec ≈ 83 minutes — acceptable for Phase 1.
"""

from __future__ import annotations

import logging
import time

import httpx

logger = logging.getLogger(__name__)

S2_API_URL = "https://api.semanticscholar.org/graph/v1"
S2_RATE_LIMIT_SECONDS = 1.1  # slightly above 1s to be safe
S2_BATCH_SIZE = 500  # max for batch endpoint
S2_TIMEOUT = 30


def fetch_citations_single(
    arxiv_id: str,
    api_key: str | None = None,
    timeout: int = S2_TIMEOUT,
) -> list[str]:
    """Fetch references for a single paper, returning arXiv IDs of cited papers.

    Args:
        arxiv_id: arXiv ID of the paper (e.g., "1706.03762").
        api_key: Optional Semantic Scholar API key for higher rate limits.
        timeout: Request timeout in seconds.

    Returns:
        List of arXiv IDs that this paper cites (empty if not found).
    """
    url = f"{S2_API_URL}/paper/ARXIV:{arxiv_id}"
    params = {"fields": "references.externalIds"}
    headers = {}
    if api_key:
        headers["x-api-key"] = api_key

    try:
        resp = httpx.get(url, params=params, headers=headers, timeout=timeout)
        if resp.status_code == 404:
            logger.debug(f"Paper not found on S2: {arxiv_id}")
            return []
        if resp.status_code == 429:
            logger.warning(f"Rate limited by S2 for {arxiv_id}")
            return []
        resp.raise_for_status()
        data = resp.json()
    except (httpx.HTTPError, ValueError) as e:
        logger.warning(f"S2 API error for {arxiv_id}: {e}")
        return []

    refs = data.get("references") or []
    cited_arxiv_ids = []
    for ref in refs:
        ext_ids = ref.get("externalIds") or {}
        ref_arxiv = ext_ids.get("ArXiv")
        if ref_arxiv:
            cited_arxiv_ids.append(ref_arxiv)

    return cited_arxiv_ids


def fetch_citations_batch(
    arxiv_ids: list[str],
    api_key: str | None = None,
    corpus_ids: set[str] | None = None,
    rate_limit: float = S2_RATE_LIMIT_SECONDS,
) -> dict[str, list[str]]:
    """Fetch citations for multiple papers, returning only intra-corpus edges.

    Args:
        arxiv_ids: List of arXiv IDs to fetch citations for.
        api_key: Optional Semantic Scholar API key.
        corpus_ids: If provided, only return citations to papers in this set.
            This filters to intra-corpus CITES edges. If None, returns all.
        rate_limit: Seconds between API requests.

    Returns:
        Dict mapping arxiv_id → list of cited arXiv IDs.
    """
    results: dict[str, list[str]] = {}
    total = len(arxiv_ids)

    for i, arxiv_id in enumerate(arxiv_ids):
        logger.info(f"Fetching citations [{i + 1}/{total}]: {arxiv_id}")

        cited = fetch_citations_single(arxiv_id, api_key=api_key)

        # Filter to intra-corpus edges if corpus specified
        if corpus_ids is not None:
            cited = [c for c in cited if c in corpus_ids]

        results[arxiv_id] = cited

        # Rate limiting (skip on last request)
        if i < total - 1:
            time.sleep(rate_limit)

    # Summary
    total_edges = sum(len(v) for v in results.values())
    papers_with_refs = sum(1 for v in results.values() if v)
    logger.info(f"Citations fetched: {total_edges} edges from {papers_with_refs}/{total} papers")

    return results


def validate_citation_graph(
    citations: dict[str, list[str]],
) -> dict[str, list[str]]:
    """Validate and clean citation data.

    Removes:
    - Self-citations (paper cites itself)
    - Duplicate edges

    Returns:
        Cleaned citation dict with list of issues found.
    """
    issues: dict[str, list[str]] = {
        "self_citations": [],
        "duplicates": [],
    }
    cleaned: dict[str, list[str]] = {}

    for paper_id, cited_ids in citations.items():
        seen = set()
        clean_cited = []
        for cited_id in cited_ids:
            if cited_id == paper_id:
                issues["self_citations"].append(paper_id)
                continue
            if cited_id in seen:
                issues["duplicates"].append(f"{paper_id} -> {cited_id}")
                continue
            seen.add(cited_id)
            clean_cited.append(cited_id)
        cleaned[paper_id] = clean_cited

    if issues["self_citations"]:
        logger.warning(f"Removed {len(issues['self_citations'])} self-citations")
    if issues["duplicates"]:
        logger.warning(f"Removed {len(issues['duplicates'])} duplicate edges")

    return cleaned


def citation_graph_stats(
    citations: dict[str, list[str]],
) -> dict[str, object]:
    """Compute basic statistics for the citation graph.

    Returns dict with:
    - total_papers: number of papers
    - total_edges: number of citation edges
    - papers_with_refs: papers that cite at least one corpus paper
    - papers_cited: papers cited by at least one corpus paper
    - avg_out_degree: average number of outgoing citations
    - max_out_degree: max outgoing citations (with paper ID)
    - avg_in_degree: average incoming citations
    - max_in_degree: max incoming citations (with paper ID)
    - isolated_papers: papers with no edges at all
    """
    all_papers = set(citations.keys())
    total_edges = sum(len(v) for v in citations.values())

    # Out-degree (citing)
    out_degrees = {p: len(c) for p, c in citations.items()}
    avg_out = total_edges / max(1, len(all_papers))
    max_out_paper = max(out_degrees, key=out_degrees.get, default="")
    max_out = out_degrees.get(max_out_paper, 0)

    # In-degree (cited by)
    in_degrees: dict[str, int] = {}
    for cited_list in citations.values():
        for cited_id in cited_list:
            in_degrees[cited_id] = in_degrees.get(cited_id, 0) + 1

    cited_papers = set(in_degrees.keys()) & all_papers
    avg_in = sum(in_degrees.get(p, 0) for p in all_papers) / max(1, len(all_papers))
    max_in_paper = max(
        all_papers,
        key=lambda p: in_degrees.get(p, 0),
        default="",
    )
    max_in = in_degrees.get(max_in_paper, 0)

    # Isolated (no in or out edges within corpus)
    papers_with_out = {p for p, c in citations.items() if c}
    papers_with_in = cited_papers
    isolated = all_papers - papers_with_out - papers_with_in

    return {
        "total_papers": len(all_papers),
        "total_edges": total_edges,
        "papers_with_refs": len(papers_with_out),
        "papers_cited": len(cited_papers),
        "avg_out_degree": round(avg_out, 2),
        "max_out_degree": (max_out_paper, max_out),
        "avg_in_degree": round(avg_in, 2),
        "max_in_degree": (max_in_paper, max_in),
        "isolated_papers": len(isolated),
        "isolated_ids": sorted(isolated),
    }
