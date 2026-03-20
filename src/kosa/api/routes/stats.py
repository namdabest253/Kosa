"""Dashboard statistics endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from kosa.api.deps import Neo4jSession

router = APIRouter()


@router.get("/dashboard")
async def dashboard(session: Neo4jSession):
    """Aggregate dashboard data: graph size, quality metrics, feedback summary."""
    # Node counts
    node_result = await session.run(
        "MATCH (n) WHERE n:Paper OR n:Technique OR n:Problem OR n:Dataset "
        "WITH labels(n)[0] AS label, count(*) AS cnt "
        "RETURN label, cnt ORDER BY cnt DESC"
    )
    node_counts = {rec["label"]: rec["cnt"] async for rec in node_result}

    # Edge counts
    edge_result = await session.run(
        "MATCH ()-[r]->() RETURN type(r) AS t, count(r) AS cnt ORDER BY cnt DESC"
    )
    edge_counts = {rec["t"]: rec["cnt"] async for rec in edge_result}

    # Confidence distribution
    conf_result = await session.run(
        "MATCH ()-[r]->() WHERE r.confidence IS NOT NULL "
        "WITH CASE "
        "  WHEN r.confidence > 0.8 THEN 'high' "
        "  WHEN r.confidence > 0.5 THEN 'medium' "
        "  ELSE 'low' END AS band, count(*) AS cnt "
        "RETURN band, cnt"
    )
    confidence_dist = {rec["band"]: rec["cnt"] async for rec in conf_result}

    # Year distribution (papers)
    year_result = await session.run(
        "MATCH (p:Paper) WHERE p.year IS NOT NULL "
        "RETURN p.year AS year, count(*) AS cnt ORDER BY year"
    )
    year_dist = {rec["year"]: rec["cnt"] async for rec in year_result}

    # Venue tier distribution
    tier_result = await session.run(
        "MATCH (p:Paper) WHERE p.venue_tier IS NOT NULL "
        "RETURN p.venue_tier AS tier, count(*) AS cnt ORDER BY tier"
    )
    tier_dist = {rec["tier"]: rec["cnt"] async for rec in tier_result}

    # Hypothesis feedback summary
    feedback_result = await session.run(
        "MATCH (h:Hypothesis) "
        "RETURN count(h) AS total, "
        "sum(coalesce(h.feedback_up, 0)) AS total_up, "
        "sum(coalesce(h.feedback_down, 0)) AS total_down, "
        "avg(h.elo_score) AS avg_elo"
    )
    fb_rec = await feedback_result.single()

    return {
        "graph": {
            "node_counts": node_counts,
            "edge_counts": edge_counts,
            "total_nodes": sum(node_counts.values()),
            "total_edges": sum(edge_counts.values()),
        },
        "quality": {
            "confidence_distribution": confidence_dist,
            "year_distribution": year_dist,
            "venue_tier_distribution": tier_dist,
        },
        "hypotheses": {
            "total": fb_rec["total"] if fb_rec else 0,
            "total_feedback_up": fb_rec["total_up"] if fb_rec else 0,
            "total_feedback_down": fb_rec["total_down"] if fb_rec else 0,
            "avg_elo": round(fb_rec["avg_elo"], 1) if fb_rec and fb_rec["avg_elo"] else None,
        },
    }
