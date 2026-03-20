"""Hypothesis listing, detail, reasoning chain, and feedback endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Query

from kosa.api.deps import Neo4jSession
from kosa.api.models import (
    EdgeAttributes,
    FeedbackRequest,
    FeedbackResponse,
    GraphData,
    GraphEdge,
    GraphNode,
    Hypothesis,
    HypothesisListResponse,
    NodeAttributes,
)

router = APIRouter()


@router.get("", response_model=HypothesisListResponse)
async def list_hypotheses(
    session: Neo4jSession,
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    sort: str = Query("elo", description="Sort by: elo, recent, feedback"),
):
    """List hypotheses ranked by Elo score."""
    order_clause = {
        "elo": "h.elo_score DESC",
        "recent": "h.created_at DESC",
        "feedback": "(h.feedback_up - h.feedback_down) DESC",
    }.get(sort, "h.elo_score DESC")

    count_result = await session.run("MATCH (h:Hypothesis) RETURN count(h) AS total")
    count_rec = await count_result.single()
    total = count_rec["total"] if count_rec else 0

    result = await session.run(
        f"MATCH (h:Hypothesis) "
        f"RETURN elementId(h) AS id, h "
        f"ORDER BY {order_clause} "
        f"SKIP $skip LIMIT $limit",
        {"skip": skip, "limit": limit},
    )

    hypotheses = []
    async for rec in result:
        props = dict(rec["h"])
        chain = props.get("reasoning_chain", [])
        if isinstance(chain, str):
            chain = [chain]
        hypotheses.append(
            Hypothesis(
                id=rec["id"],
                title=props.get("title", ""),
                description=props.get("description", ""),
                elo_score=props.get("elo_score", 1000.0),
                reasoning_chain=chain,
                feedback_up=props.get("feedback_up", 0),
                feedback_down=props.get("feedback_down", 0),
                created_at=props.get("created_at", ""),
            )
        )

    return HypothesisListResponse(hypotheses=hypotheses, total=total)


@router.get("/{hypothesis_id}")
async def get_hypothesis(session: Neo4jSession, hypothesis_id: str):
    """Get full hypothesis detail."""
    result = await session.run(
        "MATCH (h:Hypothesis) WHERE elementId(h) = $id RETURN h",
        {"id": hypothesis_id},
    )
    rec = await result.single()
    if not rec:
        return {"error": "not found"}

    props = dict(rec["h"])
    chain = props.get("reasoning_chain", [])
    if isinstance(chain, str):
        chain = [chain]

    return Hypothesis(
        id=hypothesis_id,
        title=props.get("title", ""),
        description=props.get("description", ""),
        elo_score=props.get("elo_score", 1000.0),
        reasoning_chain=chain,
        feedback_up=props.get("feedback_up", 0),
        feedback_down=props.get("feedback_down", 0),
        created_at=props.get("created_at", ""),
    )


@router.post("/{hypothesis_id}/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    session: Neo4jSession,
    hypothesis_id: str,
    feedback: FeedbackRequest,
):
    """Submit thumbs up/down feedback on a hypothesis."""
    if feedback.vote == "up":
        field = "feedback_up"
    elif feedback.vote == "down":
        field = "feedback_down"
    else:
        return FeedbackResponse(hypothesis_id=hypothesis_id, feedback_up=0, feedback_down=0)

    result = await session.run(
        f"MATCH (h:Hypothesis) WHERE elementId(h) = $id "
        f"SET h.{field} = coalesce(h.{field}, 0) + 1 "
        f"RETURN h.feedback_up AS up, h.feedback_down AS down",
        {"id": hypothesis_id},
    )
    rec = await result.single()

    return FeedbackResponse(
        hypothesis_id=hypothesis_id,
        feedback_up=rec["up"] or 0 if rec else 0,
        feedback_down=rec["down"] or 0 if rec else 0,
    )


@router.get("/{hypothesis_id}/path")
async def get_reasoning_path(session: Neo4jSession, hypothesis_id: str):
    """Return the reasoning chain as a subgraph for visualization."""
    # Get hypothesis and its linked path nodes
    result = await session.run(
        "MATCH (h:Hypothesis) WHERE elementId(h) = $id "
        "OPTIONAL MATCH (h)-[:DERIVED_FROM]->(n) "
        "OPTIONAL MATCH (n)-[r]-(m) "
        "WHERE elementId(m) IN ["
        "  x IN [(h)-[:DERIVED_FROM]->(p) | elementId(p)]"
        "] "
        "RETURN collect(DISTINCT {id: elementId(n), node: n, labels: labels(n)}) AS nodes, "
        "collect(DISTINCT {id: elementId(r), rel: r, type: type(r), "
        "src: elementId(startNode(r)), tgt: elementId(endNode(r))}) AS edges",
        {"id": hypothesis_id},
    )
    rec = await result.single()
    if not rec:
        return GraphData(nodes=[], edges=[])

    nodes = []
    for item in rec["nodes"]:
        if item["node"] is None:
            continue
        props = dict(item["node"])
        n_type = item["labels"][0] if item["labels"] else "Unknown"
        from kosa.api.routes.graph import NODE_COLORS, _safe_props

        nodes.append(
            GraphNode(
                key=item["id"],
                attributes=NodeAttributes(
                    label=props.get("name") or props.get("title") or "?",
                    node_type=n_type,
                    x=props.get("x", 0.0),
                    y=props.get("y", 0.0),
                    color=NODE_COLORS.get(n_type, "#999999"),
                    properties=_safe_props(props),
                ),
            )
        )

    edges = []
    for item in rec["edges"]:
        if item["rel"] is None:
            continue
        rel_props = dict(item["rel"])
        confidence = rel_props.get("confidence", 1.0)
        edges.append(
            GraphEdge(
                key=item["id"],
                source=item["src"],
                target=item["tgt"],
                attributes=EdgeAttributes(
                    edge_type=item["type"],
                    confidence=confidence,
                    properties={
                        k: v
                        for k, v in rel_props.items()
                        if isinstance(v, str | int | float | bool)
                    },
                ),
            )
        )

    return GraphData(nodes=nodes, edges=edges)
