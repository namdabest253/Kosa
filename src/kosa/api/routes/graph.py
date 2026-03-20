"""Graph exploration endpoints: search, neighborhood, node detail, overview, stats."""

from __future__ import annotations

from fastapi import APIRouter, Query

from kosa.api.deps import Neo4jSession
from kosa.api.models import (
    EdgeAttributes,
    GraphData,
    GraphEdge,
    GraphNode,
    GraphStats,
    NeighborhoodResponse,
    NodeAttributes,
    NodeDetail,
    SearchResponse,
    SearchResult,
    TypeCount,
)

router = APIRouter()

# Node type → color mapping (matches frontend)
NODE_COLORS = {
    "Paper": "#2196F3",
    "Technique": "#4CAF50",
    "Problem": "#F44336",
    "Dataset": "#FF9800",
}


def _node_color(node_type: str) -> str:
    return NODE_COLORS.get(node_type, "#999999")


def _edge_color(confidence: float) -> str:
    if confidence > 0.8:
        return "#555555"
    if confidence > 0.5:
        return "#999999"
    return "#cccccc"


def _node_size(props: dict) -> float:
    """Size node by significance (papers) or default."""
    sig = props.get("significance")
    if sig is not None:
        return max(1.0, float(sig) * 5)
    return 2.0


@router.get("/stats", response_model=GraphStats)
async def graph_stats(session: Neo4jSession):
    """Node/edge counts by type."""
    node_counts = []
    for label in ("Paper", "Technique", "Problem", "Dataset"):
        result = await session.run(f"MATCH (n:{label}) RETURN count(n) AS cnt")
        record = await result.single()
        node_counts.append(TypeCount(type=label, count=record["cnt"] if record else 0))

    edge_result = await session.run(
        "MATCH ()-[r]->() RETURN type(r) AS t, count(r) AS cnt ORDER BY cnt DESC"
    )
    edge_counts = [TypeCount(type=rec["t"], count=rec["cnt"]) async for rec in edge_result]

    total_nodes = sum(tc.count for tc in node_counts)
    total_edges = sum(tc.count for tc in edge_counts)

    return GraphStats(
        total_nodes=total_nodes,
        total_edges=total_edges,
        node_counts=node_counts,
        edge_counts=edge_counts,
    )


@router.get("/search", response_model=SearchResponse)
async def search_nodes(
    session: Neo4jSession,
    q: str = Query(..., min_length=1, description="Search query"),
    type: str | None = Query(None, description="Filter by node type"),
    limit: int = Query(20, ge=1, le=100),
):
    """Full-text search across node types using Neo4j fulltext indexes."""
    # Build UNION of fulltext queries per type
    index_map = {
        "Paper": "paper_search",
        "Technique": "technique_search",
        "Problem": "problem_search",
        "Dataset": "dataset_search",
    }

    if type and type in index_map:
        indexes = {type: index_map[type]}
    else:
        indexes = index_map

    results = []
    escaped_q = q.replace("\\", "\\\\").replace('"', '\\"')

    for node_type, index_name in indexes.items():
        try:
            query = (
                f'CALL db.index.fulltext.queryNodes("{index_name}", $query) '
                f"YIELD node, score "
                f"RETURN elementId(node) AS id, node.name AS name, "
                f"node.title AS title, node.description AS desc, "
                f"score, '{node_type}' AS node_type "
                f"ORDER BY score DESC LIMIT $limit"
            )
            result = await session.run(query, {"query": escaped_q, "limit": limit})
            async for rec in result:
                label = rec["name"] or rec["title"] or "?"
                snippet = (rec["desc"] or "")[:200]
                results.append(
                    SearchResult(
                        id=rec["id"],
                        label=label,
                        node_type=rec["node_type"],
                        score=rec["score"],
                        snippet=snippet,
                    )
                )
        except Exception:
            # Index may not exist yet — skip silently
            continue

    results.sort(key=lambda r: r.score, reverse=True)
    results = results[:limit]

    return SearchResponse(results=results, query=q, total=len(results))


@router.get("/neighborhood/{node_id}", response_model=NeighborhoodResponse)
async def get_neighborhood(
    session: Neo4jSession,
    node_id: str,
    depth: int = Query(1, ge=1, le=3),
    limit: int = Query(50, ge=1, le=200),
):
    """Expand neighborhood from a node. Fan-out capped per the K=50 design constraint."""
    # Get center node
    center_result = await session.run(
        "MATCH (n) WHERE elementId(n) = $id RETURN n, labels(n) AS labels",
        {"id": node_id},
    )
    center_rec = await center_result.single()
    if not center_rec:
        return NeighborhoodResponse(
            graph=GraphData(nodes=[], edges=[]),
            center_id=node_id,
            depth=depth,
        )

    center_node = center_rec["n"]
    center_type = center_rec["labels"][0] if center_rec["labels"] else "Unknown"

    nodes_map: dict[str, GraphNode] = {}
    edges_list: list[GraphEdge] = []

    # Add center node
    center_props = dict(center_node)
    nodes_map[node_id] = GraphNode(
        key=node_id,
        attributes=NodeAttributes(
            label=center_props.get("name") or center_props.get("title") or "?",
            node_type=center_type,
            x=center_props.get("x", 0.0),
            y=center_props.get("y", 0.0),
            size=_node_size(center_props),
            color=_node_color(center_type),
            properties=_safe_props(center_props),
        ),
    )

    # Expand neighbors up to depth, fan-out capped
    query = (
        "MATCH (center)-[r]-(neighbor) "
        "WHERE elementId(center) = $id "
        "RETURN elementId(neighbor) AS nid, neighbor, labels(neighbor) AS labels, "
        "r, type(r) AS rtype, elementId(r) AS rid, "
        "elementId(startNode(r)) AS src_id, elementId(endNode(r)) AS tgt_id "
        "ORDER BY r.confidence DESC "
        "LIMIT $limit"
    )

    # Count total neighbors for truncation info
    count_result = await session.run(
        "MATCH (center)-[r]-(neighbor) WHERE elementId(center) = $id "
        "RETURN count(neighbor) AS total",
        {"id": node_id},
    )
    count_rec = await count_result.single()
    total_count = count_rec["total"] if count_rec else 0

    result = await session.run(query, {"id": node_id, "limit": limit})
    async for rec in result:
        nid = rec["nid"]
        neighbor = rec["neighbor"]
        n_type = rec["labels"][0] if rec["labels"] else "Unknown"
        n_props = dict(neighbor)

        if nid not in nodes_map:
            nodes_map[nid] = GraphNode(
                key=nid,
                attributes=NodeAttributes(
                    label=n_props.get("name") or n_props.get("title") or "?",
                    node_type=n_type,
                    x=n_props.get("x", 0.0),
                    y=n_props.get("y", 0.0),
                    size=_node_size(n_props),
                    color=_node_color(n_type),
                    properties=_safe_props(n_props),
                ),
            )

        rel = rec["r"]
        confidence = dict(rel).get("confidence", 1.0)
        edges_list.append(
            GraphEdge(
                key=rec["rid"],
                source=rec["src_id"],
                target=rec["tgt_id"],
                attributes=EdgeAttributes(
                    edge_type=rec["rtype"],
                    confidence=confidence,
                    color=_edge_color(confidence),
                    size=max(0.5, confidence * 2),
                    properties=_safe_props(dict(rel)),
                ),
            )
        )

    # For depth > 1, expand from each neighbor (simplified: just one more hop)
    if depth >= 2:
        neighbor_ids = [nid for nid in nodes_map if nid != node_id]
        for nid in neighbor_ids[:10]:  # Cap second-hop expansion
            result2 = await session.run(query, {"id": nid, "limit": 10})
            async for rec in result2:
                nid2 = rec["nid"]
                neighbor2 = rec["neighbor"]
                n_type2 = rec["labels"][0] if rec["labels"] else "Unknown"
                n_props2 = dict(neighbor2)

                if nid2 not in nodes_map:
                    nodes_map[nid2] = GraphNode(
                        key=nid2,
                        attributes=NodeAttributes(
                            label=n_props2.get("name") or n_props2.get("title") or "?",
                            node_type=n_type2,
                            x=n_props2.get("x", 0.0),
                            y=n_props2.get("y", 0.0),
                            size=_node_size(n_props2),
                            color=_node_color(n_type2),
                            properties=_safe_props(n_props2),
                        ),
                    )

                rel2 = rec["r"]
                confidence2 = dict(rel2).get("confidence", 1.0)
                edges_list.append(
                    GraphEdge(
                        key=rec["rid"],
                        source=rec["src_id"],
                        target=rec["tgt_id"],
                        attributes=EdgeAttributes(
                            edge_type=rec["rtype"],
                            confidence=confidence2,
                            color=_edge_color(confidence2),
                            size=max(0.5, confidence2 * 2),
                            properties=_safe_props(dict(rel2)),
                        ),
                    )
                )

    return NeighborhoodResponse(
        graph=GraphData(nodes=list(nodes_map.values()), edges=edges_list),
        center_id=node_id,
        depth=depth,
        truncated=total_count > limit,
        total_neighbor_count=total_count,
    )


@router.get("/node/{node_id}", response_model=NodeDetail)
async def get_node(session: Neo4jSession, node_id: str):
    """Full detail for a single node."""
    result = await session.run(
        "MATCH (n) WHERE elementId(n) = $id "
        "OPTIONAL MATCH (n)-[r]-() "
        "RETURN n, labels(n) AS labels, count(r) AS neighbor_count",
        {"id": node_id},
    )
    rec = await result.single()
    if not rec:
        return NodeDetail(id=node_id, node_type="Unknown", properties={})

    props = dict(rec["n"])
    n_type = rec["labels"][0] if rec["labels"] else "Unknown"

    return NodeDetail(
        id=node_id,
        node_type=n_type,
        properties=_safe_props(props),
        neighbor_count=rec["neighbor_count"],
    )


@router.get("/overview")
async def graph_overview(session: Neo4jSession):
    """Cluster-level coarsened view using Leiden communities (if computed).

    Falls back to node-type grouping if no community labels exist.
    """
    # Check if community property exists
    check = await session.run(
        "MATCH (n) WHERE n.community IS NOT NULL RETURN count(n) AS cnt LIMIT 1"
    )
    check_rec = await check.single()
    has_communities = check_rec and check_rec["cnt"] > 0

    if has_communities:
        result = await session.run(
            "MATCH (n) WHERE n.community IS NOT NULL "
            "WITH n.community AS cid, labels(n)[0] AS ntype, count(*) AS cnt "
            "RETURN cid, collect({type: ntype, count: cnt}) AS types, sum(cnt) AS total "
            "ORDER BY total DESC LIMIT 50"
        )
        clusters = []
        async for rec in result:
            types = rec["types"]
            dominant = max(types, key=lambda t: t["count"])
            clusters.append(
                {
                    "cluster_id": rec["cid"],
                    "label": f"Cluster {rec['cid']}",
                    "node_count": rec["total"],
                    "dominant_type": dominant["type"],
                }
            )
        return {"clusters": clusters, "edges": []}

    # Fallback: group by type
    result = await session.run(
        "MATCH (n) WHERE n:Paper OR n:Technique OR n:Problem OR n:Dataset "
        "WITH labels(n)[0] AS ntype, count(*) AS cnt "
        "RETURN ntype, cnt ORDER BY cnt DESC"
    )
    clusters = []
    idx = 0
    async for rec in result:
        clusters.append(
            {
                "cluster_id": idx,
                "label": rec["ntype"],
                "node_count": rec["cnt"],
                "dominant_type": rec["ntype"],
            }
        )
        idx += 1

    return {"clusters": clusters, "edges": []}


def _safe_props(props: dict) -> dict:
    """Filter node/edge properties for JSON serialization (drop embeddings, etc.)."""
    skip = {"embedding", "x", "y"}
    out = {}
    for k, v in props.items():
        if k in skip:
            continue
        if isinstance(v, str | int | float | bool):
            out[k] = v
        elif isinstance(v, list) and all(isinstance(i, str | int | float) for i in v):
            out[k] = v
    return out
