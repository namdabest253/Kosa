"""Pydantic response models for the API.

Returns graphology-native format for graph data so the frontend
needs zero transformation.
"""

from __future__ import annotations

from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Graph data (graphology-native format)
# ---------------------------------------------------------------------------


class NodeAttributes(BaseModel):
    label: str
    node_type: str  # Paper, Technique, Problem, Dataset
    x: float = 0.0
    y: float = 0.0
    size: float = 1.0
    color: str = "#999999"
    # Optional properties surfaced for display
    properties: dict = {}


class GraphNode(BaseModel):
    key: str
    attributes: NodeAttributes


class EdgeAttributes(BaseModel):
    edge_type: str
    confidence: float = 1.0
    color: str = "#cccccc"
    size: float = 1.0
    properties: dict = {}


class GraphEdge(BaseModel):
    key: str
    source: str
    target: str
    attributes: EdgeAttributes


class GraphData(BaseModel):
    """Graphology-compatible graph serialization."""

    nodes: list[GraphNode]
    edges: list[GraphEdge]


# ---------------------------------------------------------------------------
# Neighborhood response
# ---------------------------------------------------------------------------


class NeighborhoodResponse(BaseModel):
    graph: GraphData
    center_id: str
    depth: int
    truncated: bool = False
    total_neighbor_count: int = 0


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


class SearchResult(BaseModel):
    id: str
    label: str
    node_type: str
    score: float
    snippet: str = ""


class SearchResponse(BaseModel):
    results: list[SearchResult]
    query: str
    total: int


# ---------------------------------------------------------------------------
# Node detail
# ---------------------------------------------------------------------------


class NodeDetail(BaseModel):
    id: str
    node_type: str
    properties: dict
    neighbor_count: int = 0


# ---------------------------------------------------------------------------
# Graph stats
# ---------------------------------------------------------------------------


class TypeCount(BaseModel):
    type: str
    count: int


class GraphStats(BaseModel):
    total_nodes: int
    total_edges: int
    node_counts: list[TypeCount]
    edge_counts: list[TypeCount]


# ---------------------------------------------------------------------------
# Hypotheses
# ---------------------------------------------------------------------------


class Hypothesis(BaseModel):
    id: str
    title: str
    description: str
    elo_score: float = 1000.0
    reasoning_chain: list[str] = []
    feedback_up: int = 0
    feedback_down: int = 0
    created_at: str = ""


class HypothesisListResponse(BaseModel):
    hypotheses: list[Hypothesis]
    total: int


class FeedbackRequest(BaseModel):
    vote: str  # "up" or "down"


class FeedbackResponse(BaseModel):
    hypothesis_id: str
    feedback_up: int
    feedback_down: int


# ---------------------------------------------------------------------------
# Overview (cluster-level coarsened view)
# ---------------------------------------------------------------------------


class ClusterNode(BaseModel):
    cluster_id: int
    label: str
    node_count: int
    dominant_type: str
    x: float = 0.0
    y: float = 0.0
    size: float = 1.0


class ClusterEdge(BaseModel):
    source_cluster: int
    target_cluster: int
    weight: int  # number of cross-cluster edges


class OverviewResponse(BaseModel):
    clusters: list[ClusterNode]
    edges: list[ClusterEdge]
