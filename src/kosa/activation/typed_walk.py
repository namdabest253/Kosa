"""Typed random walk for activation wave computation.

Implements significance-weighted traversal through the knowledge graph:
- Edge-type-specific transition weights
- Cumulative confidence tracking along paths
- Minimum path significance threshold (stop below 0.05)
- Super-node handling: fan-out cap K=50, specificity bonus for low-degree nodes
- Full path traces for hypothesis generation

NOT vanilla PPR — designed for heterogeneous typed graphs where
different edge types have fundamentally different traversal semantics.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field

from kosa.graph.schema import EdgeType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Edge type transition weights
# ---------------------------------------------------------------------------

# Higher weight = more likely to traverse. Weights reflect how informative
# each edge type is for discovering novel connections.
EDGE_TYPE_WEIGHTS: dict[EdgeType, float] = {
    EdgeType.MITIGATES: 1.0,  # Most valuable: technique solves problem
    EdgeType.HAS_LIMITATION: 0.9,  # Problems are discovery seeds
    EdgeType.INTRODUCES: 0.8,  # Paper → technique link
    EdgeType.IMPROVES_OVER: 0.85,  # Technique evolution chain
    EdgeType.USES: 0.7,  # Compositional structure
    EdgeType.EVALUATES_ON: 0.5,  # Less informative for hypotheses
    EdgeType.CITES: 0.3,  # Ground truth but noisy (many citations)
    EdgeType.IS_INSTANCE_OF: 0.6,  # Type hierarchy
    EdgeType.CAUSED_BY: 0.8,  # Causal chains are valuable
    EdgeType.TEMPORALLY_FOLLOWS: 0.4,  # Temporal ordering
    EdgeType.SAME_AS: 0.1,  # Entity resolution links — traverse rarely
}

# Defaults
DEFAULT_MAX_DEPTH = 4
DEFAULT_MIN_PATH_SIGNIFICANCE = 0.05
DEFAULT_FAN_OUT_CAP = 50
DEFAULT_NUM_WALKS = 100
DEFAULT_SPECIFICITY_BONUS = 1.5  # Multiplier for low-degree nodes


@dataclass
class PathNode:
    """A node visited during a walk, with context."""

    node_id: str
    node_type: str
    node_name: str
    significance: float = 0.0
    degree: int = 0


@dataclass
class PathEdge:
    """An edge traversed during a walk."""

    edge_type: EdgeType
    confidence: float = 0.0
    venue_weight: float = 0.0


@dataclass
class WalkPath:
    """A single path through the graph with cumulative scores."""

    nodes: list[PathNode] = field(default_factory=list)
    edges: list[PathEdge] = field(default_factory=list)
    cumulative_confidence: float = 1.0
    cumulative_significance: float = 1.0

    @property
    def depth(self) -> int:
        return len(self.edges)

    @property
    def score(self) -> float:
        """Combined path score: confidence × significance."""
        return self.cumulative_confidence * self.cumulative_significance

    @property
    def start_node(self) -> PathNode | None:
        return self.nodes[0] if self.nodes else None

    @property
    def end_node(self) -> PathNode | None:
        return self.nodes[-1] if self.nodes else None

    def describe(self) -> str:
        """Human-readable path description."""
        parts = []
        for i, node in enumerate(self.nodes):
            parts.append(f"{node.node_name}")
            if i < len(self.edges):
                parts.append(f" -[{self.edges[i].edge_type.value}]-> ")
        return "".join(parts) + f" (score={self.score:.4f})"


@dataclass
class ActivatedNode:
    """A node activated by the walk, with aggregated score."""

    node_id: str
    node_type: str
    node_name: str
    activation_score: float = 0.0
    visit_count: int = 0
    best_path: WalkPath | None = None
    min_depth: int = 999


@dataclass
class ActivationResult:
    """Result of running the activation wave from a seed node."""

    seed_id: str
    seed_name: str
    activated_nodes: dict[str, ActivatedNode] = field(default_factory=dict)
    all_paths: list[WalkPath] = field(default_factory=list)
    config: dict[str, object] = field(default_factory=dict)

    def top_activated(self, n: int = 20) -> list[ActivatedNode]:
        """Return top-N activated nodes by score."""
        nodes = sorted(
            self.activated_nodes.values(),
            key=lambda x: x.activation_score,
            reverse=True,
        )
        return nodes[:n]

    def by_depth(self) -> dict[int, list[ActivatedNode]]:
        """Group activated nodes by minimum depth from seed."""
        groups: dict[int, list[ActivatedNode]] = {}
        for node in self.activated_nodes.values():
            groups.setdefault(node.min_depth, []).append(node)
        for depth in groups:
            groups[depth].sort(key=lambda x: x.activation_score, reverse=True)
        return groups

    def summary(self) -> str:
        total = len(self.activated_nodes)
        by_depth = self.by_depth()
        depth_str = ", ".join(f"d{d}={len(nodes)}" for d, nodes in sorted(by_depth.items()))
        return (
            f"Activation from '{self.seed_name}': "
            f"{total} nodes activated ({depth_str}), "
            f"{len(self.all_paths)} paths explored"
        )


# ---------------------------------------------------------------------------
# Neighbor abstraction (decouples from Neo4j for testability)
# ---------------------------------------------------------------------------


@dataclass
class Neighbor:
    """A neighboring node accessible via an edge."""

    node_id: str
    node_type: str
    node_name: str
    edge_type: EdgeType
    edge_confidence: float
    edge_venue_weight: float
    node_significance: float
    node_degree: int  # total degree of the neighbor


class GraphInterface:
    """Abstract interface for graph access. Subclass for Neo4j or in-memory."""

    def get_node(self, node_id: str) -> PathNode | None:
        raise NotImplementedError

    def get_neighbors(self, node_id: str) -> list[Neighbor]:
        raise NotImplementedError


class InMemoryGraph(GraphInterface):
    """In-memory graph for testing."""

    def __init__(self):
        self.nodes: dict[str, PathNode] = {}
        self.edges: dict[str, list[Neighbor]] = {}

    def add_node(self, node_id: str, node_type: str, name: str, significance: float = 0.5):
        self.nodes[node_id] = PathNode(
            node_id=node_id,
            node_type=node_type,
            node_name=name,
            significance=significance,
        )
        if node_id not in self.edges:
            self.edges[node_id] = []

    def add_edge(
        self,
        src_id: str,
        tgt_id: str,
        edge_type: EdgeType,
        confidence: float = 0.8,
        venue_weight: float = 1.0,
    ):
        tgt = self.nodes.get(tgt_id)
        if tgt is None:
            return
        # Count degree
        degree = len(self.edges.get(tgt_id, [])) + 1

        self.edges.setdefault(src_id, []).append(
            Neighbor(
                node_id=tgt_id,
                node_type=tgt.node_type,
                node_name=tgt.node_name,
                edge_type=edge_type,
                edge_confidence=confidence,
                edge_venue_weight=venue_weight,
                node_significance=tgt.significance,
                node_degree=degree,
            )
        )

    def get_node(self, node_id: str) -> PathNode | None:
        return self.nodes.get(node_id)

    def get_neighbors(self, node_id: str) -> list[Neighbor]:
        return self.edges.get(node_id, [])


# ---------------------------------------------------------------------------
# Typed random walk
# ---------------------------------------------------------------------------


def _compute_transition_weight(
    neighbor: Neighbor,
    specificity_bonus: float = DEFAULT_SPECIFICITY_BONUS,
) -> float:
    """Compute the transition weight for traversing to a neighbor.

    weight = edge_type_weight × confidence × venue_weight × significance × specificity
    """
    edge_weight = EDGE_TYPE_WEIGHTS.get(neighbor.edge_type, 0.5)
    specificity = specificity_bonus if neighbor.node_degree <= 5 else 1.0

    return (
        edge_weight
        * neighbor.edge_confidence
        * neighbor.edge_venue_weight
        * neighbor.node_significance
        * specificity
    )


def _sample_neighbors(
    neighbors: list[Neighbor],
    k: int = DEFAULT_FAN_OUT_CAP,
    specificity_bonus: float = DEFAULT_SPECIFICITY_BONUS,
) -> list[tuple[Neighbor, float]]:
    """Sample at most K neighbors, weighted by transition weight.

    Returns list of (neighbor, weight) tuples.
    """
    if not neighbors:
        return []

    weighted = [(n, _compute_transition_weight(n, specificity_bonus)) for n in neighbors]

    if len(weighted) <= k:
        return weighted

    # Weighted sampling without replacement
    weights = [w for _, w in weighted]
    total = sum(weights)
    if total == 0:
        return weighted[:k]

    # Normalize to probabilities
    probs = [w / total for w in weights]
    indices = list(range(len(weighted)))
    sampled_indices = set()

    for _ in range(k):
        # Pick based on probability (simple approach)
        r = random.random()
        cumulative = 0.0
        for idx in indices:
            if idx in sampled_indices:
                continue
            cumulative += probs[idx]
            if r <= cumulative:
                sampled_indices.add(idx)
                break
        else:
            # Fallback: pick first unsampled
            for idx in indices:
                if idx not in sampled_indices:
                    sampled_indices.add(idx)
                    break

    return [weighted[i] for i in sampled_indices]


def typed_random_walk(
    graph: GraphInterface,
    seed_id: str,
    max_depth: int = DEFAULT_MAX_DEPTH,
    num_walks: int = DEFAULT_NUM_WALKS,
    min_path_significance: float = DEFAULT_MIN_PATH_SIGNIFICANCE,
    fan_out_cap: int = DEFAULT_FAN_OUT_CAP,
    specificity_bonus: float = DEFAULT_SPECIFICITY_BONUS,
) -> ActivationResult:
    """Run typed random walks from a seed node.

    Each walk randomly traverses the graph, weighted by edge type,
    confidence, venue weight, and node significance. Walks terminate
    when max_depth is reached or cumulative score drops below threshold.

    Args:
        graph: Graph interface for node/neighbor access.
        seed_id: Starting node ID.
        max_depth: Maximum walk depth.
        num_walks: Number of random walks to perform.
        min_path_significance: Stop exploring below this cumulative score.
        fan_out_cap: Maximum neighbors to consider per node (K).
        specificity_bonus: Weight multiplier for low-degree nodes.

    Returns:
        ActivationResult with all activated nodes and paths.
    """
    seed_node = graph.get_node(seed_id)
    if seed_node is None:
        logger.warning(f"Seed node not found: {seed_id}")
        return ActivationResult(seed_id=seed_id, seed_name="unknown")

    result = ActivationResult(
        seed_id=seed_id,
        seed_name=seed_node.node_name,
        config={
            "max_depth": max_depth,
            "num_walks": num_walks,
            "min_path_significance": min_path_significance,
            "fan_out_cap": fan_out_cap,
        },
    )

    for _ in range(num_walks):
        path = WalkPath(nodes=[seed_node])
        current_id = seed_id

        for _depth in range(max_depth):
            neighbors = graph.get_neighbors(current_id)
            if not neighbors:
                break

            # Sample neighbors (fan-out cap)
            sampled = _sample_neighbors(
                neighbors, k=fan_out_cap, specificity_bonus=specificity_bonus
            )
            if not sampled:
                break

            # Weighted random selection
            total_weight = sum(w for _, w in sampled)
            if total_weight == 0:
                break

            r = random.random() * total_weight
            cumulative = 0.0
            chosen_neighbor = sampled[0][0]
            for neighbor, weight in sampled:
                cumulative += weight
                if r <= cumulative:
                    chosen_neighbor = neighbor
                    break

            # Update path
            edge = PathEdge(
                edge_type=chosen_neighbor.edge_type,
                confidence=chosen_neighbor.edge_confidence,
                venue_weight=chosen_neighbor.edge_venue_weight,
            )
            node = PathNode(
                node_id=chosen_neighbor.node_id,
                node_type=chosen_neighbor.node_type,
                node_name=chosen_neighbor.node_name,
                significance=chosen_neighbor.node_significance,
                degree=chosen_neighbor.node_degree,
            )

            path.edges.append(edge)
            path.nodes.append(node)
            path.cumulative_confidence *= chosen_neighbor.edge_confidence
            path.cumulative_significance *= chosen_neighbor.node_significance

            # Check minimum threshold
            if path.score < min_path_significance:
                break

            current_id = chosen_neighbor.node_id

        # Record path
        if path.depth > 0:
            result.all_paths.append(path)

            # Update activated nodes
            for i, node in enumerate(path.nodes):
                if node.node_id == seed_id:
                    continue

                if node.node_id not in result.activated_nodes:
                    result.activated_nodes[node.node_id] = ActivatedNode(
                        node_id=node.node_id,
                        node_type=node.node_type,
                        node_name=node.node_name,
                    )

                activated = result.activated_nodes[node.node_id]
                activated.visit_count += 1
                activated.min_depth = min(activated.min_depth, i)

                # Accumulate activation score (normalized by walk count later)
                # Score = path score at this point in the walk
                partial_conf = 1.0
                partial_sig = 1.0
                for j in range(i):
                    if j < len(path.edges):
                        partial_conf *= path.edges[j].confidence
                    if j + 1 < len(path.nodes):
                        partial_sig *= path.nodes[j + 1].significance

                score = partial_conf * partial_sig
                if score > activated.activation_score:
                    activated.activation_score = score
                    activated.best_path = path

    logger.info(result.summary())
    return result


# ---------------------------------------------------------------------------
# Baselines for ablation study
# ---------------------------------------------------------------------------


def ppr_baseline(
    graph: GraphInterface,
    seed_id: str,
    alpha: float = 0.15,
    max_iterations: int = 50,
    top_k: int = 50,
) -> dict[str, float]:
    """Personalized PageRank baseline (untyped, no edge weights).

    Standard PPR: at each step, with probability alpha return to seed,
    otherwise follow a random outgoing edge (uniform weights).

    Returns dict of node_id → PPR score.
    """
    scores: dict[str, float] = {seed_id: 1.0}

    for _ in range(max_iterations):
        new_scores: dict[str, float] = {}
        for node_id, score in scores.items():
            neighbors = graph.get_neighbors(node_id)
            if not neighbors:
                # Dead end: return to seed
                new_scores[seed_id] = new_scores.get(seed_id, 0) + score
                continue

            # Teleport back to seed
            new_scores[seed_id] = new_scores.get(seed_id, 0) + alpha * score

            # Distribute remaining score uniformly
            spread = (1 - alpha) * score / len(neighbors)
            for n in neighbors:
                new_scores[n.node_id] = new_scores.get(n.node_id, 0) + spread

        scores = new_scores

    # Sort and return top-K
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return dict(sorted_scores[:top_k])


def embedding_similarity_baseline(
    similarities: dict[str, float],
    top_k: int = 50,
) -> dict[str, float]:
    """Embedding-only similarity baseline.

    Takes pre-computed cosine similarities from seed to all nodes.
    Returns top-K by similarity. No graph structure used.
    """
    sorted_sims = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    return dict(sorted_sims[:top_k])
