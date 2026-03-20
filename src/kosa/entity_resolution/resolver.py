"""Entity resolution: link-first-merge-cautiously deduplication.

Design (from DESIGN.md):
1. Blocking: group candidates by embedding similarity (cosine > 0.85) to avoid O(n²)
2. Candidate linking: connect duplicates with SAME_AS edge + confidence score
3. Merge threshold: only merge at confidence > 0.95
4. Below threshold: keep separate, linked by SAME_AS
5. Runs on every ingestion cycle, not just once

Phase 1: uses string-based similarity + LLM alignment (no embeddings yet).
Embeddings are added when vector indexes are populated.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from kosa.graph.schema import NodeLabel

logger = logging.getLogger(__name__)

# Thresholds
SAME_AS_THRESHOLD = 0.85  # Link with SAME_AS edge
MERGE_THRESHOLD = 0.95  # Merge into single node
MIN_BLOCK_SIMILARITY = 0.6  # Minimum for blocking candidates


@dataclass
class EntityPair:
    """A candidate pair for entity resolution."""

    name_a: str
    name_b: str
    node_type: NodeLabel
    confidence: float
    signals: dict[str, float] = field(default_factory=dict)
    should_merge: bool = False
    should_link: bool = False


@dataclass
class ResolutionResult:
    """Results of running entity resolution on a set of entities."""

    pairs_evaluated: int = 0
    same_as_links: list[EntityPair] = field(default_factory=list)
    merges: list[EntityPair] = field(default_factory=list)
    duplicate_rate: float = 0.0

    def summary(self) -> str:
        return (
            f"Entity resolution: {self.pairs_evaluated} pairs evaluated, "
            f"{len(self.same_as_links)} SAME_AS links, "
            f"{len(self.merges)} merges, "
            f"duplicate rate: {self.duplicate_rate:.1%}"
        )


# ---------------------------------------------------------------------------
# String similarity signals
# ---------------------------------------------------------------------------

# Common aliases in ML/AI terminology
ALIAS_TABLE: dict[str, set[str]] = {
    "transformer": {"transformer architecture", "transformer model"},
    "bert": {"bidirectional encoder representations from transformers"},
    "gpt": {"generative pre-trained transformer"},
    "vae": {"variational autoencoder", "variational auto-encoder"},
    "gan": {"generative adversarial network", "generative adversarial net"},
    "resnet": {"residual network", "deep residual network", "residual net"},
    "lstm": {"long short-term memory"},
    "gru": {"gated recurrent unit"},
    "cnn": {"convolutional neural network"},
    "rnn": {"recurrent neural network"},
    "mlp": {"multi-layer perceptron", "multilayer perceptron"},
    "relu": {"rectified linear unit"},
    "adam": {"adam optimizer"},
    "sgd": {"stochastic gradient descent"},
    "dropout": {"dropout regularization"},
    "batchnorm": {"batch normalization", "batch norm"},
    "layernorm": {"layer normalization", "layer norm"},
    "attention": {"attention mechanism"},
    "self-attention": {"self attention", "scaled dot-product attention"},
    "clip": {"contrastive language-image pre-training"},
    "vit": {"vision transformer"},
    "lora": {"low-rank adaptation"},
    "rlhf": {"reinforcement learning from human feedback"},
    "dpo": {"direct preference optimization"},
    "ppo": {"proximal policy optimization"},
    "moe": {"mixture of experts"},
    "rag": {"retrieval augmented generation", "retrieval-augmented generation"},
    "flash attention": {"flashattention", "flash-attention"},
    "diffusion model": {"diffusion models", "denoising diffusion"},
    "ddpm": {"denoising diffusion probabilistic model"},
    "knowledge distillation": {"model distillation", "distillation"},
    "fine-tuning": {"finetuning", "fine tuning"},
    "quantization": {"model quantization", "weight quantization"},
}

# Build reverse lookup: alias → canonical name
_REVERSE_ALIASES: dict[str, str] = {}
for canonical, aliases in ALIAS_TABLE.items():
    for alias in aliases:
        _REVERSE_ALIASES[alias.lower()] = canonical.lower()
    _REVERSE_ALIASES[canonical.lower()] = canonical.lower()


def normalize_name(name: str) -> str:
    """Normalize an entity name to canonical form."""
    lower = name.lower().strip()
    # Check alias table
    if lower in _REVERSE_ALIASES:
        return _REVERSE_ALIASES[lower]
    return lower


def string_similarity(a: str, b: str) -> float:
    """Compute string similarity between two entity names.

    Uses a combination of:
    - Exact match after normalization
    - Alias table match
    - Token overlap (Jaccard)
    - Substring containment
    """
    norm_a = normalize_name(a)
    norm_b = normalize_name(b)

    # Exact match after normalization
    if norm_a == norm_b:
        return 1.0

    # Alias match (both map to same canonical)
    canon_a = _REVERSE_ALIASES.get(norm_a)
    canon_b = _REVERSE_ALIASES.get(norm_b)
    if canon_a and canon_b and canon_a == canon_b:
        return 0.98

    # Token overlap (Jaccard similarity)
    tokens_a = set(re.split(r"[\s\-_/]+", norm_a))
    tokens_b = set(re.split(r"[\s\-_/]+", norm_b))
    if tokens_a and tokens_b:
        jaccard = len(tokens_a & tokens_b) / len(tokens_a | tokens_b)
    else:
        jaccard = 0.0

    # Substring containment (one is prefix/suffix of other)
    containment = 0.0
    if norm_a in norm_b or norm_b in norm_a:
        shorter = min(len(norm_a), len(norm_b))
        longer = max(len(norm_a), len(norm_b))
        containment = shorter / longer if longer > 0 else 0.0

    return max(jaccard, containment)


# ---------------------------------------------------------------------------
# Blocking
# ---------------------------------------------------------------------------


def block_candidates(
    entities: list[tuple[str, NodeLabel]],
    min_similarity: float = MIN_BLOCK_SIMILARITY,
) -> list[tuple[str, str, NodeLabel, float]]:
    """Group candidate duplicate pairs using string similarity blocking.

    Only compares entities of the same type. Returns pairs above threshold.

    Args:
        entities: List of (name, node_type) tuples.
        min_similarity: Minimum similarity to be a candidate pair.

    Returns:
        List of (name_a, name_b, node_type, similarity) tuples.
    """
    # Group by type
    by_type: dict[NodeLabel, list[str]] = {}
    for name, ntype in entities:
        by_type.setdefault(ntype, []).append(name)

    candidates = []
    for ntype, names in by_type.items():
        # O(n²) within each type — but blocked by type first
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                sim = string_similarity(names[i], names[j])
                if sim >= min_similarity:
                    candidates.append((names[i], names[j], ntype, sim))

    candidates.sort(key=lambda x: x[3], reverse=True)
    return candidates


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def compute_pair_confidence(
    name_a: str,
    name_b: str,
    node_type: NodeLabel,
    string_sim: float,
    embedding_sim: float | None = None,
    llm_agrees: bool | None = None,
) -> tuple[float, dict[str, float]]:
    """Compute overall confidence that two entities are the same.

    Combines multiple signals with weights:
    - String similarity: 0.4
    - Alias table match: 0.2 (binary bonus)
    - Embedding similarity: 0.3 (when available)
    - LLM alignment: 0.1 (when available)

    Returns (confidence, signal_dict).
    """
    signals: dict[str, float] = {"string_similarity": string_sim}
    weights = {"string_similarity": 0.4}

    # Alias bonus
    canon_a = _REVERSE_ALIASES.get(normalize_name(name_a))
    canon_b = _REVERSE_ALIASES.get(normalize_name(name_b))
    alias_match = 1.0 if (canon_a and canon_b and canon_a == canon_b) else 0.0
    signals["alias_match"] = alias_match
    weights["alias_match"] = 0.2

    # Embedding similarity (when available)
    if embedding_sim is not None:
        signals["embedding_similarity"] = embedding_sim
        weights["embedding_similarity"] = 0.3
    else:
        # Redistribute weight to string similarity
        weights["string_similarity"] = 0.6

    # LLM alignment (when available)
    if llm_agrees is not None:
        signals["llm_alignment"] = 1.0 if llm_agrees else 0.0
        weights["llm_alignment"] = 0.1
    else:
        weights["string_similarity"] += 0.1

    # Normalize weights
    total_weight = sum(weights.values())
    confidence = sum(signals.get(k, 0.0) * (v / total_weight) for k, v in weights.items())

    return confidence, signals


def resolve_entities(
    entities: list[tuple[str, NodeLabel]],
    embedding_similarities: dict[tuple[str, str], float] | None = None,
) -> ResolutionResult:
    """Run entity resolution on a set of entities.

    Args:
        entities: List of (name, node_type) tuples.
        embedding_similarities: Optional pre-computed embedding similarities
            for entity pairs, keyed by (name_a, name_b) or (name_b, name_a).

    Returns:
        ResolutionResult with SAME_AS links and merge candidates.
    """
    embedding_similarities = embedding_similarities or {}
    result = ResolutionResult()

    # Step 1: Blocking
    candidates = block_candidates(entities)
    logger.info(f"Blocking found {len(candidates)} candidate pairs")

    # Step 2: Score each candidate pair
    for name_a, name_b, ntype, string_sim in candidates:
        result.pairs_evaluated += 1

        # Look up embedding similarity if available
        emb_sim = embedding_similarities.get(
            (name_a, name_b),
            embedding_similarities.get((name_b, name_a)),
        )

        confidence, signals = compute_pair_confidence(
            name_a, name_b, ntype, string_sim, embedding_sim=emb_sim
        )

        pair = EntityPair(
            name_a=name_a,
            name_b=name_b,
            node_type=ntype,
            confidence=confidence,
            signals=signals,
        )

        # Step 3: Decide action
        if confidence >= MERGE_THRESHOLD:
            pair.should_merge = True
            pair.should_link = True
            result.merges.append(pair)
        elif confidence >= SAME_AS_THRESHOLD:
            pair.should_link = True
            result.same_as_links.append(pair)

    # Compute duplicate rate
    total_entities = len(entities)
    duplicates = len(result.merges) + len(result.same_as_links)
    result.duplicate_rate = duplicates / max(1, total_entities)

    logger.info(result.summary())
    return result


# ---------------------------------------------------------------------------
# Precision / recall metrics
# ---------------------------------------------------------------------------


@dataclass
class ERMetrics:
    """Precision/recall metrics for entity resolution evaluation."""

    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    @property
    def precision(self) -> float:
        total = self.true_positives + self.false_positives
        return self.true_positives / total if total > 0 else 0.0

    @property
    def recall(self) -> float:
        total = self.true_positives + self.false_negatives
        return self.true_positives / total if total > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    def summary(self) -> str:
        return (
            f"Precision: {self.precision:.3f}, "
            f"Recall: {self.recall:.3f}, "
            f"F1: {self.f1:.3f} "
            f"(TP={self.true_positives}, FP={self.false_positives}, "
            f"FN={self.false_negatives})"
        )


def evaluate_resolution(
    predicted_pairs: set[tuple[str, str]],
    ground_truth_pairs: set[tuple[str, str]],
) -> ERMetrics:
    """Evaluate entity resolution predictions against ground truth.

    Both predicted and ground truth are sets of (name_a, name_b) pairs
    that should be considered the same entity. Order within pairs doesn't
    matter — (a, b) and (b, a) are equivalent.
    """

    # Normalize pair order
    def normalize_pair(a: str, b: str) -> tuple[str, str]:
        return (min(a, b), max(a, b))

    pred = {normalize_pair(a, b) for a, b in predicted_pairs}
    gold = {normalize_pair(a, b) for a, b in ground_truth_pairs}

    tp = len(pred & gold)
    fp = len(pred - gold)
    fn = len(gold - pred)

    return ERMetrics(true_positives=tp, false_positives=fp, false_negatives=fn)
