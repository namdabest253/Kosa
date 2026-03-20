"""Tests for entity resolution module."""

from kosa.entity_resolution.resolver import (
    ERMetrics,
    block_candidates,
    compute_pair_confidence,
    evaluate_resolution,
    normalize_name,
    resolve_entities,
    string_similarity,
)
from kosa.graph.schema import NodeLabel


class TestNormalizeName:
    def test_exact_alias(self):
        assert normalize_name("VAE") == "vae"
        assert normalize_name("Generative Adversarial Network") == "gan"

    def test_case_insensitive(self):
        assert normalize_name("BERT") == "bert"
        assert normalize_name("bert") == "bert"

    def test_unknown_name(self):
        assert normalize_name("SomeNewTechnique") == "somenewtechnique"

    def test_strips_whitespace(self):
        assert normalize_name("  LSTM  ") == "lstm"


class TestStringSimilarity:
    def test_exact_match(self):
        assert string_similarity("Transformer", "Transformer") == 1.0

    def test_alias_match(self):
        sim = string_similarity("VAE", "variational autoencoder")
        assert sim >= 0.95

    def test_high_overlap(self):
        sim = string_similarity("flash attention", "FlashAttention")
        assert sim >= 0.9

    def test_partial_overlap(self):
        sim = string_similarity("attention mechanism", "self-attention mechanism")
        assert sim > 0.3  # Jaccard: 2/4 tokens overlap, but split on hyphen gives 3/5

    def test_no_overlap(self):
        sim = string_similarity("ResNet", "LSTM")
        assert sim < 0.3

    def test_substring_containment(self):
        sim = string_similarity("attention", "self-attention")
        assert sim > 0.5


class TestBlockCandidates:
    def test_same_type_blocking(self):
        entities = [
            ("VAE", NodeLabel.TECHNIQUE),
            ("variational autoencoder", NodeLabel.TECHNIQUE),
            ("ImageNet", NodeLabel.DATASET),
        ]
        candidates = block_candidates(entities)
        # Should find VAE/variational autoencoder pair
        assert len(candidates) >= 1
        names = {(c[0], c[1]) for c in candidates}
        assert ("VAE", "variational autoencoder") in names or (
            "variational autoencoder",
            "VAE",
        ) in names

    def test_different_types_not_paired(self):
        entities = [
            ("attention", NodeLabel.TECHNIQUE),
            ("attention", NodeLabel.PROBLEM),
        ]
        candidates = block_candidates(entities)
        # Different types should not be paired
        assert len(candidates) == 0

    def test_low_similarity_filtered(self):
        entities = [
            ("ResNet", NodeLabel.TECHNIQUE),
            ("LSTM", NodeLabel.TECHNIQUE),
        ]
        candidates = block_candidates(entities, min_similarity=0.8)
        assert len(candidates) == 0


class TestComputePairConfidence:
    def test_high_confidence_alias(self):
        conf, signals = compute_pair_confidence(
            "VAE",
            "variational autoencoder",
            NodeLabel.TECHNIQUE,
            string_sim=0.98,
        )
        assert conf >= 0.9
        assert signals["alias_match"] == 1.0

    def test_low_confidence_different(self):
        conf, signals = compute_pair_confidence(
            "ResNet",
            "LSTM",
            NodeLabel.TECHNIQUE,
            string_sim=0.1,
        )
        assert conf < 0.5

    def test_embedding_boost(self):
        conf_without, _ = compute_pair_confidence(
            "attention",
            "self-attention",
            NodeLabel.TECHNIQUE,
            string_sim=0.6,
        )
        conf_with, _ = compute_pair_confidence(
            "attention",
            "self-attention",
            NodeLabel.TECHNIQUE,
            string_sim=0.6,
            embedding_sim=0.95,
        )
        assert conf_with > conf_without


class TestResolveEntities:
    def test_detects_known_aliases(self):
        entities = [
            ("VAE", NodeLabel.TECHNIQUE),
            ("variational autoencoder", NodeLabel.TECHNIQUE),
            ("ResNet", NodeLabel.TECHNIQUE),
        ]
        result = resolve_entities(entities)
        # Should detect VAE/variational autoencoder as same
        all_pairs = result.same_as_links + result.merges
        assert len(all_pairs) >= 1

    def test_no_false_positives_different_entities(self):
        entities = [
            ("ResNet", NodeLabel.TECHNIQUE),
            ("LSTM", NodeLabel.TECHNIQUE),
            ("GAN", NodeLabel.TECHNIQUE),
        ]
        result = resolve_entities(entities)
        assert len(result.same_as_links) == 0
        assert len(result.merges) == 0

    def test_duplicate_rate(self):
        entities = [
            ("VAE", NodeLabel.TECHNIQUE),
            ("variational autoencoder", NodeLabel.TECHNIQUE),
            ("GAN", NodeLabel.TECHNIQUE),
            ("generative adversarial network", NodeLabel.TECHNIQUE),
            ("ResNet", NodeLabel.TECHNIQUE),
        ]
        result = resolve_entities(entities)
        # 2 duplicate pairs out of 5 entities = 40%
        assert result.duplicate_rate > 0


class TestERMetrics:
    def test_perfect_precision_recall(self):
        metrics = ERMetrics(true_positives=10, false_positives=0, false_negatives=0)
        assert metrics.precision == 1.0
        assert metrics.recall == 1.0
        assert metrics.f1 == 1.0

    def test_no_predictions(self):
        metrics = ERMetrics(true_positives=0, false_positives=0, false_negatives=5)
        assert metrics.precision == 0.0
        assert metrics.recall == 0.0
        assert metrics.f1 == 0.0

    def test_mixed(self):
        metrics = ERMetrics(true_positives=5, false_positives=2, false_negatives=3)
        assert 0.5 < metrics.precision < 1.0
        assert 0.5 < metrics.recall < 1.0


class TestEvaluateResolution:
    def test_exact_match(self):
        predicted = {("a", "b"), ("c", "d")}
        ground_truth = {("a", "b"), ("c", "d")}
        metrics = evaluate_resolution(predicted, ground_truth)
        assert metrics.precision == 1.0
        assert metrics.recall == 1.0

    def test_order_independent(self):
        predicted = {("b", "a")}
        ground_truth = {("a", "b")}
        metrics = evaluate_resolution(predicted, ground_truth)
        assert metrics.true_positives == 1

    def test_false_positives(self):
        predicted = {("a", "b"), ("c", "d")}
        ground_truth = {("a", "b")}
        metrics = evaluate_resolution(predicted, ground_truth)
        assert metrics.false_positives == 1

    def test_false_negatives(self):
        predicted = {("a", "b")}
        ground_truth = {("a", "b"), ("c", "d")}
        metrics = evaluate_resolution(predicted, ground_truth)
        assert metrics.false_negatives == 1
