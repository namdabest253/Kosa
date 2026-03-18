"""Tests for the extraction pipeline (unit tests, no API calls)."""

from unittest.mock import MagicMock, patch

from kosa.graph.schema import EdgeType, NodeLabel
from kosa.ingestion.extract import (
    ExtractedEntity,
    ExtractedRelation,
    NoveltyResult,
    PaperExtractionResult,
    extract_paper,
)


class TestExtractionDataClasses:
    def test_extracted_entity(self):
        e = ExtractedEntity(
            name="LoRA",
            entity_type=NodeLabel.TECHNIQUE,
            description="Low-rank adaptation",
            source_paper="2106.09685",
        )
        assert e.name == "LoRA"
        assert e.entity_type == NodeLabel.TECHNIQUE

    def test_extracted_relation(self):
        r = ExtractedRelation(
            source_name="LoRA paper",
            source_type=NodeLabel.PAPER,
            relation=EdgeType.INTRODUCES,
            target_name="LoRA",
            target_type=NodeLabel.TECHNIQUE,
            confidence=0.95,
            supporting_text="We introduce LoRA",
            source_paper="2106.09685",
        )
        assert r.relation == EdgeType.INTRODUCES
        assert r.confidence == 0.95

    def test_novelty_result(self):
        n = NoveltyResult("novel_technique", 0.9, "New method")
        assert n.classification == "novel_technique"

    def test_paper_extraction_result(self):
        r = PaperExtractionResult(
            arxiv_id="2106.09685",
            title="LoRA",
            abstract="...",
            authors=["A"],
            year=2021,
            venue="ICLR",
        )
        assert r.entities == []
        assert r.relations == []
        assert r.novelty is None


class TestExtractPaperMocked:
    """Test extract_paper with mocked LLM calls."""

    @patch("kosa.ingestion.extract._call_llm")
    def test_full_pipeline(self, mock_call):
        mock_call.side_effect = [
            # Entity extraction
            {
                "techniques": [{"name": "LoRA", "description": "Low-rank adaptation"}],
                "problems": [{"name": "fine-tuning cost", "description": "expensive to fine-tune"}],
                "datasets": [{"name": "GLUE", "description": "NLU benchmark"}],
            },
            # Schema alignment
            {
                "techniques": [
                    {
                        "name": "LoRA",
                        "original_name": "LoRA",
                        "description": "Low-rank adaptation",
                        "mathematical_structure": "low-rank matrix factorization",
                    }
                ],
                "problems": [
                    {
                        "name": "fine-tuning cost",
                        "original_name": "fine-tuning cost",
                        "description": "expensive to fine-tune",
                        "bottleneck_class": "parameter_count_scaling_with_model_size",
                    }
                ],
            },
            # Relation extraction
            {
                "relations": [
                    {
                        "source": "2106.09685",
                        "source_type": "paper",
                        "relation": "INTRODUCES",
                        "target": "LoRA",
                        "target_type": "technique",
                        "confidence": 0.95,
                        "supporting_text": "We propose LoRA",
                    },
                    {
                        "source": "LoRA",
                        "source_type": "technique",
                        "relation": "MITIGATES",
                        "target": "fine-tuning cost",
                        "target_type": "problem",
                        "confidence": 0.85,
                        "supporting_text": "reduces trainable parameters",
                    },
                ]
            },
            # Novelty classification
            {
                "classification": "novel_technique",
                "confidence": 0.92,
                "reasoning": "Introduces a new parameter-efficient fine-tuning method",
            },
        ]

        client = MagicMock()
        result = extract_paper(
            client=client,
            arxiv_id="2106.09685",
            title="LoRA: Low-Rank Adaptation of Large Language Models",
            abstract="We propose LoRA which freezes weights and injects low-rank matrices.",
            authors=["Edward Hu"],
            year=2021,
            venue="ICLR",
            model="gpt-4o-mini",
        )

        assert result.arxiv_id == "2106.09685"
        assert len(result.entities) == 3  # 1 technique + 1 problem + 1 dataset
        assert len(result.relations) == 2
        assert result.novelty is not None
        assert result.novelty.classification == "novel_technique"
        assert result.errors == []

        # Check entity types
        types = {e.entity_type for e in result.entities}
        assert NodeLabel.TECHNIQUE in types
        assert NodeLabel.PROBLEM in types
        assert NodeLabel.DATASET in types

        # Check schema alignment was applied
        technique = next(e for e in result.entities if e.entity_type == NodeLabel.TECHNIQUE)
        assert technique.mathematical_structure == "low-rank matrix factorization"

        problem = next(e for e in result.entities if e.entity_type == NodeLabel.PROBLEM)
        assert problem.bottleneck_class == "parameter_count_scaling_with_model_size"

    @patch("kosa.ingestion.extract._call_llm")
    def test_handles_llm_failure_gracefully(self, mock_call):
        mock_call.return_value = None  # All LLM calls fail

        client = MagicMock()
        result = extract_paper(
            client=client,
            arxiv_id="0000.00000",
            title="Test",
            abstract="Test abstract",
            authors=[],
            year=2024,
            venue=None,
        )

        assert result.entities == []
        assert result.relations == []
        assert result.novelty is None
