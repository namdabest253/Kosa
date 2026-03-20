"""Tests for extraction prompts."""

from kosa.ingestion.prompts import (
    RELATION_TYPE_MAP,
    SOURCE_TYPE_MAP,
    VALID_NOVELTY_CLASSES,
    format_entity_extraction,
    format_novelty_classifier,
    format_relation_extraction,
    format_schema_alignment,
)


class TestPromptFormatting:
    def test_entity_extraction_messages(self):
        msgs = format_entity_extraction("Test Paper", "This paper introduces X.")
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert "Test Paper" in msgs[1]["content"]
        assert "technique" in msgs[0]["content"].lower()

    def test_relation_extraction_messages(self):
        msgs = format_relation_extraction(
            "Test",
            "Abstract",
            "2401.00001",
            ["LoRA"],
            ["overfitting"],
            ["ImageNet"],
        )
        assert len(msgs) == 2
        assert "LoRA" in msgs[1]["content"]
        assert "2401.00001" in msgs[1]["content"]

    def test_schema_alignment_messages(self):
        msgs = format_schema_alignment(
            [{"name": "LoRA", "description": "low-rank adaptation"}],
            [{"name": "overfitting", "description": "model memorizes training data"}],
        )
        assert len(msgs) == 2
        assert "mathematical_structure" in msgs[0]["content"]
        assert "bottleneck_class" in msgs[0]["content"]

    def test_novelty_classifier_messages(self):
        msgs = format_novelty_classifier("Test", "Abstract")
        assert len(msgs) == 2
        assert "novel_technique" in msgs[0]["content"]

    def test_novelty_classes(self):
        assert VALID_NOVELTY_CLASSES == {
            "novel_technique",
            "significant_improvement",
            "incremental",
            "survey",
        }


class TestTypeMaps:
    def test_source_type_map_covers_all_labels(self):
        assert "paper" in SOURCE_TYPE_MAP
        assert "technique" in SOURCE_TYPE_MAP
        assert "problem" in SOURCE_TYPE_MAP
        assert "dataset" in SOURCE_TYPE_MAP

    def test_relation_type_map_covers_extracted_types(self):
        expected = {
            "INTRODUCES",
            "EVALUATES_ON",
            "HAS_LIMITATION",
            "MITIGATES",
            "IMPROVES_OVER",
            "USES",
            "IS_INSTANCE_OF",
            "CAUSED_BY",
            "TEMPORALLY_FOLLOWS",
        }
        assert set(RELATION_TYPE_MAP.keys()) == expected
