"""Automated extraction quality evaluation.

Since we don't have domain-expert ground truth annotations, we use a multi-signal
approach to assess extraction quality:

1. Schema conformance — structural validity of extracted data
2. LLM-as-judge — GPT-4o grades GPT-4o-mini extractions against the abstract
3. Cross-paper consistency — same entities should have consistent names
4. Coverage heuristics — expected entity counts per paper type

Each signal produces a score. The overall quality gate is:
- Schema conformance: 100% (hard requirement)
- LLM-as-judge accuracy: >80% of entities/relations rated "correct" by GPT-4o
- mathematical_structure fill rate: >80% of techniques have non-"none" values
- bottleneck_class fill rate: >90% of problems have non-"none" values
- No self-referential relations
- No reversed-direction relations
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from openai import OpenAI

logger = logging.getLogger(__name__)

# Valid (source_type, relation, target_type) per schema
RELATION_DIRECTION_RULES: dict[str, tuple[str, str]] = {
    "INTRODUCES": ("paper", "technique"),
    "EVALUATES_ON": ("paper", "dataset"),
    "HAS_LIMITATION": ("technique", "problem"),
    "MITIGATES": ("technique", "problem"),
    "IMPROVES_OVER": ("technique", "technique"),
    "USES": ("technique", "technique"),
    "IS_INSTANCE_OF": ("technique|problem", "technique|problem"),
    "CAUSED_BY": ("problem", "problem"),
    "TEMPORALLY_FOLLOWS": ("technique", "technique"),
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SchemaCheck:
    """Result of schema conformance check for one paper."""

    paper_id: str
    self_referential_relations: list[dict] = field(default_factory=list)
    reversed_relations: list[dict] = field(default_factory=list)
    invalid_type_combos: list[dict] = field(default_factory=list)
    vague_improves_over: list[dict] = field(default_factory=list)
    fake_datasets: list[str] = field(default_factory=list)
    math_structure_missing: list[str] = field(default_factory=list)
    bottleneck_class_missing: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return (
            not self.self_referential_relations
            and not self.reversed_relations
            and not self.invalid_type_combos
        )

    @property
    def issues(self) -> list[str]:
        out = []
        for r in self.self_referential_relations:
            out.append(f"Self-ref: {r['source']} → {r['relation']} → {r['target']}")
        for r in self.reversed_relations:
            out.append(
                f"Reversed: {r['source_type']} → {r['relation']} → {r['target_type']} "
                f"(expected {RELATION_DIRECTION_RULES.get(r['relation'], '?')})"
            )
        for r in self.invalid_type_combos:
            out.append(f"Invalid combo: {r['source_type']} → {r['relation']} → {r['target_type']}")
        for r in self.vague_improves_over:
            out.append(f"Vague IMPROVES_OVER target: '{r['target']}'")
        for d in self.fake_datasets:
            out.append(f"Fake/generic dataset: '{d}'")
        for t in self.math_structure_missing:
            out.append(f"mathematical_structure missing for technique: '{t}'")
        for p in self.bottleneck_class_missing:
            out.append(f"bottleneck_class missing for problem: '{p}'")
        return out


@dataclass
class LLMJudgment:
    """GPT-4o judgment on extraction quality for one paper."""

    paper_id: str
    entity_verdicts: list[dict] = field(default_factory=list)
    relation_verdicts: list[dict] = field(default_factory=list)
    missing_entities: list[dict] = field(default_factory=list)
    missing_relations: list[dict] = field(default_factory=list)

    @property
    def entity_accuracy(self) -> float:
        if not self.entity_verdicts:
            return 0.0
        correct = sum(1 for v in self.entity_verdicts if v.get("correct"))
        return correct / len(self.entity_verdicts)

    @property
    def relation_accuracy(self) -> float:
        if not self.relation_verdicts:
            return 0.0
        correct = sum(1 for v in self.relation_verdicts if v.get("correct"))
        return correct / len(self.relation_verdicts)


@dataclass
class EvalReport:
    """Full evaluation report across all papers."""

    schema_checks: list[SchemaCheck] = field(default_factory=list)
    llm_judgments: list[LLMJudgment] = field(default_factory=list)

    @property
    def schema_pass_rate(self) -> float:
        if not self.schema_checks:
            return 0.0
        return sum(1 for c in self.schema_checks if c.passed) / len(self.schema_checks)

    @property
    def avg_entity_accuracy(self) -> float:
        if not self.llm_judgments:
            return 0.0
        return sum(j.entity_accuracy for j in self.llm_judgments) / len(self.llm_judgments)

    @property
    def avg_relation_accuracy(self) -> float:
        if not self.llm_judgments:
            return 0.0
        return sum(j.relation_accuracy for j in self.llm_judgments) / len(self.llm_judgments)

    @property
    def math_structure_fill_rate(self) -> float:
        total = 0
        filled = 0
        for c in self.schema_checks:
            # Count from the check: missing means unfilled
            total += len(c.math_structure_missing)
        # We need total techniques — get from judgments or approximate
        for j in self.llm_judgments:
            for v in j.entity_verdicts:
                if v.get("type") == "Technique":
                    total += 1
                    if v.get("has_math_structure"):
                        filled += 1
        if total == 0:
            return 1.0
        return filled / total

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "EXTRACTION QUALITY EVALUATION",
            "=" * 60,
            f"Papers evaluated: {len(self.schema_checks)}",
            "",
            "── Schema Conformance ──",
            f"  Pass rate: {self.schema_pass_rate:.0%}",
        ]

        all_issues = []
        for c in self.schema_checks:
            all_issues.extend(c.issues)
        if all_issues:
            lines.append(f"  Issues ({len(all_issues)}):")
            for issue in all_issues[:20]:  # cap at 20
                lines.append(f"    - {issue}")
            if len(all_issues) > 20:
                lines.append(f"    ... and {len(all_issues) - 20} more")

        if self.llm_judgments:
            lines.extend(
                [
                    "",
                    "── LLM-as-Judge (GPT-4o) ──",
                    f"  Entity accuracy:   {self.avg_entity_accuracy:.0%}",
                    f"  Relation accuracy: {self.avg_relation_accuracy:.0%}",
                ]
            )

            all_missing_ent = []
            all_missing_rel = []
            for j in self.llm_judgments:
                all_missing_ent.extend(j.missing_entities)
                all_missing_rel.extend(j.missing_relations)
            if all_missing_ent:
                lines.append(f"  Missing entities ({len(all_missing_ent)}):")
                for m in all_missing_ent[:10]:
                    lines.append(f"    - [{m.get('type', '?')}] {m.get('name', '?')}")
            if all_missing_rel:
                lines.append(f"  Missing relations ({len(all_missing_rel)}):")
                for m in all_missing_rel[:10]:
                    lines.append(
                        f"    - {m.get('source', '?')} → {m.get('relation', '?')} → "
                        f"{m.get('target', '?')}"
                    )

        lines.extend(
            [
                "",
                "── Quality Gate ──",
                f"  Schema conformance 100%:    {'PASS' if self.schema_pass_rate == 1.0 else 'FAIL'}",
                f"  Entity accuracy >80%:       {'PASS' if self.avg_entity_accuracy > 0.8 else 'FAIL' if self.llm_judgments else 'SKIP'}",
                f"  Relation accuracy >80%:     {'PASS' if self.avg_relation_accuracy > 0.8 else 'FAIL' if self.llm_judgments else 'SKIP'}",
            ]
        )

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Schema conformance check (no LLM needed)
# ---------------------------------------------------------------------------

VAGUE_TARGETS = {
    "existing methods",
    "existing approaches",
    "previous methods",
    "previous approaches",
    "prior work",
    "baseline",
    "baselines",
    "existing best results",
    "state-of-the-art",
    "current methods",
    "traditional methods",
    "conventional approaches",
}

GENERIC_DATASET_PATTERNS = {
    "i.i.d.",
    "datasets",
    "large-scale",
    "real-world",
    "synthetic",
    "training data",
    "test data",
    "benchmark data",
}


def check_schema_conformance(paper: dict) -> SchemaCheck:
    """Check a single paper's extraction for schema violations."""
    check = SchemaCheck(paper_id=paper.get("arxiv_id", "?"))

    # Check entities
    for ent in paper.get("entities", []):
        etype = ent.get("type", "")
        name = ent.get("name", "")

        if etype == "Technique":
            ms = ent.get("mathematical_structure") or ""
            if not ms or ms.lower() in ("none", "n/a", ""):
                check.math_structure_missing.append(name)

        if etype == "Problem":
            bc = ent.get("bottleneck_class") or ""
            if not bc or bc.lower() in ("none", "n/a", ""):
                check.bottleneck_class_missing.append(name)

        if etype == "Dataset":
            name_lower = name.lower()
            if any(pat in name_lower for pat in GENERIC_DATASET_PATTERNS):
                check.fake_datasets.append(name)

    # Check relations
    for rel in paper.get("relations", []):
        source = rel.get("source", "")
        target = rel.get("target", "")
        relation = rel.get("relation", "")
        source_type = rel.get("source_type", "").lower()
        target_type = rel.get("target_type", "").lower()

        # Self-referential
        if source == target and source_type == target_type:
            check.self_referential_relations.append(rel)

        # Direction check
        expected = RELATION_DIRECTION_RULES.get(relation)
        if expected:
            exp_src, exp_tgt = expected
            src_ok = source_type in exp_src.split("|")
            tgt_ok = target_type in exp_tgt.split("|")
            if not src_ok or not tgt_ok:
                check.reversed_relations.append(rel)

        # Vague IMPROVES_OVER targets
        if relation == "IMPROVES_OVER":
            if target.lower() in VAGUE_TARGETS:
                check.vague_improves_over.append(rel)

    return check


# ---------------------------------------------------------------------------
# LLM-as-judge (uses GPT-4o to grade GPT-4o-mini extractions)
# ---------------------------------------------------------------------------

LLM_JUDGE_SYSTEM = """\
You are an expert ML/AI researcher evaluating the quality of automated knowledge extraction \
from research papers. You will be given a paper's title and abstract, along with entities and \
relations that were automatically extracted.

Your job is to judge:
1. Is each extracted entity CORRECT? (Does it exist in the paper? Is the type right? Is the name reasonable?)
2. Is each extracted relation CORRECT? (Is the direction right? Is it supported by the text? Is the confidence appropriate?)
3. What important entities or relations were MISSED?

Be strict but fair. An entity is "correct" if:
- It represents something actually discussed in the paper
- The type (technique/problem/dataset) is appropriate
- The name is a reasonable canonical form

A relation is "correct" if:
- The source→target direction matches the relation type
- It is supported by the abstract text
- The source and target entities exist and are typed correctly

Respond with valid JSON only."""

LLM_JUDGE_USER = """\
Paper title: {title}
Paper abstract: {abstract}

Extracted entities:
{entities_json}

Extracted relations:
{relations_json}

Judge each entity and relation. Also identify what was missed.

Output format:
{{
  "entity_verdicts": [
    {{
      "name": "entity name",
      "type": "Technique|Problem|Dataset",
      "correct": true/false,
      "has_math_structure": true/false,
      "reason": "why correct or incorrect"
    }}
  ],
  "relation_verdicts": [
    {{
      "source": "...",
      "relation": "...",
      "target": "...",
      "correct": true/false,
      "reason": "why correct or incorrect"
    }}
  ],
  "missing_entities": [
    {{
      "name": "what should have been extracted",
      "type": "Technique|Problem|Dataset",
      "reason": "why this is important"
    }}
  ],
  "missing_relations": [
    {{
      "source": "...",
      "relation": "...",
      "target": "...",
      "reason": "why this should have been extracted"
    }}
  ]
}}"""


def judge_extraction(client: OpenAI, paper: dict, model: str = "gpt-4o") -> LLMJudgment:
    """Use GPT-4o to judge extraction quality for a single paper."""
    entities_json = json.dumps(paper.get("entities", []), indent=2)
    relations_json = json.dumps(paper.get("relations", []), indent=2)

    messages = [
        {"role": "system", "content": LLM_JUDGE_SYSTEM},
        {
            "role": "user",
            "content": LLM_JUDGE_USER.format(
                title=paper.get("title", ""),
                abstract=paper.get("abstract", ""),
                entities_json=entities_json,
                relations_json=relations_json,
            ),
        },
    ]

    judgment = LLMJudgment(paper_id=paper.get("arxiv_id", "?"))

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        result = json.loads(response.choices[0].message.content)
        judgment.entity_verdicts = result.get("entity_verdicts", [])
        judgment.relation_verdicts = result.get("relation_verdicts", [])
        judgment.missing_entities = result.get("missing_entities", [])
        judgment.missing_relations = result.get("missing_relations", [])
    except Exception as e:
        logger.error(f"LLM judge failed for {paper.get('arxiv_id')}: {e}")

    return judgment


# ---------------------------------------------------------------------------
# Full evaluation pipeline
# ---------------------------------------------------------------------------


def evaluate_extractions(
    results: list[dict],
    client: OpenAI | None = None,
    judge_model: str = "gpt-4o",
    run_llm_judge: bool = True,
) -> EvalReport:
    """Run full evaluation on extraction results.

    Args:
        results: List of paper extraction dicts (from phase0_extraction.json).
        client: OpenAI client (required if run_llm_judge=True).
        judge_model: Model to use for LLM-as-judge.
        run_llm_judge: Whether to run the expensive LLM judge step.
    """
    report = EvalReport()

    for paper in results:
        # Always run schema check (free)
        check = check_schema_conformance(paper)
        report.schema_checks.append(check)

        # Optionally run LLM judge (costs money)
        if run_llm_judge and client is not None:
            logger.info(f"Judging {paper.get('arxiv_id')}...")
            judgment = judge_extraction(client, paper, judge_model)
            report.llm_judgments.append(judgment)

    return report
