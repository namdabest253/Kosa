"""Extraction prompts for GPT-4o-mini.

Four prompt types:
1. Entity extraction — extract technique, problem, dataset nodes
2. Relation extraction — extract typed edges between entities
3. Schema alignment — normalize entities, assign bottleneck_class & mathematical_structure
4. Novelty classifier — classify abstract as novel/improvement/incremental/survey
"""

from __future__ import annotations

from kosa.graph.schema import EdgeType, NodeLabel

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

ENTITY_EXTRACTION_SYSTEM = """\
You are a precise ML/AI research entity extractor. Given a paper's title and abstract, \
extract all entities of these types:

- **technique**: A specific method, algorithm, architecture, or model \
(e.g., "FlashAttention", "LoRA", "diffusion models", "PPO")
- **problem**: A known limitation, bottleneck, failure mode, or open challenge \
(e.g., "quadratic memory scaling", "catastrophic forgetting", "mode collapse")
- **dataset**: A standard evaluation dataset or benchmark \
(e.g., "ImageNet", "MMLU", "LibriSpeech")

Rules:
- Extract only entities explicitly mentioned or clearly implied by the text.
- Do NOT hallucinate entities not supported by the text.
- Use the most specific name for each entity (e.g., "FlashAttention-2" not "attention").
- For techniques, prefer the canonical name used in the paper.
- For problems, describe the actual bottleneck, not just a topic.
- Each entity should appear only once in your output.

Respond with valid JSON only. No markdown, no explanation."""

ENTITY_EXTRACTION_USER = """\
Paper title: {title}
Paper abstract: {abstract}

Extract all technique, problem, and dataset entities from this paper.

Output format:
{{
  "techniques": [
    {{"name": "...", "description": "brief description from the paper"}}
  ],
  "problems": [
    {{"name": "...", "description": "brief description from the paper"}}
  ],
  "datasets": [
    {{"name": "...", "description": "brief description if available"}}
  ]
}}"""

# ---------------------------------------------------------------------------

RELATION_EXTRACTION_SYSTEM = """\
You are a precise ML/AI research relation extractor. Given a paper and its extracted entities, \
identify relationships between them.

Valid relationship types (use EXACTLY these labels):
- INTRODUCES: paper introduces a technique (paper → technique)
- EVALUATES_ON: paper evaluates on a dataset (paper → dataset)
- HAS_LIMITATION: technique has a known limitation (technique → problem)
- MITIGATES: technique addresses/solves a problem (technique → problem)
- IMPROVES_OVER: technique improves upon another technique (technique → technique)
- IS_INSTANCE_OF: entity is a specific case of a more general entity (problem → problem, technique → technique)
- CAUSED_BY: problem is caused by another problem (problem → problem)
- TEMPORALLY_FOLLOWS: technique succeeds another chronologically (technique → technique)

Rules:
- Only extract relationships that are explicitly stated or strongly implied by the text.
- Do NOT infer relationships that require external knowledge beyond the paper.
- Every relationship must have a supporting quote or close paraphrase from the text.
- Confidence: 0.9+ for explicitly stated, 0.7-0.9 for strongly implied, 0.5-0.7 for inferred.

Respond with valid JSON only. No markdown, no explanation."""

RELATION_EXTRACTION_USER = """\
Paper title: {title}
Paper abstract: {abstract}
ArXiv ID: {arxiv_id}

Extracted entities:
- Techniques: {techniques}
- Problems: {problems}
- Datasets: {datasets}

Extract all relationships between these entities and the paper.

Output format:
{{
  "relations": [
    {{
      "source": "entity name",
      "source_type": "paper|technique|problem|dataset",
      "relation": "INTRODUCES|EVALUATES_ON|HAS_LIMITATION|MITIGATES|IMPROVES_OVER|IS_INSTANCE_OF|CAUSED_BY|TEMPORALLY_FOLLOWS",
      "target": "entity name",
      "target_type": "paper|technique|problem|dataset",
      "confidence": 0.85,
      "supporting_text": "quote or close paraphrase from abstract"
    }}
  ]
}}"""

# ---------------------------------------------------------------------------

SCHEMA_ALIGNMENT_SYSTEM = """\
You are a knowledge graph schema alignment specialist for ML/AI research. \
Given extracted entities, normalize them and add structured metadata.

For each **technique**, add:
- `mathematical_structure`: The underlying formal/mathematical structure. \
Use domain-agnostic descriptions like "eigenvector of stochastic matrix", \
"convex optimization", "fixed-point iteration", "block-sparse matrix multiplication". \
If no clear mathematical structure, use "none".

For each **problem**, add:
- `bottleneck_class`: A domain-agnostic structural description of the bottleneck. \
Do NOT use domain-specific jargon. Examples:
  - "exponential_signal_decay_across_stages" (not "vanishing gradients")
  - "quadratic_scaling_with_sequence_length" (not "attention is slow")
  - "distribution_shift_between_training_and_deployment" (not "domain adaptation")
  - "combinatorial_explosion_in_search_space" (not "NP-hard planning")

The bottleneck_class should describe the structural nature of the problem so that \
solutions from other domains with the same structural bottleneck can be connected.

Rules:
- Normalize entity names to canonical forms (e.g., "layer norm" → "LayerNorm").
- Merge obvious duplicates (e.g., "GPT3" and "GPT-3").
- Keep mathematical_structure and bottleneck_class as concise as possible.

Respond with valid JSON only."""

SCHEMA_ALIGNMENT_USER = """\
Entities to align:

Techniques:
{techniques}

Problems:
{problems}

Output format:
{{
  "techniques": [
    {{
      "name": "canonical name",
      "original_name": "name as extracted",
      "description": "...",
      "mathematical_structure": "domain-agnostic formal structure"
    }}
  ],
  "problems": [
    {{
      "name": "canonical name",
      "original_name": "name as extracted",
      "description": "...",
      "bottleneck_class": "domain_agnostic_structural_description"
    }}
  ]
}}"""

# ---------------------------------------------------------------------------

NOVELTY_CLASSIFIER_SYSTEM = """\
You are an ML/AI research novelty classifier. Given a paper's title and abstract, \
classify it into exactly one category:

- **novel_technique**: Introduces a fundamentally new method, architecture, or approach
- **significant_improvement**: Substantially improves an existing method (>10% gains, new capability)
- **incremental**: Minor improvements, ablation studies, or straightforward applications
- **survey**: Review paper, benchmark paper, or position paper

Respond with valid JSON only."""

NOVELTY_CLASSIFIER_USER = """\
Paper title: {title}
Paper abstract: {abstract}

Classify this paper's novelty level.

Output format:
{{
  "classification": "novel_technique|significant_improvement|incremental|survey",
  "confidence": 0.85,
  "reasoning": "one sentence explanation"
}}"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_NOVELTY_CLASSES = {"novel_technique", "significant_improvement", "incremental", "survey"}

# Map string type labels from LLM output to our schema enums
SOURCE_TYPE_MAP = {
    "paper": NodeLabel.PAPER,
    "technique": NodeLabel.TECHNIQUE,
    "problem": NodeLabel.PROBLEM,
    "dataset": NodeLabel.DATASET,
}

RELATION_TYPE_MAP = {
    "INTRODUCES": EdgeType.INTRODUCES,
    "EVALUATES_ON": EdgeType.EVALUATES_ON,
    "HAS_LIMITATION": EdgeType.HAS_LIMITATION,
    "MITIGATES": EdgeType.MITIGATES,
    "IMPROVES_OVER": EdgeType.IMPROVES_OVER,
    "IS_INSTANCE_OF": EdgeType.IS_INSTANCE_OF,
    "CAUSED_BY": EdgeType.CAUSED_BY,
    "TEMPORALLY_FOLLOWS": EdgeType.TEMPORALLY_FOLLOWS,
}


def format_entity_extraction(title: str, abstract: str) -> list[dict]:
    """Return OpenAI messages for entity extraction."""
    return [
        {"role": "system", "content": ENTITY_EXTRACTION_SYSTEM},
        {"role": "user", "content": ENTITY_EXTRACTION_USER.format(title=title, abstract=abstract)},
    ]


def format_relation_extraction(
    title: str,
    abstract: str,
    arxiv_id: str,
    techniques: list[str],
    problems: list[str],
    datasets: list[str],
) -> list[dict]:
    """Return OpenAI messages for relation extraction."""
    return [
        {"role": "system", "content": RELATION_EXTRACTION_SYSTEM},
        {
            "role": "user",
            "content": RELATION_EXTRACTION_USER.format(
                title=title,
                abstract=abstract,
                arxiv_id=arxiv_id,
                techniques=", ".join(techniques),
                problems=", ".join(problems),
                datasets=", ".join(datasets),
            ),
        },
    ]


def format_schema_alignment(techniques: list[dict], problems: list[dict]) -> list[dict]:
    """Return OpenAI messages for schema alignment."""
    import json

    return [
        {"role": "system", "content": SCHEMA_ALIGNMENT_SYSTEM},
        {
            "role": "user",
            "content": SCHEMA_ALIGNMENT_USER.format(
                techniques=json.dumps(techniques, indent=2),
                problems=json.dumps(problems, indent=2),
            ),
        },
    ]


def format_novelty_classifier(title: str, abstract: str) -> list[dict]:
    """Return OpenAI messages for novelty classification."""
    return [
        {"role": "system", "content": NOVELTY_CLASSIFIER_SYSTEM},
        {"role": "user", "content": NOVELTY_CLASSIFIER_USER.format(title=title, abstract=abstract)},
    ]
