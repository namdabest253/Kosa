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
(e.g., "Transformer", "LoRA", "DDPM", "PPO", "residual connections", "self-attention"). \
Use the SHORT canonical name researchers use when citing this work \
(e.g., "ResNet" not "Residual Learning Framework", "VAE" not "Auto-Encoding Variational Bayes", \
"reparameterization trick" not "Reparameterization of the Variational Lower Bound").

- **problem**: A known limitation, bottleneck, failure mode, or open challenge that the paper \
either addresses or identifies. Extract BOTH: \
(1) the problem this paper SOLVES (the motivation), and \
(2) any limitations of the proposed approach. \
Also extract problems that are implied by the motivation even if not stated as a named problem. \
For example, if a paper says "existing models require O(n²) memory", extract the problem \
"quadratic memory scaling in attention". \
(e.g., "quadratic memory scaling", "catastrophic forgetting", "mode collapse", "training instability")

- **dataset**: A specific, named evaluation dataset or benchmark. \
Must be a real, citable dataset with a proper name. \
Do NOT extract generic descriptions like "i.i.d. datasets" or "large-scale data" — \
only named benchmarks (e.g., "ImageNet", "MMLU", "LibriSpeech", "WMT 2014 En-De").

Rules:
- Extract only entities explicitly mentioned or clearly implied by the text.
- Do NOT hallucinate entities not supported by the text.
- Use SHORT canonical names (e.g., "GAN" not "Generative Adversarial Network", "ResNet" not "Deep Residual Network").
- The paper's title is NOT a technique name. Extract the technique the paper introduces, which may have the same name but is a method, not a document.
- Each entity should appear only once in your output.
- Aim for completeness: a typical ML paper mentions 3-8 techniques, 1-4 problems, and 2-6 datasets.

Respond with valid JSON only. No markdown, no explanation."""

ENTITY_EXTRACTION_USER = """\
Paper title: {title}
Paper abstract: {abstract}

Extract all technique, problem, and dataset entities from this paper.

Output format:
{{
  "techniques": [
    {{"name": "short canonical name", "description": "brief description from the paper"}}
  ],
  "problems": [
    {{"name": "short description of the bottleneck", "description": "brief description from the paper"}}
  ],
  "datasets": [
    {{"name": "exact dataset name", "description": "brief description if available"}}
  ]
}}"""

# ---------------------------------------------------------------------------

RELATION_EXTRACTION_SYSTEM = """\
You are a precise ML/AI research relation extractor. Given a paper and its extracted entities, \
identify relationships between them.

Valid relationship types with STRICT direction rules:
- INTRODUCES: The PAPER introduces a technique. Source MUST be "paper", target MUST be "technique". \
Use the paper title as the source name.
- EVALUATES_ON: The PAPER evaluates on a dataset. Source MUST be "paper", target MUST be "dataset". \
Use the paper title as the source name.
- HAS_LIMITATION: A technique has a limitation. Source MUST be "technique", target MUST be "problem".
- MITIGATES: A technique addresses a problem. Source MUST be "technique", target MUST be "problem".
- IMPROVES_OVER: A technique improves upon another SPECIFIC technique. Source and target MUST both be "technique". \
Target must be a specific named technique, NOT a vague phrase like "existing methods" or "previous approaches".
- IS_INSTANCE_OF: An entity is a specific case of a more general entity. \
Valid: technique→technique, problem→problem.
- CAUSED_BY: A problem is caused by another problem. Source and target MUST both be "problem".
- TEMPORALLY_FOLLOWS: A technique succeeds another chronologically. Source and target MUST both be "technique".

Critical rules:
- NEVER create a relation where source and target are the same entity.
- NEVER reverse the direction (e.g., technique INTRODUCES paper is WRONG).
- For INTRODUCES and EVALUATES_ON, the source is ALWAYS the paper title with source_type "paper".
- For IMPROVES_OVER, the target must be a SPECIFIC named technique (e.g., "LSTM", "VGG"), \
never a vague phrase. If the paper only claims generic improvement, skip this relation.
- Only extract relationships explicitly stated or strongly implied by the text.
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

Extract all relationships. Remember:
- For INTRODUCES: source is the paper title ("{title}") with source_type "paper"
- For EVALUATES_ON: source is the paper title ("{title}") with source_type "paper"
- NEVER create self-referential relations (source == target)
- IMPROVES_OVER target must be a specific named technique, not "existing methods"

Output format:
{{
  "relations": [
    {{
      "source": "entity or paper name",
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
- `mathematical_structure`: The underlying formal/mathematical operation or structure. \
Be SPECIFIC. Every technique has some mathematical basis — think about what it actually computes. \
Examples:
  - Transformer → "scaled dot-product attention with softmax normalization"
  - ResNet → "identity shortcut connections / skip connections"
  - VAE → "variational lower bound optimization with KL divergence regularization"
  - GAN → "minimax two-player game / adversarial optimization"
  - Adam → "adaptive first and second moment estimation for gradient descent"
  - BatchNorm → "per-batch mean/variance normalization with learned affine transform"
  - LoRA → "low-rank matrix factorization of weight updates"
  - FlashAttention → "tiled block-sparse computation with IO-aware memory management"
  - Diffusion models → "iterative denoising via learned reverse Markov chain"
  - PPO → "clipped surrogate objective for policy gradient optimization"
Do NOT use "none" unless the entity is so vague it has no identifiable mathematical basis. \
If unsure, describe what the technique computes or optimizes.

For each **problem**, add:
- `bottleneck_class`: A domain-agnostic structural description of the bottleneck. \
Do NOT use ML-specific jargon. Describe the STRUCTURAL nature of the problem so that \
solutions from entirely different fields with the same structural bottleneck can be connected. \
Examples:
  - "vanishing gradients" → "exponential_signal_decay_across_stages"
  - "attention is slow" → "quadratic_scaling_with_input_length"
  - "catastrophic forgetting" → "interference_between_sequential_learned_representations"
  - "mode collapse" → "degenerate_fixed_point_in_adversarial_dynamics"
  - "overfitting" → "model_capacity_exceeds_training_signal"
  - "distribution shift" → "mismatch_between_training_and_deployment_distributions"
  - "expensive fine-tuning" → "parameter_count_scaling_with_model_size"
NEVER use "none" for bottleneck_class. Every problem has a structural nature.

Rules:
- Normalize entity names to short canonical forms used by practitioners \
(e.g., "layer norm" → "LayerNorm", "Residual Learning Framework" → "ResNet", \
"Auto-Encoding Variational Bayes" → "VAE", "generative adversarial network" → "GAN").
- Merge obvious duplicates (e.g., "GPT3" and "GPT-3").
- Keep mathematical_structure and bottleneck_class concise (under 10 words).

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
      "mathematical_structure": "specific formal structure (NEVER 'none' for known techniques)"
    }}
  ],
  "problems": [
    {{
      "name": "canonical name",
      "original_name": "name as extracted",
      "description": "...",
      "bottleneck_class": "domain_agnostic_structural_description (NEVER 'none')"
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
