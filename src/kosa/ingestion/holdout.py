"""Temporal holdout benchmark: 20 known ML/AI innovations from 2024-2025.

Each innovation combines techniques from different subfields and should be
expressible as a graph path connecting pre-2024 nodes. Used to evaluate whether
the system can retrospectively predict known breakthroughs from pre-2024 data.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HoldoutInnovation:
    """A known 2024-2025 innovation for temporal holdout evaluation."""

    name: str
    year: int
    papers: list[str]  # arXiv IDs of the innovation papers
    source_techniques: list[str]  # pre-2024 building blocks (technique node names)
    problem_solved: str  # the problem node this addresses
    solution_description: str  # how the techniques were combined
    subfields_combined: list[str]  # which subfields were bridged
    graph_path: str  # expected path through pre-2024 KG nodes


# fmt: off
HOLDOUT_INNOVATIONS: list[HoldoutInnovation] = [
    HoldoutInnovation(
        name="Mixtral 8x7B",
        year=2024,
        papers=["2401.04088"],
        source_techniques=["Mixture of Experts", "Transformer", "instruction tuning"],
        problem_solved="computational cost of dense large language models",
        solution_description=(
            "Applied sparse Mixture of Experts routing to Transformer LLMs, activating only "
            "2 of 8 expert sub-networks per token, matching GPT-3.5 quality at 6x lower inference cost"
        ),
        subfields_combined=["optimization", "NLP"],
        graph_path="Transformer -[USES]-> self-attention, Mixture of Experts -[MITIGATES]-> computational cost scaling",
    ),
    HoldoutInnovation(
        name="Mamba (Selective State Spaces)",
        year=2024,
        papers=["2312.00752"],
        source_techniques=["state space models", "Transformer", "gating mechanisms"],
        problem_solved="quadratic scaling of attention with sequence length",
        solution_description=(
            "Replaced attention with input-dependent selective state space model (S6), achieving "
            "linear scaling in sequence length while matching Transformer quality on language tasks"
        ),
        subfields_combined=["signal processing", "NLP"],
        graph_path="state space models -[MITIGATES]-> quadratic memory scaling, Transformer -[HAS_LIMITATION]-> quadratic memory scaling",
    ),
    HoldoutInnovation(
        name="Sora (Video Generation)",
        year=2024,
        papers=["2402.17177"],
        source_techniques=["diffusion models", "Vision Transformer", "spatiotemporal patches"],
        problem_solved="generating long coherent videos from text",
        solution_description=(
            "Combined diffusion transformer (DiT) architecture with spacetime patches to generate "
            "up to 60-second videos with temporal coherence, bridging image diffusion and video understanding"
        ),
        subfields_combined=["generative", "CV"],
        graph_path="diffusion models -[USES]-> denoising, Vision Transformer -[USES]-> patch embeddings",
    ),
    HoldoutInnovation(
        name="Direct Preference Optimization (DPO)",
        year=2024,
        papers=["2305.18290"],
        source_techniques=["RLHF", "PPO", "Bradley-Terry model", "language model fine-tuning"],
        problem_solved="instability and complexity of RLHF training",
        solution_description=(
            "Derived a closed-form loss from the RL objective, eliminating the need for a separate "
            "reward model and PPO training loop, simplifying RLHF to a single classification loss"
        ),
        subfields_combined=["RL", "NLP", "optimization"],
        graph_path="PPO -[MITIGATES]-> reward hacking, RLHF -[HAS_LIMITATION]-> training instability",
    ),
    HoldoutInnovation(
        name="Medusa (Multi-Head Speculative Decoding)",
        year=2024,
        papers=["2401.10774"],
        source_techniques=["speculative decoding", "multi-head attention", "tree search"],
        problem_solved="slow autoregressive LLM inference",
        solution_description=(
            "Added multiple lightweight prediction heads to an LLM to propose several future tokens "
            "in parallel, verified via tree-based attention, achieving 2-3x speedup without quality loss"
        ),
        subfields_combined=["systems", "NLP"],
        graph_path="speculative decoding -[MITIGATES]-> sequential bottleneck, multi-head attention -[USES]-> parallel computation",
    ),
    HoldoutInnovation(
        name="DeepSeek-R1 (RL for Reasoning)",
        year=2025,
        papers=["2501.12948"],
        source_techniques=["reinforcement learning", "chain-of-thought prompting", "GRPO"],
        problem_solved="LLMs lack reliable multi-step reasoning",
        solution_description=(
            "Applied group relative policy optimization (RL) directly to a base LLM with only "
            "outcome-based rewards, producing emergent chain-of-thought reasoning and self-verification "
            "without any supervised reasoning data"
        ),
        subfields_combined=["RL", "NLP"],
        graph_path="reinforcement learning -[MITIGATES]-> lack of reasoning, chain-of-thought -[MITIGATES]-> reasoning errors",
    ),
    HoldoutInnovation(
        name="Latent Consistency Models (LCM)",
        year=2024,
        papers=["2310.04378"],
        source_techniques=["diffusion models", "consistency distillation", "adversarial training"],
        problem_solved="slow multi-step diffusion sampling",
        solution_description=(
            "Distilled a latent diffusion model into a consistency model that generates images in "
            "1-4 steps instead of 50+, combining consistency training with adversarial loss for quality"
        ),
        subfields_combined=["generative", "optimization"],
        graph_path="diffusion models -[HAS_LIMITATION]-> slow sampling, consistency distillation -[MITIGATES]-> slow sampling",
    ),
    HoldoutInnovation(
        name="Phi-3 (Small Language Models via Data Curation)",
        year=2024,
        papers=["2404.14219"],
        source_techniques=["knowledge distillation", "curriculum learning", "Transformer"],
        problem_solved="small models underperform on reasoning tasks",
        solution_description=(
            "Trained a 3.8B parameter model on heavily curated synthetic + filtered web data, "
            "matching GPT-3.5 on many benchmarks through data quality rather than model scale"
        ),
        subfields_combined=["optimization", "NLP"],
        graph_path="knowledge distillation -[MITIGATES]-> parameter count scaling, curriculum learning -[IMPROVES_OVER]-> random data sampling",
    ),
    HoldoutInnovation(
        name="Jamba (Hybrid SSM-Transformer-MoE)",
        year=2024,
        papers=["2403.19887"],
        source_techniques=["Mamba", "Transformer", "Mixture of Experts"],
        problem_solved="no single architecture excels at both long-range and in-context recall",
        solution_description=(
            "Interleaved Mamba SSM layers with Transformer attention layers and added MoE routing, "
            "getting best of both worlds: long context from SSM + strong recall from attention"
        ),
        subfields_combined=["signal processing", "NLP", "optimization"],
        graph_path="state space models -[MITIGATES]-> quadratic scaling, Mixture of Experts -[MITIGATES]-> computational cost",
    ),
    HoldoutInnovation(
        name="Ring Attention (Million-Token Context)",
        year=2024,
        papers=["2310.01889"],
        source_techniques=["FlashAttention", "distributed computing", "blockwise parallel attention"],
        problem_solved="memory limits context length to a single device",
        solution_description=(
            "Distributed attention computation across multiple devices in a ring topology, "
            "overlapping communication with computation via FlashAttention blocks, enabling "
            "context lengths proportional to the number of devices"
        ),
        subfields_combined=["systems", "optimization", "NLP"],
        graph_path="FlashAttention -[MITIGATES]-> memory bottleneck, distributed computing -[MITIGATES]-> single-device memory limit",
    ),
    HoldoutInnovation(
        name="Grouped Query Attention (GQA)",
        year=2024,
        papers=["2305.13245"],
        source_techniques=["multi-head attention", "multi-query attention", "key-value caching"],
        problem_solved="KV cache memory grows linearly with number of attention heads",
        solution_description=(
            "Grouped attention heads to share key-value projections (between MHA and MQA), "
            "reducing KV cache memory by 4-8x while retaining most of MHA quality"
        ),
        subfields_combined=["optimization", "NLP"],
        graph_path="multi-head attention -[HAS_LIMITATION]-> KV cache memory scaling, multi-query attention -[MITIGATES]-> KV cache memory",
    ),
    HoldoutInnovation(
        name="QLoRA (Quantized Low-Rank Adaptation)",
        year=2024,
        papers=["2305.14314"],
        source_techniques=["LoRA", "quantization", "NormalFloat data type"],
        problem_solved="fine-tuning large models requires too much GPU memory",
        solution_description=(
            "Combined 4-bit quantization of the base model with LoRA adapters in full precision, "
            "enabling fine-tuning of 65B models on a single 48GB GPU without quality degradation"
        ),
        subfields_combined=["optimization", "NLP"],
        graph_path="LoRA -[MITIGATES]-> parameter count scaling, quantization -[MITIGATES]-> memory requirements",
    ),
    HoldoutInnovation(
        name="Grounded SAM (Open-Set Segmentation)",
        year=2024,
        papers=["2401.14159"],
        source_techniques=["CLIP", "Segment Anything Model", "Grounding DINO"],
        problem_solved="segmentation models limited to fixed category sets",
        solution_description=(
            "Chained open-vocabulary object detection (Grounding DINO, powered by CLIP features) "
            "with promptable segmentation (SAM), enabling text-driven segmentation of any object"
        ),
        subfields_combined=["CV", "multimodal", "NLP"],
        graph_path="CLIP -[MITIGATES]-> fixed vocabulary, Segment Anything -[MITIGATES]-> manual annotation",
    ),
    HoldoutInnovation(
        name="LLaVA-NeXT (Improved Multimodal LLM)",
        year=2024,
        papers=["2401.16420"],
        source_techniques=["vision encoder", "LLM", "instruction tuning", "dynamic resolution"],
        problem_solved="multimodal LLMs struggle with high-resolution images and OCR",
        solution_description=(
            "Combined dynamic high-resolution image encoding with larger LLM backbone and "
            "improved visual instruction tuning data, substantially improving OCR, chart, "
            "and document understanding"
        ),
        subfields_combined=["CV", "NLP", "multimodal"],
        graph_path="CLIP -[USES]-> vision encoder, instruction tuning -[MITIGATES]-> task transfer gap",
    ),
    HoldoutInnovation(
        name="Kolmogorov-Arnold Networks (KAN)",
        year=2024,
        papers=["2404.19756"],
        source_techniques=["Kolmogorov-Arnold representation theorem", "B-splines", "MLP"],
        problem_solved="MLPs use fixed activation functions limiting expressiveness per parameter",
        solution_description=(
            "Replaced fixed activation functions in MLPs with learnable B-spline functions on "
            "edges (instead of nodes), based on the Kolmogorov-Arnold representation theorem, "
            "achieving better accuracy-per-parameter on scientific tasks"
        ),
        subfields_combined=["mathematics", "optimization"],
        graph_path="MLP -[HAS_LIMITATION]-> fixed activation functions, B-splines -[MITIGATES]-> limited function approximation",
    ),
    HoldoutInnovation(
        name="Quiet-STaR (Self-Taught Reasoner)",
        year=2024,
        papers=["2403.09629"],
        source_techniques=["chain-of-thought prompting", "self-play", "REINFORCE"],
        problem_solved="LLMs only reason when explicitly prompted to",
        solution_description=(
            "Trained LLMs to generate internal rationales (hidden reasoning tokens) at every "
            "position, reinforced by whether the rationale improved next-token prediction, "
            "teaching implicit reasoning without labeled reasoning data"
        ),
        subfields_combined=["NLP", "RL", "cognitive science"],
        graph_path="chain-of-thought -[MITIGATES]-> reasoning errors, REINFORCE -[USES]-> policy gradient",
    ),
    HoldoutInnovation(
        name="RAFT (Retrieval Augmented Fine-Tuning)",
        year=2024,
        papers=["2403.10131"],
        source_techniques=["RAG", "fine-tuning", "chain-of-thought prompting"],
        problem_solved="RAG models distracted by irrelevant retrieved documents",
        solution_description=(
            "Fine-tuned LLMs with a mix of relevant and distractor documents plus chain-of-thought "
            "answers, teaching the model to cite relevant passages and ignore noise during retrieval"
        ),
        subfields_combined=["NLP", "information retrieval"],
        graph_path="RAG -[HAS_LIMITATION]-> distractor sensitivity, fine-tuning -[MITIGATES]-> domain adaptation gap",
    ),
    HoldoutInnovation(
        name="Genie (Generative Interactive Environments)",
        year=2024,
        papers=["2402.15391"],
        source_techniques=["video prediction models", "VQ-VAE", "latent action models"],
        problem_solved="creating interactive environments requires manual game design",
        solution_description=(
            "Trained a world model on unlabeled internet video that learns a latent action space, "
            "enabling generation of playable 2D environments from a single image, bridging "
            "video generation and interactive simulation"
        ),
        subfields_combined=["generative", "RL", "CV"],
        graph_path="VQ-VAE -[USES]-> discrete latent space, video prediction -[MITIGATES]-> environment design cost",
    ),
    HoldoutInnovation(
        name="StripedHyena (Hybrid Sequence Model)",
        year=2024,
        papers=["2407.15326"],
        source_techniques=["Hyena operator", "gating mechanisms", "rotary positional embeddings"],
        problem_solved="pure SSMs underperform attention on recall-intensive tasks",
        solution_description=(
            "Alternated Hyena (long-convolution) layers with gated attention layers and rotary "
            "embeddings, achieving near-Transformer quality with sub-quadratic scaling"
        ),
        subfields_combined=["signal processing", "NLP", "optimization"],
        graph_path="Hyena -[MITIGATES]-> quadratic scaling, gating mechanisms -[MITIGATES]-> gradient flow problems",
    ),
    HoldoutInnovation(
        name="Test-Time Compute Scaling (o1/o3)",
        year=2024,
        papers=["2408.03314"],
        source_techniques=["chain-of-thought prompting", "beam search", "process reward models", "MCTS"],
        problem_solved="LLM reasoning quality limited by fixed forward pass computation",
        solution_description=(
            "Scaled compute at inference time by having the model generate and evaluate multiple "
            "reasoning chains, using process reward models to guide search, achieving strong "
            "improvements on math and coding benchmarks proportional to test-time compute spent"
        ),
        subfields_combined=["NLP", "RL", "search"],
        graph_path="chain-of-thought -[USES]-> step-by-step reasoning, beam search -[MITIGATES]-> greedy decoding errors",
    ),
]
# fmt: on

# Sanity checks
assert len(HOLDOUT_INNOVATIONS) == 20, f"Expected 20 innovations, got {len(HOLDOUT_INNOVATIONS)}"
assert all(i.year >= 2024 for i in HOLDOUT_INNOVATIONS), "All innovations must be from 2024-2025"
assert len({i.name for i in HOLDOUT_INNOVATIONS}) == 20, "Innovation names must be unique"


def get_source_techniques() -> set[str]:
    """Return all unique source technique names across innovations.

    These are the pre-2024 nodes that should exist in the KG for holdout evaluation.
    """
    techniques = set()
    for innovation in HOLDOUT_INNOVATIONS:
        techniques.update(innovation.source_techniques)
    return techniques


def get_problems_solved() -> set[str]:
    """Return all unique problem descriptions across innovations."""
    return {i.problem_solved for i in HOLDOUT_INNOVATIONS}
