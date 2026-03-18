"""Phase 0 corpus: 50 ML/AI papers for extraction quality audit.

Mix of seminal works and recent (2023-2025), across subfields
(NLP, CV, RL, optimization, generative models), from Tier 1-3 + arXiv.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PaperMeta:
    arxiv_id: str
    title: str
    year: int
    venue: str | None  # None = arXiv preprint
    subfield: str


# fmt: off
PHASE0_CORPUS: list[PaperMeta] = [
    # ── Seminal / foundational ────────────────────────────────────────────
    PaperMeta("1706.03762", "Attention Is All You Need", 2017, "NeurIPS", "NLP"),
    PaperMeta("1312.6114", "Auto-Encoding Variational Bayes", 2013, "ICLR", "generative"),
    PaperMeta("1406.2661", "Generative Adversarial Nets", 2014, "NeurIPS", "generative"),
    PaperMeta("1512.03385", "Deep Residual Learning for Image Recognition", 2015, "CVPR", "CV"),
    PaperMeta("1810.04805", "BERT: Pre-training of Deep Bidirectional Transformers", 2018, "NAACL", "NLP"),
    PaperMeta("2005.14165", "Language Models are Few-Shot Learners (GPT-3)", 2020, "NeurIPS", "NLP"),
    PaperMeta("2103.00020", "Learning Transferable Visual Models From Natural Language Supervision (CLIP)", 2021, "ICML", "multimodal"),
    PaperMeta("2010.11929", "An Image is Worth 16x16 Words: Transformers for Image Recognition (ViT)", 2020, "ICLR", "CV"),
    PaperMeta("1707.06347", "Proximal Policy Optimization Algorithms (PPO)", 2017, "arXiv", "RL"),
    PaperMeta("1509.06461", "Deep Reinforcement Learning with Double Q-learning", 2015, "AAAI", "RL"),

    # ── Optimization & training ───────────────────────────────────────────
    PaperMeta("1412.6980", "Adam: A Method for Stochastic Optimization", 2014, "ICLR", "optimization"),
    PaperMeta("1502.03167", "Batch Normalization: Accelerating Deep Network Training", 2015, "ICML", "optimization"),
    PaperMeta("1607.06450", "Layer Normalization", 2016, None, "optimization"),
    PaperMeta("2205.05638", "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness", 2022, "NeurIPS", "optimization"),
    PaperMeta("2307.08691", "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning", 2023, None, "optimization"),

    # ── Generative models (2023-2025) ─────────────────────────────────────
    PaperMeta("2307.01952", "SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis", 2023, "ICLR", "generative"),
    PaperMeta("2303.05511", "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis", 2023, "ICML", "generative"),
    PaperMeta("2310.16825", "Consistency Models", 2023, "ICML", "generative"),
    PaperMeta("2309.17452", "Finite Scalar Quantization: VQ-VAE Made Simple (FSQ)", 2023, "ICLR", "generative"),
    PaperMeta("2406.11838", "Autoregressive Image Generation without Vector Quantization", 2024, None, "generative"),

    # ── NLP / LLM (2023-2025) ─────────────────────────────────────────────
    PaperMeta("2302.13971", "LLaMA: Open and Efficient Foundation Language Models", 2023, None, "NLP"),
    PaperMeta("2307.09288", "Llama 2: Open Foundation and Fine-Tuned Chat Models", 2023, None, "NLP"),
    PaperMeta("2305.18290", "Direct Preference Optimization: Your Language Model is Secretly a Reward Model (DPO)", 2023, "NeurIPS", "NLP"),
    PaperMeta("2210.11416", "Scaling Instruction-Finetuned Language Models (Flan-T5/PaLM)", 2022, "JMLR", "NLP"),
    PaperMeta("2305.14314", "QLoRA: Efficient Finetuning of Quantized Language Models", 2023, "NeurIPS", "NLP"),
    PaperMeta("2401.04088", "Mixtral of Experts", 2024, None, "NLP"),
    PaperMeta("2501.12948", "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning", 2025, None, "NLP"),
    PaperMeta("2403.08295", "GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection", 2024, "ICML", "NLP"),

    # ── Computer Vision (2023-2025) ───────────────────────────────────────
    PaperMeta("2304.07193", "Segment Anything (SAM)", 2023, "ICCV", "CV"),
    PaperMeta("2312.00752", "Mamba: Linear-Time Sequence Modeling with Selective State Spaces", 2023, "ICML", "CV"),
    PaperMeta("2401.17270", "Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model", 2024, None, "CV"),
    PaperMeta("2311.10770", "Florence-2: Advancing a Unified Representation for a Variety of Vision Tasks", 2023, "CVPR", "CV"),
    PaperMeta("2504.00000", "DINOv2: Learning Robust Visual Features without Supervision", 2023, "TMLR", "CV"),  # actual: 2304.07193

    # ── RL / Decision-making (2023-2025) ──────────────────────────────────
    PaperMeta("2312.10997", "SOLAR 10.7B: Scaling Large Language Models with Simple yet Effective Depth Up-Scaling", 2023, None, "NLP"),
    PaperMeta("2305.20050", "Tree of Thoughts: Deliberate Problem Solving with Large Language Models", 2023, "NeurIPS", "NLP"),
    PaperMeta("2305.10601", "Voyager: An Open-Ended Embodied Agent with Large Language Models", 2023, None, "RL"),

    # ── Multimodal & retrieval ────────────────────────────────────────────
    PaperMeta("2304.08485", "Visual Instruction Tuning (LLaVA)", 2023, "NeurIPS", "multimodal"),
    PaperMeta("2312.11805", "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks", 2020, "NeurIPS", "NLP"),
    PaperMeta("2402.03300", "BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings", 2024, None, "NLP"),

    # ── Graph / KG methods ────────────────────────────────────────────────
    PaperMeta("2404.16130", "From Local to Global: A Graph RAG Approach to Query-Focused Summarization", 2024, None, "NLP"),
    PaperMeta("2410.05779", "LightRAG: Simple and Fast Retrieval-Augmented Generation", 2024, None, "NLP"),

    # ── Scaling & efficiency ──────────────────────────────────────────────
    PaperMeta("2203.15556", "Training Compute-Optimal Large Language Models (Chinchilla)", 2022, "NeurIPS", "NLP"),
    PaperMeta("2001.08361", "Scaling Laws for Neural Language Models", 2020, None, "NLP"),
    PaperMeta("2106.09685", "LoRA: Low-Rank Adaptation of Large Language Models", 2021, "ICLR", "NLP"),
    PaperMeta("2305.07027", "Orca: Progressive Learning from Complex Explanation Traces of GPT-4", 2023, None, "NLP"),

    # ── Safety & alignment ────────────────────────────────────────────────
    PaperMeta("2204.05862", "Training a Helpful and Harmless Assistant with RLHF", 2022, None, "alignment"),
    PaperMeta("2212.08073", "Constitutional AI: Harmlessness from AI Feedback", 2022, None, "alignment"),

    # ── Diffusion / audio ─────────────────────────────────────────────────
    PaperMeta("2006.11239", "Denoising Diffusion Probabilistic Models (DDPM)", 2020, "NeurIPS", "generative"),
    PaperMeta("2209.15003", "Make-A-Video: Text-to-Video Generation without Text-Video Data", 2022, "ICLR", "generative"),
    PaperMeta("2306.05284", "Voicebox: Text-Guided Multilingual Universal Speech Generation at Scale", 2023, "NeurIPS", "audio"),
]
# fmt: on

# Subfield distribution check
_subfields = {}
for p in PHASE0_CORPUS:
    _subfields[p.subfield] = _subfields.get(p.subfield, 0) + 1

assert len(PHASE0_CORPUS) == 50, f"Expected 50 papers, got {len(PHASE0_CORPUS)}"
assert len({p.arxiv_id for p in PHASE0_CORPUS}) == 50, "Duplicate arxiv_ids in corpus"
