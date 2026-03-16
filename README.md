# Kosa

A knowledge graph of ML/AI research that agents traverse to generate research hypotheses, project ideas, and cross-domain discoveries. Where fields cross is where discoveries happen.

## The Problem

Research improvements often take years to surface. Finite Scalar Quantization (FSQ, 2023) simplified discrete representation learning but suffered from residual magnitude decay in multi-stage settings — subsequent quantization stages received exponentially weaker signals, making them useless. The fix, R-RFSQ (2025), applied learnable scaling factors and invertible LayerNorm to maintain signal strength across stages — known techniques, applied in a novel configuration. That connection took two years to make. Meanwhile, AI models like Gemini Deep Think 3 have shown they can identify obscure cross-domain connections (e.g., linking Steiner trees to the Kirszbraun extension theorem) that human experts overlooked.

This system automates that process: build a knowledge graph of techniques, limitations, and capabilities from ML/AI papers, then have agents traverse it to find improvements and novel combinations.

## How It Works

### 1. Knowledge Graph Construction
Ingest ML/AI papers from arXiv. Extract structured nodes (techniques, capabilities, limitations, systems) and typed edges (enables, mitigates, has_limitation, improves_over) using tiered LLM extraction. Store in Neo4j with vector embeddings.

### 2. Neuron Activation Pattern
When a new technique enters the graph, an activation wave propagates outward using a typed random walk:

```
New node: "Gemini Embedding 2" (multimodal embeddings)
  → activates: "cross-modal search" (capability)
    → activates: "file search limited to filenames" (problem)
      → activates: "macOS Spotlight" (system with that limitation)
        → generates idea: "Semantic file search across all media types"
```

Edge types carry different transition weights — `mitigates` edges propagate activation more strongly than `belongs_to` edges.

### 3. Hypothesis Generation
Agents evaluate activated paths and generate ranked hypotheses:
- Does a new capability solve an existing limitation?
- Can two techniques be combined in a novel way?
- Does a new release obsolete an existing approach?

### 4. Ranking & Filtering
Hypotheses are ranked via pairwise Elo tournament, with structural feasibility checks (do prerequisite techniques exist in the graph? are component papers co-cited? do required datasets exist?) rather than unreliable LLM subjective scoring.

## Architecture

```
Ingestion → Graph Store → Activation Wave → Agent Evaluation → Ranked Output
  (arXiv)    (Neo4j)    (typed walk+embeds) (single/multi)    (hypotheses)
```

**Modules:**
- `ingestion/` — Paper parsing, entity/relation extraction via LLM (cheap model)
- `graph/` — Neo4j schema, CRUD, Leiden clustering
- `entity_resolution/` — First-class dedup with own precision/recall metrics
- `activation/` — Typed random walk, embedding similarity, blast radius computation
- `agents/` — Hypothesis and project idea generation (expensive model)
- `ranking/` — Pairwise Elo ranking, structural feasibility, novelty detection

## Tech Stack (Phase 1)

- **Python 3.11+**
- **Neo4j** — Graph storage with MERGE operations
- **nano-graphrag** — KG construction (~1100 lines Python)
- **DeepSeek V3.2** — Entity extraction (cheap, high volume)
- **Claude Sonnet** — Hypothesis generation + ranking (expensive, low volume)

## Graph Schema

**Node types:** `capability`, `system`, `technique`, `problem`, `domain`, `benchmark`

**Edge types:** `enables`, `has_limitation`, `mitigates`, `improves_over`, `is_instance_of`, `used_in`, `belongs_to`, `evaluated_on`, `caused_by`, `temporally_follows`, `same_as`

## Roadmap

**Phase 0** — Extraction quality audit on 50 papers. Must hit >80% precision, >65% recall before proceeding.

**Phase 1** — Static KG from 2K ML/AI papers. Typed random walk activation. Single agent. Human feedback from day 1. Temporal holdout validation against known 2024-2025 innovations.

**Phase 1.5** — Batch multi-agent debate on same static KG. Pairwise Elo ranking. Falsification gates.

**Phase 2** — Streaming arXiv ingestion. LightRAG incremental updates. Scaled evaluation.

**Phase 3** — Feedback-driven self-improvement. Expand beyond ML/AI domain.

## Prior Art

This project builds on ideas from:
- [SciAgentsDiscovery](https://github.com/lamm-mit/SciAgentsDiscovery) — Multi-agent KG traversal for scientific discovery
- [The AI Scientist](https://github.com/SakanaAI/AI-Scientist) — Fully automated research pipeline
- [Graph of AI Ideas](https://arxiv.org/html/2503.08549v1) — KG-based research idea generation with beam search
- [HypoChainer](https://arxiv.org/abs/2507.17209) — Chain-of-hypothesis via KG bridge nodes
- [SA-RAG](https://arxiv.org/abs/2512.15922) — Spreading activation for KG-based retrieval
- [Graphiti](https://github.com/getzep/graphiti) — Temporal knowledge graphs for AI agents
- [AGATHA](https://github.com/JSybrandt/agatha) — Biomedical hypothesis prediction via KG link prediction

## License

TBD
