# Kosa

A **constantly-updating knowledge graph of research, code, tools, and techniques** that agents traverse to generate SOTA research improvements, novel project ideas, and solutions to stated problems. New additions propagate through the graph automatically. Users can also query it directly — "what are current bottlenecks for X?", "what could solve problem Y?", "what tech improves startup idea Z?"

**Phase 1 is a beachhead: ML/AI papers from arXiv.** GitHub repos, tools, frameworks, and cross-domain sources are Phase 3 — but the schema, source-confidence model, and agent interfaces are designed for them from day 1.

## The Problem

Research improvements often take years to surface. Finite Scalar Quantization (FSQ, 2023) simplified discrete representation learning but suffered from residual magnitude decay in multi-stage settings — subsequent quantization stages received exponentially weaker signals, making them useless. The fix, R-RFSQ (2025), applied learnable scaling factors and invertible LayerNorm to maintain signal strength across stages — known techniques, applied in a novel configuration. That connection took two years to make. Meanwhile, AI models like Gemini Deep Think 3 have shown they can identify obscure cross-domain connections (e.g., linking Steiner trees to the Kirszbraun extension theorem) that human experts overlooked.

This system automates that process: build a knowledge graph of techniques, limitations, capabilities, repos, and tools, then have agents traverse it to find improvements, novel combinations, and answers to user questions.

## How It Works

### 1. Knowledge Graph Construction
Ingest papers (Phase 1), then repos/tools/frameworks (Phase 3). Extract structured nodes (techniques, capabilities, limitations, systems) and typed edges (introduces, mitigates, has_limitation, improves_over, uses) using tiered LLM extraction. Every edge carries source-confidence metadata. Store in Neo4j with vector embeddings.

### 2. Neuron Activation Pattern
When a new node enters the graph, a **propagation-on-insert** background job fires an activation wave using a typed random walk:

```
New node: "Gemini Embedding 2" (multimodal embeddings)
  → activates: "cross-modal search" (capability)
    → activates: "file search limited to filenames" (problem)
      → activates: "macOS Spotlight" (system with that limitation)
        → queues hypothesis: "Semantic file search across all media types"
```

Edge types carry different transition weights — `mitigates` edges propagate activation more strongly than `cites` edges. This is what makes the graph a *living* object: additions have consequences without waiting for a user to ask.

### 3. Four Agent Surfaces (same substrate, different traversal patterns)
All four agents share the activation + falsification + ranking pipeline. Each returns ranked results with full graph-path provenance.

- **`HypothesisAgent`** — "what research improvements does this graph imply?" Surfaces hypotheses from new-node activation waves and from explicit triggers. *(Phase 1)*
- **`BottleneckFinder(topic)`** — "what are the current bottlenecks for RAG?" Seeds activation on topic nodes, follows `has_limitation` / `caused_by` edges, ranks by unresolved-ness. *(Phase 1.5)*
- **`SolutionProposer(problem)`** — "I have this problem, what could solve it?" Embeds the problem, matches by `bottleneck_class`, traverses `mitigates` edges backwards to candidate techniques/tools. *(Phase 1.5)*
- **`ProjectImprover(idea)`** — "I have this startup idea, what research/tools improve it?" LLM decomposes the idea into components, resolves each to graph nodes, surfaces techniques/repos/papers that strengthen weak components. *(Phase 1.5)*

### 4. Ranking & Filtering
Results are ranked via pairwise Elo tournament, with **structural feasibility checks** (do prerequisite techniques exist? are component papers co-cited? do required datasets/repos exist?) rather than unreliable LLM subjective scoring. A **falsification gate runs before generation** — prerequisites, contradictions, "already tried" signals, and implementation constraints are checked on the graph path itself, so the agent never wastes tokens on paths that are already dead.

## Architecture

```
  Sources            Graph                 Substrate                   Agents                   Output
┌───────────┐      ┌───────┐      ┌──────────────────────┐      ┌─────────────────────┐      ┌───────────┐
│ arXiv  P1 │      │       │      │  Typed random walk   │  ──▶ │ HypothesisAgent     │      │           │
│ GitHub P3 │ ───▶ │ Neo4j │ ───▶ │        +             │  ──▶ │ BottleneckFinder    │ ───▶ │  Ranked   │
│ Tools  P3 │      │ + vec │      │  Falsification gate  │  ──▶ │ SolutionProposer    │      │  results  │
│ Repos  P3 │      │       │      │        +             │  ──▶ │ ProjectImprover     │      │  + paths  │
└───────────┘      └───────┘      │  Elo ranking         │      └─────────────────────┘      └───────────┘
      │                ▲          └──────────────────────┘                ▲
      │                │                     ▲                            │
      │ new-node ──────┘                     │                            │
      │ insert         fresh-implications queue ◀── propagation-on-insert (Phase 2)
      └──────────────────────────────────────────────────────────────────────
                                                         FastAPI ───▶ React + Sigma.js
```

**Backend Modules:**
- `ingestion/` — Source adapters (arXiv in Phase 1; GitHub/tools in Phase 3), citation fetching (Semantic Scholar), LLM extraction (cheap model), streaming incremental ingestion with watermark tracking, propagation-on-insert queue
- `graph/` — Neo4j schema (open to new node types), CRUD, batch loader, Leiden clustering, per-source confidence weights
- `entity_resolution/` — First-class dedup with own precision/recall metrics
- `activation/` — Typed random walk, embedding similarity, blast radius computation, falsification layer
- `agents/` — Four agent surfaces sharing the activation substrate (hypothesis, bottleneck-finder, solution-proposer, project-improver); uses expensive model only for final synthesis
- `ranking/` — Pairwise Elo ranking, structural feasibility, novelty detection, shared human-feedback store
- `api/` — FastAPI routes: graph query, four agent endpoints, activation, stats, streaming ingestion, rate limiting
- `web/` — React + TypeScript UI with Sigma.js graph visualization (details under Tech Stack)

## Tech Stack

**Core (Phase 1):**
- **Python 3.11+** (src-layout package: `kosa`)
- **Neo4j** — Graph storage with MERGE operations
- **nano-graphrag** — KG construction (~1100 lines Python)
- **GPT-4o-mini** — Entity/relation extraction (cheap, high volume)
- **GPT-4o** — Hypothesis generation, ranking, and extraction quality evaluation

**Web Layer (Phase 2):**
- **FastAPI** + **Uvicorn** — REST API with async Neo4j driver, CORS, rate limiting
- **React 18 + TypeScript** (in `web/`) — dark/light mode, keyboard shortcuts (`/` or Ctrl+K to search, `1/2/3` for tabs, Escape to close), graph controls (depth slider, double-click to expand, drag-based interactive layout)
- **Sigma.js** — interactive graph viz: pan/zoom/hover, filter panel (node types, edge types, confidence), circular initial layout with force-directed refinement
- **Recharts** — dashboard charts and statistics
- **python-igraph** — optional ForceAtlas2 layout pre-computation via `compute_layout.py` (deferred to Phase 2+)

## Graph Schema

**Phase 1 node types:** `Paper`, `Technique`, `Problem`, `Dataset`

**Designed-for node types (slot in without rework):** `Tool`, `Repo`, `Framework`, `System`, `Capability`. Registry is intentionally open so Phase 3 source expansion is additive, not a rewrite.

**Edge types:**
- Ground truth: `CITES` (paper→paper)
- High confidence: `INTRODUCES` (paper→technique), `EVALUATES_ON` (paper→dataset)
- Medium confidence: `HAS_LIMITATION`, `MITIGATES`, `IMPROVES_OVER`, `USES`, `IS_INSTANCE_OF`, `CAUSED_BY`, `TEMPORALLY_FOLLOWS` (all technique/problem→technique/problem)
- Entity resolution: `SAME_AS` (confidence-scored)
- Phase 3 additions: `IS_IMPLEMENTATION_OF` (repo/tool→technique), `DEPENDS_ON` (tool→tool/framework), `SOLVES` (tool→problem)

All edges carry provenance: confidence score, source paper, source venue, venue weight, supporting text excerpt. Source-confidence tiers are per-edge metadata from day 1 — when GitHub/HN arrive, they just add rows to the weight table.

## Roadmap

**Phase 0** — Extraction quality audit on 50 papers. Quality gate: 100% schema conformance, >80% entity accuracy, >80% relation accuracy (LLM-as-judge). **COMPLETE** — see `docs/PHASE0_AUDIT_REPORT.md`.

**Phase 1** — Static KG from 2K ML/AI papers. Typed random walk activation. Single hypothesis agent. Human feedback from day 1. Temporal holdout validation against known 2024-2025 innovations.

**Phase 1.5** — Multi-agent debate on same static KG. Pairwise Elo ranking. Falsification gates. **Add three user-prompt agents:** bottleneck-finder, solution-proposer, project-improver (share activation + falsification substrate).

**Phase 2** — FastAPI web service + React frontend. Streaming arXiv ingestion with LightRAG incremental updates. **Propagation-on-insert** — new-node arrival triggers an activation wave that populates a fresh-implications queue (this is what makes "constantly updating" real, not just "constantly growing"). Scaled evaluation.

**Phase 3** — Multi-source ingestion: GitHub repos, tools, frameworks, product launches with per-source confidence weights. New node types (`tool`, `repo`, `framework`, `system`) go live. Cross-domain validation benchmark (20 curated transfers, ≥25% recall target). Feedback-driven self-improvement (ranking-weight tuning via contextual bandit on accumulated thumbs-up/down data). Expand beyond ML/AI.

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
