AI Scientist Platform — Plan

Working title: SciNets / AI Scientist
Style: Single-founder, “vibe coding” friendly, but with enough structure that you don’t drown.

0. High-Level Overview

Goal:
Build an AI assistant that can, for multiple scientific domains:

Understand a research question

Retrieve and structure literature (multi-domain)

Build a small, per-session concept graph

Generate and rank hypotheses

Collect literature evidence for each hypothesis

Run lightweight computational experiments where possible

Produce a structured, reproducible report

Non-goals (for now):

Heavy simulations (DFT, AutoDock-scale docking)

Global, persistent knowledge graph over millions of papers

Paid sandbox services (E2B, Modal)

Fully autonomous lab (wet-lab integration, robotics, etc.)

1. Tech Stack
1.1 Backend

Language: Python 3.11+

Framework: FastAPI (REST + SSE/WebSockets)

Orchestration: LangGraph (multi-agent workflows)

Schema / Types: Pydantic models

LLMs:

Primary reasoning: Claude 3.5 Sonnet / GPT-4.1

Utility / cheap tasks: Claude Haiku / GPT-4o-mini

1.2 Tools Layer

Knowledge Tools

OpenAlex API (core literature search)

Optional later: PubMed, arXiv, Materials Project API

Computation Tools

No external sandbox initially.

Phase 1:

Template-based experiments implemented by you:

ML pipelines (regression/classification)

AI Scientist Platform — Plan

Working title: SciNets / AI Scientist
Style: Single-founder, “vibe coding” friendly, but with enough structure that you don’t drown.

0. High-Level Overview

Goal:
Build an AI assistant that can, for multiple scientific domains:

Understand a research question

Retrieve and structure literature (multi-domain)

Build a small, per-session concept graph

Generate and rank hypotheses

Collect literature evidence for each hypothesis

Run lightweight computational experiments where possible

Produce a structured, reproducible report

Non-goals (for now):

Heavy simulations (DFT, AutoDock-scale docking)

Global, persistent knowledge graph over millions of papers

Paid sandbox services (E2B, Modal)

Fully autonomous lab (wet-lab integration, robotics, etc.)

1. Tech Stack
1.1 Backend

Language: Python 3.11+

Framework: FastAPI (REST + SSE/WebSockets)

Orchestration: LangGraph (multi-agent workflows)

Schema / Types: Pydantic models

LLMs:

Primary reasoning: Claude 3.5 Sonnet / GPT-4.1

Utility / cheap tasks: Claude Haiku / GPT-4o-mini

1.2 Tools Layer

Knowledge Tools

OpenAlex API (core literature search)

Optional later: PubMed, arXiv, Materials Project API

Computation Tools

No external sandbox initially.

Phase 1:

Template-based experiments implemented by you:

ML pipelines (regression/classification)

Basic statistical tests (t-test, ANOVA, correlations)

Simple bio analyses (using BioPython on toy/public datasets)

Phase 2:

Restricted local execution (Docker container) with:

Whitelisted imports (if using python image)

Timeouts & Memory limits

Phase 3:

Docker-based sandbox on your own server:

Short-lived containers with CPU/RAM limits

Vector Store

Qdrant (per-session paper embeddings, maybe experiment notes)

Concept Graph

In-memory via NetworkX or Python dicts

Per-session only (rebuilt each run)

Stored in Postgres as JSONB if needed

1.3 Data & Storage

PostgreSQL (Neon / Supabase)

Users

Sessions

Hypotheses

Evidence items

Experiment metadata

Object Storage (Cloudflare R2 or Supabase Storage)

Plots (images)

Datasets (uploaded/generated)

Code snippets/artifacts

Redis (Upstash)

Caching OpenAlex responses

Lightweight session state / rate-limiting

1.4 Frontend

Next.js 14 (App Router)

React + TailwindCSS + shadcn/ui

SSE/WebSockets from FastAPI for streaming progress

Plotly.js for plots (via react-plotly.js)

Cytoscape.js (or similar) for small concept graph visualization

1.5 Dev/Deploy

Dev:

docker-compose for backend + Postgres + Qdrant + Redis

Next.js via npm run dev locally

Deploy:

Backend: Railway / Render (container)

Frontend: Vercel

DB: Neon / Supabase

Storage: R2 / Supabase

2. Core Architecture
2.1 Domain Packs (Multi-Domain)

Each DomainPack defines:

name: "bio", "ml", "materials", "generic_ds"

tools: allowed knowledge/computation tools for that domain

prompts: domain-specific prompt fragments (hypotheses, experiments)

constraints: hard rules (e.g., “no heavy simulations”; “API-only for materials”)

MVP packs:

bio

ml

materials (API-only, no DFT)

generic_ds

2.2 State Model
class EvidenceItem(BaseModel):
    paper_id: str
    title: str
    venue: str | None
    year: int | None
    stance: Literal["support", "contradict", "neutral"]
    strength: int  # 1–5
    key_points: list[str]
    url: str | None = None

class Hypothesis(BaseModel):
    id: str
    text: str
    domain_tags: list[str]
    novelty_score: float
    feasibility_score: float
    testability_score: float
    required_data: list[str] = []
    experiment_idea: str | None = None
    evidence: list[EvidenceItem] = []

class DiscoveryState(BaseModel):
    user_query: str
    domain_tags: list[str] = []
    plan: dict | None = None
    literature: dict | None = None
    concept_graph: dict | None = None  # or pointer/id
    hypotheses: list[Hypothesis] | None = None
    selected_hypothesis_id: str | None = None
    experiments: list[dict] | None = None
    critique: dict | None = None
    critique: dict | None = None
    done: bool = False
    # Note: Keep this state lightweight. Store heavy text/vectors in DB/Qdrant.

2.3 Agents

Orchestrator Agent

Detect domains from user query (rule + small LLM call).

Load domain packs.

Call plan_research_steps(question, domain_tags) tool.

Configure which steps (nodes) to run in LangGraph.

Literature Agent

Multi-query OpenAlex search.

Filter & rank by year, relevance, citations.

Multi-query OpenAlex search.

Filter & rank by year, relevance, citations.

Two-pass processing:

1. Triage: Cheap LLM (Haiku/4o-mini) filters papers based on title/abstract.

2. Synthesis: Strong LLM (Sonnet/GPT-4) summarizes top candidates.

evaluate strength per hypothesis,

identify contradictions,

suggest explicit next experiments.

Export:

Generate markdown report from DiscoveryState.

Optional: convert to PDF server-side (or leave to client tools).

Frontend Tasks

Final UI shape:

Agent Log (toggle)

Full Literature, Concept Graph, Hypotheses, Evidence, Experiments, Final Report sections.

Improve:

Loading/skeleton states per section.

Error messages per step.

Add:

Export buttons (download markdown; PDF optional).

Session history page (/sessions).

Tests for Phase 3

Backend:

Unit tests:

Additional experiment templates run end-to-end on dummy datasets.

Critique generator:

Given known combinations of evidence + metrics, returns structured critique without crashing.

Integration tests:

For a multi-domain question:

Some hypotheses get experiments; others only evidence.

Final report contains:

per-hypothesis critique,

recommended next steps.

Frontend:

Manual tests:

Full run:

Check all sections populate and are scrollable.

Evidence and experiments update correctly for chosen hypothesis.

Export:

Markdown file is well-structured and readable.

Sessions:

Previous runs appear and reload correctly.

5. Principles to Keep in Mind While Vibe Coding

Start from end-to-end flow, then deepen each step.

Prefer small, well-defined tools (functions) over giant prompts.

Keep the concept graph small and session-local.

Use bigger models only for planning/hypotheses/critique; cheaper ones for summarization.

Always attach evidence to hypotheses, even where no experiment runs.

UI should always answer:

What was the question?

What did the agents do?

What hypotheses did they propose?

What evidence backs them?

What should I do next?

This plan is the “plan.md” you can keep in the repo root and evolve as you build.