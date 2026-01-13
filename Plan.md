# SciNets — Architecture Plan

**Working title:** SciNets / AI Scientist  
**Style:** Single-founder, "vibe coding" friendly, but with enough structure that you don't drown.

---

## 0. High-Level Overview

### Goal
Build an AI assistant that can, for multiple scientific domains:

1. Understand a research question
2. Retrieve and structure literature (multi-domain)
3. Build a small, per-session concept graph
4. Generate and rank hypotheses with structured causal chains
5. Collect literature evidence for each hypothesis
6. **User-triggered** computational experiments on specific hypotheses
7. Produce a structured, reproducible report

### Non-goals (for now)
- Heavy simulations (DFT, AutoDock-scale docking)
- Global, persistent knowledge graph over millions of papers
- Paid sandbox services (E2B, Modal)
- Fully autonomous lab (wet-lab integration, robotics, etc.)

---

## 1. Tech Stack

### 1.1 Backend
- **Language:** Python 3.11+
- **Framework:** FastAPI (REST + SSE)
- **Orchestration:** LangGraph (multi-agent workflows)
- **Schema/Types:** Pydantic models
- **LLMs:**
  - Primary reasoning: Claude 3.5 Sonnet / GPT-4.1
  - Utility/cheap tasks: Claude Haiku / GPT-4o-mini

### 1.2 Tools Layer

**Knowledge Tools:**
- OpenAlex API (core literature search)
- DuckDuckGo (web search for latest info) - *disabled on Windows*
- Optional: PubMed, arXiv, Materials Project API

**Computation Tools:**
- Template-based experiments (ML pipelines, statistical tests)
- User-triggered only via `/run_experiment` endpoint

### 1.3 Data & Storage
- PostgreSQL (sessions, hypotheses, evidence)
- Redis (caching, rate-limiting)
- In-memory concept graph (NetworkX/dicts, per-session)

### 1.4 Frontend
- Vite + React + TypeScript
- TailwindCSS + shadcn/ui
- SSE streaming for real-time agent updates
- Cytoscape.js for concept graph visualization

---

## 2. Core Architecture

### 2.1 Agent Pipeline

```
PLAN → LITERATURE → HYPOTHESIS → EVIDENCE → CRITIQUE → END
         │
         └── User can trigger: POST /run_experiment → EXPERIMENT_SUBGRAPH
```

**Key Design Decision:** Experiments are **NOT** part of the default discovery pipeline.
They are user-triggered post-discovery on specific hypotheses.

### 2.2 Agents

| Agent | Role |
|-------|------|
| **Orchestrator** | Detect domains, create research plan |
| **Literature** | Search OpenAlex, build concept graph, summarize papers |
| **Hypothesis** | Generate hypotheses with structured causal chains |
| **Evidence** | Collect supporting/contradicting evidence per hypothesis |
| **Critique** | Structural assessment (no experiments) - verdicts: Structurally Supported/Undermined |
| **Experiment** | User-triggered only, runs on specific hypothesis via separate endpoint |

### 2.3 State Models

```python
class DiscoveryState(BaseModel):
    user_query: str
    goal: str = "discover"  # discover, survey, write
    lens: str = "none"      # disciplinary lens
    speculation: str = "medium"  # low, medium, high
    mock: bool = False
    max_papers: int = 15
    
    domain_tags: List[str] = []
    plan: Optional[dict] = None
    literature: Optional[dict] = None
    concept_graph: Optional[dict] = None
    hypotheses: List[Hypothesis] = []
    critique: Optional[dict] = None
    done: bool = False
    
    # NOTE: experiments removed from default state
    # They are handled separately via ExperimentState

class ExperimentState(BaseModel):
    """State for user-triggered experiments (separate from discovery)"""
    hypothesis_id: str
    hypothesis_text: str
    intent: str = "simulate"  # simulate, sensitivity, fit
    data_source: str = "synthetic"
    seed: int = 42
    result: Optional[dict] = None
```

### 2.4 Domain Packs

Each DomainPack defines:
- `name`: "bio", "ml", "materials", "generic_ds"
- `tools`: allowed knowledge/computation tools
- `prompts`: domain-specific prompt fragments
- `constraints`: hard rules

---

## 3. API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/run_stream` | POST | Start discovery, SSE stream of agent events |
| `/run_experiment` | POST | User-triggered experiment on a hypothesis |
| `/sessions` | GET | List past sessions |
| `/docs` | GET | Swagger UI |

---

## 4. Frontend Flow

```
Query → Clarification → Execution (SSE) → Results
                              │
                              └── "Explore Computationally" button
                                  on each hypothesis card
```

**Key Components:**
- `DiscoveryQueryStep` - Enter research question
- `DiscoveryClarificationStep` - Set goal, speculation level
- `DiscoveryExecutionStep` - Real-time agent activity display
- `HypothesisCard` - Shows hypothesis with "Explore computationally" button
- `ExperimentConfigModal` - Configure and trigger experiment

---

## 5. Principles

1. Start from end-to-end flow, then deepen each step
2. Prefer small, well-defined tools over giant prompts
3. Keep the concept graph small and session-local
4. Use stronger models only for planning/hypotheses/critique
5. Always attach evidence to hypotheses
6. Experiments are explorations, not validations
7. UI should always answer:
   - What was the question?
   - What did the agents do?
   - What hypotheses did they propose?
   - What evidence backs them?
   - What should I explore next?

---

*This plan reflects the current system architecture as of January 2026.*