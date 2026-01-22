# SciNets Codebase Overview

## 1. High-Level Architecture
SciNets is a **Agentic Scientific Discovery System** composed of:
- **Frontend**: A modern React/Vite application (`scinets-discovery-hub`) that provides an interactive interface for researchers.
- **Backend**: A FastAPI server (`backend/app`) that orchestrates a multi-agent system using **LangGraph**.
- **Core Logic**: The system transforms literature into a **Concept Graph**, explores it for "structural holes" (missing links), and generates novel hypotheses.

## 2. Frontend (`scinets-discovery-hub`)
Built with **Vite**, **React**, **TypeScript**, **Tailwind CSS**, and **shadcn/ui**.
- **Entry Point**: `src/App.tsx` defines routes:
  - `/`: Landing page.
  - `/discovery`: The main workspace where the user interacts with the agents.
- **State Management**: Uses `React Protocol` (likely custom hooks) and SSE (Server-Sent Events) to stream real-time updates from the backend.
- **Key Components**:
  - `DiscoveryQueryStep`: Initial user input.
  - `DiscoveryClarificationStep`: Refines the query.
  - `PaperCurationStep`: Allows users to select/deselect papers fetched from OpenAlex.
  - `DiscoveryExecutionStep`: Visualizes the agent workflow, concept graph, and generated hypotheses.

## 3. Backend (`backend/app`)
Built with **FastAPI** and **LangGraph**.
- **Entry Point**: `main.py`
  - `/search_papers`: Pre-fetches papers for curation.
  - `/run_stream`: Triggers the main discovery workflow (LangGraph).
  - `/run_experiment`: Isolated endpoint to run "experiments" on specific hypotheses.

### The Agentic Workflow (LangGraph)
Defined in `backend/app/graph.py`:
`Plan` -> `Literature` -> `Hypothesis` -> `Evidence` -> `Critique` -> `Decision`

#### 1. Literature Agent (`agents/literature.py`)
**Goal**: Build a knowledge base from partial information.
- **Search**: Queries **OpenAlex** (papers) and **DuckDuckGo** (web/latest info).
- **Processing**:
  - Refines queries using an LLM.
  - Generates an "Executive Abstract".
- **Graph Construction**:
  - Extracts Concepts (Nodes) and Relationships (Edges) from abstracts using an LLM.
  - **Normalization**: Clusters similar terms (e.g., "p53", "p53 protein") using heuristics + LLM.
  - **Densification**: Second pass to find connections between known nodes in the abstracts.
  - **Analysis**: Calculates Centrality and Betweenness (Bridges) using NetworkX.

#### 2. Hypothesis Agent (`agents/hypothesis.py`)
**Goal**: Generate novel, testable insights.
- **Explorer**: A ReAct agent that traverses the Concept Graph using tools:
  - `get_neighbors`: Inspects local connections.
  - `find_paths`: Finds causal chains between disparate concepts.
  - `get_central_nodes`: Orients itself.
- **Structural Hole Analysis**:
  - Detects "Communities" (Clusters) in the graph.
  - Identifies disconnected clusters.
  - **Novelty Engine**: Asks an LLM to "bridge" these clusters using a specific "Disciplinary Lens" (if provided).
- **Generation**:
  - Produces structured hypotheses (Mechanism, Causal Chain, Rationale).
  - Classifies stability (Stable vs. Speculative) based on graph grounding.

#### 3. Other Agents
- **Orchestrator**: Simple planner (Literature -> Hypothesis).
- **Evidence/Critique**: Validate findings against the literature (implied logic).

## 4. Key Mechanisms
- **Concept Graph**: The central data structure. It turns unstructured text into a structured network, allowing the system to "reason" about connection paths rather than just summarizing text.
- **Structural Holes**: The system explicitly looks for *what is missing* (disconnected clusters) rather than just reporting what is known.
- **Streaming**: The backend streams fine-grained events (`on_tool_start`, `on_tool_end`, `activity`) via SSE, so the frontend can show a "living" UI (e.g., "Reading paper...", "Building graph...").

## 5. Development Status
- **Simulation/Mocking**: The code has extensive "MOCK MODE" checks (`state.mock`), allowing for testing without API costs or latency.
- **Experiments**: Were previously part of the main loop but are now spun out into a user-triggered sidecar (`run_experiment`).
- **Memory**: Uses `MemorySaver` for checkpoints, configured in `graph.py` but potentially used statelessly in typical `main.py` runs.
