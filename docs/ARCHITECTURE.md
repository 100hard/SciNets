# SciNets Architecture Overview

SciNets is an agentic scientific discovery system with a React frontend and a FastAPI backend.

## High-Level Structure

- `scinets-discovery-hub/`: user-facing frontend
- `backend/app/`: API, orchestration, tools, and agent logic

## Frontend

The frontend is built with Vite, React, TypeScript, Tailwind CSS, and shadcn/ui.

Key responsibilities:

- Collect research questions from the user
- Display discovery progress and streamed agent events
- Present generated hypotheses and graph-backed outputs
- Handle authentication-protected views

## Backend

The backend is built with FastAPI and LangGraph.

Key responsibilities:

- Accept research queries
- Search and summarize literature
- Build concept graphs from papers and extracted relationships
- Generate and critique hypotheses
- Maintain user sessions and auth flows

## Discovery Flow

The backend follows an agent pipeline similar to:

`Plan -> Literature -> Hypothesis -> Evidence -> Critique -> Decision`

Important behavior:

- literature retrieval is evidence-first
- concept graphs are built per session
- experiments are kept separate from the default discovery path

## Core Mechanisms

### Concept Graphs

SciNets converts unstructured literature into graph-like structures of concepts and relationships. This allows the system to reason across bridges and gaps rather than only summarizing retrieved text.

### Hypothesis Generation

The system explores graph structure to surface potentially novel scientific connections, then turns those into candidate hypotheses with causal framing and supporting evidence.

### Streaming UX

The backend emits intermediate events so the frontend can show progress while discovery is running.

## Deployment Shape

- Backend: Docker-based deployment, suitable for Render
- Frontend: static build deployment, suitable for Vercel
