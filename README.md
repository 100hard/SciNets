# SciNets

SciNets is a graph-constrained scientific discovery system for literature synthesis, multi-hop reasoning, and hypothesis generation.

Instead of treating scientific exploration as plain retrieval or free-form text generation, SciNets builds a local concept graph from literature, reasons over that structure, and uses the graph to generate more grounded mechanistic hypotheses.

## Why SciNets

Scientific questions often require connecting ideas spread across multiple papers rather than summarizing a single source. SciNets is built for that setting.

It focuses on:

- literature-first discovery
- graph-backed multi-hop reasoning
- hypothesis generation with structural grounding
- evidence-aware synthesis rather than unconstrained brainstorming

## Core Workflow

```text
User Query
   ->
Literature Retrieval
   ->
Concept / Relation Extraction
   ->
Session-Level Concept Graph
   ->
Multi-Hop Path Reasoning
   ->
Hypothesis + Evidence Synthesis
```

## What the System Does

- Searches and structures scientific literature
- Builds session-level concept graphs from retrieved evidence
- Identifies bridge paths between concepts that rarely co-occur directly
- Generates hypotheses from graph structure and supporting context
- Streams intermediate reasoning activity to the frontend
- Supports authenticated user workflows for interactive discovery

## Research Paper

SciNets is based on the paper:

**SciNets: Graph-Constrained Multi-Hop Reasoning for Scientific Literature Synthesis**  
Sauhard Dubey, 2025  
[Read on arXiv](https://arxiv.org/abs/2601.09727)

If you are exploring the repo for research context, this paper is the best high-level description of the motivation, evaluation framing, and system behavior.

## Repository Structure

```text
backend/                 FastAPI backend, agent pipeline, tools, auth, storage
scinets-discovery-hub/   React + Vite frontend
docs/                    Architecture and evaluation notes
Dockerfile               Backend deployment image
docker-compose.yml       Local multi-service development setup
```

## Tech Stack

- Backend: Python, FastAPI, LangGraph, Pydantic
- Frontend: React, TypeScript, Vite, Tailwind CSS, shadcn/ui
- Data / infra: PostgreSQL, Docker
- Deployment: Render for backend, Vercel for frontend

## Local Development

### 1. Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python server.py
```

The backend expects configuration through environment variables. See `backend/.env.example`.

### 2. Frontend

```bash
cd scinets-discovery-hub
npm install
cp .env.example .env
npm run dev
```

The frontend expects configuration through environment variables. See `scinets-discovery-hub/.env.example`.

## Environment Variables

Example env files are included:

- `backend/.env.example`
- `scinets-discovery-hub/.env.example`

Production secrets should only be set in deployment platform settings and should never be committed to git.

Important production variables typically include:

- backend API keys and auth secrets
- database connection settings
- Google OAuth client configuration
- frontend API base URL

## Deployment

SciNets is structured as a split deployment:

- Backend: Docker-based deployment suitable for Render
- Frontend: static Vite deployment suitable for Vercel

The public repository intentionally keeps deployment configuration but excludes private runtime secrets and generated local artifacts.

## Documentation

- [Architecture Notes](./docs/ARCHITECTURE.md)
- [Evaluation Notes](./docs/EVALUATION.md)

## Current Scope

SciNets is best understood as a research-grade discovery and ideation system, not a fully autonomous scientific truth engine.

It is strongest when:

- the retrieved corpus is compact and relevant
- the reasoning path is structurally meaningful
- a human remains in the loop to validate important conclusions

## Repository Status

This public repository is a cleaned version of the project codebase. Sensitive local outputs, generated debug artifacts, and non-essential private runtime files were removed before publication.
