# SciNets

SciNets is an AI-assisted scientific discovery workspace built around literature retrieval, concept-graph construction, hypothesis generation, and evidence-backed exploration.

The project is split into two main parts:

- `backend/`: FastAPI backend that orchestrates the research workflow and auth flows
- `scinets-discovery-hub/`: React + Vite frontend for the user-facing product

## What It Does

- Searches and structures scientific literature
- Builds session-level concept graphs from retrieved evidence
- Generates research hypotheses and supporting rationale
- Streams intermediate agent activity to the frontend
- Supports authenticated user sessions and protected discovery workflows

## Tech Stack

- Backend: Python, FastAPI, LangGraph, Pydantic
- Frontend: React, TypeScript, Vite, Tailwind CSS
- Infra: Docker, Render, Vercel

## Repository Layout

```text
backend/                 FastAPI application and agent pipeline
scinets-discovery-hub/   Frontend application
docs/                    Project architecture and evaluation notes
Dockerfile               Container build for backend deployment
docker-compose.yml       Local multi-service development setup
```

## Local Development

### Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python server.py
```

### Frontend

```bash
cd scinets-discovery-hub
npm install
cp .env.example .env
npm run dev
```

## Environment Variables

Example environment files are included here:

- `backend/.env.example`
- `scinets-discovery-hub/.env.example`

Production secrets should live only in your deployment platform settings, not in git.

## Deployment

- Backend is designed to run on Render or any Docker-capable host
- Frontend is designed for Vercel or any static hosting setup that supports Vite builds

## Docs

- `docs/ARCHITECTURE.md`
- `docs/EVALUATION.md`

## Status

This repository is a cleaned public snapshot of the SciNets project codebase. Local debug outputs, generated artifacts, and sensitive runtime files have been removed from version control.
