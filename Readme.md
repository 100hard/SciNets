3-Week MVP Implementation Plan (Final, Improved)

Project: Research Paper Knowledge Graph Platform
Core Deliverables:

PDF upload + robust parsing

Semantic search with embeddings

Interactive knowledge graph (progressive, scalable)

Q&A with retrieval baseline + optional LLM answers

Dashboard + polish

📅 WEEK 1: Infrastructure + Upload/Parse Pipeline
Day 1: Environment & Repository Setup

Backend

Initialize FastAPI app (main.py) with /health.

Configure CORS + error handling middleware.

Database models: Paper, Section, Concept, Relation, Evidence.

Config via Pydantic (config.py) for Postgres, MinIO, Qdrant.

Frontend

Init Next.js 14 (TS, Tailwind, shadcn).

Base layout: Header, Sidebar, type definitions.

Docker

Services: Postgres (pgvector), Redis (future optional), MinIO, Qdrant, FastAPI, Next.js.

Health checks + persistent volumes.

Deliverables

Containers run successfully.

/health endpoint returns 200.

Frontend loads, DB connected, MinIO buckets created.

Day 2: Database Schema & Initial API Endpoints

Setup migrations (alembic or custom scripts).

Create tables + foreign keys + indexes.

Implement API:

GET /api/papers/

POST /api/papers/

GET /api/papers/{id}

Pydantic models for requests/responses.

Deliverables

DB migrations succeed.

Paper records can be created/retrieved.

API docs show working endpoints.

Day 3: MinIO Integration & File Upload

Storage service for PDF uploads with size/type validation.

POST /api/papers/upload: store file, create paper record (status="uploaded").

GET /api/papers/{id}/download: signed URLs.

Prepare background task hook (parse_pdf_task) — can be sync for MVP.

Deliverables

Files upload & stored.

Records created with correct file paths.

Signed URLs work.

Day 4: Frontend Upload Interface

Upload page with drag-drop (react-dropzone).

Metadata form (title, authors, venue, year).

Progress bar + toast notifications.

API integration with retry logic.

Deliverables

Functional upload UI.

Progress shown.

Success/error handled gracefully.

Day 5: Robust PDF Parsing Implementation

Parsing pipeline (three-tiered):

Fast path: PyMuPDF text extraction + heading detection (regex + font-size).

Heuristic path: fallback to paragraph splitting & offsets.

Fallback path: chunk whole text into overlapping ~250–400 token blocks with offsets (section_name="_auto_chunk_").

OCR (Tesseract) offered only for text-light PDFs.

Store sections with char_start, char_end, page, snippet.

Update paper status="parsed".

Deliverables

Parser extracts clean text for most PDFs.

Sections/chunks stored with offsets.

Fallback ensures no complete failure.

Day 6: Papers List Interface

Table view of papers: title, authors, status, upload date.

Status badges (uploaded, parsed, failed).

Polling or websockets for live status updates.

Search + filters (title, author).

Deliverables

Papers list UI working.

Status updates visible.

Day 7: Integration Testing

End-to-end test: upload → parse → display.

Handle corrupted PDFs gracefully.

Optimize memory usage for large files.

Polish UI transitions.

Deliverables

Stable pipeline.

Known failure cases logged, not crashing.

📅 WEEK 2: Embeddings + Graph
Day 8: Embedding Service with Caching & Batching

Embedding service: embed_texts(batch) using all-MiniLM-L6-v2.

Hash-based caching (sha256) to skip duplicates.

Batch API calls (32–64 chunks).

Upsert to Qdrant in batches (256–512).

Link embeddings in Evidence table.

Deliverables

Efficient embedding pipeline.

Cache saves cost on duplicates.

Evidence table links chunks to vectors.

Day 9: Semantic Search

API: GET /api/search/similarity?q=... → embed query → Qdrant search.

Return snippets, scores, metadata (paper, page, offsets).

Re-rank top 3 locally (BM25 optional).

Frontend: SearchBar + SearchResults.

Deliverables

Semantic search returns relevant results.

Results show context + source.

Day 10: Concept Extraction (Tiered)

Default: scispaCy NER for scientific entities.

Fallback: small LLM (GPT-4o-mini) with structured JSON output.

Dedup + normalize concept names.

Store Concept + Relation(paper↔concept).

Deliverables

Concepts extracted & stored.

Deduplication ensures clean graph.

Day 11: Graph API (Progressive)

Graph builder:

Nodes = Papers, Concepts.

Edges = paper↔concept, concept↔concept (basic).

API endpoints:

/api/graph/overview?limit=100 (small snapshot).

/api/graph/neighborhood/{id} (progressive expansion).

JSON format ready for Cytoscape.js.

Deliverables

Graph endpoints return structured JSON.

Supports filters + progressive loading.

Day 12: Graph Visualization

Cytoscape.js + React wrapper.

Paper nodes = rectangles; concepts = circles.

Progressive expansion (click node → load neighbors).

Sidebar shows details of selected node.

Deliverables

Interactive graph works smoothly.

Expand/focus on nodes without lag.

Day 13: Graph Filters & Search

Filters: by type (dataset, method, metric), by year, by authors.

Graph search: highlight nodes by label.

Sidebar with related papers/concepts.

Deliverables

Graph UI with filters/search.

Sidebar details implemented.

Day 14: Graph Optimization

Implement clustering (group by type or degree).

Lazy-load subgraphs for large sets.

UI polish: hover tooltips, transitions, legend.

Deliverables

Graph performs with >100 nodes.

Professional visuals.

📅 WEEK 3: Q&A + Final Polish
Day 15: Q&A Backend (Tiered Strategy)

Retrieval-only baseline: always return top-K snippets.

Optional LLM generation: synthesize concise answer with inline citations.

Structured JSON output {answer, citations[]}.

Citations link to offsets.

Deliverables

/api/qa/ask works with retrieval baseline.

LLM answers available when enabled.

Day 16: Q&A Frontend

Q&A interface with input, results, history.

Answer shows inline citations (clickable).

Hover previews for citations.

Option to “ask follow-up.”

Deliverables

Q&A UI clean + functional.

Citations clickable.

Day 17: PDF Viewer with Citation Highlighting

React-pdf/pdf.js viewer.

Highlight cited offsets.

Scroll-to-citation on click.

Deliverables

Citations link to exact PDF snippet.

Viewer responsive.

Day 18: Dashboard

Show stats: papers uploaded, concepts, Q&A asked.

Recent activity timeline.

Quick action buttons (upload, search, ask).

Deliverables

Dashboard gives meaningful overview.

Navigation integrated.

Day 19: UX Polish & Error Handling

Skeleton loaders + spinners.

Global error boundary + retry options.

Responsive design + accessibility basics (ARIA, keyboard nav).

Deliverables

MVP feels professional.

Errors handled gracefully.

Day 20: Performance Optimization

Backend: caching, DB indexes, optimized queries.

Frontend: memoization, lazy load, virtual scrolling.

Graph: viewport culling, cluster rendering.

Deliverables

Fast API (<2s avg).

Graph renders 200+ nodes smoothly.

Day 21: Deployment & Demo

Deploy backend (Render/Fly.io), frontend (Vercel).

Run migrations + seed with demo papers.

Prepare demo script with curated Q&A queries.

Documentation: user guide, API docs, deployment steps.

Deliverables

Live production demo ready.

Stable sample data.

Documentation complete.

🎯 Final MVP Checklist (with improvements)
Core

 Robust PDF parsing (fast + heuristic + fallback).

 Embedding with batching + caching.

 Semantic search.

 Progressive graph visualization.

 Q&A with retrieval baseline + optional LLM.

 PDF viewer with citation highlighting.

 Dashboard.

Technical

 Dockerized infra.

 Indexed DB.

 Efficient Qdrant usage.

 Error handling + metrics.

UX

 Always return something useful (even if LLM off).

 Responsive design.

 Polished navigation + search.