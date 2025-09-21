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

Revised MVP Plan (Post–Day 12 to Launch)
New Core Goals (what changes from here)

Move from generic “Concept” nodes to a small research ontology: Method/Model, Task, Dataset, Metric, Result, Claim, Paper, Section.

All edges are typed and carry evidence (paper id, section id, char offsets, page, snippet, confidence).

Add a 3-tier extraction pipeline (rules/lexicons → LLM structurer → verifier).

Add canonicalization (merge duplicates) and cross-paper linking.

Deliver Compare Papers, SOTA lookup, and Grounded Q&A (+ citation highlighting).

Keep UI responsive and simple; prefer fewer, high-quality nodes.

Pre-Day-13 (Safe Cleanup Plan)

Goal: stop generating noisy, untyped graph data; keep everything that already works (upload, parse, search, the current graph UI), and lay rails for typed entities.

Create a migration branch

Branch name: feature/typed-graph-prep

Keep main deployable. All changes below are in PRs to this branch.

Backend (FastAPI) — what to freeze, remove, or gate

Freeze untyped co-occurrence edges

Stop writing concept↔concept edges created purely by co-occurrence.

Keep read APIs working, but change the write path to no-op.

Add feature flag: ENABLE_UNTYPED_GRAPH=false (env var). If true, legacy writes still work.

Keep Concepts table, but mark as transitional

Do not delete the concept table or paper↔concept edges yet.

Add a new nullable type column if it doesn’t exist: enum ('unknown','method','dataset','metric','task'). Default unknown.

This lets you backfill types for existing rows later and keeps graph rendering alive.

Deprecate “auto-label” concept extraction

If you have a job that promotes arbitrary noun-phrases to concepts, guard it behind ENABLE_NOISY_CONCEPTS=false.

Leave the code path in place to avoid breakage; simply return an empty list when the flag is false.

Keep embeddings & search as-is

Do not touch Qdrant/pgvector indices or similarity search endpoints. They’ll be reused by the typed pipeline and QA.

Stabilize Evidence storage

If you currently store evidence in a separate evidence table, keep it.

Add a provenance jsonb field (if missing): { "tier": "rule|llm|verifier", "model": "…", "rules": ["…"] }.

Add a confidence float column if missing.

Add compatibility adapters

New internal module adapters/typed_to_legacy.py: maps typed nodes/edges to the old Concept/Relation shape so the existing Cytoscape view doesn’t break while you migrate.

Frontend (Next.js) — what to freeze or gate

Keep the current Cytoscape.js view

Do not remove it. Add a URL query flag ?typed=1 to switch to the new typed data once available.

Hide “Graph search by label” if it promotes noise

If the label search currently surfaces random phrases, hide it behind NEXT_PUBLIC_ENABLE_LEGACY_GRAPH_SEARCH=false.

Do not delete any UI routes

Upload, Papers list, and Graph page stay. You’ll reuse them.

APIs — deprecations without breaking

Keep:

GET /api/graph/overview

GET /api/graph/neighborhood/{id}

Internally route these through the adapter:

If ?typed=1 or server flag PREFER_TYPED_GRAPH=true, the controller pulls from the new typed graph and adapts to the legacy JSON shape; else, it returns the legacy graph.

“Do NOT remove” checklist

Do not remove parsing, sections, offsets, Qdrant code, upload flows, or current Evidence table.

Do not drop any DB columns/tables yet; we’ll migrate/dual-write first.

Do not hard-delete concept data; we’ll mine it to bootstrap typed entities.


Day 13: Ontology & DB Migration (Typed Nodes + Evidence)

Backend

Add tables:

method(id, name, aliases jsonb, created_at)

task(id, name, aliases jsonb)

dataset(id, name, aliases jsonb)

metric(id, name, unit, aliases jsonb)

result(id, paper_id, method_id, dataset_id, metric_id, split, value_numeric decimal(8,3), value_text, is_sota boolean default false, confidence float, evidence jsonb, created_at)

claim(id, paper_id, category enum('contribution','limitation','ablation','future_work','other'), text, confidence float, evidence jsonb)

paper_relation(id, src_paper_id, dst_paper_id, type enum('cites','extends','compares'), evidence jsonb, confidence float)

concept_resolution(id, type enum('method','dataset','metric','task'), canonical_id, alias_text, score float)

Modify/keep existing:

paper(id, title, year, ...)

section(id, paper_id, title, page_start, page_end, char_start, char_end, text)

evidence can be embedded as jsonb on each typed row (keep existing Evidence table if you prefer; just ensure you can store [{"section_id":..,"range":[s,e],"page":..,"snippet":"..."}]).

Indexes

Btree/GIN on (name) for each concept table, and GIN on aliases.

Btree on (paper_id, dataset_id, metric_id, method_id) in result.

Deliverables

Migrations apply cleanly.

Old “Concept/Relation” left intact for now; new tables ready.

Day 14: Tier-1 Extractors (Patterns + Lexicons) with Evidence

Backend

Create lightweight lexicons (YAML/JSON) for the first domain (pick one: MT or CV):

datasets: e.g., WMT14 En-De, ImageNet, COCO.

metrics: e.g., BLEU, ROUGE-L, Accuracy, Top-1.

tasks: e.g., machine translation, image classification.

method keywords: e.g., Transformer, ResNet, BERT.

Regex patterns:

Results: (?P<metric>BLEU|ROUGE(?:-L)?|F1|Accuracy|Top-1)\s*(=|:)\s*(?P<val>\d+(\.\d+)?)\s*(%|pts)?

Evaluate/use: (evaluate(d)? on|tested on|trained on)\s+(?P<dataset>[A-Za-z0-9\-\+\/ ]{2,})

Proposes: we\s+(propose|introduce|present)\s+(?P<method>[A-Z0-9\-\+ ]{3,})

PDF table extractor (pdfplumber or camelot) and run result patterns on table cells.

Emit typed nodes/edges with confidence (e.g., 0.6 by default), and evidence offsets (section + char ranges + snippet).

Deliverables

Endpoint POST /api/extract/{paper_id}?tiers=1 populates method/dataset/metric/result/task/claim (claims can be empty).

Unit tests on 3–5 papers; inspect evidence and values.

Day 15: Tier-2 LLM Structurer (Strict JSON) + Pydantic Validation

Backend

For sections with poor/no Tier-1 signals, call LLM with a strict schema:

{
  "paper_title": "...",
  "methods":[{"name":"...","is_new":true,"aliases":["..."]}],
  "tasks":["..."],
  "datasets":["..."],
  "metrics":["..."],
  "results":[
    {"method":"...","dataset":"...","metric":"...","value":41.8,"split":"test",
     "evidence_span":{"section_id":123,"start":100,"end":180}}
  ],
  "claims":[{"category":"limitation","text":"...","evidence_span":{"section_id":123,"start":..., "end":...}}]
}


Validate with Pydantic; annotate each extraction with tier="llm_structurer", confidence baseline 0.7.

Store exact evidence spans provided by the model.

Deliverables

POST /api/extract/{paper_id}?tiers=1,2 augments Tier-1 output; JSON validation errors handled.

Measurable coverage increase vs Tier-1 alone.

Day 16: Tier-3 Verifier (Cross-checks & Sanity)

Backend

Cross-check LLM results with raw text:

If metric value appears verbatim near the span, +0.1 confidence; else −0.2.

Normalize metrics: convert % to numeric [0–100], F1 strings like 0.87 → 87 if unit is %.

Outlier checks (e.g., BLEU > 100) are discarded or flagged.

Mark result.verified=true/false; store verifier_notes.

Deliverables

POST /api/extract/{paper_id}?tiers=1,2,3 produces verified results; audit log shows adjustments.

Day 17: Canonicalization & Resolver (Dedup Merge)

Backend

Implement resolver:

Similarity = 0.6 * cosine(embedding(name)) + 0.4 * JaroWinkler(name).

Thresholds by type: Dataset ≥ 0.82, Metric ≥ 0.90, Method ≥ 0.85, Task ≥ 0.80.

Union-find to map aliases → canonical IDs; persist in concept_resolution.

Backfill aliases into each table’s aliases.

Deliverables

POST /api/admin/canonicalize?types=method,dataset,metric,task merges and updates foreign keys in result/claim.

Before/after report: counts of merges, examples.

Day 18: Typed Graph API (Progressive) + Confidence Filters

Backend

New endpoints (JSON for Cytoscape):

/api/graph/neighborhood/{id}?types=Method,Dataset,Metric,Task&relations=proposes,evaluates_on,reports,compares&min_conf=0.6

/api/graph/overview?limit=150&min_conf=0.6

Edge rules:

Only draw typed edges; no raw co-occurrence.

Weight = number of distinct papers × avg confidence.

Frontend

Graph filters: type toggles + confidence slider.

Node panel shows: type, aliases, used-by N papers, top linked nodes, “Why?” button listing evidence snippets.

Deliverables

Cleaner, denser graph with far fewer junk nodes; interactions stay smooth.

Day 19: Compare Papers (Overlap/Diff + Results Table)

Backend

/api/compare?papers=p1,p2[,p3...] returns:

Shared: tasks/datasets/metrics.

Unique: methods/claims per paper.

Results matrix: (dataset × metric) per method, with best value per row highlighted.

Frontend

Compare view:

“What’s new” bullets (from claims category contribution or method.is_new).

Results table with column sorting (metric/dataset).

Quick links to evidence (hover to preview snippet, click to open PDF at span).

Deliverables

Two-paper comparison demo works on seed papers; shows useful differences.

Day 20: SOTA Calculator & API

Backend

/api/sota?task=...&dataset=...&metric=...:

Returns best result with is_sota=true, method, paper, value, date, evidence.

Optionally scope by year range.

Frontend

Inline SOTA badge in concept panels and compare table.

“Show SOTA over time” sparkline (optional later).

Deliverables

SOTA lookup gives sensible answers on seed set.

Day 21: Grounded QA (Answers from the Graph + Citations)

Backend

/api/qa/ask pipeline:

Intent/type detection (simple rules + embedding match) → which tables to query.

Query graph / results tables first; fall back to vector search only if needed.

Collect top-k evidence snippets (Evidence jsonb).

LLM synthesizes an answer only from provided snippets; return:

{
  "answer":"...",
  "citations":[{"paper_id":"...", "section_id":..., "page":..., "range":[s,e]}],
  "structured":{"sota":{...}}
}


Add a “no-fabrication” guard: if no evidence, return a ranked list of snippets instead of a generated answer.

Frontend

QA panel with inline citations; click jumps the PDF viewer to span.

Deliverables

Grounded QA works for “What datasets are used for machine translation?”, “State of the art BLEU on WMT14 En-De?”, “How does Paper A differ from Paper B?”

Day 22: PDF Viewer – Evidence Highlighting & Jump

Frontend

Extend your Day-17 viewer:

Accept citations[] or evidence[] and highlight spans.

“Open in context” button on any node/edge to jump to first evidence.

Deliverables

End-to-end: Ask → click citation → PDF scrolls and highlights exact text.

Day 23: Admin Tools – Quick Fixes & Audit

Frontend

Tiny admin panel:

Merge concepts (pick canonical, absorb alias).

Edit a wrong metric value/unit; re-run verifier.

Re-extract a paper with tiers flags.

Show extraction audit (which tier produced which node/edge, model version, confidence).

Deliverables

10-minute curator pass meaningfully improves quality.

Day 24: UX Polish & Robustness

Frontend

Skeleton loaders, error toasts with “retry” on extract/compare.

Confidence slider default 0.65; “Show low-confidence” toggle.

Keyboard nav basics; accessible labels on graph filters.

Backend

Rate-limit LLM calls per minute; batch extraction tasks.

Deliverables

Smooth feel; graceful failure modes.

Day 25: Performance & Indexing

Backend

Add DB indexes for hot queries (results by dataset/metric/method).

Cache SOTA responses (keyed by task+dataset+metric).

Pre-compute degree/weights for graph overview.

Frontend

Virtualize long tables (react-virtualized) in Compare view.

Deliverables

<2s for typical graph/compare/QA queries on seed set.

Day 26: Seed, Evaluate, and Tune Thresholds

Ops

Seed 15–20 papers in a single domain (e.g., MT).

Evaluate:

Precision@k for dataset/method detection on 5 papers (manual spot check).

Resolver thresholds: adjust until duplicates are rare and merges safe.

Deliverables

Short evaluation report; thresholds updated in config.

Day 27: Dashboard & Onboarding

Frontend

Dashboard:

Totals: papers uploaded, methods, datasets, metrics, results, claims.

Activity timeline (uploads/extracts).

Quick actions (Upload, Extract, Compare, Ask).

“Start here” tour highlighting Upload → Extract → Compare → Ask.

Deliverables

Researchers can discover features quickly.

Day 28: Deployment & Demo Script

Ops

Backend on Render/Fly.io; Frontend on Vercel; MinIO/Qdrant hosted where you already run them.

Env var templates and health checks.

Demo script:

Upload 2 papers → Extract (tiers 1–3),

Compare → differences + results table,

Ask QA → click citations → PDF highlight,

Show SOTA query.

Deliverables

Live demo instance seeded with the domain set.

Updated API Surface (post-Day 12)

POST /api/extract/{paper_id}?tiers=1,2,3

POST /api/admin/canonicalize?types=...

GET /api/compare?papers=p1,p2[,p3]

GET /api/sota?task=...&dataset=...&metric=...

GET /api/graph/overview?limit=&min_conf=

GET /api/graph/neighborhood/{id}?types=&relations=&min_conf=

GET /api/search/concepts?q=&type=Dataset|Method|Metric|Task

POST /api/qa/ask (returns {answer, citations[], structured})

Acceptance Criteria (what “useful for researchers” means)

Typed graph only (no untyped co-occurrence edges).

Every edge/node has evidence you can open and highlight.

Compare Papers clearly surfaces “what’s new” and tabulates results.

SOTA queries return sensible results with provenance.

QA answers are grounded—if no evidence, you return snippets instead of fabricating.

Final MVP Checklist (revised)

Core

Robust parsing with offsets (+ table extraction).

Tiered extraction with evidence, verifier, and confidence.

Canonicalized, typed research graph.

Compare Papers view.

SOTA endpoint + UI.

Grounded QA with clickable citations.

PDF viewer with span highlighting.

Dashboard and simple onboarding.

Technical

Dockerized infra, migrations, indexes.

Qdrant for embeddings; DB for graph/results.

Caching + rate limiting; audit logs.

Thresholds configurable.

UX

Clean graph with type filters + confidence slider.

“Why?” evidence everywhere.

Responsive, accessible, error-tolerant.
