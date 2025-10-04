Hybrid KG Strategy: Implementation Plan (Superset of MVP)

Overview

- Goal: generate insightful, reliable knowledge graphs from research papers across domains by combining robust structure parsers, scientific NER/RE models, schema-enforced LLM extraction, and strong normalization/aggregation.
- Why hybrid: off‑the‑shelf components (PDF parsing, base NER) provide stability; custom schema, validation, normalization, and graph aggregation provide scientific nuance and control.

Architecture (at a glance)

- Parsing: Docling or GROBID for structure (sections/tables/figures) with PyMuPDF fallback and optional OCR.
- Entities: spaCy en_core_web_trf + SciSpaCy (default en_core_sci_md for lighter builds) with UMLS/Wikidata linking and domain lexicons.
- Relations: supervised patterns (DependencyMatcher/weakly‑supervised rules) plus LLM fallback with JSON‑schema guardrails.
- Validation: schema‑first JSON validation + deterministic verifier for numeric values and units.
- Aggregation: alias resolution, measurement/effect size computation, confidence scoring, provenance.
- Feedback: human‑in‑the‑loop queue for low‑confidence or conflicting triples.

Phased Plan

0) Foundations (Infra & Data)

- Confirm DB schema supports: concept tables (method/dataset/metric/task with `aliases jsonb`), `result` (paper_id, method_id, dataset_id, metric_id, split, value_numeric, unit, evidence jsonb, confidence), `claim`, `paper_relation`, and `concept_resolution` for alias merges.
- Add migrations if missing; ensure GIN indexes on `aliases`, btree on `(paper_id, dataset_id, metric_id, method_id)`.
- Add object storage buckets (MinIO) and vector DB (Qdrant) for semantic search.

1) Robust Parsing Tier

- Integrate Docling or GROBID (preferred) for section and table/figure extraction; keep PyMuPDF as fast path and OCR as last resort.
- Persist sections with offsets: `page`, `char_start/char_end`, `title`, `text`, and capture table cell coordinates.
- Evidence anchors: standardize evidence JSON as `[ {section_id, page, range:[s,e], snippet, source:'section|table'} ]`.

2) Entity Extraction Tier (NER + Linking)

- Register two NLP pipelines with lazy load:
  - General: spaCy `en_core_web_trf` for syntax.
  - Scientific: SciSpaCy `en_core_sci_md` (switchable to `lg` in prod) with abbreviation detector + linker (UMLS/Wikidata).
- Add lexicon/rule support per domain (datasets, metrics, tasks, common methods) to boost recall and precision.
- Produce entity candidates with span, label, canonical hint, and model provenance.

3) Relation Extraction Tier (Rules + Model + LLM Fallback)

- Implement DependencyMatcher patterns for proposes/evaluates_on/reports; apply over sentence spans.
- Optional: integrate a lightweight SciBERT/DyGIE++‑style finetuned model for RE (kept modular).
- LLM fallback: extract relations in hard cases using function‑calling with strict JSON schema; retry/repair loop on invalid output.

4) Schema‑First Validation & Deterministic Verifier

- Define JSON Schemas for entity and relation candidates; validate all LLM outputs before persistence.
- Deterministic checks:
  - Parse numbers and units; normalize percentages and ranges.
  - Cross‑check reported values appear in evidence text/table cells; lower confidence or reject otherwise.
  - Sanity bounds (e.g., accuracy ≤ 100, BLEU ≤ 100) with verifier notes.

5) Normalization, Ontology, and Canonicalization

- Normalize metric names/units to canonical forms; map abbreviations via SciSpaCy linker and curated maps.
- Implement canonical resolver: score = 0.6·embedding(sim) + 0.4·string(sim); union‑find to merge aliases; persist in `concept_resolution` and backfill `aliases`.

6) Graph Aggregation & Insight Enrichment

- Edge metadata includes `measurement` (value, unit), `split`, `confidence`, and `provenance` (tiers/models).
- Compute effect sizes across methods/datasets when multiple numeric outcomes exist; surface in `metadata.insights`.
- Rank edges by confidence × recency × evidence strength; expose to frontend.

7) Human‑in‑the‑Loop QA

- Add a review queue for low‑confidence items or conflicts; enable quick approve/edit/merge with provenance.
- Track reviewer overrides and feed lexicon/resolver improvements.

8) APIs and Jobs

- Endpoints: `/api/papers`, `/api/extract/{paper_id}?tiers=1,2,3`, `/api/graph/overview`, `/api/graph/neighborhood`, `/api/claims`.
- Background tasks: parsing, NLP, validation, resolver, aggregation; idempotent and resumable via content hashes.

9) Observability & Evaluation

- Structured logs with timings per tier and rejection reasons; metrics for coverage/precision.
- Small gold set per domain; evaluate precision/recall for entities/relations and numeric correctness.

10) Build & Ops Notes

- Default to SciSpaCy `en_core_sci_md` to avoid large downloads; allow switching to `lg` in production.
- Cache model wheels if possible; increase pip retries/timeouts; or prebuild a base image with models.
- Make LLM base URL/model configurable; fail fast on missing keys.

Acceptance Checks

- Upload → parse → extract → graph endpoints complete within expected time on sample PDFs.
- Graph edges show measurements/effect sizes with evidence snippets and confidence.
- No schema drift errors; invalid LLM outputs are repaired or rejected.

Run Guide (docker compose)

- Build backend with retries and optional smaller SciSpaCy model:
  - `docker compose build backend`
  - If model downloads are flaky, rebuild or pre‑pull models; consider switching to `en_core_sci_md` in requirements.
- Start services:
  - `docker compose up -d`
- Health and quick smoke:
  - `curl http://localhost:8000/health`
  - Upload a PDF via UI or `POST /api/papers/upload`, then `POST /api/extract/{paper_id}?tiers=1,2,3`, then query `/api/graph/overview`.

What Changed vs. Previous Plan

- Explicit hybrid approach with robust parsing, schema‑enforced LLM use, deterministic numeric verification, and effect‑size aggregation.
- Reduced build fragility by recommending SciSpaCy `md` default and wheel caching.
- Clear acceptance gates and evaluation harness for quality tracking.

—

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

Default: run registered spaCy and SciSpaCy pipelines (en_core_web_trf and en_core_sci_lg) for broad coverage.

Fallback: rely on Tier-1 lexicons and heuristics; log missed spans for curator review (no LLM dependency).

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

Add a 3-tier extraction pipeline (rules/lexicons → spaCy/SciSpaCy structurer → deterministic verifier).

Add canonicalization (merge duplicates) and cross-paper linking.

Deliver Compare Papers, SOTA lookup, and Grounded Q&A (+ citation highlighting).

Keep UI responsive and simple; prefer fewer, high-quality nodes.


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

Day 15: Tier-2 spaCy/SciSpaCy Pipeline + Arbitration

Backend

Register dual NLP pipelines via settings (e.g., spaCy en_core_web_trf, SciSpaCy en_core_sci_lg) with lazy loading.

Add SciSpaCy abbreviation detector to the scientific pipeline and share a preprocessing component for normalization and citation stripping.

Implement Doc caching keyed by content hash so repeated uploads reuse parsed documents.

Run both pipelines with nlp.pipe (n_process tuned to CPUs); collect entity spans with text/start/end/label/score/model metadata.

Apply arbitration rules: prefer Tier-1 lexicon hits, resolve overlapping spans by higher score plus longer span, drop low-confidence or noise tokens.

Map surviving spans into ontology types through the mapper and record provenance tier='spacy_structurer'.

Use spaCy DependencyMatcher patterns to emit proposes/evaluates_on/reports relations with sentence-level evidence offsets.

Deliverables

POST /api/extract/{paper_id}?tiers=1,2 merges Tier-1 and Tier-2 outputs; cached docs keep re-runs fast.

Captured spans expose model and confidence; relations include evidence snippets ready for the verifier.

Day 16: Tier-3 Deterministic Verifier & Evidence Scoring

Backend

Cross-check numeric results against source sentences or table cells; require the value to appear in text or drop and reduce confidence.

Normalize metrics and units (convert percentages to decimals, handle splits, parse ranges and confidence intervals).

Flag outliers (e.g., BLEU > 100, accuracy > 1); add verifier_notes for manual review and stash rejected spans for QA.

Compute a confidence recipe based on evidence tier (tables highest, rule hits next, spaCy-only lowest) and store tier='deterministic_verifier' on adjustments.

Persist rejection logs so analysts can extend lexicons or rules when the verifier strips useful items.

Deliverables

POST /api/extract/{paper_id}?tiers=1,2,3 outputs verified entities and relations; verifier report lists adjustments and rejections.

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

Answer synthesizer (template today, optional LLM later) builds a response only from provided snippets and returns:

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

Rate-limit extraction jobs and spaCy workers; batch pipeline tasks.

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

### Graph insight fields

Graph node responses now surface richer metadata for trend analysis:

* `metadata.papers_by_year` – array of `{year, paper_count}` objects summarizing supporting papers per calendar year.
* `metadata.best_outcome` / `metadata.worst_outcome` – contextualized result payloads including metric name, dataset, numeric/text value, verification status, and provenance identifiers.

Graph edges expose Tier-3 grounded summaries alongside existing evidence:

* `metadata.insights` – list of concise natural-language findings (`summary`, `paper_id`, `paper_title`, `paper_year`, `confidence`, `claim_text`, `metric`, `dataset`, `task`, `value_text`, `value_numeric`).

Clients should treat these fields as additive signals to drive UI highlights or downstream analytics; the core schema (`papers`, `contexts`, `evidence`) remains unchanged.

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
