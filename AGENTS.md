Repository Agent Guide (Scope: entire repo)

Purpose

- Deliver reliable, insightful knowledge graphs (KG) from uploaded research papers across domains using a hybrid extraction strategy. This file guides agents and contributors on process, checkpoints, and quality bars.

Success Criteria

- User-facing
  - Upload → parse → extract → graph renders with measurements/evidence within expected latency.
  - Edges show normalized metrics/units and confidence; evidence snippets are present.
  - Low-confidence items are queued for review, not silently dropped or accepted.
- System-level
  - No schema drift errors; invalid LLM outputs are repaired or rejected before persistence.
  - Migrations applied; indexes present; pipelines idempotent and resumable.
  - Build is stable (models cached or smaller defaults) and observability is in place.

Roles (RACI placeholders)

- Lead (A/R): owns acceptance, prioritization, and releases.
- Backend (R): services, DB schema, extraction pipeline, migrations.
- ML/NLP (R): NER/RE models, LLM prompts/schemas, normalization logic.
- Frontend (R): graph UI, uploads, review queue, insights rendering.
- DevOps (R): Docker images, caching, CI, env/secrets, monitoring.
- QA/Analyst (C): quality sets, manual review loop, lexicon upkeep.

Execution Checklist (by phase)

0) Foundations (Infra & Schema)

- [ ] Define and apply DB migrations for:
      - [ ] method/dataset/metric/task tables with `aliases jsonb`
      - [ ] result(id, paper_id, method_id, dataset_id, metric_id, split, value_numeric, unit, value_text, is_sota, confidence, evidence jsonb)
      - [ ] claim, paper_relation, concept_resolution
      - [ ] Indexes: GIN on aliases; btree on (paper_id, dataset_id, metric_id, method_id)
- [ ] Configure MinIO buckets and Qdrant collections
- [ ] Ensure `.env` contains DB, MinIO, Qdrant, and LLM settings
- Acceptance: migrations run cleanly; health endpoint OK; buckets/collections exist

1) Robust Parsing Tier

- [ ] Integrate Docling or GROBID (preferred) with PyMuPDF fallback; OCR last-resort
- [ ] Persist sections with offsets (page, char_start/end, title, text); capture table cells
- [ ] Standardize evidence JSON: `[ {section_id, page, range:[s,e], snippet, source} ]`
- Acceptance: 5 diverse PDFs parse; sections/tables persisted with offsets; failures fall back gracefully

2) Entity Extraction (NER + Linking)

- [ ] Register spaCy `en_core_web_trf` and SciSpaCy (`en_core_sci_md` default; `lg` optional)
- [ ] Add abbreviation detector and UMLS/Wikidata linker
- [ ] Implement domain lexicons (datasets, metrics, tasks, frequent methods)
- [ ] Emit entity candidates with spans, labels, canonical hints, provenance
- Acceptance: entities extracted on sample set; abbreviations resolved; candidates include provenance

3) Relation Extraction (Rules/Model + LLM Fallback)

- [ ] DependencyMatcher patterns for proposes/evaluates_on/reports
- [ ] Optional: plug a RE model (SciBERT/DyGIE++) behind a feature flag
- [ ] LLM fallback via function-calling with strict JSON Schemas and repair loop
- Acceptance: relations extracted with sentence/table evidence; invalid LLM outputs are retried/repaired

4) Schema Validation & Deterministic Verifier

- [ ] JSON Schemas for all candidate payloads (LLM enforced)
- [ ] Deterministic checks: numeric parsing, unit normalization, value-in-evidence verification, sanity bounds
- [ ] Confidence recipe by evidence source (table > rule > spacy-only > LLM)
- Acceptance: rejects/adjusts logged with reasons; numeric anomalies flagged; no invalid payloads reach DB

5) Normalization, Ontology, Canonicalization

- [ ] Canonical metric/units mapping; percent handling; range parsing
- [ ] Resolver: similarity = 0.6 cosine(name embedding) + 0.4 string (Jaro/Winkler)
- [ ] Union-find merge; persist in `concept_resolution`; backfill `aliases` to concept tables
- Acceptance: duplicates merged; aliases populated; lookups resolve to canonical IDs

6) Graph Aggregation & Insights

- [ ] Edge metadata includes measurement (value, unit), split, confidence, provenance
- [ ] Compute effect sizes where multiple outcomes exist; stash in `metadata.insights`
- [ ] Rank edges by confidence × recency × evidence strength
- Acceptance: `/api/graph/overview` returns enriched metadata; UI renders insights

7) Human-in-the-Loop QA

- [ ] Create review queue endpoint/UI for low-confidence/conflicts
- [ ] Record reviewer overrides and feed lexicon/resolver updates
- Acceptance: reviewers can correct and approve within the app; overrides persisted

8) APIs & Jobs

- [ ] Endpoints: `/api/papers`, `/api/extract/{paper_id}?tiers=1,2,3`, `/api/graph/overview`, `/api/graph/neighborhood`, `/api/claims`
- [ ] Background tasks are idempotent, resumable by content hash
- Acceptance: e2e flow works on sample PDFs with retries and resume

9) Observability & Evaluation

- [ ] Structured logs per tier with timings and rejection reasons
- [ ] Metrics: coverage of entities/relations, numeric correctness rate, latency per phase
- [ ] Gold set per domain; precision/recall eval job
- Acceptance: dashboards show green SLOs; eval report tracked over time

10) Build & Ops

- [ ] Prefer SciSpaCy `en_core_sci_md` for default builds; gate `lg` via env/flag
- [ ] Cache wheels/models; increase pip retries/timeouts or prebuild base image
- [ ] CI pipeline: run unit tests, lint, type-check; optionally subset of integration tests
- Acceptance: reproducible builds; CI green; clear troubleshooting docs

Config & Commands (quick reference)

- Required env vars: `DATABASE_URL`, `MINIO_*`, `QDRANT_URL`, `LLM_BASE_URL`, `LLM_MODEL`, `OPENAI_API_KEY` (if applicable)
- Compose:
  - Build: `docker compose build backend`
  - Run: `docker compose up -d`
  - Logs: `docker compose logs -f backend`
  - Health: `curl http://localhost:8000/health`

Testing & Validation

- Unit: normalization, schema validation, resolver, parsers on fixtures
- Integration: upload → parse → extract (tiers) → overview graph
- Quality: compare to gold set; track numeric verification pass-rate and evidence presence

Troubleshooting

- SciSpaCy model download flaky → switch to `en_core_sci_md` and/or cache wheels; rebuild without cache if needed
- Schema drift (e.g., unknown fields) → run migrations; verify models and Pydantic schemas align
- Invalid LLM JSON → confirm function-calling schema, retry/repair loop enabled, and timeouts sane
- Missing metrics/units → check normalization maps and verifier logs

Definitions of Done (DoD)

- All acceptance criteria in phases 0–6 pass on the sample set
- Observability in place; CI green; docker build/run reproducible
- README run guide accurate; AGENTS checklist reflects the current system

Non-Goals (for now)

- Training large domain models in-repo; prefer pluggable, flag-guarded models
- Full PDF figure/table semantic understanding beyond cell extraction (can be phased in)

