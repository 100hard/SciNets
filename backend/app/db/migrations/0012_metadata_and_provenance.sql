-- Ensure ontology and concept tables have flexible metadata storage
ALTER TABLE methods
    ADD COLUMN IF NOT EXISTS metadata JSONB NOT NULL DEFAULT '{}'::jsonb;
ALTER TABLE datasets
    ADD COLUMN IF NOT EXISTS metadata JSONB NOT NULL DEFAULT '{}'::jsonb;
ALTER TABLE metrics
    ADD COLUMN IF NOT EXISTS metadata JSONB NOT NULL DEFAULT '{}'::jsonb;
ALTER TABLE tasks
    ADD COLUMN IF NOT EXISTS metadata JSONB NOT NULL DEFAULT '{}'::jsonb;
ALTER TABLE concepts
    ADD COLUMN IF NOT EXISTS aliases JSONB NOT NULL DEFAULT '[]'::jsonb,
    ADD COLUMN IF NOT EXISTS metadata JSONB NOT NULL DEFAULT '{}'::jsonb;

-- Index metadata payloads for fast filtering
CREATE INDEX IF NOT EXISTS idx_methods_metadata ON methods USING GIN (metadata);
CREATE INDEX IF NOT EXISTS idx_datasets_metadata ON datasets USING GIN (metadata);
CREATE INDEX IF NOT EXISTS idx_metrics_metadata ON metrics USING GIN (metadata);
CREATE INDEX IF NOT EXISTS idx_tasks_metadata ON tasks USING GIN (metadata);
CREATE INDEX IF NOT EXISTS idx_concepts_metadata ON concepts USING GIN (metadata);
CREATE INDEX IF NOT EXISTS idx_concepts_aliases ON concepts USING GIN (aliases);

-- Add provenance to stored evidence and triples
ALTER TABLE triple_candidates
    ADD COLUMN IF NOT EXISTS provenance JSONB NOT NULL DEFAULT '{}'::jsonb;
ALTER TABLE evidence
    ADD COLUMN IF NOT EXISTS provenance JSONB NOT NULL DEFAULT '{}'::jsonb;

-- Store claim categories as free-form text
ALTER TABLE claims
    ALTER COLUMN category TYPE TEXT USING category::text;

