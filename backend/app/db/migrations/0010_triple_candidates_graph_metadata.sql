ALTER TABLE triple_candidates
    ADD COLUMN IF NOT EXISTS graph_metadata JSONB NOT NULL DEFAULT '{}'::jsonb;
