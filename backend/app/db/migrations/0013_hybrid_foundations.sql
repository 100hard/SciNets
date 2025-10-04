-- Ensure measurement-centric tables match the hybrid plan foundations

-- Guarantee measurement metadata on results
ALTER TABLE IF EXISTS results
    ADD COLUMN IF NOT EXISTS unit TEXT;

ALTER TABLE IF EXISTS results
    ALTER COLUMN evidence SET DEFAULT '[]'::jsonb;

ALTER TABLE IF EXISTS results
    ALTER COLUMN evidence SET NOT NULL;

-- Recreate critical indexes if they do not exist
CREATE INDEX IF NOT EXISTS idx_results_paper_dataset_metric_method
    ON results (paper_id, dataset_id, metric_id, method_id);

CREATE INDEX IF NOT EXISTS idx_methods_aliases ON methods USING GIN (aliases);
CREATE INDEX IF NOT EXISTS idx_tasks_aliases ON tasks USING GIN (aliases);
CREATE INDEX IF NOT EXISTS idx_datasets_aliases ON datasets USING GIN (aliases);
CREATE INDEX IF NOT EXISTS idx_metrics_aliases ON metrics USING GIN (aliases);

CREATE INDEX IF NOT EXISTS idx_claims_paper_id ON claims (paper_id);
CREATE INDEX IF NOT EXISTS idx_paper_relations_src ON paper_relations (src_paper_id);
CREATE INDEX IF NOT EXISTS idx_paper_relations_dst ON paper_relations (dst_paper_id);
CREATE INDEX IF NOT EXISTS idx_concept_resolutions_alias
    ON concept_resolutions (lower(alias_text));

CREATE UNIQUE INDEX IF NOT EXISTS idx_concept_resolutions_unique
    ON concept_resolutions (resolution_type, canonical_id, lower(alias_text));
