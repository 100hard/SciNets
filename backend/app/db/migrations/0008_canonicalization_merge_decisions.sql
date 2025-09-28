DO $$
BEGIN
    CREATE TYPE canonicalization_decision_source AS ENUM (
        'hard',
        'llm'
    );
EXCEPTION
    WHEN duplicate_object THEN NULL;
END$$;

CREATE TABLE IF NOT EXISTS canonicalization_merge_decisions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    resolution_type concept_resolution_type NOT NULL,
    left_id UUID NOT NULL,
    right_id UUID NOT NULL,
    score DOUBLE PRECISION,
    decision_source canonicalization_decision_source NOT NULL,
    verdict TEXT NOT NULL,
    rationale TEXT,
    adjudicator_metadata JSONB,
    mapping_version INT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_canonicalization_merge_decisions_resolution
    ON canonicalization_merge_decisions (resolution_type, mapping_version);
CREATE INDEX IF NOT EXISTS idx_canonicalization_merge_decisions_left
    ON canonicalization_merge_decisions (left_id);
CREATE INDEX IF NOT EXISTS idx_canonicalization_merge_decisions_right
    ON canonicalization_merge_decisions (right_id);
