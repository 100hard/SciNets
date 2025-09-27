CREATE TABLE IF NOT EXISTS triple_candidates (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    paper_id UUID NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    section_id UUID REFERENCES sections(id) ON DELETE SET NULL,
    subject_text TEXT NOT NULL,
    relation_text TEXT NOT NULL,
    object_text TEXT NOT NULL,
    subject_span INT[] NOT NULL,
    object_span INT[] NOT NULL,
    subject_type_guess TEXT NOT NULL,
    object_type_guess TEXT NOT NULL,
    relation_type_guess TEXT NOT NULL,
    evidence_text TEXT NOT NULL,
    triple_conf DOUBLE PRECISION NOT NULL,
    schema_match_score DOUBLE PRECISION NOT NULL,
    tier TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_triple_candidates_paper ON triple_candidates (paper_id);
CREATE INDEX IF NOT EXISTS idx_triple_candidates_section ON triple_candidates (section_id);
