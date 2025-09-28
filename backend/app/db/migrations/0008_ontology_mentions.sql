-- Mention-level storage for ontology canonicalization
CREATE TABLE IF NOT EXISTS ontology_mentions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    resolution_type concept_resolution_type NOT NULL,
    entity_id UUID NOT NULL,
    paper_id UUID NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    section_id UUID REFERENCES sections(id) ON DELETE SET NULL,
    surface TEXT NOT NULL,
    normalized_surface TEXT NOT NULL,
    mention_type TEXT,
    context_snippet TEXT,
    evidence_start INTEGER,
    evidence_end INTEGER,
    context_embedding DOUBLE PRECISION[],
    first_seen_year INTEGER,
    is_acronym BOOLEAN NOT NULL DEFAULT FALSE,
    has_digit BOOLEAN NOT NULL DEFAULT FALSE,
    is_shared BOOLEAN NOT NULL DEFAULT FALSE,
    source TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_ontology_mentions_entity ON ontology_mentions (entity_id);
CREATE INDEX IF NOT EXISTS idx_ontology_mentions_paper ON ontology_mentions (paper_id);
CREATE INDEX IF NOT EXISTS idx_ontology_mentions_surface
    ON ontology_mentions (resolution_type, normalized_surface);
