-- Sections table captures parsed chunks from papers
CREATE TABLE IF NOT EXISTS sections (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    paper_id UUID NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    title TEXT,
    content TEXT NOT NULL,
    char_start INTEGER,
    char_end INTEGER,
    page_number INTEGER,
    snippet TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_sections_paper_id ON sections (paper_id);
CREATE INDEX IF NOT EXISTS idx_sections_paper_page ON sections (paper_id, page_number);

-- Concepts extracted from papers
CREATE TABLE IF NOT EXISTS concepts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    paper_id UUID NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    type TEXT,
    description TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_concepts_paper_id_name ON concepts (paper_id, lower(name));
CREATE INDEX IF NOT EXISTS idx_concepts_type ON concepts (type);

-- Relations link concepts within the context of a paper
CREATE TABLE IF NOT EXISTS relations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    paper_id UUID NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    concept_id UUID NOT NULL REFERENCES concepts(id) ON DELETE CASCADE,
    related_concept_id UUID REFERENCES concepts(id) ON DELETE SET NULL,
    section_id UUID REFERENCES sections(id) ON DELETE SET NULL,
    relation_type TEXT NOT NULL,
    description TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_relations_paper_id ON relations (paper_id);
CREATE INDEX IF NOT EXISTS idx_relations_concept_id ON relations (concept_id);
CREATE INDEX IF NOT EXISTS idx_relations_related_concept_id ON relations (related_concept_id);
CREATE INDEX IF NOT EXISTS idx_relations_section_id ON relations (section_id);

-- Evidence ties content back to vectors and relations for retrieval
CREATE TABLE IF NOT EXISTS evidence (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    paper_id UUID NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    section_id UUID REFERENCES sections(id) ON DELETE SET NULL,
    concept_id UUID REFERENCES concepts(id) ON DELETE SET NULL,
    relation_id UUID REFERENCES relations(id) ON DELETE SET NULL,
    snippet TEXT NOT NULL,
    vector_id TEXT,
    embedding_model TEXT,
    score DOUBLE PRECISION,
    metadata JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_evidence_paper_id ON evidence (paper_id);
CREATE INDEX IF NOT EXISTS idx_evidence_section_id ON evidence (section_id);
CREATE INDEX IF NOT EXISTS idx_evidence_concept_id ON evidence (concept_id);
CREATE INDEX IF NOT EXISTS idx_evidence_relation_id ON evidence (relation_id);

