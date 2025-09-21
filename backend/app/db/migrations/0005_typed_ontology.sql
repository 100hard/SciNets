-- Enums for typed research structures
DO $$
BEGIN
    CREATE TYPE claim_category AS ENUM (
        'contribution',
        'limitation',
        'ablation',
        'future_work',
        'other'
    );
EXCEPTION
    WHEN duplicate_object THEN NULL;
END$$;

DO $$
BEGIN
    CREATE TYPE paper_relation_type AS ENUM (
        'cites',
        'extends',
        'compares'
    );
EXCEPTION
    WHEN duplicate_object THEN NULL;
END$$;

DO $$
BEGIN
    CREATE TYPE concept_resolution_type AS ENUM (
        'method',
        'dataset',
        'metric',
        'task'
    );
EXCEPTION
    WHEN duplicate_object THEN NULL;
END$$;

-- Core ontology entities
CREATE TABLE IF NOT EXISTS methods (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT NOT NULL,
    aliases JSONB NOT NULL DEFAULT '[]'::jsonb,
    description TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS tasks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT NOT NULL,
    aliases JSONB NOT NULL DEFAULT '[]'::jsonb,
    description TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS datasets (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT NOT NULL,
    aliases JSONB NOT NULL DEFAULT '[]'::jsonb,
    description TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT NOT NULL,
    unit TEXT,
    aliases JSONB NOT NULL DEFAULT '[]'::jsonb,
    description TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Result statements tie together ontology entities with evidence
CREATE TABLE IF NOT EXISTS results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    paper_id UUID NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    method_id UUID REFERENCES methods(id) ON DELETE SET NULL,
    dataset_id UUID REFERENCES datasets(id) ON DELETE SET NULL,
    metric_id UUID REFERENCES metrics(id) ON DELETE SET NULL,
    task_id UUID REFERENCES tasks(id) ON DELETE SET NULL,
    split TEXT,
    value_numeric NUMERIC(8,3),
    value_text TEXT,
    is_sota BOOLEAN NOT NULL DEFAULT FALSE,
    confidence DOUBLE PRECISION,
    evidence JSONB NOT NULL DEFAULT '[]'::jsonb,
    verified BOOLEAN,
    verifier_notes TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Claims summarise findings with typed evidence
CREATE TABLE IF NOT EXISTS claims (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    paper_id UUID NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    category claim_category NOT NULL,
    text TEXT NOT NULL,
    confidence DOUBLE PRECISION,
    evidence JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Paper-to-paper relationships with citations and provenance
CREATE TABLE IF NOT EXISTS paper_relations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    src_paper_id UUID NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    dst_paper_id UUID NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    relation_type paper_relation_type NOT NULL,
    confidence DOUBLE PRECISION,
    evidence JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Canonicalisation mappings between raw aliases and ontology entities
CREATE TABLE IF NOT EXISTS concept_resolutions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    resolution_type concept_resolution_type NOT NULL,
    canonical_id UUID NOT NULL,
    alias_text TEXT NOT NULL,
    score DOUBLE PRECISION,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes for fast lookup of ontology concepts and resolutions
CREATE INDEX IF NOT EXISTS idx_methods_name ON methods (lower(name));
CREATE INDEX IF NOT EXISTS idx_tasks_name ON tasks (lower(name));
CREATE INDEX IF NOT EXISTS idx_datasets_name ON datasets (lower(name));
CREATE INDEX IF NOT EXISTS idx_metrics_name ON metrics (lower(name));

CREATE INDEX IF NOT EXISTS idx_methods_aliases ON methods USING GIN (aliases);
CREATE INDEX IF NOT EXISTS idx_tasks_aliases ON tasks USING GIN (aliases);
CREATE INDEX IF NOT EXISTS idx_datasets_aliases ON datasets USING GIN (aliases);
CREATE INDEX IF NOT EXISTS idx_metrics_aliases ON metrics USING GIN (aliases);

CREATE INDEX IF NOT EXISTS idx_results_paper_dataset_metric_method
    ON results (paper_id, dataset_id, metric_id, method_id);

CREATE INDEX IF NOT EXISTS idx_claims_paper_id ON claims (paper_id);
CREATE INDEX IF NOT EXISTS idx_paper_relations_src ON paper_relations (src_paper_id);
CREATE INDEX IF NOT EXISTS idx_paper_relations_dst ON paper_relations (dst_paper_id);
CREATE INDEX IF NOT EXISTS idx_concept_resolutions_alias
    ON concept_resolutions (lower(alias_text));
CREATE UNIQUE INDEX IF NOT EXISTS idx_concept_resolutions_unique
    ON concept_resolutions (resolution_type, canonical_id, lower(alias_text));

