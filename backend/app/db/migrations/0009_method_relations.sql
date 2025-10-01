CREATE TABLE IF NOT EXISTS method_relations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    paper_id UUID NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    method_id UUID NOT NULL REFERENCES methods(id) ON DELETE CASCADE,
    dataset_id UUID REFERENCES datasets(id) ON DELETE SET NULL,
    task_id UUID REFERENCES tasks(id) ON DELETE SET NULL,
    relation_type TEXT NOT NULL,
    confidence DOUBLE PRECISION,
    evidence JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_method_relations_paper_id
    ON method_relations (paper_id);

CREATE INDEX IF NOT EXISTS idx_method_relations_method_id
    ON method_relations (method_id);

CREATE INDEX IF NOT EXISTS idx_method_relations_dataset_id
    ON method_relations (dataset_id);

CREATE INDEX IF NOT EXISTS idx_method_relations_task_id
    ON method_relations (task_id);
