-- Extend concept resolution enum and ontology tables for new entity types
DO $$
BEGIN
    ALTER TYPE concept_resolution_type ADD VALUE IF NOT EXISTS 'application';
EXCEPTION
    WHEN duplicate_object THEN NULL;
END$$;

DO $$
BEGIN
    ALTER TYPE concept_resolution_type ADD VALUE IF NOT EXISTS 'research_area';
EXCEPTION
    WHEN duplicate_object THEN NULL;
END$$;

CREATE TABLE IF NOT EXISTS applications (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT NOT NULL,
    aliases JSONB NOT NULL DEFAULT '[]'::jsonb,
    description TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS research_areas (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT NOT NULL,
    aliases JSONB NOT NULL DEFAULT '[]'::jsonb,
    description TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_applications_name ON applications (lower(name));
CREATE INDEX IF NOT EXISTS idx_research_areas_name ON research_areas (lower(name));
CREATE INDEX IF NOT EXISTS idx_applications_aliases ON applications USING GIN (aliases);
CREATE INDEX IF NOT EXISTS idx_research_areas_aliases ON research_areas USING GIN (aliases);
