DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_name = 'methods' AND column_name = 'aliases'
    ) THEN
        ALTER TABLE methods
            ADD COLUMN aliases JSONB NOT NULL DEFAULT '[]'::jsonb;
    ELSE
        IF EXISTS (
            SELECT 1
            FROM information_schema.columns
            WHERE table_name = 'methods'
              AND column_name = 'aliases'
              AND udt_name <> 'jsonb'
        ) THEN
            ALTER TABLE methods
                ALTER COLUMN aliases TYPE JSONB
                USING COALESCE(aliases::jsonb, '[]'::jsonb);
        END IF;
        ALTER TABLE methods
            ALTER COLUMN aliases SET NOT NULL,
            ALTER COLUMN aliases SET DEFAULT '[]'::jsonb;
    END IF;
END$$;

ALTER TABLE methods
    ADD COLUMN IF NOT EXISTS metadata JSONB NOT NULL DEFAULT '{}'::jsonb;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_name = 'datasets' AND column_name = 'aliases'
    ) THEN
        ALTER TABLE datasets
            ADD COLUMN aliases JSONB NOT NULL DEFAULT '[]'::jsonb;
    ELSE
        IF EXISTS (
            SELECT 1
            FROM information_schema.columns
            WHERE table_name = 'datasets'
              AND column_name = 'aliases'
              AND udt_name <> 'jsonb'
        ) THEN
            ALTER TABLE datasets
                ALTER COLUMN aliases TYPE JSONB
                USING COALESCE(aliases::jsonb, '[]'::jsonb);
        END IF;
        ALTER TABLE datasets
            ALTER COLUMN aliases SET NOT NULL,
            ALTER COLUMN aliases SET DEFAULT '[]'::jsonb;
    END IF;
END$$;

ALTER TABLE datasets
    ADD COLUMN IF NOT EXISTS metadata JSONB NOT NULL DEFAULT '{}'::jsonb;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_name = 'metrics' AND column_name = 'aliases'
    ) THEN
        ALTER TABLE metrics
            ADD COLUMN aliases JSONB NOT NULL DEFAULT '[]'::jsonb;
    ELSE
        IF EXISTS (
            SELECT 1
            FROM information_schema.columns
            WHERE table_name = 'metrics'
              AND column_name = 'aliases'
              AND udt_name <> 'jsonb'
        ) THEN
            ALTER TABLE metrics
                ALTER COLUMN aliases TYPE JSONB
                USING COALESCE(aliases::jsonb, '[]'::jsonb);
        END IF;
        ALTER TABLE metrics
            ALTER COLUMN aliases SET NOT NULL,
            ALTER COLUMN aliases SET DEFAULT '[]'::jsonb;
    END IF;
END$$;

ALTER TABLE metrics
    ADD COLUMN IF NOT EXISTS metadata JSONB NOT NULL DEFAULT '{}'::jsonb;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_name = 'tasks' AND column_name = 'aliases'
    ) THEN
        ALTER TABLE tasks
            ADD COLUMN aliases JSONB NOT NULL DEFAULT '[]'::jsonb;
    ELSE
        IF EXISTS (
            SELECT 1
            FROM information_schema.columns
            WHERE table_name = 'tasks'
              AND column_name = 'aliases'
              AND udt_name <> 'jsonb'
        ) THEN
            ALTER TABLE tasks
                ALTER COLUMN aliases TYPE JSONB
                USING COALESCE(aliases::jsonb, '[]'::jsonb);
        END IF;
        ALTER TABLE tasks
            ALTER COLUMN aliases SET NOT NULL,
            ALTER COLUMN aliases SET DEFAULT '[]'::jsonb;
    END IF;
END$$;

ALTER TABLE tasks
    ADD COLUMN IF NOT EXISTS metadata JSONB NOT NULL DEFAULT '{}'::jsonb;
ALTER TABLE concepts
    ADD COLUMN IF NOT EXISTS aliases JSONB NOT NULL DEFAULT '[]'::jsonb,
    ADD COLUMN IF NOT EXISTS metadata JSONB NOT NULL DEFAULT '{}'::jsonb;

CREATE INDEX IF NOT EXISTS idx_methods_metadata ON methods USING GIN (metadata);
CREATE INDEX IF NOT EXISTS idx_datasets_metadata ON datasets USING GIN (metadata);
CREATE INDEX IF NOT EXISTS idx_metrics_metadata ON metrics USING GIN (metadata);
CREATE INDEX IF NOT EXISTS idx_tasks_metadata ON tasks USING GIN (metadata);
CREATE INDEX IF NOT EXISTS idx_concepts_metadata ON concepts USING GIN (metadata);
CREATE INDEX IF NOT EXISTS idx_methods_aliases ON methods USING GIN (aliases);
CREATE INDEX IF NOT EXISTS idx_datasets_aliases ON datasets USING GIN (aliases);
CREATE INDEX IF NOT EXISTS idx_metrics_aliases ON metrics USING GIN (aliases);
CREATE INDEX IF NOT EXISTS idx_tasks_aliases ON tasks USING GIN (aliases);
CREATE INDEX IF NOT EXISTS idx_concepts_aliases ON concepts USING GIN (aliases);

-- Add provenance to stored evidence and triples
ALTER TABLE triple_candidates
    ADD COLUMN IF NOT EXISTS provenance JSONB NOT NULL DEFAULT '{}'::jsonb;
ALTER TABLE evidence
    ADD COLUMN IF NOT EXISTS provenance JSONB NOT NULL DEFAULT '{}'::jsonb;

-- Store claim categories as free-form text
ALTER TABLE claims
    ALTER COLUMN category TYPE TEXT USING category::text;

