ALTER TABLE papers
    ADD COLUMN IF NOT EXISTS file_path TEXT,
    ADD COLUMN IF NOT EXISTS file_name TEXT,
    ADD COLUMN IF NOT EXISTS file_size BIGINT,
    ADD COLUMN IF NOT EXISTS file_content_type TEXT;

CREATE INDEX IF NOT EXISTS idx_papers_file_path ON papers (file_path);
