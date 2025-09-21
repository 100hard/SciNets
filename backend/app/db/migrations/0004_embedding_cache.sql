-- Embedding cache stores reusable vectors keyed by text hash and model name
CREATE TABLE IF NOT EXISTS embedding_cache (
    model TEXT NOT NULL,
    text_hash TEXT NOT NULL,
    embedding DOUBLE PRECISION[] NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (model, text_hash)
);

CREATE INDEX IF NOT EXISTS idx_embedding_cache_updated_at ON embedding_cache (updated_at DESC);
