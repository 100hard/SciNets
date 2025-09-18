-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Papers table
CREATE TABLE IF NOT EXISTS papers (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title TEXT NOT NULL,
    authors TEXT,
    venue TEXT,
    year INTEGER,
    status TEXT NOT NULL DEFAULT 'uploaded',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_papers_created_at ON papers (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_papers_title ON papers (title);

