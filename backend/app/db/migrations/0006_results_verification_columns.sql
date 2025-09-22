-- Ensure verification columns exist on results for legacy databases
ALTER TABLE IF EXISTS results
    ADD COLUMN IF NOT EXISTS verified BOOLEAN;

ALTER TABLE IF EXISTS results
    ADD COLUMN IF NOT EXISTS verifier_notes TEXT;
