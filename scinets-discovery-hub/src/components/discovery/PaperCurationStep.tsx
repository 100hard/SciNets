import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import {
  ArrowLeft, ArrowRight, Check, Lock, Unlock,
  Upload, Plus, X, FileText, Calendar, Info
} from "lucide-react";
import { cn } from "../../lib/utils";
import { Button } from "@/components/ui/button";

interface CandidatePaper {
  id: string;
  title: string;
  year: number;
  venue: string;
  rationale: string;
  selected: boolean;
  locked: boolean;
}

interface PaperCurationStepProps {
  query: string;
  candidatePapers?: CandidatePaper[];
  onSubmit: (papers: CandidatePaper[]) => void;
  onBack: () => void;
}

// Fallback removed - we trust props
// const fallbackCandidates: CandidatePaper[] = [];

export const PaperCurationStep = ({
  query,
  candidatePapers,
  onSubmit,
  onBack,
}: PaperCurationStepProps) => {
  // Use candidatePapers from API, empty if none
  const [papers, setPapers] = useState<CandidatePaper[]>(candidatePapers || []);
  const [manualTitle, setManualTitle] = useState("");
  const [showAddForm, setShowAddForm] = useState(false);

  // Update papers when candidatePapers prop changes
  useEffect(() => {
    if (candidatePapers) {
      setPapers(candidatePapers);
    }
  }, [candidatePapers]);

  const selectedCount = papers.filter(p => p.selected).length;
  const lockedCount = papers.filter(p => p.locked).length;

  const toggleSelect = (id: string) => {
    setPapers(prev => prev.map(p =>
      p.id === id ? { ...p, selected: !p.selected } : p
    ));
  };

  const toggleLock = (id: string) => {
    setPapers(prev => prev.map(p =>
      p.id === id ? { ...p, locked: !p.locked, selected: p.locked ? p.selected : true } : p
    ));
  };

  const removePaper = (id: string) => {
    setPapers(prev => prev.filter(p => p.id !== id));
  };

  const addManualPaper = () => {
    if (!manualTitle.trim()) return;

    const newPaper: CandidatePaper = {
      id: `manual-${Date.now()}`,
      title: manualTitle.trim(),
      year: new Date().getFullYear(),
      venue: "User Added",
      rationale: "Manually added by user",
      selected: true,
      locked: false,
    };

    setPapers(prev => [newPaper, ...prev]);
    setManualTitle("");
    setShowAddForm(false);
  };

  const handleSubmit = () => {
    onSubmit(papers.filter(p => p.selected));
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      className="w-full max-w-4xl mx-auto"
    >
      {/* Header */}
      <div className="mb-6">
        <h2 className="text-xl font-medium text-foreground mb-2">Curate Paper Corpus</h2>
        <p className="text-sm text-foreground-muted">
          The Literature Agent identified these candidate papers. Select the papers to include in the analysis.
        </p>
      </div>

      {/* Query context */}
      <div className="mb-6 px-4 py-3 border border-border rounded-lg bg-background/30">
        <p className="text-xs text-foreground-muted mb-1">Research Query</p>
        <p className="text-sm text-foreground">{query}</p>
      </div>

      {/* Stats bar */}
      <div className="mb-4 flex items-center justify-between">
        <div className="flex items-center gap-4 text-xs text-foreground-muted">
          <span>{selectedCount} selected</span>
          <span>{lockedCount} locked</span>
          <span>{papers.length} total</span>
        </div>
        <button
          onClick={() => setShowAddForm(!showAddForm)}
          className={cn(
            "flex items-center gap-1.5 px-3 py-1.5 rounded text-xs transition-colors",
            "border border-border hover:border-foreground/20",
            showAddForm && "bg-foreground/5"
          )}
        >
          <Plus className="w-3 h-3" />
          Add Paper
        </button>
      </div>

      {/* Add paper form */}
      {showAddForm && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: "auto" }}
          exit={{ opacity: 0, height: 0 }}
          className="mb-4 p-4 border border-border rounded-lg bg-background/30"
        >
          <div className="flex gap-3">
            <input
              type="text"
              value={manualTitle}
              onChange={(e) => setManualTitle(e.target.value)}
              placeholder="Enter paper title or paste DOI..."
              className="flex-1 px-3 py-2 text-sm bg-background border border-border rounded focus:outline-none focus:border-foreground/30"
              onKeyDown={(e) => e.key === "Enter" && addManualPaper()}
            />
            <Button
              onClick={addManualPaper}
              disabled={!manualTitle.trim()}
              size="sm"
              variant="outline"
            >
              Add
            </Button>
            <Button
              variant="ghost"
              size="sm"
              className="text-foreground-muted"
            >
              <Upload className="w-4 h-4" />
            </Button>
          </div>
        </motion.div>
      )}

      {/* Frozen notice */}
      <div className="mb-4 flex items-start gap-2 px-4 py-3 border border-border rounded-lg bg-foreground/5">
        <Info className="w-4 h-4 text-foreground-muted flex-shrink-0 mt-0.5" />
        <p className="text-xs text-foreground-muted">
          The selected papers will be <strong>frozen</strong> before analysis proceeds.
          Lock papers to ensure they're always included regardless of relevance scoring.
        </p>
      </div>

      {/* Paper list */}
      <div className="border border-border rounded-lg overflow-hidden mb-6">
        {/* Table header */}
        <div className="grid grid-cols-[auto_1fr_80px_120px_auto_auto] gap-4 px-4 py-2 bg-foreground/5 border-b border-border text-xs font-medium text-foreground-muted">
          <div className="w-6"></div>
          <div>Title</div>
          <div>Year</div>
          <div>Venue</div>
          <div className="w-6"></div>
          <div className="w-6"></div>
        </div>

        {/* Paper rows */}
        <div className="divide-y divide-border max-h-[400px] overflow-auto">
          {papers.length === 0 && (
            <div className="p-8 text-center text-foreground-muted">
              <p className="mb-2">No papers found for this query.</p>
              <p className="text-xs">Try manually adding a paper or going back to refine your query.</p>
            </div>
          )}
          {papers.map((paper) => (
            <div
              key={paper.id}
              className={cn(
                "grid grid-cols-[auto_1fr_80px_120px_auto_auto] gap-4 px-4 py-3 items-start transition-colors",
                paper.selected ? "bg-background" : "bg-background/50 opacity-60"
              )}
            >
              {/* Checkbox */}
              <button
                onClick={() => toggleSelect(paper.id)}
                className={cn(
                  "w-6 h-6 rounded border flex items-center justify-center transition-colors",
                  paper.selected
                    ? "bg-foreground text-background border-foreground"
                    : "border-border hover:border-foreground/30"
                )}
              >
                {paper.selected && <Check className="w-3.5 h-3.5" />}
              </button>

              {/* Title & Rationale */}
              <div>
                <p className="text-sm text-foreground mb-1">{paper.title}</p>
                <p className="text-xs text-foreground-muted">{paper.rationale}</p>
              </div>

              {/* Year */}
              <div className="flex items-center gap-1 text-xs text-foreground-muted">
                <Calendar className="w-3 h-3" />
                {paper.year}
              </div>

              {/* Venue */}
              <p className="text-xs text-foreground-muted truncate">{paper.venue}</p>

              {/* Lock button */}
              <button
                onClick={() => toggleLock(paper.id)}
                className={cn(
                  "w-6 h-6 rounded flex items-center justify-center transition-colors",
                  paper.locked
                    ? "text-foreground"
                    : "text-foreground-muted hover:text-foreground"
                )}
                title={paper.locked ? "Unlock paper" : "Lock paper (always include)"}
              >
                {paper.locked ? <Lock className="w-3.5 h-3.5" /> : <Unlock className="w-3.5 h-3.5" />}
              </button>

              {/* Remove button */}
              {paper.venue === "User Added" && (
                <button
                  onClick={() => removePaper(paper.id)}
                  className="w-6 h-6 rounded flex items-center justify-center text-foreground-muted hover:text-red-500 transition-colors"
                >
                  <X className="w-3.5 h-3.5" />
                </button>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Navigation */}
      <div className="flex items-center justify-between">
        <Button variant="ghost" onClick={onBack}>
          <ArrowLeft className="w-4 h-4 mr-2" />
          Back
        </Button>
        <Button
          onClick={handleSubmit}
          disabled={selectedCount === 0}
        >
          Proceed with {selectedCount} papers
          <ArrowRight className="w-4 h-4 ml-2" />
        </Button>
      </div>
    </motion.div>
  );
};

export type { CandidatePaper };
