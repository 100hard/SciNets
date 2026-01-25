import { useState } from "react";
import { motion } from "framer-motion";
import { Search, Plus, X, FileText, ArrowRight } from "lucide-react";
import { cn } from "../../lib/utils.ts";

interface DiscoveryQueryStepProps {
  onSubmit: (query: string, papers: string[]) => void;
  isQuotaExceeded?: boolean;
}

export const DiscoveryQueryStep = ({ onSubmit, isQuotaExceeded }: DiscoveryQueryStepProps) => {
  const [query, setQuery] = useState("");
  const [papers, setPapers] = useState<string[]>([]);
  const [paperInput, setPaperInput] = useState("");
  const [showPaperInput, setShowPaperInput] = useState(false);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim() && !isQuotaExceeded) {
      onSubmit(query, papers);
    }
  };

  const addPaper = () => {
    if (paperInput.trim() && !papers.includes(paperInput.trim())) {
      setPapers([...papers, paperInput.trim()]);
      setPaperInput("");
    }
  };

  const removePaper = (paper: string) => {
    setPapers(papers.filter(p => p !== paper));
  };

  const suggestions = [
    // Biomed (Humanized)
    { text: "How does lack of sleep connect to Alzheimer's?", type: "demo" },
    // AI / ML (Technical)
    { text: "Identify safety gaps in multi-agent Reinforcement Learning", type: "search" },
    // Climate (Technical)
    { text: "Propose carbon capture mechanisms using basalt weathering", type: "search" },
    // Social / Econ (Humanized)
    { text: "Why do Universal Basic Income pilots often fail?", type: "demo" },
    // Materials (Technical)
    { text: "Discover perovskite candidates for stable solar cells", type: "search" },
  ];

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="max-w-2xl mx-auto"
    >
      <div className="text-center mb-8">
        <h1 className="text-2xl font-medium text-foreground mb-2">
          What would you like to discover?
        </h1>
        <p className="text-sm text-foreground-muted">
          Describe your research goal and let our agents explore the literature
        </p>
      </div>

      <form onSubmit={handleSubmit} className="space-y-4">
        {/* Main Query Input */}
        <div className="relative">
          <div className={cn(
            "flex items-center gap-3 px-4 py-4 rounded-lg",
            "border border-border bg-background-elevated",
            "focus-within:border-foreground/20 transition-colors"
          )}>
            <Search className="w-5 h-5 text-foreground-muted flex-shrink-0" />
            <textarea
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Describe your research goal..."
              rows={2}
              className={cn(
                "flex-1 bg-transparent resize-none",
                "text-foreground placeholder:text-foreground-muted",
                "text-sm outline-none"
              )}
            />
          </div>
        </div>

        {/* Suggestions */}
        {!query && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="flex flex-wrap gap-2 justify-center"
          >
            {suggestions.map((s) => (
              <button
                key={s.text}
                type="button"
                onClick={() => setQuery(s.text)}
                className={cn(
                  "px-4 py-2 rounded-full text-xs transition-all border",
                  "bg-background-elevated border-border text-foreground-muted",
                  "hover:border-foreground/20 hover:text-foreground hover:bg-foreground/5 shadow-sm"
                )}
              >
                {s.text}
              </button>
            ))}
          </motion.div>
        )}

        {/* Specific Papers Section */}
        <div className="pt-4 border-t border-border">
          <button
            type="button"
            onClick={() => setShowPaperInput(!showPaperInput)}
            className={cn(
              "flex items-center gap-2 text-xs text-foreground-muted",
              "hover:text-foreground transition-colors"
            )}
          >
            <Plus className={cn(
              "w-3 h-3 transition-transform",
              showPaperInput && "rotate-45"
            )} />
            {showPaperInput ? "Hide abstract input" : "Add paper abstract to analyze (optional)"}
          </button>

          {showPaperInput && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              className="mt-3 space-y-3"
            >
              <div className="flex gap-2">
                <div className={cn(
                  "flex-1 flex items-center gap-2 px-3 py-2 rounded",
                  "border border-border bg-background",
                  "focus-within:border-foreground/20 transition-colors"
                )}>
                  <FileText className="w-3.5 h-3.5 text-foreground-muted self-start mt-2" />
                  <textarea
                    value={paperInput}
                    onChange={(e) => setPaperInput(e.target.value)}
                    onKeyDown={(e) => {
                      // Allow Shift+Enter for newlines, Enter for submit
                      if (e.key === "Enter" && !e.shiftKey) {
                        e.preventDefault();
                        addPaper();
                      }
                    }}
                    placeholder="Paste abstract of the paper you want to analyze..."
                    rows={3}
                    className={cn(
                      "flex-1 bg-transparent resize-none",
                      "text-xs text-foreground placeholder:text-foreground-muted",
                      "outline-none py-1"
                    )}
                  />
                </div>
                <button
                  type="button"
                  onClick={addPaper}
                  disabled={!paperInput.trim()}
                  className={cn(
                    "px-3 py-2 rounded text-xs",
                    "border border-border",
                    "hover:border-foreground/20 transition-colors",
                    "disabled:opacity-30"
                  )}
                >
                  Add
                </button>
              </div>

              {papers.length > 0 && (
                <div className="flex flex-wrap gap-2">
                  {papers.map((paper) => (
                    <span
                      key={paper}
                      className={cn(
                        "inline-flex items-center gap-1.5 px-2 py-1 rounded",
                        "text-xs text-foreground",
                        "bg-foreground/5 border border-border"
                      )}
                    >
                      <FileText className="w-3 h-3 text-foreground-muted" />
                      <span className="max-w-[400px] truncate" title={paper}>{paper.substring(0, 50)}...</span>
                      <button
                        type="button"
                        onClick={() => removePaper(paper)}
                        className="text-foreground-muted hover:text-foreground"
                      >
                        <X className="w-3 h-3" />
                      </button>
                    </span>
                  ))}
                </div>
              )}
            </motion.div>
          )}
        </div>

        {/* Submit */}
        <button
          type="submit"
          disabled={!query.trim() || isQuotaExceeded}
          className={cn(
            "w-full flex items-center justify-center gap-2",
            "px-4 py-3 rounded-lg",
            "bg-foreground text-background",
            "text-sm font-medium",
            "hover:bg-foreground/90 transition-colors",
            "disabled:opacity-30 disabled:cursor-not-allowed"
          )}
        >
          {isQuotaExceeded ? "Quota Limit Reached" : "Start Discovery"}
          {!isQuotaExceeded && <ArrowRight className="w-4 h-4" />}
        </button>
      </form>
    </motion.div>
  );
};
