import { useState } from "react";
import { motion } from "framer-motion";
import { Bot, ArrowRight, Target, FileSearch } from "lucide-react";
import { cn } from "@/lib/utils";

export interface ClarificationAnswers {
  timeline: string;
  goal: string;
  depth: string;
  guidance?: string;
}

interface DiscoveryClarificationStepProps {
  query: string;
  onSubmit: (answers: ClarificationAnswers) => void;
  onBack: () => void;
}

const goalOptions = [
  { value: "discover", label: "Discover novel ideas", desc: "Find gaps and unexplored connections", icon: Target },
  { value: "survey", label: "Literature survey", desc: "Comprehensive overview of the field", icon: FileSearch },
  { value: "write", label: "Help me write", desc: "Draft summaries and synthesis", icon: FileSearch },
];

const depthOptions = [
  { value: "quick", label: "Quick scan", desc: "~20 papers, key findings only" },
  { value: "standard", label: "Standard", desc: "~50 papers, detailed analysis" },
  { value: "deep", label: "Deep dive", desc: "100+ papers, exhaustive coverage" },
];

export const DiscoveryClarificationStep = ({
  query,
  onSubmit,
  onBack
}: DiscoveryClarificationStepProps) => {
  const [answers, setAnswers] = useState<ClarificationAnswers>({
    timeline: "recent",
    goal: "discover",
    depth: "standard",
    guidance: ""
  });



  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit(answers);
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="max-w-xl mx-auto"
    >
      {/* Header removed as per request */}

      <div className="mb-6 rounded-lg border border-border bg-foreground/5 p-4 space-y-3">
        <div className="flex items-center gap-2 text-xs font-semibold text-foreground-muted uppercase tracking-wider">
          Scientific Discovery Process
        </div>
        <p className="text-sm text-foreground/80 leading-relaxed">
          SciNets employs a <strong>multi-agent cognitive architecture</strong> to simulate scientific discovery. It aggregates literature to construct a <strong>probabilistic causal graph</strong>, utilizes graph algorithms to identify structural holes, and synthesizes <strong>empirically-constrained hypotheses</strong> via multi-hop reasoning.
        </p>


        <div className="pt-3 mt-3 border-t border-dashed border-border/50 flex items-start gap-2 text-[11px] text-foreground-muted/80 italic">
          <span>Note: Deep multi-hop reasoning is computationally intensive. Completing a full discovery cycle typically takes 5-10 minutes.</span>
        </div>
      </div>

      <form onSubmit={handleSubmit} className="space-y-8">

        {/* 1. Research Guidance (Restored) */}
        <div>
          <label className="flex items-center gap-2 text-xs font-semibold text-foreground-muted uppercase tracking-wider mb-3">
            Optional: Research Guidance
          </label>
          <div className="relative">
            <textarea
              value={answers.guidance}
              onChange={(e) => setAnswers({ ...answers, guidance: e.target.value })}
              placeholder="Share any constraints, focus areas, or context you want SciNets to consider before starting..."
              className={cn(
                "w-full h-32 px-4 py-3 rounded-lg bg-background border border-border",
                "text-sm text-foreground placeholder:text-foreground-muted/50",
                "focus:ring-1 focus:ring-foreground/20 focus:border-foreground/30 outline-none transition-all",
                "resize-none"
              )}
            />
            <p className="absolute bottom-3 right-4 text-[10px] text-foreground-muted pointer-events-none">
              Leave blank for default exploration
            </p>
          </div>
        </div>

        {/* Actions */}
        <div className="pt-4 border-t border-border/50 flex gap-3">
          <button
            type="button"
            onClick={onBack}
            className="px-5 py-3 rounded-lg text-sm border border-border text-foreground-muted hover:text-foreground hover:border-foreground/20 transition-colors"
          >
            Back
          </button>
          <button
            type="submit"
            className="flex-1 flex items-center justify-center gap-2 px-6 py-3 rounded-lg bg-foreground text-background text-sm font-medium hover:bg-foreground/90 transition-all shadow-lg hover:shadow-xl active:scale-[0.99]"
          >
            Begin Exploration
            <ArrowRight className="w-4 h-4 ml-1" />
          </button>
        </div>
      </form>
    </motion.div>
  );
};
