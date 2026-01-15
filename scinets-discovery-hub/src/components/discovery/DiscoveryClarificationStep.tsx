import { useState } from "react";
import { motion } from "framer-motion";
import { Bot, ArrowRight, Calendar, Target, FileSearch } from "lucide-react";
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

const timelineOptions = [
  { value: "recent", label: "Recent (last 5 years)", desc: "Focus on cutting-edge research" },
  { value: "decade", label: "Last decade", desc: "Broader historical context" },
  { value: "all", label: "All time", desc: "Comprehensive coverage" },
];

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
    guidance: ""
  });

  const timelineOptions = [
    {
      value: "recent",
      label: "Recent (last 5 years)",
      desc: "Focus on emerging mechanisms"
    },
    {
      value: "decade",
      label: "Last decade",
      desc: "Balance mature theories with recent data"
    },
    {
      value: "all",
      label: "All time",
      desc: "Comprehensive historical coverage"
    },
  ];

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
      {/* Agent Message */}
      <div className="flex gap-4 mb-8">
        <div className="w-10 h-10 rounded-full bg-foreground/10 flex items-center justify-center flex-shrink-0">
          <Bot className="w-5 h-5 text-foreground" />
        </div>
        <div className="flex-1 space-y-1">
          <p className="text-lg font-medium text-foreground">
            I'm ready to research: <span className="text-foreground italic">"{query}"</span>
          </p>
          <p className="text-sm text-foreground-muted">
            Configure the search parameters below.
          </p>
        </div>
      </div>

      <form onSubmit={handleSubmit} className="space-y-8">

        {/* 1. Timeline Scope (Kept as simple filter) */}
        <div>
          <label className="flex items-center gap-2 text-xs font-semibold text-foreground-muted uppercase tracking-wider mb-3">
            <Calendar className="w-3.5 h-3.5" />
            Timeline Scope
          </label>
          <div className="grid grid-cols-3 gap-3">
            {timelineOptions.map((option) => (
              <button
                key={option.value}
                type="button"
                onClick={() => setAnswers({ ...answers, timeline: option.value })}
                className={cn(
                  "text-left px-3 py-3 rounded-lg border transition-all",
                  answers.timeline === option.value
                    ? "border-foreground/30 bg-foreground/5 shadow-sm"
                    : "border-border hover:border-foreground/20"
                )}
              >
                <p className="text-sm font-medium text-foreground">{option.label}</p>
                <p className="text-[10px] text-foreground-muted mt-1 leading-snug">
                  {option.desc}
                </p>
              </button>
            ))}
          </div>
        </div>

        {/* 2. Research Guidance (New) */}
        <div>
          <label className="flex items-center gap-2 text-xs font-semibold text-foreground-muted uppercase tracking-wider mb-3">
            <FileSearch className="w-3.5 h-3.5" />
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
