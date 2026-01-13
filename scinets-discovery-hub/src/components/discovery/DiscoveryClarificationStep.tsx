import { useState } from "react";
import { motion } from "framer-motion";
import { Bot, ArrowRight, Calendar, Target, FileSearch } from "lucide-react";
import { cn } from "@/lib/utils";

export interface ClarificationAnswers {
  timeline: string;
  goal: string;
  // REMOVED: runExperiments - experiments are now post-discovery user-triggered only
  depth: string;
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
    goal: "discover",
    // REMOVED: runExperiments
    depth: "standard",
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit(answers);
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="max-w-2xl mx-auto"
    >
      {/* Agent Message */}
      <div className="flex gap-3 mb-8">
        <div className="w-8 h-8 rounded-full bg-foreground/10 flex items-center justify-center flex-shrink-0">
          <Bot className="w-4 h-4 text-foreground" />
        </div>
        <div className="flex-1">
          <p className="text-sm text-foreground mb-1">
            I'll help you explore: <span className="text-foreground-muted">"{query}"</span>
          </p>
          <p className="text-xs text-foreground-muted">
            A few quick questions to optimize my search...
          </p>
        </div>
      </div>

      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Timeline */}
        <div>
          <label className="flex items-center gap-2 text-xs text-foreground-muted mb-3">
            <Calendar className="w-3.5 h-3.5" />
            What timeline should I focus on?
          </label>
          <div className="grid grid-cols-3 gap-2">
            {timelineOptions.map((option) => (
              <button
                key={option.value}
                type="button"
                onClick={() => setAnswers({ ...answers, timeline: option.value })}
                className={cn(
                  "text-left px-3 py-2.5 rounded-lg border transition-all",
                  answers.timeline === option.value
                    ? "border-foreground/30 bg-foreground/5"
                    : "border-border hover:border-foreground/20"
                )}
              >
                <p className="text-xs font-medium text-foreground">{option.label}</p>
                <p className="text-[10px] text-foreground-muted mt-0.5">{option.desc}</p>
              </button>
            ))}
          </div>
        </div>

        {/* Goal */}
        <div>
          <label className="flex items-center gap-2 text-xs text-foreground-muted mb-3">
            <Target className="w-3.5 h-3.5" />
            What's your primary goal?
          </label>
          <div className="space-y-2">
            {goalOptions.map((option) => (
              <button
                key={option.value}
                type="button"
                onClick={() => setAnswers({ ...answers, goal: option.value })}
                className={cn(
                  "w-full text-left px-3 py-3 rounded-lg border transition-all",
                  "flex items-center gap-3",
                  answers.goal === option.value
                    ? "border-foreground/30 bg-foreground/5"
                    : "border-border hover:border-foreground/20"
                )}
              >
                <div className={cn(
                  "w-8 h-8 rounded flex items-center justify-center",
                  answers.goal === option.value ? "bg-foreground/10" : "bg-background"
                )}>
                  <option.icon className="w-4 h-4 text-foreground-muted" />
                </div>
                <div>
                  <p className="text-xs font-medium text-foreground">{option.label}</p>
                  <p className="text-[10px] text-foreground-muted">{option.desc}</p>
                </div>
              </button>
            ))}
          </div>
        </div>

        {/* Depth */}
        <div>
          <label className="flex items-center gap-2 text-xs text-foreground-muted mb-3">
            <FileSearch className="w-3.5 h-3.5" />
            How deep should I go?
          </label>
          <div className="grid grid-cols-3 gap-2">
            {depthOptions.map((option) => (
              <button
                key={option.value}
                type="button"
                onClick={() => setAnswers({ ...answers, depth: option.value })}
                className={cn(
                  "text-left px-3 py-2.5 rounded-lg border transition-all",
                  answers.depth === option.value
                    ? "border-foreground/30 bg-foreground/5"
                    : "border-border hover:border-foreground/20"
                )}
              >
                <p className="text-xs font-medium text-foreground">{option.label}</p>
                <p className="text-[10px] text-foreground-muted mt-0.5">{option.desc}</p>
              </button>
            ))}
          </div>
        </div>

        {/* NOTE: Experiment toggle REMOVED - experiments are now post-discovery user-triggered only */}

        {/* Actions */}
        <div className="flex gap-3 pt-2">
          <button
            type="button"
            onClick={onBack}
            className={cn(
              "px-4 py-2.5 rounded-lg text-xs",
              "border border-border",
              "text-foreground-muted hover:text-foreground",
              "hover:border-foreground/20 transition-colors"
            )}
          >
            Back
          </button>
          <button
            type="submit"
            className={cn(
              "flex-1 flex items-center justify-center gap-2",
              "px-4 py-2.5 rounded-lg",
              "bg-foreground text-background",
              "text-xs font-medium",
              "hover:bg-foreground/90 transition-colors"
            )}
          >
            Begin Exploration
            <ArrowRight className="w-3.5 h-3.5" />
          </button>
        </div>
      </form>
    </motion.div>
  );
};
