import { motion } from "framer-motion";
import { ArrowUpRight, Clock, Target, Rocket, HelpCircle } from "lucide-react";
import { DecisionSummary } from "@/lib/types";
import { Hypothesis as UIHypothesis } from "./HypothesisCard";
import { cn } from "../../lib/utils.ts";

interface HypothesisPrioritizationProps {
    summary: DecisionSummary;
    hypotheses: UIHypothesis[];
    onSelectHypothesis: (id: string) => void;
}

export function HypothesisPrioritization({ summary, hypotheses, onSelectHypothesis }: HypothesisPrioritizationProps) {
    // Take top 3-4 hypotheses for comparison to fit screen
    const comparisonHypotheses = hypotheses.slice(0, 4);

    const dims = [
        { key: 'mechanistic_coherence', label: 'Coherence' },
        { key: 'empirical_support', label: 'Empirical Support' },
        { key: 'experimental_tractability', label: 'Testability' },
        { key: 'translational_relevance', label: 'Translation' },
    ] as const;

    const getValueColor = (val: string | undefined) => {
        if (!val) return "bg-muted text-muted-foreground";
        switch (val) {
            case "High": return "bg-emerald-500/20 text-emerald-400 border-emerald-500/30";
            case "Medium": return "bg-amber-500/20 text-amber-400 border-amber-500/30";
            case "Low": return "bg-red-500/20 text-red-400 border-red-500/30";
            default: return "bg-muted text-muted-foreground";
        }
    };

    return (
        <div className="mb-12">
            <h2 className="text-lg font-semibold mb-4 px-1">Hypothesis Comparative Matrix</h2>
            <div className="overflow-x-auto rounded-lg border border-border/50 bg-card/30 backdrop-blur-sm">
                <table className="w-full text-sm">
                    <thead>
                        <tr className="border-b border-border/50 bg-secondary/20">
                            <th className="p-4 text-left font-medium text-muted-foreground w-[150px]">
                                Criteria
                            </th>
                            {comparisonHypotheses.map((h, i) => (
                                <th key={h.id} className="p-4 text-left min-w-[200px] border-l border-border/50">
                                    <div className="flex items-start justify-between gap-2">
                                        <span className="font-mono text-xs text-muted-foreground">H{i + 1}</span>
                                        <button
                                            onClick={() => onSelectHypothesis(h.id)}
                                            className="text-primary hover:underline flex items-center gap-1 text-[10px]"
                                        >
                                            View <ArrowUpRight className="w-3 h-3" />
                                        </button>
                                    </div>
                                    <div
                                        className="mt-2 text-xs font-normal text-foreground line-clamp-2 leading-relaxed h-[3em]"
                                        title={h.statement}
                                    >
                                        {h.statement}
                                    </div>
                                </th>
                            ))}
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-border/50">
                        {dims.map((dim) => (
                            <tr key={dim.key} className="hover:bg-foreground/5 transition-colors">
                                <td className="p-4 font-medium text-foreground/80 flex items-center gap-2">
                                    {dim.label}
                                </td>
                                {comparisonHypotheses.map(h => (
                                    <td key={`${h.id}-${dim.key}`} className="p-4 border-l border-border/50">
                                        <span className={cn(
                                            "px-2 py-1 rounded text-xs font-medium border inline-block text-center min-w-[60px]",
                                            getValueColor(h.strength_profile?.[dim.key])
                                        )}>
                                            {h.strength_profile?.[dim.key] || "N/A"}
                                        </span>
                                    </td>
                                ))}
                            </tr>
                        ))}
                        {/* Status Row */}
                        <tr className="bg-secondary/5 font-medium">
                            <td className="p-4 text-foreground/80">Status</td>
                            {comparisonHypotheses.map(h => (
                                <td key={`status-${h.id}`} className="p-4 border-l border-border/50">
                                    <div className="flex items-center gap-1.5">
                                        <div className={cn("w-2 h-2 rounded-full",
                                            h.status === 'supported' ? 'bg-emerald-400' :
                                                h.status === 'mixed' ? 'bg-amber-400' : 'bg-slate-400'
                                        )} />
                                        <span className="text-xs text-foreground capitalize">{h.status}</span>
                                    </div>
                                </td>
                            ))}
                        </tr>
                    </tbody>
                </table>
            </div>
            {hypotheses.length > 4 && (
                <p className="text-xs text-center text-muted-foreground mt-2 italic">
                    Showing top 4 of {hypotheses.length} hypotheses
                </p>
            )}
        </div>
    );
}
