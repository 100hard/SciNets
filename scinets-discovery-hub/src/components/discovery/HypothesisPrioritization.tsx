import { motion } from "framer-motion";
import { ArrowUpRight, Clock, Target, Rocket } from "lucide-react";
import { DecisionSummary } from "@/lib/types";
import { Hypothesis as UIHypothesis } from "./HypothesisCard";
import { cn } from "@/lib/utils";

interface HypothesisPrioritizationProps {
    summary: DecisionSummary;
    hypotheses: UIHypothesis[];
    onSelectHypothesis: (id: string) => void;
}

export function HypothesisPrioritization({ summary, hypotheses, onSelectHypothesis }: HypothesisPrioritizationProps) {

    const getHypothesisById = (id: string) => hypotheses.find(h => h.id === id);

    const PrioritySection = ({ title, ids, icon: Icon, color }: { title: string, ids: string[], icon: any, color: string }) => {
        if (!ids || ids.length === 0) return null;

        return (
            <div className="flex-1 min-w-[300px]">
                <h3 className={cn("text-xs font-bold uppercase tracking-wider mb-3 flex items-center gap-2", color)}>
                    <Icon className="w-4 h-4" /> {title}
                </h3>
                <div className="space-y-3">
                    {ids.map(id => {
                        const h = getHypothesisById(id);
                        if (!h) return null;
                        return (
                            <motion.div
                                key={id}
                                whileHover={{ scale: 1.02 }}
                                onClick={() => onSelectHypothesis(id)}
                                className="p-3 bg-card border border-border/50 rounded cursor-pointer hover:border-primary/30 transition-all group"
                            >
                                <div className="flex justify-between items-start gap-2">
                                    <span className="font-mono text-xs text-muted-foreground group-hover:text-foreground transition-colors">
                                        H{hypotheses.findIndex(hyp => hyp.id === id) + 1}
                                    </span>
                                    {h.strength_profile && (
                                        <span className="text-[10px] bg-secondary px-1.5 py-0.5 rounded text-muted-foreground border border-border">
                                            Exp: {h.strength_profile.experimental_tractability}
                                        </span>
                                    )}
                                </div>
                                <p className="text-sm font-medium mt-1 line-clamp-2 leading-snug">
                                    {h.statement}
                                </p>
                                <div className="mt-2 flex items-center text-[10px] text-muted-foreground gap-1 group-hover:text-primary transition-colors">
                                    View Analysis <ArrowUpRight className="w-3 h-3" />
                                </div>
                            </motion.div>
                        );
                    })}
                </div>
            </div>
        );
    };

    return (
        <div className="mb-12">
            <h2 className="text-lg font-semibold mb-4 px-1">Hypothesis Prioritization</h2>
            <div className="flex flex-wrap gap-6 bg-secondary/20 p-6 rounded-lg border border-border/50">
                <PrioritySection
                    title="Best for Near-Term Testing"
                    ids={summary.near_term_focus}
                    icon={Clock}
                    color="text-emerald-400"
                />
                <PrioritySection
                    title="Best for Long-Term Theory"
                    ids={summary.long_term_focus}
                    icon={Target}
                    color="text-blue-400"
                />
                <PrioritySection
                    title="High Risk / High Reward"
                    ids={summary.high_risk_high_reward}
                    icon={Rocket}
                    color="text-purple-400"
                />
            </div>
        </div>
    );
}
