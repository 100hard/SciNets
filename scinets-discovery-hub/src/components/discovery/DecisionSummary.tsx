
import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
    AlertTriangle, ArrowRight, ShieldAlert,
    ChevronDown, ChevronUp, CheckCircle2
} from "lucide-react";
import { DecisionSummary as IDecisionSummary } from "@/lib/types";
import { cn } from "../../lib/utils";

interface DecisionSummaryProps {
    summary: IDecisionSummary;
}

export function DecisionSummary({ summary }: DecisionSummaryProps) {
    const [showPlanning, setShowPlanning] = useState(false);

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-card/50 border border-border rounded-lg p-6 mb-8 backdrop-blur-sm relative overflow-hidden"
        >
            {/* 1. Synthesis Paragraph */}
            <div className="mb-6">
                <h2 className="text-lg font-bold flex items-center gap-2 mb-3 text-foreground">
                    Executive Synthesis
                </h2>
                <div className="prose prose-sm text-foreground/90 max-w-none leading-relaxed py-1">
                    {summary.primary_hypothesis_reason}
                </div>
            </div>

            {/* 2. Key Risks (Bullet Points) */}
            <div className="mb-6">
                <h3 className="text-sm font-bold uppercase tracking-wider text-muted-foreground mb-3 flex items-center gap-2">
                    <ShieldAlert className="w-4 h-4" /> Strategic Risks
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-2">
                    {summary.key_risks.slice(0, 6).map((risk, i) => (
                        <div key={i} className="flex items-start gap-2 text-sm text-foreground/80">
                            <div className="w-1.5 h-1.5 rounded-full bg-amber-500 mt-2" />
                            <span>{risk}</span>
                        </div>
                    ))}
                </div>
            </div>

            {/* 3. Advanced Planning Toggle */}
            <div className="border-t border-border/50 pt-4">
                <button
                    onClick={() => setShowPlanning(!showPlanning)}
                    className="flex items-center gap-2 text-sm font-medium text-primary hover:text-primary/80 transition-colors"
                >
                    {showPlanning ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                    {showPlanning ? "Hide Advanced Planning" : "View Advanced Planning (Timeline & Next Steps)"}
                </button>

                <AnimatePresence>
                    {showPlanning && (
                        <motion.div
                            initial={{ height: 0, opacity: 0 }}
                            animate={{ height: "auto", opacity: 1 }}
                            exit={{ height: 0, opacity: 0 }}
                            className="overflow-hidden"
                        >
                            <div className="pt-6 grid grid-cols-1 md:grid-cols-2 gap-8">
                                {/* Next Steps */}
                                <div>
                                    <h3 className="text-xs font-bold uppercase tracking-wider text-muted-foreground mb-4 flex items-center gap-2">
                                        <ArrowRight className="w-3.5 h-3.5" /> Recommended Sequence
                                    </h3>
                                    <ul className="space-y-3">
                                        {summary.recommended_next_steps.map((step, i) => (
                                            <li key={i} className="flex items-start gap-3 p-3 rounded bg-secondary/30 border border-border/50">
                                                <span className="flex items-center justify-center w-5 h-5 rounded-full bg-primary/10 text-primary text-[10px] font-bold border border-primary/20 shrink-0 mt-0.5">
                                                    {i + 1}
                                                </span>
                                                <span className="text-sm text-foreground/90">{step}</span>
                                            </li>
                                        ))}
                                    </ul>
                                </div>

                                {/* Success Criteria / Meta */}
                                <div>
                                    <h3 className="text-xs font-bold uppercase tracking-wider text-muted-foreground mb-4 flex items-center gap-2">
                                        <CheckCircle2 className="w-3.5 h-3.5" /> Success Markers
                                    </h3>
                                    <div className="p-4 rounded bg-emerald-500/5 border border-emerald-500/10 space-y-3">
                                        <div className="flex justify-between items-center text-sm">
                                            <span className="text-muted-foreground">System Confidence</span>
                                            <span className="font-medium text-emerald-400">{summary.system_confidence}</span>
                                        </div>
                                        <div className="flex justify-between items-center text-sm">
                                            <span className="text-muted-foreground">Evidence Base</span>
                                            <span className="font-medium text-foreground">{summary.evidence_level}</span>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </motion.div>
                    )}
                </AnimatePresence>
            </div>
        </motion.div>
    );
}
