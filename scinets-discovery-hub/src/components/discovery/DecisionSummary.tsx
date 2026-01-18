import { motion } from "framer-motion";
import { CheckCircle2, AlertTriangle, ArrowRight, Activity, ShieldAlert } from "lucide-react";
import { DecisionSummary as IDecisionSummary } from "@/lib/types";
import { cn } from "@/lib/utils";

interface DecisionSummaryProps {
    summary: IDecisionSummary;
}

export function DecisionSummary({ summary }: DecisionSummaryProps) {
    const getConfidenceColor = (level: string) => {
        switch (level) {
            case "High": return "text-emerald-400";
            case "Moderate": return "text-amber-400";
            case "Low": return "text-destructive";
            default: return "text-muted-foreground";
        }
    };

    const getEvidenceBadge = (level: string) => {
        const colors = {
            Strong: "bg-emerald-500/10 text-emerald-400 border-emerald-500/20",
            Moderate: "bg-amber-500/10 text-amber-400 border-amber-500/20",
            Weak: "bg-destructive/10 text-destructive border-destructive/20",
            Inconclusive: "bg-purple-500/10 text-purple-400 border-purple-500/20",
        };
        return colors[level as keyof typeof colors] || colors.Inconclusive;
    };

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-card/50 border border-border rounded-lg p-6 mb-8 backdrop-blur-sm relative overflow-hidden group"
        >
            {/* Background Accent Grid */}
            <div className="absolute inset-0 bg-grid-white/[0.02] -z-10" />
            <div className="absolute top-0 left-0 w-1 h-full bg-primary/50 group-hover:bg-primary transition-colors" />

            {/* Header */}
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-6 border-b border-border/40 pb-4">
                <div>
                    <h2 className="text-xl font-bold flex items-center gap-2">
                        <Activity className="w-5 h-5 text-primary" />
                        SciNets Decision Summary
                    </h2>
                    <p className="text-sm text-muted-foreground mt-1">
                        Executive Brief • System Confidence: <span className={cn("font-medium", getConfidenceColor(summary.system_confidence))}>{summary.system_confidence}</span>
                    </p>
                </div>
                <div className={cn("px-3 py-1 rounded-full border text-xs font-mono uppercase tracking-wider", getEvidenceBadge(summary.evidence_level))}>
                    Evidence Level: {summary.evidence_level}
                </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                {/* Column 1: Primary Action */}
                <div className="lg:col-span-1 border-r border-dashed border-border/40 pr-8">
                    <h3 className="text-sm font-semibold uppercase tracking-wider text-muted-foreground mb-3 flex items-center gap-2">
                        <CheckCircle2 className="w-4 h-4" /> Primary Actionable Hypothesis
                    </h3>
                    <div className="text-lg font-medium text-foreground mb-2">
                        Hypothesis {summary.primary_hypothesis_id.slice(0, 8)}...
                        {/* Ideally we map ID to name, but for summary brief ID is ok if mapped elsewhere */}
                    </div>
                    <p className="text-sm text-foreground/80 leading-relaxed italic border-l-2 border-primary/20 pl-3">
                        "{summary.primary_hypothesis_reason}"
                    </p>
                </div>

                {/* Column 2: Risks & Uncertainties */}
                <div className="lg:col-span-1 border-r border-dashed border-border/40 pr-8">
                    <h3 className="text-sm font-semibold uppercase tracking-wider text-muted-foreground mb-3 flex items-center gap-2">
                        <ShieldAlert className="w-4 h-4" /> Key Risks
                    </h3>
                    <ul className="space-y-2">
                        {summary.key_risks.map((risk, i) => (
                            <li key={i} className="flex items-start gap-2 text-sm text-foreground/80">
                                <AlertTriangle className="w-3 h-3 text-amber-500 mt-1 shrink-0" />
                                <span>{risk}</span>
                            </li>
                        ))}
                    </ul>
                </div>

                {/* Column 3: Recommended Next Steps */}
                <div className="lg:col-span-1">
                    <h3 className="text-sm font-semibold uppercase tracking-wider text-muted-foreground mb-3 flex items-center gap-2">
                        <ArrowRight className="w-4 h-4" /> Recommended Next Steps
                    </h3>
                    <ul className="space-y-2">
                        {summary.recommended_next_steps.map((step, i) => (
                            <li key={i} className="flex items-start gap-2 text-sm text-foreground/80 bg-primary/5 p-2 rounded border border-primary/10">
                                <div className="w-1.5 h-1.5 rounded-full bg-primary mt-1.5 shrink-0" />
                                <span>{step}</span>
                            </li>
                        ))}
                    </ul>
                </div>
            </div>
        </motion.div>
    );
}
