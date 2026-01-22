import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
    Beaker, X, Loader2, Play, FlaskConical,
    TrendingUp, AlertTriangle, Settings, ChevronRight
} from "lucide-react";
import { cn } from "@/lib/utils";

interface ExperimentConfigModalProps {
    isOpen: boolean;
    onClose: () => void;
    hypothesisId: string;
    hypothesisText: string;
    onRunExperiment: (config: ExperimentConfig) => void;
}

export interface ExperimentConfig {
    intent: "validate_direction" | "probe_sensitivity" | "stress_test";
    dataSource: "synthetic" | "public_dataset" | "user_provided";
    seed: number;
}

export interface ExperimentResult {
    stability_score?: number;
    sensitivity?: string;
    failure_modes?: string[];
    behavioral_pattern?: string;
    consistency_check?: string;
    plot_base64?: string;
}

export interface LocalizedCritique {
    summary: string;
    verdict: "Consistent" | "Inconsistent" | "Inconclusive";
    full_output?: {
        behavioral_interpretation: string;
        consistency_assessment: string;
        failure_modes_identified: string[];
        limitations: string[];
        next_exploration: string;
    };
}

const intentOptions = [
    {
        value: "validate_direction" as const,
        label: "Validate causal direction",
        desc: "Check if A → B holds",
        icon: TrendingUp
    },
    {
        value: "probe_sensitivity" as const,
        label: "Probe sensitivity",
        desc: "How stable under perturbation?",
        icon: Settings
    },
    {
        value: "stress_test" as const,
        label: "Stress-test assumptions",
        desc: "Find failure modes",
        icon: AlertTriangle
    },
];

const dataSourceOptions = [
    { value: "synthetic" as const, label: "Synthetic data", desc: "Generated for this hypothesis" },
    { value: "public_dataset" as const, label: "Public dataset", desc: "sklearn or similar" },
    { value: "user_provided" as const, label: "User-provided", desc: "Coming soon", disabled: true },
];

export const ExperimentConfigModal = ({
    isOpen,
    onClose,
    hypothesisId,
    hypothesisText,
    onRunExperiment,
}: ExperimentConfigModalProps) => {
    const [config, setConfig] = useState<ExperimentConfig>({
        intent: "validate_direction",
        dataSource: "synthetic",
        seed: 42,
    });

    const handleRun = () => {
        onRunExperiment(config);
        onClose();
    };

    if (!isOpen) return null;

    return (
        <AnimatePresence>
            <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm"
                onClick={onClose}
            >
                <motion.div
                    initial={{ opacity: 0, scale: 0.95, y: 20 }}
                    animate={{ opacity: 1, scale: 1, y: 0 }}
                    exit={{ opacity: 0, scale: 0.95, y: 20 }}
                    onClick={(e) => e.stopPropagation()}
                    className="w-full max-w-lg mx-4 bg-background border border-border rounded-xl shadow-2xl overflow-hidden"
                >
                    {/* Header */}
                    <div className="px-6 py-4 border-b border-border flex items-center justify-between">
                        <div className="flex items-center gap-3">
                            <div className="w-8 h-8 rounded-lg bg-purple-500/10 flex items-center justify-center">
                                <FlaskConical className="w-4 h-4 text-purple-400" />
                            </div>
                            <div>
                                <h3 className="text-sm font-medium text-foreground">Explore Computationally</h3>
                                <p className="text-[10px] text-foreground-muted">Exploratory, not validation</p>
                            </div>
                        </div>
                        <button
                            onClick={onClose}
                            className="w-8 h-8 rounded-lg hover:bg-foreground/5 flex items-center justify-center transition-colors"
                        >
                            <X className="w-4 h-4 text-foreground-muted" />
                        </button>
                    </div>

                    {/* Hypothesis preview */}
                    <div className="px-6 py-3 bg-foreground/5 border-b border-border">
                        <p className="text-xs text-foreground-muted line-clamp-2">
                            {hypothesisText}
                        </p>
                    </div>

                    {/* Config form */}
                    <div className="px-6 py-4 space-y-5">
                        {/* Intent */}
                        <div>
                            <label className="text-xs font-medium text-foreground mb-2 block">
                                Experiment Intent
                            </label>
                            <div className="space-y-2">
                                {intentOptions.map((option) => (
                                    <button
                                        key={option.value}
                                        onClick={() => setConfig({ ...config, intent: option.value })}
                                        className={cn(
                                            "w-full text-left px-3 py-2.5 rounded-lg border transition-all",
                                            "flex items-center gap-3",
                                            config.intent === option.value
                                                ? "border-purple-500/30 bg-purple-500/5"
                                                : "border-border hover:border-foreground/20"
                                        )}
                                    >
                                        <div className={cn(
                                            "w-7 h-7 rounded flex items-center justify-center",
                                            config.intent === option.value ? "bg-purple-500/10" : "bg-foreground/5"
                                        )}>
                                            <option.icon className={cn(
                                                "w-3.5 h-3.5",
                                                config.intent === option.value ? "text-purple-400" : "text-foreground-muted"
                                            )} />
                                        </div>
                                        <div>
                                            <p className="text-xs font-medium text-foreground">{option.label}</p>
                                            <p className="text-[10px] text-foreground-muted">{option.desc}</p>
                                        </div>
                                    </button>
                                ))}
                            </div>
                        </div>

                        {/* Data Source */}
                        <div>
                            <label className="text-xs font-medium text-foreground mb-2 block">
                                Data Source
                            </label>
                            <div className="grid grid-cols-3 gap-2">
                                {dataSourceOptions.map((option) => (
                                    <button
                                        key={option.value}
                                        disabled={option.disabled}
                                        onClick={() => setConfig({ ...config, dataSource: option.value })}
                                        className={cn(
                                            "text-left px-3 py-2 rounded-lg border transition-all",
                                            config.dataSource === option.value
                                                ? "border-purple-500/30 bg-purple-500/5"
                                                : "border-border hover:border-foreground/20",
                                            option.disabled && "opacity-50 cursor-not-allowed"
                                        )}
                                    >
                                        <p className="text-xs font-medium text-foreground">{option.label}</p>
                                        <p className="text-[10px] text-foreground-muted">{option.desc}</p>
                                    </button>
                                ))}
                            </div>
                        </div>

                        {/* Seed */}
                        <div>
                            <label className="text-xs font-medium text-foreground mb-2 block">
                                Reproducibility Seed
                            </label>
                            <input
                                type="number"
                                value={config.seed}
                                onChange={(e) => setConfig({ ...config, seed: parseInt(e.target.value) || 42 })}
                                className="w-24 px-3 py-2 text-xs border border-border rounded-lg bg-background text-foreground focus:outline-none focus:border-purple-500/50"
                            />
                            <p className="text-[10px] text-foreground-muted mt-1">Fixed seed ensures reproducibility</p>
                        </div>

                        {/* Disclaimer */}
                        <div className="p-3 rounded-lg bg-amber-500/5 border border-amber-500/20">
                            <p className="text-[10px] text-amber-400 flex items-start gap-1.5">
                                <AlertTriangle className="w-3 h-3 flex-shrink-0 mt-0.5" />
                                <span>
                                    This is an exploratory consistency check, not scientific validation.
                                    Results inform thinking but do not prove the hypothesis.
                                </span>
                            </p>
                        </div>
                    </div>

                    {/* Footer */}
                    <div className="px-6 py-4 border-t border-border flex justify-end gap-3">
                        <button
                            onClick={onClose}
                            className="px-4 py-2 text-xs text-foreground-muted hover:text-foreground transition-colors"
                        >
                            Cancel
                        </button>
                        <button
                            onClick={handleRun}
                            className={cn(
                                "flex items-center gap-2 px-4 py-2 rounded-lg",
                                "bg-purple-500 text-white text-xs font-medium",
                                "hover:bg-purple-600 transition-colors"
                            )}
                        >
                            <Play className="w-3.5 h-3.5" />
                            Run Experiment
                        </button>
                    </div>
                </motion.div>
            </motion.div>
        </AnimatePresence>
    );
};


// Inline Experiment Display Component
interface ExperimentInlineDisplayProps {
    isLoading: boolean;
    logs: string[];
    result: ExperimentResult | null;
    critique: LocalizedCritique | null;
}

export const ExperimentInlineDisplay = ({
    isLoading,
    logs,
    result,
    critique,
}: ExperimentInlineDisplayProps) => {
    return (
        <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            className="mt-4 border border-purple-500/20 rounded-lg bg-purple-500/5 overflow-hidden"
        >
            <div className="px-4 py-3 border-b border-purple-500/20 flex items-center gap-2">
                <FlaskConical className="w-3.5 h-3.5 text-purple-400" />
                <span className="text-xs font-medium text-purple-400">Exploratory Experiment</span>
                {isLoading && <Loader2 className="w-3 h-3 text-purple-400 animate-spin ml-auto" />}
            </div>

            {/* Agent Logs */}
            {logs.length > 0 && (
                <div className="px-4 py-2 border-b border-purple-500/10 bg-background/50">
                    <div className="space-y-1 max-h-24 overflow-auto font-mono text-[10px] text-foreground-muted">
                        {logs.map((log, i) => (
                            <div key={i} className="flex items-start gap-2">
                                <ChevronRight className="w-3 h-3 text-purple-400" />
                                <span>{log}</span>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* Results */}
            {result && (
                <div className="px-4 py-3 space-y-3">
                    {/* Behavioral Metrics */}
                    <div className="grid grid-cols-2 gap-3">
                        {result.stability_score !== undefined && (
                            <div className="p-2 rounded bg-background/50 border border-border">
                                <p className="text-[10px] text-foreground-muted">Stability</p>
                                <p className="text-sm font-mono text-foreground">{(result.stability_score * 100).toFixed(0)}%</p>
                            </div>
                        )}
                        {result.sensitivity && (
                            <div className="p-2 rounded bg-background/50 border border-border">
                                <p className="text-[10px] text-foreground-muted">Sensitivity</p>
                                <p className="text-sm font-mono text-foreground capitalize">{result.sensitivity}</p>
                            </div>
                        )}
                        {result.consistency_check && (
                            <div className="p-2 rounded bg-background/50 border border-border">
                                <p className="text-[10px] text-foreground-muted">Consistency</p>
                                <p className="text-sm font-mono text-foreground capitalize">{result.consistency_check}</p>
                            </div>
                        )}
                    </div>

                    {/* Behavioral Pattern */}
                    {result.behavioral_pattern && (
                        <div>
                            <p className="text-[10px] text-foreground-muted mb-1">Behavioral Pattern</p>
                            <p className="text-xs text-foreground">{result.behavioral_pattern}</p>
                        </div>
                    )}

                    {/* Failure Modes */}
                    {result.failure_modes && result.failure_modes.length > 0 && (
                        <div>
                            <p className="text-[10px] text-foreground-muted mb-1">Failure Modes</p>
                            <div className="flex flex-wrap gap-1">
                                {result.failure_modes.map((mode, i) => (
                                    <span key={i} className="px-2 py-0.5 text-[10px] rounded bg-red-500/10 text-red-400 border border-red-500/20">
                                        {mode}
                                    </span>
                                ))}
                            </div>
                        </div>
                    )}

                    {/* Plot */}
                    {result.plot_base64 && (
                        <div className="mt-2">
                            <img
                                src={`data:image/png;base64,${result.plot_base64}`}
                                alt="Experiment plot"
                                className="w-full rounded border border-border"
                            />
                        </div>
                    )}
                </div>
            )}

            {/* Localized Critique */}
            {critique && (
                <div className="px-4 py-3 border-t border-purple-500/10 bg-background/30">
                    <div className="flex items-center gap-2 mb-2">
                        <span className="text-[10px] font-medium text-foreground-muted">Localized Assessment</span>
                        <span className={cn(
                            "text-[10px] px-2 py-0.5 rounded-full border",
                            critique.verdict === "Consistent" && "bg-emerald-500/10 text-emerald-400 border-emerald-500/20",
                            critique.verdict === "Inconsistent" && "bg-red-500/10 text-red-400 border-red-500/20",
                            critique.verdict === "Inconclusive" && "bg-amber-500/10 text-amber-400 border-amber-500/20"
                        )}>
                            {critique.verdict}
                        </span>
                    </div>
                    <p className="text-xs text-foreground-muted">{critique.summary}</p>

                    {/* Disclaimer */}
                    <p className="text-[10px] text-foreground-muted/60 mt-2 italic">
                        This is exploratory and does not validate the hypothesis.
                    </p>
                </div>
            )}
        </motion.div>
    );
};
