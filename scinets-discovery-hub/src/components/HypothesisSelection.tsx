
import React, { useState } from "react";
import { Hypothesis } from "@/lib/types";
import { Check, ArrowRight } from "lucide-react";

interface HypothesisSelectionProps {
    hypotheses: Hypothesis[];
    onConfirm: (selectedIds: string[]) => void;
    onRegenerate?: (seedId: string) => void;
}

export const HypothesisSelection: React.FC<HypothesisSelectionProps> = ({
    hypotheses,
    onConfirm,
    onRegenerate
}) => {
    const [selectedIds, setSelectedIds] = useState<string[]>([]);

    // Auto-select top 3 by default if none selected? Or force choice?
    // Let's start with nothing selected to force agency, or maybe top 1.

    const toggleSelection = (id: string) => {
        if (selectedIds.includes(id)) {
            setSelectedIds(prev => prev.filter(x => x !== id));
        } else {
            if (selectedIds.length < 2) {
                setSelectedIds(prev => [...prev, id]);
            }
        }
    };

    const handleConfirm = () => {
        if (selectedIds.length === 0) return;
        onConfirm(selectedIds);
    };

    return (
        <div className="w-full max-w-6xl mx-auto p-6 animate-fade-in">
            <div className="text-center mb-8">
                <h2 className="text-3xl font-light text-foreground mb-3">
                    Preliminary <span className="text-primary font-medium">Hypothesis Sketches</span>
                </h2>
                <p className="text-muted-foreground text-lg max-w-2xl mx-auto">
                    SciNets has generated initial candidate mechanisms based on the literature graph.
                    Select <span className="text-primary font-medium">1 or 2</span> hypotheses to <span className="text-primary">deeply investigate</span>.
                </p>
            </div>

            {hypotheses.length === 0 && (
                <div className="flex flex-col items-center justify-center py-20 text-muted-foreground animate-pulse">
                    <div className="w-16 h-16 rounded-full border-4 border-primary/20 border-t-primary animate-spin mb-6"></div>
                    <p className="text-xl">Synthesizing Hypothesis Sketches...</p>
                    <p className="text-sm mt-2">Integrating literature and generating diverse candidates.</p>
                </div>
            )}

            {hypotheses.length > 0 && (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-10">
                    {hypotheses.map((h, i) => {
                        const isSelected = selectedIds.includes(h.id);
                        const isDisabled = !isSelected && selectedIds.length >= 2;
                        const score = (h.novelty_score * 0.4 + h.feasibility_score * 0.3 + h.testability_score * 0.3).toFixed(2);

                        return (
                            <div
                                key={h.id}
                                onClick={() => !isDisabled && toggleSelection(h.id)}
                                className={`
                                    relative group rounded-xl border-2 transition-all duration-300
                                    ${isDisabled ? "opacity-50 cursor-not-allowed grayscale-[0.5]" : "cursor-pointer hover:shadow-lg hover:-translate-y-1"}
                                    ${isSelected
                                        ? "border-primary bg-primary/5 shadow-primary/20"
                                        : "border-border/50 bg-card hover:border-primary/50"}
                                `}
                            >
                                {/* Mechanism Tag */}
                                {h.mechanism_class && (
                                    <div className="absolute -top-3 left-4 px-3 py-1 bg-background border border-border rounded-full text-xs font-medium text-foreground/80 shadow-sm z-10">
                                        {h.mechanism_class}
                                    </div>
                                )}

                                {/* Selection Checkbox */}
                                <div className={`
                                    absolute top-4 right-4 w-6 h-6 rounded-full border flex items-center justify-center transition-colors
                                    ${isSelected ? "bg-primary border-primary" : "border-muted-foreground/30 group-hover:border-primary/50"}
                                `}>
                                    {isSelected && <Check className="w-3.5 h-3.5 text-primary-foreground" />}
                                </div>

                                <div className="p-6 pt-8 flex flex-col h-full">
                                    {/* Text */}
                                    <h3 className="text-lg font-medium text-foreground leading-snug mb-4 line-clamp-4">
                                        {h.text}
                                    </h3>

                                    <div className="mt-auto space-y-4">
                                        {/* Key Concepts (Chain) */}
                                        {h.causal_chain?.nodes && (
                                            <div className="flex flex-wrap gap-2">
                                                {h.causal_chain.nodes.slice(0, 3).map((node, idx) => (
                                                    <span key={idx} className="text-xs px-2 py-1 rounded-md bg-secondary/50 text-secondary-foreground/80 border border-secondary">
                                                        {node}
                                                    </span>
                                                ))}
                                                {h.causal_chain.nodes.length > 3 && (
                                                    <span className="text-xs px-2 py-1 text-muted-foreground">+{h.causal_chain.nodes.length - 3}</span>
                                                )}
                                            </div>
                                        )}

                                        {/* Metrics */}
                                        <div className="flex items-center justify-between pt-4 border-t border-border/50 text-xs text-muted-foreground">
                                            <div className="flex gap-3">
                                                <span title="Novelty">Nov: {(h.novelty_score * 100).toFixed(0)}%</span>
                                                <span title="Feasibility">Feas: {(h.feasibility_score * 100).toFixed(0)}%</span>
                                            </div>
                                            <div className="font-mono text-primary/80">
                                                Score: {score}
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        );
                    })}
                </div>
            )}

            <div className="flex justify-center">
                <button
                    onClick={handleConfirm}
                    disabled={selectedIds.length === 0}
                    className={`
                        px-8 py-3 rounded-lg text-lg font-medium flex items-center gap-2 transition-all
                        ${selectedIds.length > 0
                            ? "bg-primary text-primary-foreground hover:opacity-90 shadow-lg shadow-primary/25"
                            : "bg-secondary text-muted-foreground cursor-not-allowed"}
                    `}
                >
                    {selectedIds.length > 0 ? `Deep Dive into ${selectedIds.length} Hypotheses` : "Select 1 or 2 to Proceed"}
                    <ArrowRight className="w-5 h-5" />
                </button>
            </div>
        </div>
    );
};
