import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  ChevronDown, ChevronUp, CheckCircle, AlertCircle,
  Minus, ArrowRight, Link2, FlaskConical, Target
} from "lucide-react";
import { cn } from "@/lib/utils";
import {
  ExperimentConfigModal,
  ExperimentInlineDisplay,
  ExperimentConfig,
  ExperimentResult,
  LocalizedCritique
} from "./ExperimentConfigModal";

interface MechanismNode {
  id: string;
  label: string;
}

interface Evidence {
  paper: string;
  stance: "supporting" | "contradicting" | "neutral";
  confidence: number;
  excerpt?: string;
}

interface Constraint {
  text: string;
  type: 'hard' | 'soft';
  importance: number;
}

interface HypothesisRationale {
  disconnected_clusters: string[];
  missing_link: string;
  field_assumption: string;
  structural_reason: string;
  // Adaptive Fields
  epistemic_tension?: string;
  belief_a?: string;
  belief_b?: string;
  consistency_constraint?: string;
  rationale_type?: 'tension' | 'gap' | 'opportunity';
}

export interface Hypothesis {
  id: string;
  statement: string;
  status: "supported" | "mixed" | "abstained" | "failed";
  supportingCount: number;
  contradictingCount: number;
  mechanismChain: MechanismNode[];
  evidence: Evidence[];
  groundingExplanation: string;
  synthesis: string;
  rationale_gap?: HypothesisRationale;
  mechanism_class?: string;
  constraints?: Constraint[];
  strength_profile?: {
    mechanistic_coherence: 'High' | 'Medium' | 'Low';
    empirical_support: 'High' | 'Medium' | 'Low';
    experimental_tractability: 'High' | 'Medium' | 'Low';
    translational_relevance: 'High' | 'Medium' | 'Low';
  };
  confidence_roadmap?: string[];
}

interface HypothesisCardProps {
  hypothesis: Hypothesis;
  index: number;
  isExpanded: boolean;
  onToggle: () => void;
  threadId?: string | null;
}

const statusConfig = {
  supported: {
    label: "Supported",
    color: "bg-emerald-500/10 text-emerald-400 border-emerald-500/20"
  },
  mixed: {
    label: "Mixed",
    color: "bg-amber-500/10 text-amber-400 border-amber-500/20"
  },
  abstained: {
    label: "Abstained",
    color: "bg-foreground/10 text-foreground-muted border-foreground/20"
  },
  failed: {
    label: "Data Failure",
    color: "bg-red-500/10 text-red-500 border-red-500/20"
  },
};

export const HypothesisCard = ({
  hypothesis,
  index,
  isExpanded,
  onToggle,
  threadId,
}: HypothesisCardProps) => {
  const [showSynthesis, setShowSynthesis] = useState(false);
  const [showExperimentModal, setShowExperimentModal] = useState(false);
  const [experimentLoading, setExperimentLoading] = useState(false);
  const [experimentLogs, setExperimentLogs] = useState<string[]>([]);
  const [experimentResult, setExperimentResult] = useState<ExperimentResult | null>(null);
  const [experimentCritique, setExperimentCritique] = useState<LocalizedCritique | null>(null);

  const config = statusConfig[hypothesis.status];

  const supportingEvidence = hypothesis.evidence.filter(e => e.stance === "supporting");
  const contradictingEvidence = hypothesis.evidence.filter(e => e.stance === "contradicting");
  const neutralEvidence = hypothesis.evidence.filter(e => e.stance === "neutral");

  const handleRunExperiment = async (experimentConfig: ExperimentConfig) => {
    setExperimentLoading(true);
    setExperimentLogs([]);
    setExperimentResult(null);
    setExperimentCritique(null);

    try {
      const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8005';
      const activeThreadId = threadId || `local-${Date.now()}`;
      const response = await fetch(`${API_BASE}/run_experiment`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          thread_id: activeThreadId,
          hypothesis_id: hypothesis.id,
          hypothesis_text: hypothesis.statement,
          intent: experimentConfig.intent,
          data_source: experimentConfig.dataSource,
          seed: experimentConfig.seed,
        }),
      });

      const reader = response.body?.getReader();
      if (!reader) throw new Error('No response body');

      const decoder = new TextDecoder();
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const text = decoder.decode(value);
        const lines = text.split('\n').filter(line => line.startsWith('data: '));

        for (const line of lines) {
          const data = line.slice(6);
          if (data === '[DONE]') continue;

          try {
            const parsed = JSON.parse(data);

            if (parsed.type === 'activity') {
              setExperimentLogs(prev => [...prev, parsed.data.action]);
            } else if (parsed.type === 'log') {
              // Handle detailed logs
              setExperimentLogs(prev => [...prev, parsed.data]);
            } else if (parsed.type === 'experiment_result') {
              const result = parsed.data;
              if (result.experiment_result?.metrics) {
                setExperimentResult(result.experiment_result.metrics);
              }
              if (result.localized_critique) {
                setExperimentCritique(result.localized_critique);
              }
            } else if (parsed.type === 'error') {
              setExperimentLogs(prev => [...prev, `Error: ${parsed.data}`]);
            }
          } catch (e) {
            // Skip non-JSON lines
          }
        }
      }
    } catch (error) {
      setExperimentLogs(prev => [...prev, `Failed: ${error}`]);
    } finally {
      setExperimentLoading(false);
    }
  };

  const hasExperiment = experimentResult || experimentLoading || experimentLogs.length > 0;

  return (
    <>
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: index * 0.1 }}
        className="border border-border rounded-lg bg-background/30 overflow-hidden"
      >
        {/* Collapsed header */}
        <button
          onClick={onToggle}
          className="w-full px-4 py-4 flex items-start justify-between hover:bg-foreground/5 transition-colors text-left"
        >
          <div className="flex-1 mr-4">
            <div className="flex items-center gap-3 mb-2">
              <span className="text-xs text-foreground-muted font-medium">H{index + 1}</span>
              <span className={cn(
                "text-[10px] px-2 py-0.5 rounded-full border font-medium",
                config.color
              )}>
                {config.label}
              </span>
              {hypothesis.mechanism_class && (
                <span className="text-[10px] px-2 py-0.5 rounded-full border bg-blue-500/10 text-blue-400 border-blue-500/20 font-medium">
                  {hypothesis.mechanism_class}
                </span>
              )}
              {hasExperiment && (
                <span className="text-[10px] px-2 py-0.5 rounded-full border bg-purple-500/10 text-purple-400 border-purple-500/20 font-medium">
                  Explored
                </span>
              )}
            </div>
            <p className="text-sm text-foreground leading-relaxed">
              {hypothesis.statement}
            </p>
            <div className="flex items-center gap-4 mt-2 text-xs text-foreground-muted">
              <span className="flex items-center gap-1">
                <CheckCircle className="w-3 h-3 text-emerald-400" />
                {hypothesis.supportingCount} supporting
              </span>
              <span className="flex items-center gap-1">
                <AlertCircle className="w-3 h-3 text-red-400" />
                {hypothesis.contradictingCount} contradicting
              </span>
              {neutralEvidence.length > 0 && (
                <span className="flex items-center gap-1">
                  <Minus className="w-3 h-3 text-foreground-muted" />
                  {neutralEvidence.length} neutral
                </span>
              )}
            </div>
          </div>
          {isExpanded ? (
            <ChevronUp className="w-4 h-4 text-foreground-muted flex-shrink-0 mt-1" />
          ) : (
            <ChevronDown className="w-4 h-4 text-foreground-muted flex-shrink-0 mt-1" />
          )}
        </button>

        {/* Expanded content */}
        <AnimatePresence>
          {isExpanded && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: "auto", opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              className="overflow-hidden"
            >
              <div className="px-4 pb-4 space-y-6 border-t border-border pt-4">

                {/* NEW: Why this hypothesis exists (Adaptive Layout) */}
                {hypothesis.rationale_gap && (
                  <div>
                    {/* Header with Adaptive Icon/Color */}
                    <div className="mb-4">
                      <h4 className={cn(
                        "text-xs font-medium mb-1 flex items-center gap-2",
                        hypothesis.rationale_gap.rationale_type === 'tension' ? "text-amber-400" :
                          hypothesis.rationale_gap.rationale_type === 'opportunity' ? "text-emerald-400" :
                            "text-purple-400"
                      )}>
                        <Link2 className="w-3.5 h-3.5" />
                        {hypothesis.rationale_gap.rationale_type === 'tension' ? "Epistemic Tension Resolved" :
                          hypothesis.rationale_gap.rationale_type === 'opportunity' ? "Exploratory Opportunity" :
                            "Structural Gap Identified"}
                      </h4>

                      {/* Adaptive Description */}
                      {hypothesis.rationale_gap.epistemic_tension && hypothesis.rationale_gap.rationale_type === 'tension' ? (
                        <p className="text-sm font-medium text-foreground/90 pl-5 italic border-l-2 border-amber-500/30">
                          "{hypothesis.rationale_gap.epistemic_tension}"
                        </p>
                      ) : hypothesis.rationale_gap.rationale_type === 'opportunity' ? (
                        <p className="text-[11px] text-foreground-muted pl-5">
                          This hypothesis leverages emerging evidence in a sparse domain.
                        </p>
                      ) : (
                        <p className="text-[11px] text-foreground-muted pl-5">
                          This hypothesis bridges two disconnected research clusters.
                        </p>
                      )}
                    </div>

                    {/* LAYOUT 1: TENSION (Two Truths + Constraint) */}
                    {hypothesis.rationale_gap.rationale_type === 'tension' && (
                      <>
                        {(hypothesis.rationale_gap.belief_a || hypothesis.rationale_gap.belief_b) && (
                          <div className="mb-6 pl-5">
                            <span className="text-[10px] uppercase tracking-wider text-foreground-muted font-bold mb-2 block">
                              The Field's Conflict
                            </span>
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                              <div className="p-3 bg-foreground/5 rounded border border-foreground/10">
                                <span className="text-[10px] text-foreground-muted block mb-1">Establish Belief A</span>
                                <p className="text-xs text-foreground/80">{hypothesis.rationale_gap.belief_a || "..."}</p>
                              </div>
                              <div className="p-3 bg-foreground/5 rounded border border-foreground/10">
                                <span className="text-[10px] text-foreground-muted block mb-1">Establish Belief B</span>
                                <p className="text-xs text-foreground/80">{hypothesis.rationale_gap.belief_b || "..."}</p>
                              </div>
                            </div>
                          </div>
                        )}

                        {/* Constraint */}
                        {hypothesis.rationale_gap.consistency_constraint && (
                          <div className="pl-5 mb-4">
                            <div className="p-3 rounded border border-amber-500/20 bg-amber-500/5">
                              <span className="text-[10px] uppercase tracking-wider text-amber-400/80 font-bold mb-1 block flex items-center gap-1">
                                <AlertCircle className="w-3 h-3" />
                                REQUIRED RESOLUTION
                              </span>
                              <p className="text-xs text-foreground/90 leading-snug">
                                {hypothesis.rationale_gap.consistency_constraint}
                              </p>
                            </div>
                          </div>
                        )}
                      </>
                    )}

                    {/* LAYOUT 2 & 3: GAP / OPPORTUNITY (The Grid) */}
                    {(hypothesis.rationale_gap.rationale_type === 'gap' || hypothesis.rationale_gap.rationale_type === 'opportunity' || !hypothesis.rationale_gap.rationale_type) && (
                      <div className="pl-5 mb-6">
                        <span className="text-[10px] uppercase tracking-wider text-foreground-muted font-bold mb-2 block">
                          {hypothesis.rationale_gap.rationale_type === 'opportunity' ? "Emerging Evidence Context" : "Why Existing Models Fail"}
                        </span>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                          {/* Disconnected Clusters (Visual) */}
                          <div className="p-3 rounded border border-purple-500/20 bg-purple-500/5">
                            <span className="text-[10px] uppercase tracking-wider text-purple-400/80 font-bold mb-1 block">
                              {hypothesis.rationale_gap.rationale_type === 'opportunity' ? "Relevant Domains" : "Disconnected Clusters"}
                            </span>
                            <div className="flex items-center justify-center gap-2 mb-3 py-2 border-b border-purple-500/10 border-dashed">
                              <span className="text-[10px] px-2 py-1 rounded bg-purple-500/10 text-purple-300 border border-purple-500/10 truncate max-w-[45%]">
                                {hypothesis.rationale_gap.disconnected_clusters[0] || "Component A"}
                              </span>
                              <span className="text-[10px] text-purple-400/40 tracking-widest font-mono">
                                {hypothesis.rationale_gap.rationale_type === 'opportunity' ? "→" : "✕✕✕"}
                              </span>
                              <span className="text-[10px] px-2 py-1 rounded bg-purple-500/10 text-purple-300 border border-purple-500/10 truncate max-w-[45%]">
                                {hypothesis.rationale_gap.disconnected_clusters[1] || "Component B"}
                              </span>
                            </div>
                          </div>

                          {/* Missing Link */}
                          <div className="p-3 rounded border border-purple-500/20 bg-purple-500/5">
                            <span className="text-[10px] uppercase tracking-wider text-purple-400/80 font-bold mb-1 block">
                              {hypothesis.rationale_gap.rationale_type === 'opportunity' ? "The Opportunity" : "Missing Link"}
                            </span>
                            <p className="text-xs text-foreground/90 leading-snug">
                              {hypothesis.rationale_gap.missing_link}
                            </p>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                )}




                {/* NEW: 4-Axis Strength Profile */}
                {hypothesis.strength_profile && (
                  <div className="mb-6 grid grid-cols-2 md:grid-cols-4 gap-2">
                    {Object.entries(hypothesis.strength_profile).map(([key, value]) => (
                      <div key={key} className="p-2 bg-card border border-border/50 rounded flex flex-col items-center text-center">
                        <span className="text-[10px] uppercase text-muted-foreground font-bold tracking-wider mb-1">
                          {key.replace('_', ' ')}
                        </span>
                        <span className={cn(
                          "text-xs font-medium px-2 py-0.5 rounded-full border",
                          value === 'High' ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/20" :
                            value === 'Medium' ? "bg-amber-500/10 text-amber-400 border-amber-500/20" :
                              "bg-red-500/10 text-red-400 border-red-500/20"
                        )}>
                          {value}
                        </span>
                      </div>
                    ))}
                  </div>
                )}

                {/* NEW: Hypothesis Pressure (Constraints) */}
                {hypothesis.constraints && hypothesis.constraints.length > 0 && (
                  <div className="pl-5 mb-6">
                    <span className="text-[10px] uppercase tracking-wider text-red-500/70 font-bold mb-2 block flex items-center gap-2">
                      <span className="w-1.5 h-1.5 rounded-full bg-red-400 animate-pulse" />
                      Hypothesis Pressure (Hard Constraints)
                    </span>
                    <div className="space-y-2">
                      {hypothesis.constraints.map((constraint, i) => (
                        <div key={i} className="flex items-start gap-2 p-2 rounded bg-red-500/5 border border-red-500/10">
                          <span className="text-xs font-mono text-red-400 font-bold mt-0.5">{i + 1}.</span>
                          <p className="text-xs text-foreground/90">{constraint.text}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* A. Mechanistic Chain */}
                <div>
                  <h4 className="text-xs font-medium text-foreground mb-3 flex items-center gap-2">
                    <Link2 className="w-3.5 h-3.5 text-foreground-muted" />
                    Mechanistic Chain
                  </h4>
                  <div className="flex items-center gap-2 flex-wrap">
                    {hypothesis.mechanismChain.map((node, i) => (
                      <div key={node.id} className="flex items-center gap-2">
                        <button
                          className={cn(
                            "px-3 py-1.5 rounded border border-border bg-foreground/5",
                            "text-xs text-foreground hover:border-foreground/30 transition-colors"
                          )}
                        >
                          {node.label}
                        </button>
                        {i < hypothesis.mechanismChain.length - 1 && (
                          <ArrowRight className="w-3.5 h-3.5 text-foreground-muted" />
                        )}
                      </div>
                    ))}
                  </div>
                </div>

                {/* B. Evidence Table */}
                <div>
                  <h4 className="text-xs font-medium text-foreground mb-3">Evidence</h4>

                  {/* Supporting */}
                  {supportingEvidence.length > 0 && (
                    <div className="mb-3">
                      <div className="flex items-center gap-2 mb-2">
                        <CheckCircle className="w-3.5 h-3.5 text-emerald-400" />
                        <span className="text-xs text-foreground-muted">Supporting ({supportingEvidence.length})</span>
                      </div>
                      <div className="border border-border rounded divide-y divide-border">
                        {supportingEvidence.map((ev, i) => (
                          <div key={i} className="px-3 py-2 flex items-center justify-between">
                            <span className="text-xs text-foreground">{ev.paper}</span>
                            <div className="flex items-center gap-2">
                              <div className="w-16 h-1.5 bg-foreground/10 rounded overflow-hidden">
                                <div
                                  className="h-full bg-emerald-400/60"
                                  style={{ width: `${ev.confidence * 100}%` }}
                                />
                              </div>
                              <span className="text-[10px] text-foreground-muted w-8">
                                {Math.round(ev.confidence * 100)}%
                              </span>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Contradicting */}
                  {contradictingEvidence.length > 0 && (
                    <div className="mb-3">
                      <div className="flex items-center gap-2 mb-2">
                        <AlertCircle className="w-3.5 h-3.5 text-red-400" />
                        <span className="text-xs text-foreground-muted">Contradicting ({contradictingEvidence.length})</span>
                      </div>
                      <div className="border border-border rounded divide-y divide-border">
                        {contradictingEvidence.map((ev, i) => (
                          <div key={i} className="px-3 py-2 flex items-center justify-between">
                            <span className="text-xs text-foreground">{ev.paper}</span>
                            <div className="flex items-center gap-2">
                              <div className="w-16 h-1.5 bg-foreground/10 rounded overflow-hidden">
                                <div
                                  className="h-full bg-red-400/60"
                                  style={{ width: `${ev.confidence * 100}%` }}
                                />
                              </div>
                              <span className="text-[10px] text-foreground-muted w-8">
                                {Math.round(ev.confidence * 100)}%
                              </span>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Neutral */}
                  {neutralEvidence.length > 0 && (
                    <div>
                      <div className="flex items-center gap-2 mb-2">
                        <Minus className="w-3.5 h-3.5 text-foreground-muted" />
                        <span className="text-xs text-foreground-muted">Neutral ({neutralEvidence.length})</span>
                      </div>
                      <div className="border border-border rounded divide-y divide-border">
                        {neutralEvidence.map((ev, i) => (
                          <div key={i} className="px-3 py-2 flex items-center justify-between">
                            <span className="text-xs text-foreground">{ev.paper}</span>
                            <div className="flex items-center gap-2">
                              <div className="w-16 h-1.5 bg-foreground/10 rounded overflow-hidden">
                                <div
                                  className="h-full bg-foreground/30"
                                  style={{ width: `${ev.confidence * 100}%` }}
                                />
                              </div>
                              <span className="text-[10px] text-foreground-muted w-8">
                                {Math.round(ev.confidence * 100)}%
                              </span>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>

                {/* C. Grounding Status */}
                <div>
                  <h4 className="text-xs font-medium text-foreground mb-2">Grounding Status</h4>
                  <p className="text-xs text-foreground-muted leading-relaxed">
                    {hypothesis.groundingExplanation}
                  </p>
                </div>

                {/* D. Actionability Roadmap (NEW) */}
                {hypothesis.confidence_roadmap && hypothesis.confidence_roadmap.length > 0 && (
                  <div className="p-3 bg-purple-500/5 border border-purple-500/10 rounded">
                    <h4 className="text-xs font-bold uppercase tracking-wider text-purple-400 mb-2 flex items-center gap-2">
                      <Target className="w-3.5 h-3.5" /> What would increase confidence?
                    </h4>
                    <ul className="space-y-1.5">
                      {hypothesis.confidence_roadmap.map((item, i) => (
                        <li key={i} className="flex items-start gap-2 text-xs text-foreground/80">
                          <span className="w-1 h-1 rounded-full bg-purple-400 mt-1.5 shrink-0" />
                          {item}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {/* D. Natural Language Explanation */}
                <div>
                  <button
                    onClick={() => setShowSynthesis(!showSynthesis)}
                    className="flex items-center gap-2 text-xs text-foreground-muted hover:text-foreground transition-colors"
                  >
                    {showSynthesis ? (
                      <ChevronUp className="w-3.5 h-3.5" />
                    ) : (
                      <ChevronDown className="w-3.5 h-3.5" />
                    )}
                    <span className="font-medium">Generated Synthesis</span>
                  </button>

                  <AnimatePresence>
                    {showSynthesis && (
                      <motion.div
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: "auto", opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        className="overflow-hidden"
                      >
                        <div className="mt-3 p-3 border border-border rounded bg-foreground/5">
                          <p className="text-xs text-foreground-muted leading-relaxed italic">
                            {hypothesis.synthesis}
                          </p>
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>

                {/* E. Explore Computationally Button */}
                <div className="pt-2 border-t border-border">
                  <button
                    onClick={() => setShowExperimentModal(true)}
                    className={cn(
                      "flex items-center gap-2 px-4 py-2.5 rounded-lg",
                      "bg-purple-500/10 border border-purple-500/20",
                      "text-xs font-medium text-purple-400",
                      "hover:bg-purple-500/20 hover:border-purple-500/30 transition-colors"
                    )}
                  >
                    <FlaskConical className="w-3.5 h-3.5" />
                    Explore computationally
                  </button>
                </div>

                {/* F. Inline Experiment Results (if any) */}
                {hasExperiment && (
                  <ExperimentInlineDisplay
                    isLoading={experimentLoading}
                    logs={experimentLogs}
                    result={experimentResult}
                    critique={experimentCritique}
                  />
                )}
              </div>

            </motion.div >
          )}
        </AnimatePresence >
      </motion.div >

      {/* Experiment Config Modal */}
      < ExperimentConfigModal
        isOpen={showExperimentModal}
        onClose={() => setShowExperimentModal(false)}
        hypothesisId={hypothesis.id}
        hypothesisText={hypothesis.statement}
        onRunExperiment={handleRunExperiment}
      />
    </>
  );
};

