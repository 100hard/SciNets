import { useState } from "react";
import { motion } from "framer-motion";
import { FileText, Network, ExternalLink, Star, Beaker, Target } from "lucide-react";
import { cn } from "@/lib/utils";
import type { GraphNode, GraphEdge } from "@/pages/Discovery";
import type { ClarificationAnswers } from "./DiscoveryClarificationStep";
import type { CandidatePaper } from "./PaperCurationStep";
import type { Hypothesis as APIHypothesis, ConceptGraph } from "@/lib/types";
import { HypothesisCard, Hypothesis as UIHypothesis } from "./HypothesisCard";
import { SystemReflection, ReflectionData } from "./SystemReflection";

interface DiscoveryResultsProps {
  query: string;
  answers: ClarificationAnswers;
  curatedPapers: CandidatePaper[];
  nodes: GraphNode[];
  edges: GraphEdge[];
  onNodeSelect: (nodeId: string) => void;
  // New props from real API
  hypotheses?: APIHypothesis[];
  conceptGraph?: ConceptGraph | null;
}

// Convert backend hypothesis to UI format
function convertHypothesis(apiHypothesis: APIHypothesis, index: number): UIHypothesis {
  const supportingCount = (apiHypothesis.evidence || []).filter(e => e.stance === 'support').length;
  const contradictingCount = (apiHypothesis.evidence || []).filter(e => e.stance === 'contradict').length;

  // Determine status based on evidence
  let status: 'supported' | 'mixed' | 'abstained';
  const evidenceList = apiHypothesis.evidence || [];
  if (evidenceList.length === 0) {
    status = 'abstained';
  } else if (contradictingCount === 0 && supportingCount > 0) {
    status = 'supported';
  } else if (contradictingCount > 0) {
    status = 'mixed';
  } else {
    status = 'abstained';
  }

  // Convert evidence to UI format
  const evidence = (apiHypothesis.evidence || []).map(e => ({
    paper: `${e.title}${e.venue ? ` - ${e.venue}` : ''}${e.year ? ` (${e.year})` : ''}`,
    stance: e.stance === 'support' ? 'supporting' as const :
      e.stance === 'contradict' ? 'contradicting' as const : 'neutral' as const,
    confidence: e.strength / 5, // Convert 1-5 to 0-1
  }));

  // Create mechanism chain from domain tags or required data
  const mechanismChain = (apiHypothesis.domain_tags || []).slice(0, 4).map((tag, i) => ({
    id: `m${i}`,
    label: tag,
  }));

  return {
    id: apiHypothesis.id,
    statement: apiHypothesis.text,
    status,
    supportingCount,
    contradictingCount,
    mechanismChain,
    evidence,
    groundingExplanation: apiHypothesis.evidence_summary ||
      `This hypothesis has ${supportingCount} supporting and ${contradictingCount} contradicting pieces of evidence.`,
    synthesis: apiHypothesis.experiment_idea ||
      `Novelty: ${(apiHypothesis.novelty_score * 100).toFixed(0)}% | Feasibility: ${(apiHypothesis.feasibility_score * 100).toFixed(0)}% | Testability: ${(apiHypothesis.testability_score * 100).toFixed(0)}%`,
  };
}

// Fallback mock hypotheses for when no real data available
const mockHypotheses: UIHypothesis[] = [
  {
    id: "h1",
    statement: "Sleep deprivation accelerates amyloid-β accumulation through impaired glymphatic clearance, with chronic sleep restriction reducing clearance efficiency by 40-60%.",
    status: "supported",
    supportingCount: 4,
    contradictingCount: 0,
    mechanismChain: [
      { id: "m1", label: "Sleep Deprivation" },
      { id: "m2", label: "Reduced Slow-Wave Activity" },
      { id: "m3", label: "Impaired Glymphatic Flow" },
      { id: "m4", label: "Aβ Accumulation" },
    ],
    evidence: [
      { paper: "Xie et al. 2013 - Science", stance: "supporting", confidence: 0.92 },
      { paper: "Holth et al. 2019 - Science", stance: "supporting", confidence: 0.88 },
    ],
    groundingExplanation: "This hypothesis is well-grounded in multiple independent experimental studies.",
    synthesis: "The glymphatic system provides a clearance pathway for metabolic waste including amyloid-β.",
  },
];

const mockReflection: ReflectionData = {
  overallConfidence: "moderate",
  stronglySupported: [
    "Glymphatic clearance is reduced during wakefulness and enhanced during sleep",
    "Sleep deprivation leads to measurable increases in brain amyloid-β levels",
  ],
  unresolvedBridges: [
    "The causal direction between sleep disruption and tau pathology remains unclear",
  ],
  suggestedNextSteps: [
    "Investigate longitudinal studies tracking sleep quality and AD biomarker progression",
  ],
  caveats: [
    "Analysis based on available papers; broader corpus may yield different conclusions",
  ],
};

export const DiscoveryResults = ({
  query,
  answers,
  curatedPapers,
  nodes,
  edges,
  onNodeSelect,
  hypotheses: apiHypotheses = [],
  conceptGraph = null,
}: DiscoveryResultsProps) => {
  // Convert API hypotheses to UI format, fallback to mock if empty
  const displayHypotheses: UIHypothesis[] = apiHypotheses.length > 0
    ? apiHypotheses.map((h, i) => convertHypothesis(h, i))
    : mockHypotheses;

  const [expandedHypotheses, setExpandedHypotheses] = useState<string[]>(
    displayHypotheses.length > 0 ? [displayHypotheses[0].id] : []
  );

  const toggleHypothesis = (id: string) => {
    setExpandedHypotheses(prev =>
      prev.includes(id) ? prev.filter(h => h !== id) : [...prev, id]
    );
  };

  const entityCount = conceptGraph
    ? conceptGraph.nodes.length
    : nodes.filter(n => n.type === "entity" || n.type === "mechanism").length;

  const edgeCount = conceptGraph ? conceptGraph.edges.length : edges.length;

  // Generate reflection based on real hypotheses if available
  const reflection: ReflectionData = apiHypotheses.length > 0 ? {
    overallConfidence: apiHypotheses.some(h =>
      h.evidence.filter(e => e.stance === 'support').length >
      h.evidence.filter(e => e.stance === 'contradict').length
    ) ? "moderate" : "low",
    stronglySupported: apiHypotheses
      .filter(h => h.evidence.filter(e => e.stance === 'support').length > 0)
      .slice(0, 3)
      .map(h => h.text.slice(0, 100) + (h.text.length > 100 ? '...' : '')),
    unresolvedBridges: apiHypotheses
      .filter(h => h.evidence.filter(e => e.stance === 'contradict').length > 0)
      .slice(0, 2)
      .map(h => `Conflicting evidence for: ${h.text.slice(0, 60)}...`),
    suggestedNextSteps: [
      "Explore related papers for broader context",
      "Run experiments to validate key hypotheses",
      ...apiHypotheses.slice(0, 2).map(h => h.experiment_idea || `Test hypothesis: ${h.id}`).filter(Boolean),
    ],
    caveats: [
      `Analysis based on ${curatedPapers.length} papers`,
      apiHypotheses.length === 0 ? "No hypotheses generated yet" : `Generated ${apiHypotheses.length} hypotheses`,
    ],
  } : mockReflection;

  return (
    <div className="h-full overflow-auto space-y-6 pb-8">
      {/* Summary Header */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="border border-border rounded-lg bg-background/30 p-6"
      >
        <h2 className="text-lg font-medium text-foreground mb-2">Discovery Complete</h2>
        <p className="text-sm text-foreground-muted mb-4">{query}</p>

        <div className="grid grid-cols-4 gap-4">
          <div className="text-center">
            <p className="text-2xl font-medium text-foreground">{curatedPapers.length}</p>
            <p className="text-xs text-foreground-muted">Papers Analyzed</p>
          </div>
          <div className="text-center">
            <p className="text-2xl font-medium text-foreground">{entityCount}</p>
            <p className="text-xs text-foreground-muted">Entities Found</p>
          </div>
          <div className="text-center">
            <p className="text-2xl font-medium text-foreground">{displayHypotheses.length}</p>
            <p className="text-xs text-foreground-muted">Hypotheses</p>
          </div>
          <div className="text-center">
            <p className="text-2xl font-medium text-foreground">{edgeCount}</p>
            <p className="text-xs text-foreground-muted">Connections</p>
          </div>
        </div>

        {/* Score badges for top hypothesis if available */}
        {apiHypotheses.length > 0 && (
          <div className="mt-4 pt-4 border-t border-border">
            <p className="text-xs text-foreground-muted mb-2">Top Hypothesis Scores</p>
            <div className="flex gap-3">
              <div className="flex items-center gap-1.5 px-2 py-1 rounded bg-foreground/5">
                <Star className="w-3 h-3 text-amber-400" />
                <span className="text-xs text-foreground">
                  Novelty: {(apiHypotheses[0].novelty_score * 100).toFixed(0)}%
                </span>
              </div>
              <div className="flex items-center gap-1.5 px-2 py-1 rounded bg-foreground/5">
                <Beaker className="w-3 h-3 text-blue-400" />
                <span className="text-xs text-foreground">
                  Feasibility: {(apiHypotheses[0].feasibility_score * 100).toFixed(0)}%
                </span>
              </div>
              <div className="flex items-center gap-1.5 px-2 py-1 rounded bg-foreground/5">
                <Target className="w-3 h-3 text-green-400" />
                <span className="text-xs text-foreground">
                  Testability: {(apiHypotheses[0].testability_score * 100).toFixed(0)}%
                </span>
              </div>
            </div>
          </div>
        )}
      </motion.div>

      {/* Hypotheses - Primary focus */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="space-y-3"
      >
        <div className="flex items-center gap-2 px-1">
          <FileText className="w-4 h-4 text-foreground-muted" />
          <h3 className="text-sm font-medium text-foreground">Hypotheses</h3>
          <span className="text-xs text-foreground-muted ml-auto">
            {apiHypotheses.length > 0 ? "From discovery" : "Example hypotheses"} - Click to expand
          </span>
        </div>

        {displayHypotheses.map((hypothesis, index) => (
          <HypothesisCard
            key={hypothesis.id}
            hypothesis={hypothesis}
            index={index}
            isExpanded={expandedHypotheses.includes(hypothesis.id)}
            onToggle={() => toggleHypothesis(hypothesis.id)}
          />
        ))}
      </motion.div>

      {/* System Reflection */}
      <SystemReflection reflection={reflection} />

      {/* Knowledge Graph - Placeholder only */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
      >
        <button
          className={cn(
            "w-full flex items-center justify-between px-4 py-4 rounded-lg",
            "border border-dashed border-border bg-background/20",
            "hover:border-foreground/20 transition-colors group"
          )}
          onClick={() => {
            console.log("Knowledge graph would open here", { nodes, edges, conceptGraph });
          }}
        >
          <div className="flex items-center gap-3">
            <Network className="w-5 h-5 text-foreground-muted group-hover:text-foreground transition-colors" />
            <div className="text-left">
              <span className="text-sm font-medium text-foreground">View reasoning in knowledge graph</span>
              <p className="text-xs text-foreground-muted">
                Explore {entityCount} nodes and {edgeCount} connections
              </p>
            </div>
          </div>
          <ExternalLink className="w-4 h-4 text-foreground-muted group-hover:text-foreground transition-colors" />
        </button>
      </motion.div>
    </div>
  );
};
