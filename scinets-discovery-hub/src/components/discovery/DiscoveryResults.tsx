import { useState } from "react";
import { motion } from "framer-motion";
import { FileText, Network, ExternalLink, Star, Beaker, Target, BookOpen, Lightbulb } from "lucide-react";
import { cn } from "@/lib/utils";
import type { GraphNode, GraphEdge } from "@/pages/Discovery";
import type { ClarificationAnswers } from "./DiscoveryClarificationStep";
import type { CandidatePaper } from "./PaperCurationStep";
import type { Hypothesis as APIHypothesis, ConceptGraph, DecisionSummary as IDecisionSummary } from "@/lib/types";
import { HypothesisCard, Hypothesis as UIHypothesis } from "./HypothesisCard";
import { SystemReflection, ReflectionData } from "./SystemReflection";
import { DecisionSummary } from "./DecisionSummary";


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
  literatureCount?: number;
  threadId?: string | null;
  decision_summary?: IDecisionSummary;
}

interface ExperimentConfig {
  hypothesisId: string;
  technique: string;
  parameters: Record<string, any>;
}

// Convert backend hypothesis to UI format
function convertHypothesis(apiHypothesis: APIHypothesis, index: number): UIHypothesis {
  const supportingCount = (apiHypothesis.evidence || []).filter(e => e.stance === 'support').length;
  const contradictingCount = (apiHypothesis.evidence || []).filter(e => e.stance === 'contradict').length;

  // Determine status based on evidence
  let status: 'supported' | 'mixed' | 'abstained' | 'failed';
  const evidenceList = apiHypothesis.evidence || [];
  const evidenceStatus = (apiHypothesis as any).evidence_status; // Access dynamic property

  if (evidenceStatus === 'failed_external') {
    status = 'failed';
  } else if (evidenceList.length === 0) {
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
    rationale_gap: apiHypothesis.rationale_gap,
    mechanism_class: apiHypothesis.mechanism_class,
    strength_profile: apiHypothesis.strength_profile,
    confidence_roadmap: apiHypothesis.confidence_roadmap,
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

export const DiscoveryResults = ({
  query,
  answers,
  curatedPapers,
  nodes,
  edges,
  onNodeSelect,
  hypotheses: apiHypotheses = [],
  conceptGraph = null,
  literatureCount = 0,
  threadId = null,
  decision_summary,
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

  // Use real backend count if available (fixes "0 Papers Analyzed" bug)
  const displayPaperCount = literatureCount > 0 ? literatureCount : curatedPapers.length;

  const handleRunExperiment = (config: ExperimentConfig) => {
    // Pass content to HypothesisCard if needed, but currently HypothesisCard handles its own experiment logic
    console.log("Run experiment", config);
  };


  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-12">
      {/* 1. Global Header */}
      <div className="flex flex-col gap-4">
        <div className="flex items-center justify-between">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-primary/10 text-primary text-xs font-medium">
            Discovery Complete
          </div>
          <div className="flex items-center gap-4 text-xs font-mono text-muted-foreground">
            <span title="Papers Analyzed" className="flex items-center gap-1.5">{displayPaperCount} Papers</span>
            <span title="Hypotheses Generated" className="flex items-center gap-1.5">{apiHypotheses.length} Hypotheses</span>
            <span title="Connections Explored" className="flex items-center gap-1.5">{edgeCount} Connections</span>
          </div>
        </div>
        <motion.h1
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-3xl md:text-4xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-foreground to-foreground/70"
        >
          {query}
        </motion.h1>
      </div>

      {/* 2. Detailed Hypothesis List (Analysis Details) - MOVED TO TOP */}
      <div className="space-y-4">
        <div className="flex items-center justify-between px-1">
          <h2 className="text-xl font-semibold flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-primary" />
            Analysis Details
          </h2>
          <span className="text-xs text-muted-foreground">Click cards to expand</span>
        </div>

        <div className="space-y-6">
          {displayHypotheses.map((hypothesis, index) => (
            <div id={`hypothesis-${hypothesis.id}`} key={hypothesis.id} className="scroll-mt-24">
              <HypothesisCard
                hypothesis={hypothesis}
                index={index}
                isExpanded={expandedHypotheses.includes(hypothesis.id)}
                onToggle={() => toggleHypothesis(hypothesis.id)}
                threadId={threadId}
              />
            </div>
          ))}
        </div>
      </div>



      {/* 4. Decision Summary - MOVED TO BOTTOM */}
      {decision_summary && (
        <DecisionSummary summary={decision_summary} />
      )}

      {/* 5. Graph Access */}
      <div className="pt-8 border-t border-border">
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
            <Network className="w-5 h-5 text-muted-foreground group-hover:text-foreground transition-colors" />
            <div className="text-left">
              <span className="text-sm font-medium text-foreground">View reasoning in knowledge graph</span>
              <p className="text-xs text-muted-foreground">
                Explore {entityCount} nodes and {edgeCount} connections
              </p>
            </div>
          </div>
          <ExternalLink className="w-4 h-4 text-muted-foreground group-hover:text-foreground transition-colors" />
        </button>
      </div>

    </div>
  );
};
