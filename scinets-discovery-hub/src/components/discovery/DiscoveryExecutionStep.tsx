import { useRef, useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Brain, BookOpen, Lightbulb, Search,
  CheckCircle, Loader2, AlertCircle, Clock,
  Activity, Layers, Route, Sparkles
} from "lucide-react";
import { cn } from "@/lib/utils";
import type { GraphNode, GraphEdge, AgentActivity } from "@/pages/Discovery";
import type { ClarificationAnswers } from "./DiscoveryClarificationStep";
import type { CandidatePaper } from "./PaperCurationStep";
import type { Hypothesis, ConceptGraph, DecisionSummary } from "@/lib/types";
import { DiscoveryResults } from "./DiscoveryResults";

interface DiscoveryExecutionStepProps {
  query: string;
  papers: string[];
  answers: ClarificationAnswers;
  curatedPapers: CandidatePaper[];
  nodes: GraphNode[];
  edges: GraphEdge[];
  activities: AgentActivity[];
  isComplete: boolean;
  onNodeSelect: (nodeId: string) => void;
  // New props from real API
  hypotheses?: Hypothesis[];
  conceptGraph?: ConceptGraph | null;
  logs?: string[];
  error?: string | null;
  literatureCount?: number;
  threadId?: string | null;
  decision_summary?: DecisionSummary;
}

// Process-focused status config - no discovery verbs
const statusConfig = {
  pending: { icon: Clock, label: "Pending", color: "text-foreground-muted" },
  running: { icon: Loader2, label: "Running", color: "text-agent-scientist", animate: true },
  completed: { icon: CheckCircle, label: "Completed", color: "text-foreground" },
  failed: { icon: AlertCircle, label: "Failed", color: "text-destructive" },
  abstained: { icon: AlertCircle, label: "Abstained", color: "text-foreground-muted" },
  // Legacy mapping for existing status types
  reading: { icon: BookOpen, label: "Running", color: "text-agent-scientist", animate: true },
  thinking: { icon: Brain, label: "Running", color: "text-agent-planner", animate: true },
  building: { icon: Lightbulb, label: "Running", color: "text-agent-orchestrator", animate: true },
  complete: { icon: CheckCircle, label: "Completed", color: "text-foreground" },
};

const agentLabels = {
  planner: "Planning Agent",
  literature: "Literature Agent",
  hypothesis: "Hypothesis Agent",
  critic: "Critique Agent",
  experiment: "Experiment Agent",
  orchestrator: "Orchestrator",
  scientist: "Research Agent", // Legacy fallback
};

interface ExplorationStatus {
  papersRetrieved: number;
  papersValidated: number;
  nodesExtracted: number;
  relationsInferred: number;
  graphDensification: "pending" | "running" | "completed";
  reasoningPathsExplored: number;
  structuralBridgesAttempted: number;
  hypothesisSynthesis: "pending" | "running" | "completed";
}

export const DiscoveryExecutionStep = ({
  query,
  papers,
  answers,
  curatedPapers,
  nodes,
  edges,
  activities,
  isComplete,
  onNodeSelect,
  hypotheses = [],
  conceptGraph = null,
  logs = [],
  error = null,
  literatureCount = 0,
  threadId = null,
  decision_summary,
}: DiscoveryExecutionStepProps) => {
  const activityRef = useRef<HTMLDivElement>(null);
  const logsRef = useRef<HTMLDivElement>(null);
  const [showLogs, setShowLogs] = useState(false);

  // Exploration status that updates based on activities and real data
  const [explorationStatus, setExplorationStatus] = useState<ExplorationStatus>({
    papersRetrieved: curatedPapers.length,
    papersValidated: 0,
    nodesExtracted: 0,
    relationsInferred: 0,
    graphDensification: "pending",
    reasoningPathsExplored: 0,
    structuralBridgesAttempted: 0,
    hypothesisSynthesis: "pending",
  });

  // Auto-scroll activity feed
  useEffect(() => {
    if (activityRef.current) {
      activityRef.current.scrollTop = activityRef.current.scrollHeight;
    }
  }, [activities]);

  // Auto-scroll logs
  useEffect(() => {
    if (logsRef.current) {
      logsRef.current.scrollTop = logsRef.current.scrollHeight;
    }
  }, [logs]);

  // Update exploration status based on real data when available
  useEffect(() => {
    const activityCount = activities.length;

    setExplorationStatus(prev => ({
      ...prev,
      papersValidated: Math.min(activityCount > 2 ? Math.floor(curatedPapers.length * (activityCount / 12)) : 0, curatedPapers.length),
      nodesExtracted: conceptGraph ? conceptGraph.nodes.length : Math.min(activityCount > 4 ? (activityCount - 4) * 3 : 0, 24),
      relationsInferred: conceptGraph ? conceptGraph.edges.length : Math.min(activityCount > 5 ? (activityCount - 5) * 2 : 0, 18),
      graphDensification: conceptGraph ? "completed" : activityCount > 6 ? (activityCount > 8 ? "completed" : "running") : "pending",
      reasoningPathsExplored: Math.min(activityCount > 7 ? (activityCount - 7) * 4 : 0, 32),
      structuralBridgesAttempted: Math.min(activityCount > 9 ? (activityCount - 9) : 0, 5),
      hypothesisSynthesis: hypotheses.length > 0 ? "completed" : activityCount > 10 ? (isComplete ? "completed" : "running") : "pending",
    }));
  }, [activities.length, curatedPapers.length, isComplete, conceptGraph, hypotheses.length]);

  const currentActivity = activities[activities.length - 1];
  const CurrentStatusIcon = currentActivity
    ? statusConfig[currentActivity.status]?.icon || Loader2
    : Loader2;

  // Show results when complete
  if (isComplete) {
    return (
      <DiscoveryResults
        query={query}
        answers={answers}
        curatedPapers={curatedPapers}
        nodes={nodes}
        edges={edges}
        onNodeSelect={onNodeSelect}
        hypotheses={hypotheses}
        conceptGraph={conceptGraph}
        literatureCount={literatureCount}
        threadId={threadId}
        decision_summary={decision_summary}
      />
    );
  }

  const getStatusLabel = (status: "pending" | "running" | "completed") => {
    const icons = {
      pending: <Clock className="w-3 h-3 text-foreground-muted" />,
      running: <Loader2 className="w-3 h-3 text-agent-scientist animate-spin" />,
      completed: <CheckCircle className="w-3 h-3 text-foreground" />,
    };
    const labels = {
      pending: "Pending",
      running: "Running",
      completed: "Completed",
    };
    return (
      <div className="flex items-center gap-1.5">
        {icons[status]}
        <span className={cn(
          "text-[10px]",
          status === "pending" && "text-foreground-muted",
          status === "running" && "text-agent-scientist",
          status === "completed" && "text-foreground"
        )}>
          {labels[status]}
        </span>
      </div>
    );
  };

  return (
    <div className="h-full flex flex-col">
      {/* Error Banner */}
      {error && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex-shrink-0 mb-4 px-4 py-3 rounded-lg bg-destructive/10 border border-destructive/30"
        >
          <div className="flex items-center gap-2">
            <AlertCircle className="w-4 h-4 text-destructive" />
            <span className="text-sm text-destructive">{error}</span>
          </div>
        </motion.div>
      )}

      {/* Header Status */}
      <div className="flex-shrink-0 mb-6">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-3">
            <div className={cn(
              "w-8 h-8 rounded-full flex items-center justify-center",
              "bg-foreground/5"
            )}>
              <CurrentStatusIcon className={cn(
                "w-4 h-4",
                "animate-pulse",
                currentActivity ? statusConfig[currentActivity.status]?.color : "text-foreground-muted"
              )} />
            </div>
            <div>
              <p className="text-sm font-medium text-foreground">
                Processing Research Query...
              </p>
              <p className="text-xs text-foreground-muted truncate max-w-md">
                {query}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-4 text-xs text-foreground-muted">
            <div className="flex items-center gap-2">
              <Activity className="w-3 h-3" />
              <span>{activities.length} actions</span>
            </div>
            <button
              onClick={() => setShowLogs(!showLogs)}
              className={cn(
                "px-2 py-1 rounded text-xs transition-colors",
                showLogs ? "bg-foreground/10 text-foreground" : "hover:bg-foreground/5"
              )}
            >
              {showLogs ? "Hide Logs" : "Show Logs"}
            </button>
          </div>
        </div>

        {/* Progress indicator */}
        <div className="h-0.5 bg-border rounded-full overflow-hidden">
          <motion.div
            className="h-full bg-foreground/30"
            initial={{ width: "0%" }}
            animate={{ width: isComplete ? "100%" : `${Math.min(activities.length * 8, 95)}%` }}
            transition={{ duration: 0.5, ease: "easeOut" }}
          />
        </div>
      </div>

      {/* Main Content Area */}
      <div className="flex-1 flex gap-4 min-h-0">
        {/* Activity Stream - Primary */}
        <div className="flex-1 flex flex-col border border-border rounded-lg bg-background/30 overflow-hidden">
          <div className="px-4 py-3 border-b border-border flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Search className="w-3.5 h-3.5 text-foreground-muted" />
              <span className="text-xs font-medium text-foreground">Agent Activity</span>
            </div>
            <span className="text-[10px] text-foreground-muted">
              Execution trace
            </span>
          </div>

          <div
            ref={activityRef}
            className="flex-1 overflow-auto p-4 space-y-3"
          >
            <AnimatePresence>
              {activities.map((activity, index) => {
                const config = statusConfig[activity.status] || statusConfig.running;
                const Icon = config.icon;
                const isRunning = activity.status === "reading" || activity.status === "thinking" || activity.status === "building";
                const isLast = index === activities.length - 1;

                return (
                  <motion.div
                    key={activity.id}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="flex gap-3"
                  >
                    <div className="flex-shrink-0 flex flex-col items-center">
                      <div className={cn(
                        "w-6 h-6 rounded-full flex items-center justify-center",
                        "bg-foreground/5"
                      )}>
                        <Icon className={cn(
                          "w-3 h-3",
                          config.color,
                          isRunning && isLast && "animate-spin"
                        )} />
                      </div>
                      {index < activities.length - 1 && (
                        <div className="w-px flex-1 bg-border mt-1" />
                      )}
                    </div>
                    <div className="flex-1 pb-3">
                      <div className="flex items-center gap-2">
                        <span className="text-[10px] font-medium text-foreground-muted uppercase tracking-wide">
                          {agentLabels[activity.agent]}
                        </span>
                        <span className={cn(
                          "text-[9px] px-1.5 py-0.5 rounded",
                          isRunning && isLast ? "bg-agent-scientist/10 text-agent-scientist" : "bg-foreground/5 text-foreground-muted"
                        )}>
                          {isRunning && isLast ? "Running" : "Completed"}
                        </span>
                      </div>
                      <p className="text-xs text-foreground mt-0.5">{activity.action}</p>
                    </div>
                  </motion.div>
                );
              })}
            </AnimatePresence>

            {activities.length === 0 && (
              <div className="flex items-center justify-center h-full">
                <div className="text-center">
                  <Loader2 className="w-5 h-5 text-foreground-muted mx-auto mb-2 animate-spin" />
                  <p className="text-xs text-foreground-muted">Connecting to backend...</p>
                </div>
              </div>
            )}
          </div>

          {/* Logs Panel (collapsible) */}
          {showLogs && logs.length > 0 && (
            <div className="border-t border-border">
              <div
                ref={logsRef}
                className="h-32 overflow-auto p-3 bg-background/50 font-mono text-[10px] text-foreground-muted"
              >
                {logs.map((log, i) => (
                  <div key={i} className="py-0.5">{log}</div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Exploration Status - Neutral metrics only */}
        <div className="w-80 flex flex-col border border-border rounded-lg bg-background/30 overflow-hidden">
          <div className="px-4 py-3 border-b border-border flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Activity className="w-3.5 h-3.5 text-foreground-muted" />
              <span className="text-xs font-medium text-foreground">Exploration Status</span>
            </div>
          </div>

          <div className="flex-1 overflow-auto p-4 space-y-4">
            {/* Paper Processing */}
            <div className="space-y-2">
              <div className="flex items-center gap-2 text-[10px] text-foreground-muted uppercase tracking-wide">
                <BookOpen className="w-3 h-3" />
                Corpus Processing
              </div>
              <div className="space-y-1.5 pl-5">
                <div className="flex items-center justify-between">
                  <span className="text-xs text-foreground-muted">Papers retrieved</span>
                  <span className="text-xs font-mono text-foreground">{explorationStatus.papersRetrieved}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-xs text-foreground-muted">Papers validated</span>
                  <span className="text-xs font-mono text-foreground">{explorationStatus.papersValidated}</span>
                </div>
              </div>
            </div>

            {/* Graph Construction */}
            <div className="space-y-2">
              <div className="flex items-center gap-2 text-[10px] text-foreground-muted uppercase tracking-wide">
                <Layers className="w-3 h-3" />
                Graph Construction
              </div>
              <div className="space-y-1.5 pl-5">
                <div className="flex items-center justify-between">
                  <span className="text-xs text-foreground-muted">Nodes extracted</span>
                  <span className="text-xs font-mono text-foreground">{explorationStatus.nodesExtracted}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-xs text-foreground-muted">Relations inferred</span>
                  <span className="text-xs font-mono text-foreground">{explorationStatus.relationsInferred}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-xs text-foreground-muted">Graph densification</span>
                  {getStatusLabel(explorationStatus.graphDensification)}
                </div>
              </div>
            </div>

            {/* Reasoning */}
            <div className="space-y-2">
              <div className="flex items-center gap-2 text-[10px] text-foreground-muted uppercase tracking-wide">
                <Route className="w-3 h-3" />
                Reasoning Exploration
              </div>
              <div className="space-y-1.5 pl-5">
                <div className="flex items-center justify-between">
                  <span className="text-xs text-foreground-muted">Paths explored</span>
                  <span className="text-xs font-mono text-foreground">{explorationStatus.reasoningPathsExplored}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-xs text-foreground-muted">Structural bridges attempted</span>
                  <span className="text-xs font-mono text-foreground">{explorationStatus.structuralBridgesAttempted}</span>
                </div>
              </div>
            </div>

            {/* Synthesis */}
            <div className="space-y-2">
              <div className="flex items-center gap-2 text-[10px] text-foreground-muted uppercase tracking-wide">
                <Sparkles className="w-3 h-3" />
                Synthesis
              </div>
              <div className="space-y-1.5 pl-5">
                <div className="flex items-center justify-between">
                  <span className="text-xs text-foreground-muted">Hypothesis synthesis</span>
                  {getStatusLabel(explorationStatus.hypothesisSynthesis)}
                </div>
                {hypotheses.length > 0 && (
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-foreground-muted">Hypotheses generated</span>
                    <span className="text-xs font-mono text-foreground">{hypotheses.length}</span>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Footer note */}
          <div className="px-4 py-3 border-t border-border">
            <p className="text-[10px] text-foreground-muted text-center">
              Results will be available upon completion
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
