import { useRef, useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Brain, BookOpen, Lightbulb, Search,
  CheckCircle, Loader2, AlertCircle, Clock,
  Activity, Layers, Route, Sparkles, FlaskConical,
  MessageSquare, ChevronRight
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
  hypotheses?: Hypothesis[];
  conceptGraph?: ConceptGraph | null;
  logs?: string[];
  error?: string | null;
  literatureCount?: number;
  threadId?: string | null;
  decision_summary?: DecisionSummary;
  onReset?: () => void;
}

// Process-focused status config
const statusConfig = {
  pending: { icon: Clock, label: "Pending", color: "text-foreground-muted" },
  // Removed "running" as it is not a valid AgentActivity status
  completed: { icon: CheckCircle, label: "Completed", color: "text-foreground" },
  failed: { icon: AlertCircle, label: "Failed", color: "text-destructive" },
  abstained: { icon: AlertCircle, label: "Abstained", color: "text-foreground-muted" },
  reading: { icon: BookOpen, label: "Reading", color: "text-blue-500", animate: true },
  thinking: { icon: Brain, label: "Reasoning", color: "text-purple-500", animate: true },
  building: { icon: Layers, label: "Building", color: "text-indigo-500", animate: true },
  experimenting: { icon: FlaskConical, label: "Experimenting", color: "text-orange-500", animate: true },
  complete: { icon: CheckCircle, label: "Completed", color: "text-green-500" },
};

const agentLabels = {
  planner: "Planning Agent",
  literature: "Literature Agent",
  hypothesis: "Hypothesis Agent",
  critic: "Critique Agent",
  experiment: "Experiment Agent",
  orchestrator: "Orchestrator",
  scientist: "Research Agent",
  decision: "Decision Agent",
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
  onReset,
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

  // Auto-scroll activity list
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
        onReset={onReset}
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

  // Process logs to find the latest meaningful "thought"
  const getThinkingMessage = (logs: string[], agentType: string) => {
    // 1. Scan backwards for the last "clean" message
    for (let i = logs.length - 1; i >= 0; i--) {
      let log = logs[i];

      // FILTER: Skip technical/raw logs
      if (
        log.includes("[Result]") ||
        log.includes("content='{") ||
        log.includes('"{') ||
        log.includes("http") ||
        log.includes("Error:") ||
        log.length > 150 // Skip overly long dumps
      ) {
        continue;
      }

      // CLEANUP: Remove brackets and internal prefixes
      log = log.replace(/\[.*?\]/g, "").trim();
      log = log.replace(/^log:\s*/i, "");

      // MAPPING: Beautify common backend terms
      if (log.toLowerCase().includes("openalex")) return "Querying global knowledge graph...";
      if (log.toLowerCase().includes("retrieving")) return "Retrieving academic sources...";
      if (log.toLowerCase().includes("analyzing")) return "Synthesizing information...";

      // If fairly clean, return it
      if (log.length > 5) return log;
    }

    // Default fallbacks if no clean logs found
    const defaults: Record<string, string> = {
      literature: "Reviewing academic papers...",
      planner: "Structuring research strategy...",
      scientist: "Analyzing data patterns...",
      node: "Processing...",
      orchestrator: "Coordinating agent workflow...",
      critic: "Verifying consistency...",
      hypothesis: "Formulating theories..."
    };

    return defaults[agentType] || "Thinking...";
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
            {/* Simple Header Icon */}
            <div className={cn(
              "w-8 h-8 rounded-full flex items-center justify-center",
              "bg-foreground/5"
            )}>
              <Loader2 className={cn(
                "w-4 h-4 text-primary animate-spin"
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
            {/* New Reset Button */}
            {onReset && (
              <button
                onClick={onReset}
                className="px-3 py-1.5 rounded-md bg-destructive/10 text-destructive hover:bg-destructive/20 border border-destructive/20 transition-colors flex items-center gap-1.5 font-medium"
              >
                <AlertCircle className="w-3.5 h-3.5" />
                Stop & Reset
              </button>
            )}

            {/* Stats */}
            <div className="flex items-center gap-2">
              <Activity className="w-3 h-3" />
              <span>{activities.length} steps</span>
            </div>
            <button
              onClick={() => setShowLogs(!showLogs)}
              className={cn(
                "px-2 py-1 rounded text-xs transition-colors",
                showLogs ? "bg-foreground/10 text-foreground" : "hover:bg-foreground/5"
              )}
            >
              {showLogs ? "Hide Log History" : "View Log History"}
            </button>
          </div>
        </div>

        {/* Progress indicator */}
        <div className="h-0.5 bg-border rounded-full overflow-hidden">
          <motion.div
            className="h-full bg-foreground/30"
            initial={{ width: "0%" }}
            animate={{ width: isComplete ? "100%" : `${Math.min(activities.length * 8, 95)}% ` }}
            transition={{ duration: 0.5, ease: "easeOut" }}
          />
        </div>
      </div>

      {/* Main Content Area */}
      <div className="flex-1 flex gap-4 min-h-0">
        {/* Activity List - Minimal with Enhanced Active State */}
        <div className="flex-1 flex flex-col border border-border rounded-lg bg-background/30 overflow-hidden">
          <div className="px-4 py-3 border-b border-border flex items-center justify-between">
            <div className="flex items-center gap-2">
              <MessageSquare className="w-3.5 h-3.5 text-foreground-muted" />
              <span className="text-xs font-medium text-foreground">Agent Activity</span>
            </div>
          </div>

          <div
            ref={activityRef}
            className="flex-1 overflow-auto p-4 space-y-3"
          >
            <AnimatePresence>
              {activities.map((activity, index) => {
                // Determine config and running state safely
                const config = statusConfig[activity.status] || statusConfig.completed;
                const Icon = config.icon;

                // Correctly check running state against valid types
                const isRunning = activity.status === "reading" || activity.status === "thinking" || activity.status === "building";
                const isLast = index === activities.length - 1;

                // Grab the very last log line if this is the active agent to show "Thinking"
                // const lastLog = isLast && isRunning && logs.length > 0 ? logs[logs.length - 1] : null;
                const thinkingMessage = isLast && isRunning ? getThinkingMessage(logs, activity.agent) : null;

                return (
                  <motion.div
                    key={activity.id}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="flex gap-3"
                  >
                    <div className="flex-shrink-0 flex flex-col items-center w-6">
                      {/* Vertical connector line */}
                      {index < activities.length - 1 && (
                        <div className="w-px flex-1 bg-border mt-1" />
                      )}
                    </div>

                    <div className="flex-1 pb-3">
                      <div className="flex items-center gap-3">
                        <span className="text-[10px] font-medium text-foreground-muted uppercase tracking-wide w-24 flex-shrink-0">
                          {agentLabels[activity.agent] || activity.agent}
                        </span>

                        <div className="flex-1">
                          <div className="flex items-center gap-2">
                            <span className={cn(
                              "text-[9px] px-1.5 py-0.5 rounded",
                              isRunning && isLast ? "bg-primary/10 text-primary" : "bg-foreground/5 text-foreground-muted"
                            )}>
                              {isRunning && isLast ? "Running" : "Completed"}
                            </span>
                            <p className="text-xs text-foreground font-medium">{activity.action}</p>
                          </div>

                          {/* ENHANCEMENT: Automated "Thinking" Line for Active Agent */}
                          {isLast && isRunning && (
                            <motion.div
                              initial={{ opacity: 0, height: 0 }}
                              animate={{ opacity: 1, height: "auto" }}
                              className="mt-2 pl-2 border-l-2 border-primary/20"
                            >
                              <div className="flex items-center gap-2 text-[10px] text-foreground-muted font-mono">
                                <Loader2 className="w-3 h-3 animate-spin text-primary" />
                                <motion.span
                                  key={thinkingMessage} // Animate text changes
                                  initial={{ opacity: 0 }}
                                  animate={{ opacity: 0.7 }}
                                  className="opacity-70"
                                >
                                  {thinkingMessage}
                                </motion.span>
                              </div>
                            </motion.div>
                          )}
                        </div>
                      </div>
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

          {/* Logs Panel (Hidden by default, user can toggle for full history) */}
          {showLogs && logs.length > 0 && (
            <div className="border-t border-border">
              <div
                ref={logsRef}
                className="h-32 overflow-auto p-3 bg-background/50 font-mono text-[10px] text-foreground-muted"
              >
                {logs.map((log, i) => (
                  <div key={i} className="py-0.5 border-l-2 border-transparent hover:border-foreground/20 pl-1 active:bg-foreground/5">
                    <span className="opacity-50 mr-2">{i + 1}.</span>
                    {log}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Exploration Status - Minimal Metrics */}
        <div className="w-80 flex flex-col border border-border rounded-lg bg-background/30 overflow-hidden">
          <div className="px-4 py-3 border-b border-border flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Activity className="w-3.5 h-3.5 text-foreground-muted" />
              <span className="text-xs font-medium text-foreground">Exploration Status</span>
            </div>
          </div>

          <div className="flex-1 overflow-auto p-4 space-y-4">
            {/* Minimal Metrics - No fancy graphs, just clean data */}
            <div className="space-y-4 text-xs">
              <div className="flex justify-between items-center py-2 border-b border-border/50">
                <span className="text-foreground-muted">Papers Processed</span>
                <span className="font-mono text-foreground">{explorationStatus.papersRetrieved}</span>
              </div>
              <div className="flex justify-between items-center py-2 border-b border-border/50">
                <span className="text-foreground-muted">Concepts Extracted</span>
                <span className="font-mono text-foreground">{explorationStatus.nodesExtracted}</span>
              </div>
              <div className="flex justify-between items-center py-2 border-b border-border/50">
                <span className="text-foreground-muted">Relations Inferred</span>
                <span className="font-mono text-foreground">{explorationStatus.relationsInferred}</span>
              </div>
              <div className="flex justify-between items-center py-2 border-b border-border/50">
                <span className="text-foreground-muted">Hypotheses</span>
                <span className="font-mono text-foreground">{hypotheses.length || "-"}</span>
              </div>
            </div>

            {/* Active Phase Indicator */}
            <div className="pt-4">
              <div className="text-[10px] uppercase text-foreground-muted tracking-wide mb-2">Current Phase</div>
              <div className="bg-foreground/5 rounded p-2 text-xs font-medium flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-primary animate-pulse" />
                {activities.length > 0 ? "Processing" : "Waiting"}
                <span className="ml-auto opacity-50 font-normal">
                  {explorationStatus.graphDensification === "running" ? "Densifying Graph" :
                    explorationStatus.hypothesisSynthesis === "running" ? "Synthesizing" : "Active"}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
