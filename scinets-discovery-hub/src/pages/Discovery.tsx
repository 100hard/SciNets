import { useState, useRef, useEffect } from "react";
import { NetworkBackground } from "@/components/NetworkBackground";
import { Header } from "@/components/Header";
import { DiscoveryQueryStep } from "@/components/discovery/DiscoveryQueryStep";
import { DiscoveryClarificationStep, ClarificationAnswers } from "@/components/discovery/DiscoveryClarificationStep";
import { PaperCurationStep, CandidatePaper } from "@/components/discovery/PaperCurationStep";
import { DiscoveryExecutionStep } from "@/components/discovery/DiscoveryExecutionStep";
import { startDiscoveryStream, SSECallback } from "@/lib/api";
import type { Hypothesis, DiscoveryResult, ActivityEvent, ConceptGraph } from "@/lib/types";

export interface GraphNode {
  id: string;
  label: string;
  type: "paper" | "entity" | "hypothesis" | "mechanism";
  x: number;
  y: number;
  pinned?: boolean;
  isNew?: boolean;
}

export interface GraphEdge {
  source: string;
  target: string;
  type: "cites" | "relates" | "supports" | "contradicts";
}

export interface AgentActivity {
  id: string;
  agent: "planner" | "scientist" | "critic" | "orchestrator";
  action: string;
  status: "reading" | "thinking" | "building" | "complete";
  timestamp: Date;
  nodeId?: string;
}

type DiscoveryStep = "query" | "clarification" | "curation" | "execution";

const Discovery = () => {
  const [step, setStep] = useState<DiscoveryStep>("query");
  const [query, setQuery] = useState("");
  const [papers, setPapers] = useState<string[]>([]);
  const [answers, setAnswers] = useState<ClarificationAnswers | null>(null);
  const [curatedPapers, setCuratedPapers] = useState<CandidatePaper[]>([]);
  const [nodes, setNodes] = useState<GraphNode[]>([]);
  const [edges, setEdges] = useState<GraphEdge[]>([]);
  const [activities, setActivities] = useState<AgentActivity[]>([]);
  const [isComplete, setIsComplete] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Real API result state
  const [threadId, setThreadId] = useState<string | null>(null);
  const [hypotheses, setHypotheses] = useState<Hypothesis[]>([]);
  const [conceptGraph, setConceptGraph] = useState<ConceptGraph | null>(null);
  const [logs, setLogs] = useState<string[]>([]);

  // Abort controller for cancellation
  const abortControllerRef = useRef<AbortController | null>(null);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      abortControllerRef.current?.abort();
    };
  }, []);

  const handleQuerySubmit = (q: string, p: string[]) => {
    setQuery(q);
    setPapers(p);
    setStep("clarification");
  };

  const handleClarificationSubmit = (a: ClarificationAnswers) => {
    setAnswers(a);
    setStep("curation");
  };

  const handlePaperCurationSubmit = (selectedPapers: CandidatePaper[]) => {
    setCuratedPapers(selectedPapers);
    setStep("execution");
    startDiscovery(answers!, selectedPapers);
  };

  const startDiscovery = async (clarificationAnswers: ClarificationAnswers, selectedPapers: CandidatePaper[]) => {
    // Reset state
    setNodes([]);
    setEdges([]);
    setActivities([]);
    setLogs([]);
    setHypotheses([]);
    setConceptGraph(null);
    setIsComplete(false);
    setError(null);

    let activityCounter = 0;

    const callbacks: SSECallback = {
      onThreadId: (id) => {
        setThreadId(id);
      },

      onActivity: (activity: ActivityEvent) => {
        const newActivity: AgentActivity = {
          id: `act-${activityCounter++}`,
          agent: activity.agent,
          action: activity.action,
          status: activity.status,
          timestamp: new Date(),
        };
        setActivities(prev => [...prev, newActivity]);
      },

      onLog: (message: string) => {
        setLogs(prev => [...prev, message]);
      },

      onResult: (result: Partial<DiscoveryResult>) => {
        // Update hypotheses if present
        if (result.hypotheses && result.hypotheses.length > 0) {
          setHypotheses(result.hypotheses);
        }

        // Update concept graph if present
        if (result.concept_graph) {
          setConceptGraph(result.concept_graph);

          // Convert to display nodes/edges
          const graphNodes: GraphNode[] = result.concept_graph.nodes.map((node, i) => ({
            id: node.id,
            label: node.label,
            type: (node.type === 'paper' ? 'paper' :
              node.type === 'hypothesis' ? 'hypothesis' :
                node.type === 'mechanism' ? 'mechanism' : 'entity') as GraphNode['type'],
            x: 100 + (i % 5) * 80 + Math.random() * 40,
            y: 100 + Math.floor(i / 5) * 80 + Math.random() * 40,
            isNew: true,
          }));

          const graphEdges: GraphEdge[] = result.concept_graph.edges.map(edge => ({
            source: edge.source,
            target: edge.target,
            type: (edge.relation === 'cites' ? 'cites' :
              edge.relation === 'supports' ? 'supports' :
                edge.relation === 'contradicts' ? 'contradicts' : 'relates') as GraphEdge['type'],
          }));

          setNodes(graphNodes);
          setEdges(graphEdges);

          // Remove isNew flag after animation
          setTimeout(() => {
            setNodes(prev => prev.map(n => ({ ...n, isNew: false })));
          }, 2000);
        }
      },

      onError: (errorMsg: string) => {
        setError(errorMsg);
        console.error('Discovery error:', errorMsg);
      },

      onInterrupt: (data) => {
        console.log('Workflow interrupted, next nodes:', data.next);
        // Could prompt user for input here
      },

      onDone: () => {
        setIsComplete(true);
      },
    };

    // Map depth to approximate paper count for backend
    const depthToMaxPapers: Record<string, number> = {
      quick: 5,
      standard: 10,
      deep: 20,
    };

    // Start the SSE stream
    abortControllerRef.current = await startDiscoveryStream(
      {
        query,
        goal: clarificationAnswers.goal as 'discover' | 'survey' | 'write',
        run_experiments: clarificationAnswers.runExperiments,
        documents: selectedPapers.map(p => p.doi || p.title),
        // Note: depth/timeline are frontend-only for now
        // Backend could accept max_papers if we extend RunRequest
      },
      callbacks
    );
  };

  const handleNodeSelect = (nodeId: string) => {
    console.log("Selected node:", nodeId);
  };

  return (
    <div className="min-h-screen bg-background relative">
      <Header />
      <NetworkBackground />

      <div className="relative z-10 pt-24 px-6 pb-6 h-screen flex flex-col">
        {step === "query" && (
          <div className="flex-1 flex items-center justify-center">
            <DiscoveryQueryStep onSubmit={handleQuerySubmit} />
          </div>
        )}

        {step === "clarification" && (
          <div className="flex-1 flex items-center justify-center">
            <DiscoveryClarificationStep
              query={query}
              onSubmit={handleClarificationSubmit}
              onBack={() => setStep("query")}
            />
          </div>
        )}

        {step === "curation" && (
          <div className="flex-1 flex items-center justify-center overflow-auto py-8">
            <PaperCurationStep
              query={query}
              onSubmit={handlePaperCurationSubmit}
              onBack={() => setStep("clarification")}
            />
          </div>
        )}

        {step === "execution" && answers && (
          <div className="flex-1 min-h-0">
            <DiscoveryExecutionStep
              query={query}
              papers={papers}
              answers={answers}
              curatedPapers={curatedPapers}
              nodes={nodes}
              edges={edges}
              activities={activities}
              isComplete={isComplete}
              onNodeSelect={handleNodeSelect}
              hypotheses={hypotheses}
              conceptGraph={conceptGraph}
              logs={logs}
              error={error}
            />
          </div>
        )}
      </div>
    </div>
  );
};

export default Discovery;
