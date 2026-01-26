import { useState, useRef, useEffect } from "react";
import { NetworkBackground } from "@/components/NetworkBackground";
import { Header } from "@/components/Header";
import { DiscoveryQueryStep } from "@/components/discovery/DiscoveryQueryStep";
import { DiscoveryClarificationStep, ClarificationAnswers } from "@/components/discovery/DiscoveryClarificationStep";
import { PaperCurationStep, CandidatePaper } from "@/components/discovery/PaperCurationStep";
import { DiscoveryExecutionStep } from "@/components/discovery/DiscoveryExecutionStep";
import { startDiscoveryStream, resumeDiscoveryStream, SSECallback } from "@/lib/api";
import type { Hypothesis, DiscoveryResult, ActivityEvent, ConceptGraph, DecisionSummary } from "@/lib/types";
import { Loader2 } from "lucide-react";
import { SearchOverlay } from "@/components/discovery/SearchOverlay";
import { HypothesisSelection } from "@/components/HypothesisSelection";
import { useToast } from "@/hooks/use-toast";
import type { QuotaInfo } from "@/lib/types";
import { useAuth } from "@/context/AuthContext";
import { cn } from "../lib/utils.ts";
import { config } from "../config";

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
  agent: "planner" | "scientist" | "critic" | "orchestrator" | "literature" | "hypothesis" | "experiment" | "decision";
  action: string;
  status: "reading" | "thinking" | "building" | "complete" | "failed" | "abstained";
  timestamp: Date;
  nodeId?: string;
}

type DiscoveryStep = "query" | "clarification" | "searching" | "curation" | "execution" | "selection";

const Discovery = () => {
  const { user } = useAuth();
  const { toast } = useToast();

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
  const [decisionSummary, setDecisionSummary] = useState<DecisionSummary | undefined>(undefined);
  const [literatureCount, setLiteratureCount] = useState<number>(0);
  const [logs, setLogs] = useState<string[]>([]);

  // Prevent double execution
  const [isDiscovering, setIsDiscovering] = useState(false);
  const [isSearching, setIsSearching] = useState(false);

  // Persistence Key - Force refresh v3
  const STORAGE_KEY = "scinets_discovery_state_v3_clean";

  // Quota Check
  const isQuotaExceeded = user?.quota && user.quota.used >= user.quota.limit;
  const quotaResetHours = user?.quota?.resets_in_hours || 0;

  const handleReset = () => {
    // Clear storage
    localStorage.removeItem(STORAGE_KEY);
    // Reset State
    setStep("query");
    setQuery("");
    setPapers([]);
    setAnswers(null);
    setCuratedPapers([]);
    setNodes([]);
    setEdges([]);
    setActivities([]);
    setIsComplete(false);
    setError(null);
    setThreadId(null);
    setHypotheses([]);
    setConceptGraph(null);
    setDecisionSummary(undefined);
    setLiteratureCount(0);
    setLogs([]);
    setIsDiscovering(false);
    setIsSearching(false);
    setNumHypotheses(3); // Reset config
  };

  // Abort controller for cancellation
  const abortControllerRef = useRef<AbortController | null>(null);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      abortControllerRef.current?.abort();
    };
  }, []);

  // 1. Load State on Mount
  useEffect(() => {
    const savedState = localStorage.getItem(STORAGE_KEY);
    if (savedState) {
      try {
        const parsed = JSON.parse(savedState);
        // Only restore if valid
        if (parsed.step) setStep(parsed.step);
        if (parsed.query) setQuery(parsed.query);
        if (parsed.papers) setPapers(parsed.papers);
        if (parsed.answers) setAnswers(parsed.answers);
        if (parsed.curatedPapers) setCuratedPapers(parsed.curatedPapers);
        if (parsed.nodes) setNodes(parsed.nodes);
        if (parsed.edges) setEdges(parsed.edges);
        if (parsed.activities) {
          // Fix Date parsing
          const restoredActivities = parsed.activities.map((a: any) => ({
            ...a,
            timestamp: new Date(a.timestamp)
          }));
          setActivities(restoredActivities);
        }
        if (parsed.isComplete) setIsComplete(parsed.isComplete);
        if (parsed.threadId) setThreadId(parsed.threadId);
        if (parsed.hypotheses) setHypotheses(parsed.hypotheses);
        if (parsed.conceptGraph) setConceptGraph(parsed.conceptGraph);
        if (parsed.decisionSummary) setDecisionSummary(parsed.decisionSummary);
        if (parsed.literatureCount) setLiteratureCount(parsed.literatureCount);
        if (parsed.logs) setLogs(parsed.logs);
        // Don't restore isDiscovering/isSearching to avoid stuck loading states
      } catch (e) {
        console.error("Failed to parse saved state", e);
        localStorage.removeItem(STORAGE_KEY);
      }
    }
  }, []);

  // 2. Save State on Change
  useEffect(() => {
    // Debounce slightly or just save on every change (React batches updates)
    // Only save if we have some meaningful state
    if (step === "query" && !query) return;

    const stateToSave = {
      step,
      query,
      papers,
      answers,
      curatedPapers,
      nodes,
      edges,
      activities,
      isComplete,
      threadId,
      hypotheses,
      conceptGraph,
      decisionSummary,
      literatureCount,
      logs
    };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(stateToSave));
  }, [
    step, query, papers, answers, curatedPapers, nodes, edges,
    activities, isComplete, threadId, hypotheses, conceptGraph,
    decisionSummary, literatureCount, logs
  ]);

  // New State for config
  const [numHypotheses, setNumHypotheses] = useState(3);

  const handleQuerySubmit = (q: string, p: string[]) => {
    setQuery(q);
    setPapers(p);
    setStep("clarification");
  };

  const handleClarificationSubmit = async (a: ClarificationAnswers) => {
    setAnswers(a);
    setNumHypotheses(a.numHypotheses); // Capture from clarification step
    setError(null);

    // TRANSITION TO SEARCHING STEP (reusing Execution UI)
    setStep("searching");

    // Simulate initial agent activity
    setActivities([
      {
        id: "act-search-1",
        agent: "scientist",
        action: "Initializing Literature Search...",
        status: "thinking",
        timestamp: new Date()
      }
    ]);
    setLogs([
      "Literature Agent initialized.",
      `Query optimized: "${query}"`,
      "Connecting to OpenAlex Knowledge Graph..."
    ]);

    // Fetch real papers from backend
    try {
      // import { config } from "../config"; // Removed invalid import

      // ... inside component
      // const API_URL = config.API_URL; // Actually the lines below use it directly
      const API_URL = config.API_URL;
      const response = await fetch(`${API_URL}/search_papers`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: query, max_papers: 10 }),
        credentials: "include"
      });

      if (!response.ok) {
        throw new Error(`Paper search failed: ${response.status}`);
      }

      const data = await response.json();

      const fetchedPapers: CandidatePaper[] = data.papers
        .filter((p: any) => !p.title.includes("Demo Paper")) // Hard block on demo data
        .map((p: any) => ({
          id: p.id,
          title: p.title,
          year: p.year,
          venue: p.venue,
          rationale: p.abstract?.substring(0, 100) + '...' || p.rationale,
          selected: true,
          locked: false
        }));

      // Create candidate papers from manual abstracts
      const manualCandidates: CandidatePaper[] = papers.map((abstract, index) => ({
        id: `manual-${index}-${Date.now()}`,
        title: `[User Input] ${abstract.substring(0, 50)}...`,
        year: new Date().getFullYear(),
        venue: "User Provided",
        rationale: abstract, // Store full abstract here
        selected: true,
        locked: true
      }));

      // Combine with fetched papers
      const allCandidates = [...manualCandidates, ...fetchedPapers];

      // Simulate completion before transition
      setActivities(prev => [...prev, {
        id: "act-search-complete",
        agent: "scientist",
        action: `Found ${fetchedPapers.length} papers from OpenAlex and ${manualCandidates.length} user inputs.`,
        status: "complete",
        timestamp: new Date()
      }]);
      setLogs(prev => [...prev, "Search complete. Transitioning to curation..."]);

      // Small delay to let user see "Complete" status
      setTimeout(() => {
        setCuratedPapers(allCandidates);
        setStep("curation");
      }, 1500);

    } catch (err) {
      console.error("Paper fetch failed:", err);
      // Fallback: show curation with manual papers only (if any)
      const manualCandidates: CandidatePaper[] = papers.map((abstract, index) => ({
        id: `manual-${index}-${Date.now()}`,
        title: `[User Input] ${abstract.substring(0, 50)}...`,
        year: new Date().getFullYear(),
        venue: "User Provided",
        rationale: abstract,
        selected: true,
        locked: true
      }));

      setCuratedPapers(manualCandidates);
      setStep("curation");
    }
  };

  const handlePaperCurationSubmit = (selectedPapers: CandidatePaper[]) => {
    if (isDiscovering) return; // Prevent double-click
    setCuratedPapers(selectedPapers);
    setStep("execution");
    startDiscovery(answers!, selectedPapers);
  };

  // Define Callbacks for both Start and Resume
  const createCallbacks = (isPreviewMode: boolean): SSECallback => {
    let activityCounter = activities.length; // Continue counting

    return {
      onThreadId: (id) => {
        // Only set thread ID if not already set (retains original session)
        if (!threadId) setThreadId(id);
      },
      onActivity: (activity: ActivityEvent) => {
        const newActivity: AgentActivity = {
          id: `act-${Date.now()}`,
          agent: activity.agent,
          action: activity.action,
          status: activity.status,
          timestamp: new Date(),
        };
        setActivities(prev => [...prev, newActivity]);
      },
      onLog: ((msg: string) => setLogs(prev => [...prev, msg])),
      onResult: (result: Partial<DiscoveryResult>) => {
        if (result.hypotheses && result.hypotheses.length > 0) {
          setHypotheses(result.hypotheses);
        }
        if (result.concept_graph) {
          setConceptGraph(result.concept_graph);

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
        }
        if (result.decision_summary) setDecisionSummary(result.decision_summary);
        if (result.literature && (result.literature as any).papers) {
          setLiteratureCount((result.literature as any).papers.length);
        }
      },
      onError: (msg) => {
        setError(msg);
        setIsDiscovering(false);
      },
      onInterrupt: (data) => {
        console.log("INTERRUPT RECEIVED");
        setIsDiscovering(false); // Stop "loading" state
        setStep("selection"); // Switch to selection screen
      },
      onDone: () => {
        setIsComplete(true);
        setIsDiscovering(false);
      },
      onQuota: (info: QuotaInfo) => {
        toast({
          title: "Weekly Quota Update",
          description: `You have used ${info.used} of ${info.limit} runs. ${info.remaining} runs remaining.`,
          duration: 5000,
        });
      }
    };
  };

  const startDiscovery = async (clarificationAnswers: ClarificationAnswers, selectedPapers: CandidatePaper[]) => {
    if (isDiscovering) return;
    setIsDiscovering(true);

    // Reset UI for new run
    setNodes([]);
    setEdges([]);
    setActivities([]);
    setLogs([]);
    setHypotheses([]);
    setConceptGraph(null);
    setIsComplete(false);
    setError(null);

    const callbacks = createCallbacks(true); // Preview Mode

    try {
      abortControllerRef.current = await startDiscoveryStream(
        {
          query,
          documents: selectedPapers.map(p => p.id.startsWith("manual-") ? p.rationale : (p.id || p.title)),
          max_papers: 10,
          timeline: clarificationAnswers.timeline,
          guidance: clarificationAnswers.guidance,
          num_hypotheses: numHypotheses,
          goal: 'discover',
        },
        callbacks
      );
    } catch (err) {
      console.error("Discovery failed to start:", err);
      setError("Failed to start discovery: " + (err as Error).message);
      setIsDiscovering(false);
    }
  };

  const handleSelectionConfirm = async (selectedIds: string[]) => {
    // User selected hypotheses, now resume deep mode
    if (!threadId) {
      setError("Session lost. Please restart.");
      return;
    }

    setStep("execution"); // Go back to graph view
    setIsDiscovering(true); // Re-enable loading
    setIsComplete(false);

    // Add activity log for resume
    setActivities(prev => [...prev, {
      id: `act-resume-${Date.now()}`,
      agent: "orchestrator",
      action: `Resuming deep investigation on ${selectedIds.length} hypotheses...`,
      status: "thinking",
      timestamp: new Date()
    }]);

    const callbacks = createCallbacks(false); // Deep Mode

    try {
      abortControllerRef.current = await resumeDiscoveryStream(
        threadId,
        selectedIds,
        callbacks
      );
    } catch (err) {
      setError("Failed to resume: " + (err as Error).message);
      setIsDiscovering(false);
    }
  };


  const handleNodeSelect = (nodeId: string) => {
    console.log("Selected node:", nodeId);
  };

  return (
    <div className="min-h-screen bg-background relative">
      <Header />
      <NetworkBackground />

      <div className={cn(
        "relative z-10 pt-24 px-6 pb-6 min-h-[100dvh] flex flex-col",
        (step === "execution" || step === "searching") && !isComplete && "lg:h-screen lg:overflow-hidden"
      )}>

        {/* QUOTA BANNER */}
        {isQuotaExceeded && step === "query" && (
          <div className="max-w-4xl mx-auto w-full mb-6 p-4 bg-red-500/10 border border-red-500/50 rounded-lg flex items-center gap-4 text-red-200">
            <div className="p-2 bg-red-500/20 rounded-full">
              <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10" /><line x1="12" x2="12" y1="8" y2="12" /><line x1="12" x2="12.01" y1="16" y2="16" /></svg>
            </div>
            <div>
              <h3 className="font-semibold">Quota Limit Exceeded</h3>
              <p className="text-sm opacity-90">You have used {user?.quota?.used} of {user?.quota?.limit} discoveries. Resets in {quotaResetHours} hours.</p>
            </div>
          </div>
        )}

        {step === "query" && !isSearching && (
          <div className="flex-1 flex items-center justify-center relative">
            <DiscoveryQueryStep onSubmit={handleQuerySubmit} isQuotaExceeded={isQuotaExceeded || false} />
          </div>
        )}

        {step === "clarification" && !isSearching && (
          <div className="flex-1 flex items-center justify-center">
            <DiscoveryClarificationStep
              query={query}
              onSubmit={handleClarificationSubmit}
              onBack={() => setStep("query")}
            />
          </div>
        )}

        {/* Reuse Execution Step for Searching Phase */}
        {(step === "execution" || step === "searching") && answers && (
          <div className="flex-1 min-h-0">
            <DiscoveryExecutionStep
              query={query}
              papers={papers}
              answers={answers}
              curatedPapers={curatedPapers}
              literatureCount={literatureCount} // Pass real count
              nodes={nodes}
              edges={edges}
              activities={activities}
              isComplete={isComplete}
              onNodeSelect={handleNodeSelect}
              hypotheses={hypotheses}
              conceptGraph={conceptGraph}
              logs={logs}
              error={error}
              threadId={threadId}
              decision_summary={decisionSummary}
              onReset={handleReset}
            />
          </div>
        )}

        {step === "selection" && (
          <div className="flex-1 overflow-auto py-8">
            <HypothesisSelection
              hypotheses={hypotheses}
              onConfirm={handleSelectionConfirm}
            />
          </div>
        )}

        {step === "curation" && (
          <div className="flex-1 flex items-center justify-center overflow-auto py-8">
            <PaperCurationStep
              query={query}
              candidatePapers={curatedPapers}
              onSubmit={handlePaperCurationSubmit}
              onBack={() => setStep("clarification")}
            />
          </div>
        )}
      </div>
    </div>
  );
};

export default Discovery;
