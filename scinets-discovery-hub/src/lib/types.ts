// Backend API types matching Pydantic models

export interface EvidenceItem {
    paper_id: string;
    title: string;
    venue?: string;
    year?: number;
    stance: 'support' | 'contradict' | 'neutral';
    strength: number; // 1-5
    key_points: string[];
    url?: string;
}

export interface Constraint {
    text: string;
    type: 'hard' | 'soft';
    importance: number;
}

export interface HypothesisRationale {
    disconnected_clusters: string[];
    missing_link: string;
    field_assumption: string;
    structural_reason: string;
    // New Tension Fields
    epistemic_tension?: string;
    belief_a?: string;
    belief_b?: string;
    consistency_constraint?: string;
    rationale_type?: 'tension' | 'gap' | 'opportunity';
}

export interface Hypothesis {
    id: string;
    text: string;
    domain_tags: string[];
    novelty_score: number;
    feasibility_score: number;
    testability_score: number;
    search_query?: string;
    required_data: string[];
    experiment_idea?: string;
    evidence_summary?: string;
    evidence: EvidenceItem[];
    // New Rationale Section
    rationale_gap?: HypothesisRationale;
    // Orthogonality Label
    mechanism_class?: string;
    // New Constraints
    constraints?: Constraint[];

    // New Strength Profile
    strength_profile?: HypothesisStrengthProfile;

    // New Actionability Roadmap
    confidence_roadmap?: string[];

    // New Causal Chain (Preview Mode)
    causal_chain?: {
        nodes: string[];
        edges: any[];
    };

    // Legacy support
    mechanisms?: string[];
}

export interface HypothesisStrengthProfile {
    mechanistic_coherence: 'High' | 'Medium' | 'Low';
    empirical_support: 'High' | 'Medium' | 'Low';
    experimental_tractability: 'High' | 'Medium' | 'Low';
    translational_relevance: 'High' | 'Medium' | 'Low';
}

export interface DecisionSummary {
    primary_hypothesis_id: string;
    primary_hypothesis_reason: string;
    evidence_level: 'Strong' | 'Moderate' | 'Weak' | 'Inconclusive';
    key_risks: string[];
    recommended_next_steps: string[];
    system_confidence: 'High' | 'Moderate' | 'Low';

    // Prioritization Lists (IDs)
    near_term_focus: string[];
    long_term_focus: string[];
    high_risk_high_reward: string[];
}

export interface ExperimentPlan {
    id: string;
    hypothesis_id: string;
    type: 'synthetic' | 'benchmark' | 'ablation';
    goal: string;
    method: string;
    metrics: string[];
    cost_estimate: 'low' | 'medium' | 'high';
}

export interface Experiment {
    hypothesis_id: string;
    plan_id?: string;
    code_snippet?: string;
    metrics?: Record<string, unknown>;
    plot_url?: string;
    plot_base64?: string;
}

export interface ConceptGraphNode {
    id: string;
    label: string;
    type: string;
    source?: string;
}

export interface ConceptGraphEdge {
    source: string;
    target: string;
    relation: string;
    weight?: number;
}

export interface ConceptGraph {
    nodes: ConceptGraphNode[];
    edges: ConceptGraphEdge[];
}

export interface Critique {
    summary: string;
    strengths: string[];
    weaknesses: string[];
    suggestions: string[];
}

// Full discovery result from backend
export interface DiscoveryResult {
    user_query: string;
    plan?: Record<string, unknown>;
    literature?: Record<string, unknown>;
    domain_tags: string[];
    concept_graph?: ConceptGraph;
    hypotheses: Hypothesis[];
    experiment_plans: ExperimentPlan[];
    experiments: Experiment[];
    critique?: Critique;

    // New: Decision
    decision_summary?: DecisionSummary;
}

// SSE Event types
export type SSEEventType = 'activity' | 'log' | 'result' | 'error' | 'interrupt' | 'quota';

export interface QuotaInfo {
    used: number;
    limit: number;
    remaining: number;
}

export interface ActivityEvent {
    agent: 'planner' | 'scientist' | 'critic' | 'orchestrator' | 'literature' | 'hypothesis' | 'experiment' | 'decision';
    action: string;
    status: 'reading' | 'thinking' | 'building' | 'complete';
}

export interface SSEEvent {
    type: SSEEventType | '[DONE]';
    data: unknown;
    thread_id?: string;
}

// API Request types
export interface RunRequest {
    query: string;
    goal?: 'discover' | 'survey' | 'write';
    lens?: string;
    speculation?: 'low' | 'medium' | 'high';
    run_experiments?: boolean;
    documents?: string[];
    thread_id?: string;
    feedback?: string;
    mock?: boolean;
    timeline?: string;
    max_papers?: number;
    guidance?: string;
    num_hypotheses?: number;
    selected_hypothesis_ids?: string[];
}

export interface ExperimentRequest {
    thread_id: string;
    hypothesis_id: string;
    plan_id: string;
}

export interface UserQuota {
    used: number;
    limit: number;
    resets_in_hours: number;
}

export interface User {
    id: string;
    email: string;
    quota?: UserQuota;
}
