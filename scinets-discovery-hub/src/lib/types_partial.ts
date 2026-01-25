    url?: string;
}

export interface HypothesisRationale {
    disconnected_clusters: string[];
    missing_link: string;
    field_assumption: string;
    structural_reason: string;
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
}
