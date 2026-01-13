
from typing import Literal, List, Dict, Optional, Any
from pydantic import BaseModel, Field

class EvidenceItem(BaseModel):
    paper_id: str
    title: str
    venue: str | None = None
    year: int | None = None
    stance: Literal["support", "contradict", "neutral"]
    strength: int = Field(..., ge=1, le=5, description="Strength of evidence from 1 to 5")
    key_points: List[str]
    url: str | None = None

class CausalChain(BaseModel):
    """Structured representation of a causal chain (no string parsing needed)."""
    nodes: List[str] = Field(description="Ordered list of concepts in the causal chain")
    relations: List[str] = Field(default=[], description="Relations between consecutive nodes (length = len(nodes) - 1)")
    source: str = Field(default="graph", description="Origin: 'graph', 'exploration', 'structural_hole'")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Confidence in this chain")

class Hypothesis(BaseModel):
    id: str
    text: str
    domain_tags: List[str]
    novelty_score: float
    feasibility_score: float
    testability_score: float
    search_query: Optional[str] = None # Keywords for evidence search
    required_data: List[str] = []
    experiment_idea: str | None = None
    causal_chain: Optional[CausalChain] = None # Structured causal mechanism
    evidence_summary: Optional[str] = None # Textual summary of evidence
    evidence: List[EvidenceItem] = []
    
    # Failure-mode classification (Tier 2)
    # stable: well-grounded in graph, confident chain
    # speculative: novel but plausible, moderate confidence
    # fragile: depends on weak or single edges
    # unstable: contradictory evidence or low confidence
    stability_class: Literal["stable", "speculative", "fragile", "unstable"] = "speculative"
    stability_reason: Optional[str] = None  # Human-readable explanation

class ExperimentPlan(BaseModel):
    id: str
    hypothesis_id: str
    type: Literal["synthetic", "benchmark", "ablation"]
    goal: str
    method: str
    metrics: List[str]
    cost_estimate: Literal["low", "medium", "high"]

class Experiment(BaseModel):
    id: str = ""
    hypothesis_id: str
    plan_id: Optional[str] = None # Link to the plan if applicable
    status: str = "pending"
    code_snippet: Optional[str] = None
    metrics: Optional[dict] = None
    plot_url: Optional[str] = None
    plot_base64: Optional[str] = None
    result_summary: Optional[str] = None

class Insight(BaseModel):
    content: str
    domain: str
    confidence: float = 0.0
    source: Optional[str] = None  # Track insight origin (e.g., 'experiment_critique')


# =============================================================================
# ExperimentState - For on-demand experiments (user-triggered only)
# =============================================================================
class ExperimentState(BaseModel):
    """
    State for user-triggered experiments (NOT part of default discovery).
    
    Experiments are optional, user-initiated exploratory tools.
    They inform thinking, do not validate hypotheses, do not override literature evidence.
    """
    hypothesis_id: str
    hypothesis_text: str
    intent: Literal["validate_direction", "probe_sensitivity", "stress_test"]
    user_constraints: dict = {}
    data_source: Literal["synthetic", "public_dataset", "user_provided"] = "synthetic"
    seed: int = 42  # Fixed by default for reproducibility
    
    # Results (populated after experiment runs)
    experiment_result: Optional[Experiment] = None
    localized_critique: Optional[dict] = None


# =============================================================================
# DiscoveryState - Main discovery pipeline state
# =============================================================================
class DiscoveryState(BaseModel):
    """
    Main state for the discovery pipeline.
    
    NOTE: Experiments are NOT part of this state.
    They are handled separately via ExperimentState and POST /run_experiment.
    """
    user_query: str
    goal: str = "discover"           # discover, survey, write
    lens: str = "none"               # Disciplinary lens (e.g., "game theory")
    speculation: str = "medium"      # low, medium, high
    # REMOVED: run_experiments - experiments are now user-triggered only
    mock: bool = False               # Enable mock mode for testing
    human_feedback: Optional[str] = None # User feedback for interrupt/resume
    documents: List[str] = []        # User-provided papers/context
    
    # Evaluation Config
    evaluation_mode: bool = False
    evaluation_strategy: str = "full" # full, rag, random, shortest, no_diversity
    experiment_id: Optional[str] = None # For tracking logs (evaluation runs)
    max_papers: int = 15 # Default paper limit (increased for denser graphs)
    
    # Tier 2: Citation expansion (optional, weighted not dominant)
    enable_citation_expansion: bool = False  # Set to True to expand via citations
    
    domain_tags: List[str] = []
    plan: Optional[dict] = None
    literature: Optional[dict] = None
    concept_graph: Optional[dict] = None
    hypotheses: List[Hypothesis] = []
    selected_hypothesis_id: Optional[str] = None
    # REMOVED: experiment_plans, selected_experiment_plan_id, experiments
    critique: Optional[dict] = None
    done: bool = False

    # Discovery Benchmark Metrics
    exploration_trace: Optional[str] = None
    structural_hole_analysis: Optional[str] = None
    symbolic_paths: List[List[str]] = []
    grounded_paths: List[List[str]] = []
    stance_counts: Dict[str, int] = {"support": 0, "contradict": 0, "neutral": 0}
    grounding_metrics: Dict[str, Any] = {}
    bridge_attempted: bool = False
