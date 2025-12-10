
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
    evidence_summary: Optional[str] = None # Textual summary of evidence (e.g. graph paths)
    evidence: List[EvidenceItem] = []

class ExperimentPlan(BaseModel):
    id: str
    hypothesis_id: str
    type: Literal["synthetic", "benchmark", "ablation"]
    goal: str
    method: str
    metrics: List[str]
    cost_estimate: Literal["low", "medium", "high"]

class Experiment(BaseModel):
    hypothesis_id: str
    plan_id: Optional[str] = None # Link to the plan if applicable
    code_snippet: Optional[str] = None
    metrics: Optional[dict] = None
    plot_url: Optional[str] = None
    plot_base64: Optional[str] = None

class Insight(BaseModel):
    content: str
    domain: str
    confidence: float = 0.0

class DiscoveryState(BaseModel):
    user_query: str
    goal: str = "discover"           # discover, survey, write
    lens: str = "none"               # Disciplinary lens (e.g., "game theory")
    speculation: str = "medium"      # low, medium, high
    run_experiments: bool = False    # Whether to run python experiments
    mock: bool = False               # Enable mock mode for testing
    human_feedback: Optional[str] = None # User feedback for interrupt/resume
    documents: List[str] = []        # User-provided papers/context
    domain_tags: List[str] = []
    plan: Optional[dict] = None
    literature: Optional[dict] = None
    concept_graph: Optional[dict] = None
    hypotheses: List[Hypothesis] = []
    selected_hypothesis_id: Optional[str] = None
    experiment_plans: List[ExperimentPlan] = [] # Proposed plans
    selected_experiment_plan_id: Optional[str] = None # Chosen plan for execution
    experiments: List[Experiment] = [] # Completed experiments
    critique: Optional[dict] = None
    done: bool = False

