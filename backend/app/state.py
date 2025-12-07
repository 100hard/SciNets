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
    required_data: List[str] = []
    experiment_idea: str | None = None
    evidence: List[EvidenceItem] = []

class Experiment(BaseModel):
    hypothesis_id: str
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
    documents: List[str] = []        # User-provided papers/context
    domain_tags: List[str] = []
    plan: Optional[dict] = None
    literature: Optional[dict] = None
    concept_graph: Optional[dict] = None
    hypotheses: List[Hypothesis] = []
    selected_hypothesis_id: Optional[str] = None
    experiments: List[Experiment] = []
    critique: Optional[dict] = None
    done: bool = False

