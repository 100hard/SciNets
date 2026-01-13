"""
On-demand Experiment Subgraph (User-triggered only)

This graph is NOT part of the default discovery pipeline.
It runs only when a user explicitly requests an experiment via POST /run_experiment.

Epistemic principle:
"Experiments are optional, user-initiated exploratory tools, not part of scientific discovery.
They inform thinking, do not validate hypotheses, do not override literature evidence."
"""

from langgraph.graph import StateGraph, END
from app.state import ExperimentState
from app.agents.experiment import experiment_node
from langchain_core.runnables import RunnableConfig
from langchain_core.callbacks import adispatch_custom_event
from app.llm import get_llm
from langchain_core.messages import SystemMessage, HumanMessage


async def localized_critique_node(state: ExperimentState, config: RunnableConfig) -> dict:
    """
    Localized critique for a single experiment result.
    
    This critique applies ONLY to:
    - The specific hypothesis being tested
    - The specific experiment that was run
    
    It does NOT re-evaluate other hypotheses or the overall discovery.
    """
    if not state.experiment_result:
        return {"localized_critique": {"summary": "No experiment result to critique."}}
    
    await adispatch_custom_event("log", {
        "message": f"[Localized Critique] Analyzing experiment for hypothesis: {state.hypothesis_id}"
    }, config=config)
    
    exp = state.experiment_result
    
    from pydantic import BaseModel, Field
    from typing import List, Literal
    
    class LocalizedCritiqueOutput(BaseModel):
        """Critique specific to a single experiment."""
        behavioral_interpretation: str = Field(description="What behavioral patterns were observed?")
        consistency_assessment: str = Field(description="How consistent is behavior with the hypothesized mechanism?")
        failure_modes_identified: List[str] = Field(description="What failure modes or edge cases were found?")
        limitations: List[str] = Field(description="Limitations of this exploratory experiment")
        structural_verdict: Literal[
            "Consistent", 
            "Inconsistent", 
            "Inconclusive"
        ] = Field(description="Structural consistency verdict (NOT scientific validation)")
        next_exploration: str = Field(description="Suggested next exploratory step if any")
    
    system_prompt = f"""You are reviewing an EXPLORATORY EXPERIMENT result.

EPISTEMIC FRAMING:
This is NOT validation. Experiments inform thinking but do NOT validate hypotheses.
Do NOT claim the hypothesis is "proven" or "validated".

HYPOTHESIS: {state.hypothesis_text}
EXPERIMENT INTENT: {state.intent}

Analyze the experiment result and provide a localized critique.
Focus on behavioral patterns, consistency, and failure modes."""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"""
Experiment Result:
- Status: {exp.status}
- Metrics: {exp.metrics}
- Code executed: {exp.code_snippet[:500] if exp.code_snippet else 'N/A'}...
""")
    ]
    
    llm = get_llm(temperature=0.1)
    structured_llm = llm.with_structured_output(LocalizedCritiqueOutput)
    
    try:
        critique = await structured_llm.ainvoke(messages)
        return {
            "localized_critique": {
                "summary": critique.behavioral_interpretation,
                "verdict": critique.structural_verdict,
                "full_output": critique.model_dump()
            }
        }
    except Exception as e:
        return {
            "localized_critique": {
                "summary": f"Critique failed: {e}",
                "verdict": "Inconclusive"
            }
        }


def create_experiment_graph():
    """
    On-demand experiment graph (user-triggered only).
    
    Pipeline: EXPERIMENT → LOCALIZED_CRITIQUE → END
    
    This runs for a single hypothesis at a time.
    """
    workflow = StateGraph(ExperimentState)
    
    workflow.add_node("experiment", experiment_node)
    workflow.add_node("localized_critique", localized_critique_node)
    
    workflow.set_entry_point("experiment")
    workflow.add_edge("experiment", "localized_critique")
    workflow.add_edge("localized_critique", END)
    
    from langgraph.checkpoint.memory import MemorySaver
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)
