from app.state import DiscoveryState
from app.llm import get_llm
from langchain_core.messages import HumanMessage, SystemMessage
import json
from langchain_core.runnables import RunnableConfig
from langchain_core.callbacks import adispatch_custom_event

async def critique_node(state: DiscoveryState, config: RunnableConfig) -> dict:
    """
    Critique Agent: Reviews hypotheses and evidence to provide structural assessment.
    
    NOTE: Experiments are NOT part of the default discovery pipeline.
    This critique focuses on EVIDENCE-BASED structural assessment only.
    For experiment critique, use the localized_critique_node in experiment_graph.py.
    """
    await adispatch_custom_event("log", {"message": f"[Critique] Analyzing hypotheses and evidence..."}, config=config)
    
    # MOCK MODE
    if state.mock:
        await adispatch_custom_event("log", {"message": "[Critique] MOCK MODE: Returning dummy critique."}, config=config)
        return {
            "critique": {
                "summary": "Mock Critique: Structurally assessed.",
                "recommendation": "Requires Refinement: Mock suggestion.",
                "full_output": {
                    "interpretation": "Mock behavioral interpretation",
                    "structural_assessment": "Exploratory assessment",
                    "limitations": [],
                    "suggestions": ["Mock exploration step"],
                    "decision": "Requires Refinement",
                    "confidence": 0.5
                }
            },
            "done": True
        }

    # Check for hypotheses
    if not state.hypotheses:
        return {"critique": {"summary": "No hypotheses to critique."}, "done": True}
    
    # Select primary hypothesis for critique
    selected_id = state.selected_hypothesis_id
    hypothesis = next((h for h in state.hypotheses if h.id == selected_id), state.hypotheses[0]) if selected_id else state.hypotheses[0]
    
    # Build evidence context from Literature Agent output
    evidence_context = "No specific literature evidence found."
    if hypothesis.evidence:
        support_count = sum(1 for e in hypothesis.evidence if e.stance == 'support')
        contradict_count = sum(1 for e in hypothesis.evidence if e.stance == 'contradict')
        neutral_count = sum(1 for e in hypothesis.evidence if e.stance == 'neutral')
        
        evidence_counts = f"{support_count} Support, {contradict_count} Contradict, {neutral_count} Neutral"
        evidence_details = "\n".join([f"- {e.title} ({e.stance} {e.strength}/5)" for e in hypothesis.evidence[:5]])
        evidence_context = f"Literature Stance: {evidence_counts}\nKey Papers:\n{evidence_details}"
        if getattr(hypothesis, 'evidence_summary', None):
            evidence_context += f"\nSummary: {hypothesis.evidence_summary}"
    
    # Stability classification context
    stability_context = ""
    if hasattr(hypothesis, 'stability_class'):
        stability_context = f"Stability Classification: {hypothesis.stability_class}"
        if hypothesis.stability_reason:
            stability_context += f" ({hypothesis.stability_reason})"

    # Handle External API Failure specifically
    evidence_status = getattr(hypothesis, "evidence_status", "complete")
    if evidence_status == "failed_external":
        await adispatch_custom_event("log", {"message": "[Critique] Verdict: Inconclusive (External API Failure)"}, config=config)
        return {
            "critique": {
                "summary": "Evidence gathering failed due to external API errors.",
                "recommendation": "Inconclusive: Retry later when external services are stable.",
                "full_output": {
                    "interpretation": "Cannot assess due to missing evidence.",
                    "structural_assessment": "Analysis blocked by external API failure.",
                    "limitations": ["External API (OpenAlex) 500/503 errors"],
                    "suggestions": ["Retry evidence gathering"],
                    "decision": "Inconclusive",
                    "confidence": 0.0
                }
            },
            "done": True
        }

    from pydantic import BaseModel, Field
    from typing import List, Literal
    
    class CritiqueOutput(BaseModel):
        """
        Epistemically neutral critique output.
        Evaluates STRUCTURAL support from evidence, not scientific truth.
        """
        interpretation: str = Field(description="Interpretation of the hypothesis in light of evidence (patterns, gaps, uncertainties).")
        structural_assessment: str = Field(description="How well does the literature evidence structurally support/undermine the hypothesis mechanism?")
        limitations: List[str] = Field(description="List of limitations and caveats.")
        suggestions: List[str] = Field(description="Specific suggestions for refinement or further exploration.")
        # EPISTEMICALLY NEUTRAL VERDICTS
        decision: Literal[
            "Structurally Supported",    # Evidence aligns with mechanism
            "Structurally Undermined",   # Evidence contradicts mechanism
            "Requires Refinement",       # Partial/mixed evidence
            "Inconclusive"               # Insufficient evidence to assess
        ] = Field(description="Structural verdict based on evidence (not scientific truth claim).")
        confidence: float = Field(description="Confidence in this structural assessment (0.0 - 1.0).")

    system_msg = f"""You are a Senior Principal Investigator reviewing a hypothesis and its literature evidence.
    
EPISTEMIC FRAMING:
SciNets is an exploratory, graph-constrained synthesis system that surfaces plausible 
mechanistic hypotheses and their structural support or failure modes, WITHOUT claiming scientific truth.

CONTEXT:
- Domain: {', '.join(state.domain_tags) if state.domain_tags else 'General'}
- Lens: {state.lens}
- User Goal: {state.goal}

YOUR TASK:
1. Assess STRUCTURAL SUPPORT: Does the literature evidence align with the hypothesized mechanism?
2. Consider the stability classification of the hypothesis.
3. Identify gaps, uncertainties, and limitations.
4. Provide a STRUCTURAL VERDICT (not truth claim):
   - "Structurally Supported": Evidence aligns with mechanism
   - "Structurally Undermined": Evidence contradicts mechanism
   - "Requires Refinement": Partial/mixed evidence
   - "Inconclusive": Insufficient evidence to assess

DO NOT claim the hypothesis is "proven" or "validated". This is exploratory synthesis.
"""

    messages = [
        SystemMessage(content=system_msg),
        HumanMessage(content=f"""
Hypothesis: {hypothesis.text}

{stability_context}

Literature Evidence Context:
{evidence_context}

Causal Chain: {' -> '.join(hypothesis.causal_chain.nodes) if hypothesis.causal_chain else 'Not specified'}
""")
    ]
        
    llm = get_llm(temperature=0.1)  # Low temp for rigorous critique
    structured_llm = llm.with_structured_output(CritiqueOutput)
    
    try:
        critique = await structured_llm.ainvoke(messages)
        
        # Store insight to memory
        from app.memory import MemoryManager
        from app.state import Insight
        
        memory = MemoryManager()
        
        insight_content = f"Hypothesis: {hypothesis.text}\nVerdict: {critique.decision} (Conf: {critique.confidence})\nFinding: {critique.interpretation}"
        
        insight = Insight(
            content=insight_content,
            domain=state.domain_tags[0] if state.domain_tags else "general",
            confidence=critique.confidence,
            source="evidence_critique"  # Changed from experiment_critique
        )
        memory.store_insight(insight)
        
        await adispatch_custom_event("log", {"message": f"[Critique] Verdict: {critique.decision} (Confidence: {critique.confidence})"}, config=config)

        return {
            "critique": {
                "summary": critique.interpretation,
                "recommendation": f"{critique.decision}: {critique.suggestions[0] if critique.suggestions else 'No specific suggestion'}",
                "full_output": critique.model_dump()
            },
            "done": True
        }
        
    except Exception as e:
        await adispatch_custom_event("log", {"message": f"[Critique] Analysis failed: {e}"}, config=config)
        return {
            "critique": {"summary": "Automated critique failed.", "error": str(e)},
            "done": True
        }
