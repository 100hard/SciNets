from app.state import DiscoveryState, DecisionSummary, HypothesisStrengthProfile
from app.llm import get_llm
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.callbacks import adispatch_custom_event
from pydantic import BaseModel, Field
from typing import List, Literal

class DecisionOutput(BaseModel):
    """
    Structured output for the Decision Agent.
    Aggregates hypotheses into a decision-ready summary.
    """
    # 1. Individual Hypothesis Profiles (Map by Index 0-N)
    profiles: List[HypothesisStrengthProfile] = Field(description="Strength profile for each hypothesis in order")
    confidence_roadmaps: List[List[str]] = Field(description="List of 2-3 specific actions/evidence that would increase confidence for each hypothesis (in order)")
    
    # 2. Decision Summary
    primary_hypothesis_index: int = Field(description="Index of the most actionable hypothesis (0-based)")
    primary_hypothesis_reason: str = Field(description="Why this is the primary choice (e.g. 'Best balance of novelty and support')")
    evidence_level: Literal["Strong", "Moderate", "Weak", "Inconclusive"]
    key_risks: List[str] = Field(description="List of critical risks (upstream assumptions, lack of causality)")
    recommended_next_steps: List[str] = Field(description="Specific, actionable next steps (e.g., 'Prospective metabolomics')")
    system_confidence: Literal["High", "Moderate", "Low"]
    
    # 3. Prioritization
    near_term_focus_indices: List[int] = Field(description="Indices of hypotheses best for near-term testing")
    long_term_focus_indices: List[int] = Field(description="Indices of hypotheses best for long-term theory building")
    high_risk_high_reward_indices: List[int] = Field(description="Indices of speculative but high-impact hypotheses")

async def decision_node(state: DiscoveryState, config: RunnableConfig) -> dict:
    """
    Decision Agent: Synthesizes hypotheses into an executive decision summary.
    Run AFTER evidence and critique agents.
    """
    await adispatch_custom_event("log", {"message": "[Decision Agent] Synthesizing decision summary..."}, config=config)
    
    if not state.hypotheses:
        await adispatch_custom_event("log", {"message": "[Decision Agent] No hypotheses to analyze."}, config=config)
        return {"done": True}

    # Prepare Hypothesis Context
    hypotheses_text = ""
    for i, h in enumerate(state.hypotheses):
        evidence_summary = h.evidence_summary or "No evidence summary."
        support_count = len([e for e in h.evidence if e.stance == 'support'])
        contradict_count = len([e for e in h.evidence if e.stance == 'contradict'])
        
        hypotheses_text += f"""
---
[H{i}] {h.text}
ID: {h.id}
Status: {h.stability_class}
Evidence: {support_count} Support, {contradict_count} Contradict
Summary: {evidence_summary}
Rationale: {h.rationale_gap.rationale_type if h.rationale_gap else 'N/A'}
---
"""

    system_msg = f"""You are a Strategic Research Director. Your goal is to synthesize the current findings into a DECISION-READY summary.

CONTEXT:
User Query: {state.user_query}
Goal: {state.goal}

YOUR TASKS:
1. PROFILE: Assign a 4-axis strength profile to EACH hypothesis (Mechanistic, Empirical, Tractable, Translational).
2. ROADMAP: For EACH hypothesis, list 2-3 specific items (experiments/data) that would most increase confidence.
3. PRIORITIZE: Rank hypotheses for different use cases (Near-term, Long-term, High-risk).
4. DECIDE: Select ONE primary actionable hypothesis and justify it.
5. RECOMMEND: Provide concrete next steps (experiments, assays, studies) for the overall project.

GUIDELINES:
- "Abstained" or "Neutral" evidence is valuable signal -> Treat as "Inconclusive".
- Do NOT inflate confidence. If evidence is weak, say so.
- Translational relevance means: "Can this solve a real problem?"
- Tractability means: "Can we test this with current tools?"

OUTPUT FORMAT:
Return a structured JSON object satisfying the DecisionOutput schema.
"""

    messages = [
        SystemMessage(content=system_msg),
        HumanMessage(content=f"Analyze these hypotheses and generate the decision summary:\n{hypotheses_text}")
    ]
    
    llm = get_llm(temperature=0.2) # Low temp for stable decisioning
    # FIX: Use json schema for serialization safety
    structured_llm = llm.with_structured_output(DecisionOutput.model_json_schema())
    
    try:
        # A. Generate Decision Data
        raw_output = await structured_llm.ainvoke(messages)
        output = DecisionOutput(**raw_output)
        
        # B. Post-Process State Updates
        updated_hypotheses = []
        for i, h in enumerate(state.hypotheses):
            # Attach strength profile (convert Pydantic to dict for state storage)
            if i < len(output.profiles):
                h.strength_profile = output.profiles[i].model_dump()
            # Attach confidence roadmap
            if i < len(output.confidence_roadmaps):
                h.confidence_roadmap = output.confidence_roadmaps[i]
            
            updated_hypotheses.append(h)
            
        # Map indices back to IDs
        def map_indices(indices):
            return [state.hypotheses[i].id for i in indices if 0 <= i < len(state.hypotheses)]
            
        decision_summary = DecisionSummary(
            primary_hypothesis_id=state.hypotheses[output.primary_hypothesis_index].id if state.hypotheses and 0 <= output.primary_hypothesis_index < len(state.hypotheses) else (state.hypotheses[0].id if state.hypotheses else ""),
            primary_hypothesis_reason=output.primary_hypothesis_reason,
            evidence_level=output.evidence_level,
            key_risks=output.key_risks,
            recommended_next_steps=output.recommended_next_steps,
            system_confidence=output.system_confidence,
            near_term_focus=map_indices(output.near_term_focus_indices),
            long_term_focus=map_indices(output.long_term_focus_indices),
            high_risk_high_reward=map_indices(output.high_risk_high_reward_indices)
        )
        
        await adispatch_custom_event("log", {"message": f"[Decision Agent] Decision: Primary H{output.primary_hypothesis_index} - {output.evidence_level}"}, config=config)
        
        return {
            "hypotheses": updated_hypotheses,
            "decision_summary": decision_summary
        }
        
    except Exception as e:
        await adispatch_custom_event("log", {"message": f"[Decision Agent] Failed: {e}"}, config=config)
        print(f"[Decision Agent] Error: {e}")
        return {"done": True}
