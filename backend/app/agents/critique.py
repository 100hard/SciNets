from app.state import DiscoveryState
from app.llm import get_llm
from langchain_core.messages import HumanMessage, SystemMessage
import json
from langchain_core.runnables import RunnableConfig
from langchain_core.callbacks import adispatch_custom_event

async def critique_node(state: DiscoveryState, config: RunnableConfig) -> dict:
    """
    Critique Agent: Reviews the experiment results (including plots) and concludes the session.
    """
    await adispatch_custom_event("log", {"message": f"[Critique] Analyzing results for hypothesis: {state.selected_hypothesis_id}"}, config=config)
    
    # 1. Selection Logic
    if state.mock:
        await adispatch_custom_event("log", {"message": "[Critique] MOCK MODE: Returning dummy critique."}, config=config)
        return {
            "critique": {
                "summary": "Mock Critique: Validated.",
                "recommendation": "Accept: Mock suggestion.",
                "full_output": {
                    "interpretation": "Mock Interpretation",
                    "validity": "Valid",
                    "comparison": "Consistent",
                    "limitations": [],
                    "suggestions": ["Mock Step"],
                    "decision": "Accept",
                    "confidence": 0.99
                }
            },
            "done": True
        }

    if not state.experiments:
        return {"critique": {"summary": "No experiments were run."}}
        
    selected_id = state.selected_hypothesis_id
    # Select the experiment matching the hypothesis, or fall back to the last one
    exp = next((e for e in state.experiments if e.hypothesis_id == selected_id), state.experiments[-1])
    
    # Select the hypothesis object
    hypothesis = next((h for h in state.hypotheses if h.id == selected_id), state.hypotheses[0])
    
    metrics = exp.metrics
    plot_b64 = exp.plot_base64
    
    # 2. Sanity Checks
    if not metrics or "error" in metrics:
        await adispatch_custom_event("log", {"message": "[Critique] Metrics indicate failure or error."}, config=config)
        # We still proceed to let the LLM analyze the failure
        
    # 3. Evidence Context (Literature)
    evidence_context = "No specific literature evidence found."
    if hypothesis.evidence:
        evidence_counts = f"{sum(1 for e in hypothesis.evidence if e.stance == 'support')} Support, {sum(1 for e in hypothesis.evidence if e.stance == 'contradict')} Contradict"
        evidence_details = "\n".join([f"- {e.title} ({e.stance} {e.strength}/5)" for e in hypothesis.evidence[:3]])
        evidence_context = f"Literature Stance: {evidence_counts}\nKey Papers:\n{evidence_details}"
        if getattr(hypothesis, 'evidence_summary', None):
            evidence_context += f"\nSummary: {hypothesis.evidence_summary}"

    
    # 4. Prepare Prompt with Context (Domain, Goal, Lens)
    from pydantic import BaseModel, Field
    from typing import List
    
    class CritiqueOutput(BaseModel):
        interpretation: str = Field(description="Scientific interpretation of the results.")
        validity: str = Field(description="Comment on validity (sample size, p-values, potential flaws).")
        comparison: str = Field(description="How does this result compare to the literature evidence?")
        limitations: List[str] = Field(description="List of limitations.")
        suggestions: List[str] = Field(description="Specific suggestions for next steps or new hypotheses.")
        decision: str = Field(description="Verdict: 'Accept', 'Reject', 'Refine', or 'Inconclusive'.")
        confidence: float = Field(description="Confidence in this verdict (0.0 - 1.0).")

    system_msg = f"""You are a Senior Principal Investigator reviewing an experiment.
    
    CONTEXT:
    - Domain: {', '.join(state.domain_tags)}
    - Lens: {state.lens}
    - User Goal: {state.goal}
    
    YOUR TASK:
    1. Analyze the Experiment Metrics & Plot (if present).
    2. Compare them against the Literature Evidence.
    3. Critique the VALIDITY of the experiment (too simple? overfitting? leakage?).
    4. Provide a scientific verdict and Next Steps.
    """
    
    plot_instruction = ""
    if plot_b64:
        plot_instruction = "\n\nVISUAL INSPECTION: Analyze the attached plot. Describe trends, anomalies, and if it supports the metrics."

    messages = [
        SystemMessage(content=system_msg),
        HumanMessage(content=f"""
        Hypothesis: {hypothesis.text}
        
        Literature Context:
        {evidence_context}
        
        Experiment Results:
        - Code Status: {exp.status}
        - Metrics: {json.dumps(metrics, indent=2)}
        {plot_instruction}
        """)
    ]
    
    if plot_b64:
        # Multimodal
        messages.append(
            HumanMessage(
                content=[
                    {"type": "text", "text": "Please inspect this result plot:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{plot_b64}"}},
                ]
            )
        )
        
    llm = get_llm(temperature=0.1) # Low temp for rigorous critique
    structured_llm = llm.with_structured_output(CritiqueOutput)
    
    try:
        critique = await structured_llm.ainvoke(messages)
        
        # 5. Store Rich Insight as Memory
        from app.memory import MemoryManager
        from app.state import Insight
        
        memory = MemoryManager()
        
        # Composite content for memory
        insight_content = f"Hypothesis: {hypothesis.text}\nVerdict: {critique.decision} (Conf: {critique.confidence})\nFinding: {critique.interpretation}"
        
        insight = Insight(
            content=insight_content,
            domain=state.domain_tags[0] if state.domain_tags else "general",
            confidence=critique.confidence,
            source="experiment_critique"
        )
        memory.store_insight(insight)
        
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
