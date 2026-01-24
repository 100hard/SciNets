from app.state import DiscoveryState, Hypothesis
from app.llm import get_cheap_llm
from langchain_core.messages import HumanMessage, SystemMessage
import json
from langchain_core.runnables import RunnableConfig
from langchain_core.callbacks import adispatch_custom_event
from pydantic import BaseModel, Field
from typing import List, Literal, Optional
import hashlib

# =============================================================================
# Deterministic Scoring Logic
# =============================================================================

def calculate_deterministic_metrics(hypothesis: Hypothesis) -> dict:
    """
    Calculates deterministic scoring metrics based on evidence and stability.
    """
    evidence = hypothesis.evidence or []
    
    # Weights
    w_support = 1.0
    w_contradict = 1.5 # Contradiction weighs heavier
    w_neutral = 0.1
    
    current_support = sum(e.strength for e in evidence if e.stance == 'support')
    current_contradict = sum(e.strength for e in evidence if e.stance == 'contradict')
    current_neutral = sum(1 for e in evidence if e.stance == 'neutral')
    
    # Evidence Density (0-10 scale approximation)
    raw_density = len(evidence)
    density_score = min(10, raw_density * 2)
    
    # Weighted Score
    # Range: Negative (Undermined) to Positive (Supported)
    weighted_score = (current_support * w_support) - (current_contradict * w_contradict) + (current_neutral * w_neutral)
    
    # Stability Penalty
    stability_map = {
        "stable": 0,
        "speculative": -1,
        "fragile": -2,
        "unstable": -4
    }
    penalty = stability_map.get(hypothesis.stability_class, 0)
    
    final_score = weighted_score + penalty
    
    # Verdict Assignment
    verdict = "Requires Refinement"
    if not evidence and hypothesis.evidence_status != "failed_external":
        verdict = "Inconclusive"
    elif hypothesis.evidence_status == "failed_external":
        verdict = "Inconclusive"
    elif final_score > 4:
        verdict = "Structurally Supported"
    elif final_score < -2:
        verdict = "Structurally Undermined"
    else:
        # Mixed or weak evidence
        verdict = "Requires Refinement"
        
    # Confidence Calculation (Heuristic)
    # Higher density + extreme score (high pos or high neg) = higher confidence
    # 0.5 base, +/- based on evidence
    confidence = 0.5
    if raw_density > 0:
        confidence += min(0.4, raw_density * 0.05) # Add up to 0.4 for quantity
        # Penalty for mixed signals (if both support and contradict exist)
        if current_support > 0 and current_contradict > 0:
            confidence -= 0.1
            
    confidence = max(0.0, min(1.0, confidence))
    
    return {
        "support_count": current_support,
        "contradict_count": current_contradict,
        "neutral_count": current_neutral,
        "weighted_score": final_score,
        "decision": verdict,
        "confidence": confidence
    }

def generate_critique_hash(hypothesis: Hypothesis) -> str:
    """Generates a hash of the hypothesis text and evidence for caching."""
    data = (hypothesis.text + "".join([e.paper_id + e.stance for e in hypothesis.evidence])).encode("utf-8")
    return hashlib.md5(data).hexdigest()

# =============================================================================
# Minimal LLM Schemas
# =============================================================================

class CritiqueNarrative(BaseModel):
    """
    Minimal narrative output. 
    The verdict and confidence are already decided deterministically.
    This just provides the human-readable explanation.
    """
    interpretation: str = Field(description="1-2 sentences interpreting the evidence patterns.")
    limitations: List[str] = Field(description="List of 2-3 specific limitations.")
    suggestions: List[str] = Field(description="1 specific suggestion for next steps.")
    risk_assessment: str = Field(description="One sentence on the biggest scientific risk.")

# =============================================================================
# Critique Node
# =============================================================================

from app.telemetry.cost_tracker import CostTracker

async def critique_node(state: DiscoveryState, config: RunnableConfig) -> dict:
    """
    Refactored Critique Agent (Cost-Optimized).
    """
    CostTracker.get_instance().start_step("critique", "critique_analysis")
    try:
        await adispatch_custom_event("log", {"message": f"[Critique] Batch analysis of {len(state.hypotheses)} hypotheses..."}, config=config)
        
        # Use cheap LLM for narration (Cost Reduction)
        llm = get_cheap_llm(temperature=0.2) 
        structured_llm = llm.with_structured_output(CritiqueNarrative)
        
        updated_hypotheses = []
        
        for hypothesis in state.hypotheses:
            CostTracker.get_instance().push_hypothesis_context(hypothesis.id)
            try:
                # 1. Deterministic Scoring
                metrics = calculate_deterministic_metrics(hypothesis)
                
                # 2. Check Cache
                current_hash = generate_critique_hash(hypothesis)
                
                # Check if we have a valid cached critique
                existing_critique = hypothesis.critique
                if existing_critique and existing_critique.get("hash") == current_hash:
                     # Cache Hit
                     print(f"[Critique] Cache Hit for {hypothesis.id[:8]}")
                     updated_hypotheses.append(hypothesis)
                     continue
                
                # 3. Prepare Prompt for Minimal Narration
                # We treat the generic fallback specially
                if metrics["decision"] == "Inconclusive" and hypothesis.evidence_status == "failed_external":
                     critique_data = {
                        "interpretation": "Cannot assess due to external API failures gathering evidence.",
                        "structural_assessment": "Analysis blocked by external API failure.",
                        "limitations": ["External API (OpenAlex) errors"],
                        "suggestions": ["Retry evidence gathering later"],
                        "decision": "Inconclusive",
                        "confidence": 0.0,
                        "weighted_score": 0,
                        "hash": current_hash
                     }
                elif metrics["decision"] == "Inconclusive" and not hypothesis.evidence:
                     critique_data = {
                        "interpretation": "No direct literature evidence found.",
                        "structural_assessment": "Lack of evidence prevents structural assessment.",
                        "limitations": ["Literature gap", "Search query specificity"],
                        "suggestions": ["Broaden search terms", "Check adjacent fields"],
                        "decision": "Inconclusive",
                        "confidence": 0.1,
                        "weighted_score": metrics["weighted_score"],
                        "hash": current_hash
                     }
                else:
                    # LLM Call for Narrative
                    evidence_summary = "\n".join([f"- {e.title} ({e.stance} {e.strength}/5)" for e in hypothesis.evidence[:5]])
                    
                    system_msg = """You are a scientific editor. 
Based on the provided Hypothesis, Evidence Summary, and Deterministic Verdict, write a concise structural critique.
Do NOT change the verdict. Focus on explaining WHY this verdict was reached based on the evidence."""
    
                    user_msg = f"""
Hypothesis: {hypothesis.text}
    
Evidence Summary:
{evidence_summary}
    
Deterministic Verdict: {metrics['decision']}
Weighted Score: {metrics['weighted_score']}
Support/Contradict: {metrics['support_count']} / {metrics['contradict_count']}
    
Task:
1. Write 'interpretation': A 1-2 sentence structured summary of how the evidence supports/refutes the hypothesis.
2. Write 'limitations': 2-3 specific limitations of the current evidence.
3. Write 'suggestions': 1 concrete next step.
4. Write 'risk_assessment': One sentence on the main risk.
"""
                    narrative = await structured_llm.ainvoke([
                        SystemMessage(content=system_msg),
                        HumanMessage(content=user_msg)
                    ])
                    
                    critique_data = {
                        "interpretation": narrative.interpretation,
                        "structural_assessment": f"Verdict: {metrics['decision']}. {narrative.risk_assessment}",
                        "limitations": narrative.limitations,
                        "suggestions": narrative.suggestions,
                        "decision": metrics["decision"],
                        "confidence": metrics["confidence"],
                        "weighted_score": metrics["weighted_score"],
                        "hash": current_hash,
                        "risk_assessment": narrative.risk_assessment
                    }
                
                # 4. Update Hypothesis
                hypothesis.critique = critique_data
                
                # Store insight to memory (Legacy support)
                from app.memory import MemoryManager
                from app.state import Insight
                
                memory = MemoryManager()
                insight_content = f"Hypothesis: {hypothesis.text}\nVerdict: {critique_data['decision']} (Conf: {critique_data['confidence']})\nFinding: {critique_data['interpretation']}"
                insight = Insight(
                    content=insight_content,
                    domain=state.domain_tags[0] if state.domain_tags else "general",
                    confidence=metrics["confidence"],
                    source="evidence_critique"
                )
                # We don't await this as it might be synchronous or we don't care about result
                try:
                    memory.store_insight(insight)
                except:
                    pass
    
                updated_hypotheses.append(hypothesis)
                
            except Exception as e:
                print(f"[Critique] Error processing {hypothesis.id[:8]}: {e}")
                # Fallback
                hypothesis.critique = {
                    "interpretation": "Critique generation failed.",
                    "decision": "Inconclusive",
                    "confidence": 0.0,
                    "error": str(e)
                }
                updated_hypotheses.append(hypothesis)
            finally:
                CostTracker.get_instance().pop_hypothesis_context()
    
        # Legacy Compatibility: Populate state.critique with the first hypothesis's critique
        legacy_critique = None
        if updated_hypotheses:
            legacy_critique = {
                "summary": updated_hypotheses[0].critique.get("interpretation", ""),
                "recommendation": f"{updated_hypotheses[0].critique.get('decision', 'Unsure')}: {updated_hypotheses[0].critique.get('suggestions', [''])[0]}",
                "full_output": updated_hypotheses[0].critique
            }
    
        await adispatch_custom_event("log", {"message": "[Critique] Batch analysis complete."}, config=config)
    
        return {
            "hypotheses": updated_hypotheses,
            "critique": legacy_critique, 
            "done": True
        }
    finally:
        CostTracker.get_instance().end_step("critique", "critique_analysis")
