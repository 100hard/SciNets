from app.state import DiscoveryState, EvidenceItem, Hypothesis
from app.tools.openalex import search_papers, reconstruct_abstract
from app.llm import get_cheap_llm
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import Literal, List
from langchain_core.runnables import RunnableConfig
from langchain_core.callbacks import adispatch_custom_event
import asyncio

class EvidenceClassification(BaseModel):
    stance: Literal["support", "contradict", "neutral"] = Field(description="Does the paper support, contradict, or is neutral to the hypothesis?")
    strength: int = Field(description="Strength of evidence (1-5)", ge=1, le=5)
    key_points: list[str] = Field(description="Key points from the paper relevant to the hypothesis")


async def gather_evidence_for_hypothesis(
    hypothesis: Hypothesis,
    llm,
    structured_llm,
    config: RunnableConfig
) -> Hypothesis:
    """
    Hypothesis-Conditional Evidence Evaluator.
    
    This function evaluates how the literature corpus RELATES to a specific hypothesis.
    It runs ONLY after hypotheses exist and operates independently for each hypothesis.
    
    EPISTEMIC BOUNDARY (STRICTLY ENFORCED):
    - This agent MUST NOT modify the concept graph
    - This agent MUST NOT extract new concepts
    - This agent MUST NOT perform structural reasoning
    - This agent is STANCE-AGNOSTIC until it receives a hypothesis
    
    Role separation:
    - Literature Agent answers: "What exists in the corpus?"
    - Evidence Agent answers: "How does the corpus relate to this hypothesis?"
    
    Returns updated hypothesis with evidence items attached.
    """
    # Skip if already has evidence (idempotency)
    if hypothesis.evidence:
        return hypothesis
    
    await adispatch_custom_event("log", {
        "message": f"[Evidence] Gathering evidence for H: {hypothesis.text[:60]}..."
    }, config=config)
    
    # 1. Search OpenAlex with refined query
    query = hypothesis.search_query or hypothesis.text
    if len(query.split()) < 3:
        query += " scientific papers"
    
    print(f"  > [H:{hypothesis.id[:8]}] Query: {query}")
    
    # Main search (increased from 4 to 6)
    papers = await search_papers(query, limit=6)
    
    # ADVERSARIAL SEARCH: Find contradictions
    contradiction_terms = ["limitations", "controversy", "challenges", "contradicts", "fails", "negative results"]
    key_words = [w for w in query.split() if len(w) > 4][:3]
    contradiction_query = f"{' '.join(key_words)} ({' OR '.join(contradiction_terms)})"
    
    try:
        contradiction_papers = await search_papers(contradiction_query, limit=3)
        if contradiction_papers:
            print(f"  > [H:{hypothesis.id[:8]}] Found {len(contradiction_papers)} contradiction papers")
            for p in contradiction_papers:
                if not any(existing['id'] == p['id'] for existing in papers):
                    papers.append(p)
    except Exception as e:
        print(f"[Evidence] Adversarial search failed for {hypothesis.id[:8]}: {e}")
    
    if not papers:
        print(f"[Evidence] No papers found for hypothesis {hypothesis.id[:8]}")
        return hypothesis
    
    # 2. Classify each paper
    from langchain_core.messages import SystemMessage, HumanMessage
    
    system_msg = """You are a critical scientist. Evaluate if the abstract supports or contradicts the hypothesis.
    
    DEFINITIONS:
    - "support": Abstract explicitly matches the hypothesis mechanism or outcome.
    - "contradict": Abstract explicitly refutes the mechanism or shows opposite outcome.
    - "neutral": Abstract is irrelevant, tangential, or inconclusive.
    
    STRENGTH SCALE (1-5):
    1: Tenuous/Weak (e.g. indirect inference)
    5: Definitive/Strong (e.g. direct experimental trial matching exact variables)
    """
    
    async def classify_paper(paper):
        abstract = reconstruct_abstract(paper.get("abstract"))
        if not abstract:
            return None
        
        messages = [
            SystemMessage(content=system_msg),
            HumanMessage(content=f"Hypothesis: {hypothesis.text}\n\nAbstract: {abstract}")
        ]
        
        try:
            result = await structured_llm.ainvoke(messages)
            return EvidenceItem(
                paper_id=paper["id"],
                title=paper["title"],
                venue=paper["host_venue"],
                year=paper["publication_year"],
                stance=result.stance,
                strength=result.strength,
                key_points=result.key_points,
                url=paper["landing_page_url"]
            )
        except Exception as e:
            print(f"[Evidence] Error classifying paper: {e}")
            return None
    
    # Process papers in parallel
    await adispatch_custom_event("log", {"message": f"[Evidence] Analyzing {len(papers)} papers for H:{hypothesis.text[:30]}..."}, config=config)
    evidence_items = await asyncio.gather(*[classify_paper(p) for p in papers])
    evidence_items = [e for e in evidence_items if e is not None]
    
    # 3. Compute summary
    if evidence_items:
        num_support = sum(1 for e in evidence_items if e.stance == "support")
        num_contradict = sum(1 for e in evidence_items if e.stance == "contradict")
        num_neutral = sum(1 for e in evidence_items if e.stance == "neutral")
        
        print(f"[Evidence] H:{hypothesis.id[:8]} Aggregates: +{num_support} / -{num_contradict} ~{num_neutral}")
        await adispatch_custom_event("log", {"message": f"[Evidence] Result for H:{hypothesis.text[:20]}... : +{num_support} (Support), -{num_contradict} (Contradict)"}, config=config)
        
        # Generate verdict
        summary_prompt = f"""Given the following evidence items, write a 1-sentence VERDICT on the hypothesis.
        Mention if it is broadly supported, disputed, or lacks specific data.
        
        Evidence:
        {[f"- {e.title} ({e.stance} {e.strength}/5): {'; '.join(e.key_points)}" for e in evidence_items]}
        """
        try:
            summary_res = await llm.ainvoke(summary_prompt)
            verdict = summary_res.content
        except:
            verdict = "Analysis complete."
        
        # Update hypothesis
        hypothesis.evidence = evidence_items
        hypothesis.evidence_summary = verdict
    
    return hypothesis


async def evidence_node(state: DiscoveryState, config: RunnableConfig) -> dict:
    """
    Evidence Agent: Hypothesis-Conditional Evaluator
    
    Finds and classifies evidence for ALL hypotheses (top 3).
    Processes each hypothesis independently for epistemic symmetry.
    
    ROLE CLARIFICATION:
    - Runs ONLY after hypotheses are generated
    - Evaluates each hypothesis independently
    - Reuses literature retrieval utilities with different objective
    - STANCE-AGNOSTIC until hypotheses exist
    
    STRICTLY PROHIBITED:
    - Graph modification
    - Concept extraction  
    - Structural reasoning
    - Influencing hypothesis ranking
    
    EPISTEMIC BOUNDARY:
    - Literature Agent: "What exists in the corpus?"
    - Evidence Agent: "How does the corpus relate to this specific hypothesis?"
    """
    if not state.hypotheses:
        return {}
    
    # MOCK MODE
    if state.mock:
        await adispatch_custom_event("log", {"message": "[Evidence] MOCK MODE: Returning dummy evidence for all hypotheses."}, config=config)
        dummy_ev = EvidenceItem(
            paper_id="mock-1",
            title="Mock Evidence Paper",
            stance="support",
            strength=5,
            key_points=["Mock Point 1", "Mock Point 2"]
        )
        return {"hypotheses": [
            h.model_copy(update={"evidence": [dummy_ev]}) 
            for h in state.hypotheses
        ]}
    
    # Check if all have evidence already (idempotency)
    if all(h.evidence for h in state.hypotheses[:3]):
        await adispatch_custom_event("log", {"message": "[Evidence] Skipping (all hypotheses already have evidence)"}, config=config)
        return {}
    
    await adispatch_custom_event("log", {
        "message": f"[Evidence] Gathering evidence for {min(3, len(state.hypotheses))} hypotheses..."
    }, config=config)
    
    llm = get_cheap_llm()
    structured_llm = llm.with_structured_output(EvidenceClassification)
    
    # Process top 3 hypotheses in PARALLEL
    hypotheses_to_process = state.hypotheses[:3]
    
    updated_hypotheses = await asyncio.gather(*[
        gather_evidence_for_hypothesis(h, llm, structured_llm, config)
        for h in hypotheses_to_process
    ])
    
    # Merge: updated top 3 + remaining unchanged
    final_hypotheses = list(updated_hypotheses) + state.hypotheses[3:]
    
    await adispatch_custom_event("log", {
        "message": f"[Evidence] Completed evidence gathering for {len(updated_hypotheses)} hypotheses."
    }, config=config)
    
    return {"hypotheses": final_hypotheses}

