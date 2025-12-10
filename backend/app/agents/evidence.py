from app.state import DiscoveryState, EvidenceItem
from app.tools.openalex import search_papers, reconstruct_abstract
from app.llm import get_cheap_llm
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.runnables import RunnableConfig
from langchain_core.callbacks import adispatch_custom_event

class EvidenceClassification(BaseModel):
    stance: Literal["support", "contradict", "neutral"] = Field(description="Does the paper support, contradict, or is neutral to the hypothesis?")
    strength: int = Field(description="Strength of evidence (1-5)", ge=1, le=5)
    key_points: list[str] = Field(description="Key points from the paper relevant to the hypothesis")

async def evidence_node(state: DiscoveryState, config: RunnableConfig) -> dict:
    """
    Evidence Agent: Finds and classifies evidence for the selected hypothesis.
    """
    selected_id = state.selected_hypothesis_id
    if not state.hypotheses:
        return {}
        
    hypothesis = next((h for h in state.hypotheses if h.id == selected_id), None)
    if not hypothesis:
        return {}

    # IDEMPOTENCY CHECK
    if hypothesis.evidence:
        await adispatch_custom_event("log", {"message": "[Evidence] Skipping (already cached)"}, config=config)
        return {}

    if state.mock:
        await adispatch_custom_event("log", {"message": "[Evidence] MOCK MODE: Returning dummy evidence."}, config=config)
        dummy_ev = EvidenceItem(
            paper_id="mock-1",
            title="Mock Evidence Paper",
            stance="support",
            strength=5,
            key_points=["Mock Point 1", "Mock Point 2"]
        )
        return {"hypotheses": [
            h if h.id != selected_id else h.copy(update={"evidence": [dummy_ev]}) 
            for h in state.hypotheses
        ]}

    await adispatch_custom_event("log", {"message": f"[Evidence] Searching for evidence for: {hypothesis.text}"}, config=config)
    
    # 1. Search OpenAlex with refined query
    # Prefer pre-computed search query if available, else use text
    # Limit increased to 5 for better coverage
    query = hypothesis.search_query or hypothesis.text
    
    # Optional: Augment query if it's too short for a paper search
    if len(query.split()) < 3:
        query += " scientific papers"
        
    print(f"  > Query: {query}")
    papers = await search_papers(query, limit=5)
    
    if not papers:
        print("[Evidence] No papers found.")
        # Mark hypothesis status if possible (assuming field exists or just log)
        # hypothesis.evidence_status = "no_papers_found" 
        return {"hypotheses": state.hypotheses} 

    
    evidence_items = []
    llm = get_cheap_llm()
    structured_llm = llm.with_structured_output(EvidenceClassification)
    
    tasks = []
    
    async def process_paper(paper):
        abstract = reconstruct_abstract(paper.get("abstract"))
        if not abstract: return None
            
        # 2. Classify Stance (Tightened Prompt)
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

    import asyncio
    evidence_items = await asyncio.gather(*[process_paper(p) for p in papers])
    evidence_items = [e for e in evidence_items if e is not None]
    
    # 3. Compute Aggregates & Summary
    if evidence_items:
        # Metrics
        num_support = sum(1 for e in evidence_items if e.stance == "support")
        num_contradict = sum(1 for e in evidence_items if e.stance == "contradict")
        num_neutral = sum(1 for e in evidence_items if e.stance == "neutral")
        
        # Weighted Score: Support(+Strength) - Contradict(-Strength)
        support_score = sum(e.strength for e in evidence_items if e.stance == "support") - \
                        sum(e.strength for e in evidence_items if e.stance == "contradict")
        
        print(f"[Evidence] Aggregates: +{num_support} / -{num_contradict} ~{num_neutral} (Score: {support_score})")
        
        # Synthesis LLM Pass
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
            
        # Attach to hypothesis (assuming dynamic fields allowed or exists)
        # We can store this in a 'result_summary' or similar if strictly typed
        updated_hypotheses = []
        for h in state.hypotheses:
            if h.id == selected_id:
                h.evidence = evidence_items
                if hasattr(h, "evidence_summary"): # Backward compatibility check
                     h.evidence_summary = verdict
                # Also store the score if desired
                # h.evidence_score = support_score
            updated_hypotheses.append(h)
    else:
        updated_hypotheses = state.hypotheses

    return {"hypotheses": updated_hypotheses}
