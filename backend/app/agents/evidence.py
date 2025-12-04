from app.state import DiscoveryState, EvidenceItem
from app.tools.openalex import search_papers, reconstruct_abstract
from app.llm import get_cheap_llm
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import Literal

class EvidenceClassification(BaseModel):
    stance: Literal["support", "contradict", "neutral"] = Field(description="Does the paper support, contradict, or is neutral to the hypothesis?")
    strength: int = Field(description="Strength of evidence (1-5)", ge=1, le=5)
    key_points: list[str] = Field(description="Key points from the paper relevant to the hypothesis")

async def evidence_node(state: DiscoveryState) -> dict:
    """
    Evidence Agent: Finds and classifies evidence for the selected hypothesis.
    """
    selected_id = state.selected_hypothesis_id
    if not state.hypotheses:
        return {}
        
    hypothesis = next((h for h in state.hypotheses if h.id == selected_id), None)
    if not hypothesis:
        return {}

    print(f"[Evidence] Searching for evidence for: {hypothesis.text}")
    
    # 1. Search OpenAlex with hypothesis text
    papers = await search_papers(hypothesis.text, limit=3)
    
    evidence_items = []
    llm = get_cheap_llm()
    structured_llm = llm.with_structured_output(EvidenceClassification)
    
    for paper in papers:
        abstract = reconstruct_abstract(paper.get("abstract"))
        if not abstract:
            continue
            
        # 2. Classify Stance
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a critical scientist. Evaluate if the following abstract supports or contradicts the hypothesis."),
            ("human", f"Hypothesis: {hypothesis.text}\n\nAbstract: {abstract}")
        ])
        
        chain = prompt | structured_llm
        try:
            result = await chain.ainvoke({})
            
            item = EvidenceItem(
                paper_id=paper["id"],
                title=paper["title"],
                venue=paper["host_venue"],
                year=paper["publication_year"],
                stance=result.stance,
                strength=result.strength,
                key_points=result.key_points,
                url=paper["landing_page_url"]
            )
            evidence_items.append(item)
            print(f"[Evidence] Classified {paper['title']}: {result.stance}")
            
        except Exception as e:
            print(f"[Evidence] Error classifying paper: {e}")

    # Update the hypothesis in the list
    # Note: In a real DB, we'd update the record. Here we update the state list.
    updated_hypotheses = []
    for h in state.hypotheses:
        if h.id == selected_id:
            h.evidence = evidence_items
        updated_hypotheses.append(h)
        
    return {"hypotheses": updated_hypotheses}
