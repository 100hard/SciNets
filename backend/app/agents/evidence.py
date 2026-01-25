from app.telemetry.cost_tracker import CostTracker
from app.state import DiscoveryState, Hypothesis, EvidenceItem
from langchain_core.runnables import RunnableConfig
from langchain_core.callbacks import adispatch_custom_event
from app.llm import get_cheap_llm
from app.tools.openalex import search_papers, reconstruct_abstract
from pydantic import BaseModel, Field
from typing import Literal
import asyncio
from app.logging_config import get_logger

log = get_logger(__name__)

async def gather_evidence_for_hypothesis(
    hypothesis: Hypothesis,
    llm,
    structured_llm,
    config: RunnableConfig
) -> Hypothesis:
    # Context Attribution
    CostTracker.get_instance().push_hypothesis_context(hypothesis.id)
    try:
        # ... existing logic ...
        # (Snippet truncated for brevity, but I must match exact content or replace the whole function body carefully)
        # Actually, let's just insert the context handling around the whole body logic
        
        # Skip if already has evidence (idempotency)
        if hypothesis.evidence:
            return hypothesis
        
        await adispatch_custom_event("log", {
            "message": f"[Evidence] Gathering evidence for H: {hypothesis.text[:60]}..."
        }, config=config)
        
        # ... (rest of logic) ...
        # Since I cannot replace "rest of logic" with a comment, I will target the START and END of the function.
        # But wait, replace_file_content requires me to provide exact content.
        # I will replace the whole function signature and body start, and then do another call for the end? No.
        # I should replace the whole file or large chunks.
        # Given the file size is ~254 lines, I can probably do it in 2 chunks or 1 big chunk if I am careful.
        # Let's try to target specific blocks.
        
        # ACTUALLY, I can just wrap the internal logic.
        pass # Placeholder
    finally:
        CostTracker.get_instance().pop_hypothesis_context() # This is hard to inject with replace_file_content if I don't replace everything.

# Let's try a different strategy.
# 1. Modify imports.
# 2. Modify evidence_node start/end.
# 3. Modify gather_evidence_for_hypothesis start/end.
pass

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
    """
    CostTracker.get_instance().push_hypothesis_context(hypothesis.id)
    try:
        # Skip if already has evidence (idempotency)
        if hypothesis.evidence:
            return hypothesis
        
        await adispatch_custom_event("log", {
            "message": f"[Evidence] Gathering evidence for H: {hypothesis.text[:60]}..."
        }, config=config)
        
        # 1. Search OpenAlex with refined query
        query = hypothesis.search_query or hypothesis.text
        
        # FIX #3: Harden Evidence Retrieval (Domain Constraints)
        # If the hypothesis is about specific domains, FORCE those keywords in the search
        domain_keywords = []
        lower_text = hypothesis.text.lower()
        if "multi-agent" in lower_text or "marl" in lower_text or "multiagent" in lower_text:
            domain_keywords.append('(multi-agent OR multiagent OR MARL)')
        if "reinforcement learning" in lower_text or "rl " in lower_text:
             domain_keywords.append('("reinforcement learning" OR RL)')
             
        # Append constraints if they exist and aren't already in the query
        for k in domain_keywords:
            if k.split('(')[0].strip() not in query: # fast heuristic check
                 query += f" AND {k}"

        if len(query.split()) < 3:
            query += " scientific papers"
        
        print(f"  > [H:{hypothesis.id[:8]}] Query: {query}")
        
        papers = []
        try:
            # Main search (increased from 4 to 6)
            papers = await search_papers(query, limit=6)
            
            # ADVERSARIAL SEARCH: Find contradictions (Only if primary search works)
            if papers:
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
                    # We continue with primary papers
            
        except Exception as e:
            error_msg = str(e)
            print(f"[Evidence] Primary search failed for {hypothesis.id[:8]}: {e}")
            # CHECK: Is this an external API failure?
            if "500" in error_msg or "502" in error_msg or "503" in error_msg or "ConnectError" in error_msg:
                 hypothesis.evidence_status = "failed_external"
                 hypothesis.evidence_summary = "Evidence gathering failed due to external API errors (OpenAlex)."
                 await adispatch_custom_event("log", {"message": f"[Evidence] API Failure for H:{hypothesis.id[:8]}. Marking as failed_external."}, config=config)
                 return hypothesis
    
        if not papers:
            print(f"[Evidence] No papers found for hypothesis {hypothesis.id[:8]}")
            hypothesis.evidence_status = "partial" # No evidence found, but not an error
            return hypothesis
        
        # 1.5 SEMANTIC FILTER (Fix #3 from User - Quality Tier)
        # Filter papers that are domain-irrelevant using lightweight overlap check
        # (Jaccard Similarity) to avoid wasting LLM calls on junk.
        def jaccard_similarity(text1, text2):
            if not text1 or not text2: return 0.0
            stop = {"the", "a", "an", "and", "or", "of", "in", "for", "with", "to", "is", "are", "on", "at", "by", "from", "be", "this", "that"}
            s1 = set(w.lower() for w in text1.split() if w.lower() not in stop and len(w)>2)
            s2 = set(w.lower() for w in text2.split() if w.lower() not in stop and len(w)>2)
            if not s1 or not s2: return 0.0
            return len(s1.intersection(s2)) / len(s1.union(s2))

        if papers:
            original_count = len(papers)
            scored_papers = []
            hyp_text = hypothesis.text + " " + (hypothesis.search_query or "")
            
            for p in papers:
                abstract = reconstruct_abstract(p.get("abstract"))
                title = p.get("title", "")
                content = f"{title} {abstract}"
                score = jaccard_similarity(hyp_text, content)
                p["_rel_score"] = score
                scored_papers.append(p)
            
            # Sort by score
            scored_papers.sort(key=lambda x: x["_rel_score"], reverse=True)
            
            # Filter: Keep top 5, but drop any with extremely low score (< 0.03) unless list becomes empty
            filtered_papers = [p for p in scored_papers if p["_rel_score"] > 0.03]
            
            # If aggressive filtering killed everything, keep top 2 regardless
            if not filtered_papers and scored_papers:
                 filtered_papers = scored_papers[:2]
            elif len(filtered_papers) > 5:
                 filtered_papers = filtered_papers[:5]
                 
            print(f"[Evidence] Semantic Filter: Kept {len(filtered_papers)}/{original_count} papers (Top Score: {scored_papers[0]['_rel_score']:.3f})")
            papers = filtered_papers
        
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
            hypothesis.evidence_status = "complete" # Success
    
            # DETAILED LOGGING
            log.info("evidence_gathered", 
                     hypothesis_id=hypothesis.id,
                     found=len(evidence_items),
                     support=num_support,
                     contradict=num_contradict,
                     neutral=num_neutral,
                     top_evidence=[{"title": e.title, "stance": e.stance} for e in evidence_items[:3]]
            )
        else:
            hypothesis.evidence_status = "partial" # Papers found but classification failed for all?
        
        return hypothesis
    finally:
        CostTracker.get_instance().pop_hypothesis_context()


async def evidence_node(state: DiscoveryState, config: RunnableConfig) -> dict:
    """
    Evidence Agent: Hypothesis-Conditional Evaluator
    """
    CostTracker.get_instance().start_step("evidence", "evidence_processing")
    try:
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
    finally:
         CostTracker.get_instance().end_step("evidence", "evidence_processing")

