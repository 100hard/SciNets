from app.state import DiscoveryState
from app.llm import get_cheap_llm
from app.domains import get_domain_packs
from app.memory import MemoryManager
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List

class DomainClassification(BaseModel):
    domains: List[str] = Field(description="List of relevant domains (e.g., 'bio', 'ml', 'materials')")

async def plan_node(state: DiscoveryState) -> dict:
    """
    Orchestrator Agent: Detects domains and plans the research steps, using long-term memory.
    """
    print(f"[Orchestrator] Planning for query: {state.user_query}")
    
    # 0. Retrieve Memory
    memory = MemoryManager()
    insights = memory.retrieve_insights(state.user_query)
    memory_context = ""
    if insights:
        memory_context = "Past Insights:\n" + "\n".join([f"- {i.content}" for i in insights])
        print(f"[Orchestrator] Using {len(insights)} past insights.")
    
    # MOCK MODE Check
    if state.mock:
        print("[Orchestrator] MOCK MODE: Returning dummy domains.")
        return {
            "plan": {"steps": ["literature", "hypothesis", "experiment"] if state.run_experiments else ["literature", "hypothesis"], "current_step": "literature"},
            "domain_tags": ["general", "mock-domain"]
        }
    
    # 1. Detect Domains
    llm = get_cheap_llm()
    structured_llm = llm.with_structured_output(DomainClassification)
    
    
    available_domains = list(get_domain_packs().keys())
    available_domains_str = ", ".join(available_domains)
    
    from langchain_core.messages import SystemMessage, HumanMessage
    
    # FIX: Add State context (Goal/Speculation)
    system_msg = f"""You are a scientific research assistant. Classify the following query into one or more of these domains: {available_domains_str}.
    
    CONTEXT:
    - User Goal: {state.goal} (e.g. 'survey' = broad/standard, 'discover' = novel/edge)
    - Speculation Level: {state.speculation}
    
    GUIDELINES:
    1. If the goal is 'discover', consider adding exploratory domains like 'ml' or 'theory' if applicable.
    2. If the lens is '{state.lens}', prioritize domains related to it.
    3. Return ONLY domains from the valid list: {available_domains_str}.
    """
    
    messages = [
        SystemMessage(content=system_msg),
        HumanMessage(content=f"Query: {state.user_query}\nDisciplinary Lens: {state.lens}\n\n{memory_context}")
    ]
    
    try:
        result = await structured_llm.ainvoke(messages)
        raw_domains = result.domains or []
    except Exception as e:
        print(f"[Orchestrator] Domain classification failed: {e}")
        raw_domains = []
        
    # FIX: Post-processing & Normalization
    normalized_domains = []
    # Create lookup map for case-insensitive matching
    domain_lookup = {d.lower(): d for d in available_domains}
    
    for d in raw_domains:
        d_clean = d.strip().lower()
        if d_clean in domain_lookup:
            normalized_domains.append(domain_lookup[d_clean])
            
    # Fallback
    if not normalized_domains:
        normalized_domains = ["general"] if "general" in domain_lookup else [available_domains[0]]
        
    # Deduplicate preserving order
    detected_domains = list(dict.fromkeys(normalized_domains))
    print(f"[Orchestrator] Detected domains: {detected_domains}")
    
    # 2. Create Plan (Dynamic)
    steps = ["literature", "hypothesis"]
    
    # Only add experiment if enabled and not just a survey
    if state.run_experiments and state.goal != "survey":
        steps.append("experiment")
        
    plan = {
        "steps": steps,
        "current_step": "literature"
    }
    
    return {
        "plan": plan,
        "domain_tags": detected_domains
    }
