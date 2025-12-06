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
    
    # 1. Detect Domains
    llm = get_cheap_llm()
    structured_llm = llm.with_structured_output(DomainClassification)
    
    available_domains = ", ".join(get_domain_packs().keys())
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"You are a scientific research assistant. Classify the following query into one or more of these domains: {available_domains}. Consider past insights if relevant."),
        ("human", f"Query: {state.user_query}\nDisciplinary Lens: {state.lens}\n\n{memory_context}")
    ])
    
    chain = prompt | structured_llm
    result = await chain.ainvoke({})
    
    detected_domains = result.domains
    print(f"[Orchestrator] Detected domains: {detected_domains}")
    
    # 2. Create Plan
    # For now, we still use a linear plan, but we store the domains for downstream agents.
    plan = {
        "steps": ["literature", "hypothesis", "experiment"],
        "current_step": "literature"
    }
    
    return {
        "plan": plan,
        "domain_tags": detected_domains
    }
