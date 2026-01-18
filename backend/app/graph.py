from langgraph.graph import StateGraph, END
from app.state import DiscoveryState
from app.agents.orchestrator import plan_node
from app.agents.literature import literature_node
from app.agents.hypothesis import hypothesis_node
from app.agents.evidence import evidence_node
# NOTE: experiment_node removed from default pipeline (now user-triggered only)
from app.agents.critique import critique_node
from app.agents.decision import decision_node

from langgraph.checkpoint.memory import MemorySaver

def create_graph():
    """
    Main discovery graph.
    
    Pipeline: PLAN → LITERATURE → HYPOTHESIS → EVIDENCE → CRITIQUE → DECISION → END
    
    NOTE: Experiments are NOT part of the default discovery pipeline.
    They are user-triggered only via POST /run_experiment endpoint.
    """
    workflow = StateGraph(DiscoveryState)
    
    # Add nodes (NO experiment_node in default pipeline)
    workflow.add_node("plan", plan_node)
    workflow.add_node("literature", literature_node)
    workflow.add_node("hypothesis", hypothesis_node)
    workflow.add_node("evidence", evidence_node)
    workflow.add_node("critique", critique_node)
    workflow.add_node("decision", decision_node)
    
    # Add edges - SEQUENTIAL flow
    workflow.set_entry_point("plan")
    workflow.add_edge("plan", "literature")
    
    def route_literature(state: DiscoveryState):
        if state.goal == "survey":
            return "critique"
        return "hypothesis"

    workflow.add_conditional_edges(
        "literature",
        route_literature,
        {
            "critique": "critique",
            "hypothesis": "hypothesis"
        }
    )
    
    # Sequential: Hypothesis → Evidence → Critique → Decision
    # (No more parallel experiment branch)
    workflow.add_edge("hypothesis", "evidence")
    workflow.add_edge("evidence", "critique")
    workflow.add_edge("critique", "decision")
    workflow.add_edge("decision", END)
    
    # MemorySaver for state persistence (Verified on Windows)
    memory = MemorySaver()
    
    return workflow.compile(checkpointer=memory)

