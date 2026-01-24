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

# Singleton Memory for In-Memory Persistence across requests
global_memory = MemorySaver()

def create_graph(memory=None):
    """
    Main discovery graph.
    
    Pipeline: PLAN -> LITERATURE -> HYPOTHESIS -> EVIDENCE -> CRITIQUE -> DECISION -> END
    
    NOTE: Experiments are NOT part of the default discovery pipeline.
    They are user-triggered only via POST /run_experiment endpoint.
    """
    workflow = StateGraph(DiscoveryState)
    
    # Add nodes (NO experiment_node in default pipeline)
    workflow.add_node("plan", plan_node)
    workflow.add_node("literature", literature_node)
    
    # Split Node for Loop/Resume Support
    workflow.add_node("hypothesis_preview", hypothesis_node)
    workflow.add_node("hypothesis_deep", hypothesis_node)

    workflow.add_node("evidence", evidence_node)
    workflow.add_node("critique", critique_node)
    workflow.add_node("decision", decision_node)
    
    # Add edges - SEQUENTIAL flow
    workflow.set_entry_point("plan")
    workflow.add_edge("plan", "literature")
    
    def route_literature(state: DiscoveryState):
        if state.goal == "survey":
            return "critique"
        return "hypothesis_preview"

    workflow.add_conditional_edges(
        "literature",
        route_literature,
        {
            "critique": "critique",
            "hypothesis_preview": "hypothesis_preview"
        }
    )
    
    # NEW FLOW: Preview -> Interrupt -> Deep -> Evidence
    workflow.add_edge("hypothesis_preview", "hypothesis_deep")
    workflow.add_edge("hypothesis_deep", "evidence")
    workflow.add_edge("evidence", "critique")
    workflow.add_edge("critique", "decision")
    workflow.add_edge("decision", END)
    
    # MemorySaver for state persistence (Verified on Windows)
    # Use passed memory (global) or create new one (fallback)
    checkpointer = memory if memory else MemorySaver()
    
    return workflow.compile(checkpointer=checkpointer, interrupt_after=["hypothesis_preview"])

