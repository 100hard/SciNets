from langgraph.graph import StateGraph, END
from app.state import DiscoveryState
from app.agents.orchestrator import plan_node
from app.agents.literature import literature_node
from app.agents.hypothesis import hypothesis_node
from app.agents.evidence import evidence_node
from app.agents.experiment import experiment_node
from app.agents.critique import critique_node

from langgraph.checkpoint.memory import MemorySaver

def create_graph():
    workflow = StateGraph(DiscoveryState)
    
    # Add nodes
    workflow.add_node("plan", plan_node)
    workflow.add_node("literature", literature_node)
    workflow.add_node("hypothesis", hypothesis_node)
    workflow.add_node("evidence", evidence_node)
    workflow.add_node("experiment", experiment_node)
    workflow.add_node("critique", critique_node)
    
    # Add edges
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
    # workflow.add_edge("literature", "hypothesis")  <-- Removed linear edge
    workflow.add_edge("hypothesis", "evidence")
    workflow.add_edge("evidence", "experiment")
    workflow.add_edge("experiment", "critique")
    workflow.add_edge("critique", END)
    
    # Add Checkpointer for Human In The Loop
    memory = MemorySaver()
    
    # Interrupt before 'critique' to allow user to review hypotheses/experiments
    return workflow.compile(checkpointer=memory, interrupt_before=["critique"])
