import asyncio
import os
import sys
from dotenv import load_dotenv

# Ensure we can import app
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
load_dotenv()

from app.state import DiscoveryState
from app.agents.literature import literature_node
from langchain_core.runnables import RunnableConfig, RunnableLambda

async def debug_literature():
    print("DEBUG: Starting Single Query Literature Test")
    
    q_text = "Can synaptic plasticity and neural aging mechanisms inform catastrophic forgetting in artificial neural networks?"
    state = DiscoveryState(
        user_query=q_text,
        goal="discover",
        mode="hypothesis",
        lens="neuroscience",
        max_papers=5,
        experiment_id="DEBUG_Q1"
    )
    
    config = RunnableConfig(configurable={"thread_id": "debug_thread"})
    
    try:
        print("DEBUG: Invoking literature_node...")
        lit_runnable = RunnableLambda(literature_node)
        lit_update = await lit_runnable.ainvoke(state, config)
        
        print("DEBUG: Literature Node returned.")
        if lit_update:
            lit = lit_update.get("literature", [])
            graph = lit_update.get("concept_graph", {})
            print(f"DEBUG: Papers found: {len(lit)}")
            print(f"DEBUG: Graph nodes: {len(graph.get('nodes', []))}")
            if len(lit) > 0:
                print(f"DEBUG: First paper: {lit[0].get('title')}")
        else:
            print("DEBUG: Returned None")
            
    except Exception as e:
        print(f"DEBUG: EXCEPTION caught: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_literature())
