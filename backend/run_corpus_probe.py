
import asyncio
import os
import sys
import json
import logging
from uuid import uuid4
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.graph import create_graph
from app.state import DiscoveryState

# Settings
QUERY = "loss landscape geometry and generalization"
DOMAIN = "ml"
REPEATS = 3
METHOD = "full"

async def run_iteration(N, run_idx):
    try:
        print(f"--> Starting Run N={N}, Iteration {run_idx}...")
        state = DiscoveryState(
            user_query=QUERY,
            goal="discover",
            speculation="medium",
            domain_tags=[DOMAIN],
            evaluation_mode=True,
            evaluation_strategy=METHOD,
            max_papers=N,
            experiment_id=f"probe-n{N}-r{run_idx}"
        )
        
        app = create_graph()
        config = {"configurable": {"thread_id": str(uuid4())}}
        
        # Run Graph
        final_state = await app.ainvoke(state, config=config)
        
        # Extract Metrics
        graph = final_state.get("concept_graph", {})
        nodes = len(graph.get("nodes", []))
        edges = len(graph.get("edges", []))
        
        hypotheses = final_state.get("hypotheses", [])
        
        # Default Metrics
        top_path_len = 0
        avg_ppe = 0.0
        
        # Try to read granular metrics from eval_data.jsonl if available
        # (Since hypothesizer logs validation details there)
        if os.path.exists("eval_data.jsonl"):
            import pandas as pd
            try:
                # Naive reverse read to find our ID
                with open("eval_data.jsonl", "r") as f:
                    lines = f.readlines()
                    for line in reversed(lines):
                        try:
                            d = json.loads(line)
                            if d.get("query_id") == state.experiment_id and d.get("hypothesis_rank") == 1:
                                top_path_len = d.get("path_length", 0)
                                edges_data = d.get("edges", [])
                                if edges_data:
                                    total_p = sum(len(e.get("supporting_papers", [])) for e in edges_data)
                                    avg_ppe = total_p / len(edges_data)
                                break
                        except: pass
            except: pass
            
        # Fallback if no log found but hypotheses exist (e.g. from graph state directly if we stored it)
        # But for 'full' method, hypothesis existence implies path found.
        # Length? We can guess from text if log missing? No, let's rely on log or 0.
        
        has_valid_path = top_path_len >= 3
        
        return {
            "N": N,
            "Run": run_idx,
            "Nodes": nodes,
            "Edges": edges,
            "HasValidPath": has_valid_path,
            "TopPathLen": top_path_len,
            "PapersPerEdge": avg_ppe,
            "Error": False
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "N": N,
            "Run": run_idx,
            "Nodes": 0,
            "Edges": 0,
            "HasValidPath": False,
            "TopPathLen": 0,
            "PapersPerEdge": 0,
            "Error": True
        }

async def run_probe():
    print(f"=== POST-FIX PROBE: {QUERY} ===")
    
    # Initialize Result File
    csv_file = "probe_results.csv"
    with open(csv_file, "w") as f:
        f.write("N,Run,Nodes,Edges,HasValidPath,TopPathLen,PapersPerEdge\n")

    # Re-Test N=10
    print("\n[POST-FIX PROBE] Re-Testing N=10 Papers...")
    
    # Run once
    res = await run_iteration(10, 1)
    
    # Log to CSV
    row = f"{res['N']},{res['Run']},{res['Nodes']},{res['Edges']},{res['HasValidPath']},{res['TopPathLen']},{res['PapersPerEdge']:.2f}"
    print(f"  Result: {row}")
    with open(csv_file, "a") as f:
        f.write(row + "\n")

if __name__ == "__main__":
    # Clean previous logs
    if os.path.exists("eval_data.jsonl"):
        try: os.remove("eval_data.jsonl") 
        except: pass
        
    asyncio.run(run_probe())
