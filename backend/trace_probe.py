
import asyncio
import os
import sys
import json
import networkx as nx
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.graph import create_graph
from app.state import DiscoveryState

QUERY = "How does loss landscape geometry influence generalization in overparameterized neural networks?"

async def run_trace():
    print(f"=== TRACE PROBE: {QUERY} ===")
    state = DiscoveryState(
        user_query=QUERY,
        goal="discover",
        speculation="medium",
        domain_tags=["ml"],
        evaluation_mode=True,
        evaluation_strategy="full", # Force FULL to get the path
        max_papers=10,
        experiment_id="trace-ml1"
    )
    
    app = create_graph()
    
    # Run
    final_state = await app.ainvoke(state, config={"configurable": {"thread_id": "trace-run"}})
    
    # Extract Data
    hypotheses = final_state.get("hypotheses", [])
    graph_data = final_state.get("concept_graph", {})
    
    # Extract Data from Graph directly
    graph_data = final_state.get("concept_graph", {})
    nodes = graph_data.get("nodes", [])
    edges = graph_data.get("edges", [])
    
    # Build NX Graph
    G = nx.DiGraph()
    for n in nodes: G.add_node(n['id'])
    for e in edges: G.add_edge(e['source'], e['target'], papers=e.get('history', []))
    
    with open("debug_nodes.txt", "w") as f:
        for n in nodes: f.write(n['id'] + "\n")
    
    # Find Source/Target

    source = next((n['id'] for n in nodes if "loss" in n['id'].lower() or "landscape" in n['id'].lower()), None)
    target = next((n['id'] for n in nodes if "generalization" in n['id'].lower() or "test error" in n['id'].lower()), None)
    
    print(f"Source: {source}, Target: {target}")
    
    top_path = []
    edges_out = []
    
    if source and target and nx.has_path(G, source, target):
        try:
            top_path = nx.shortest_path(G, source, target)
        except: pass
        
    # Get edge details for the path
    if top_path:
        for u, v in zip(top_path[:-1], top_path[1:]):
            edge_data = G.get_edge_data(u, v)
            if edge_data:
                edges_out.append({
                    "source": u,
                    "target": v,
                    "papers": edge_data.get("papers", [])
                })
    
    # Dump to JSON
    trace_data = {
        "top_path": top_path,
        "alternatives": [],
        "edges": edges_out
    }
    
    with open("ml1_trace.json", "w") as f:
        json.dump(trace_data, f, indent=2)
        
    print(f"Trace dumped. Path len: {len(top_path)}")

if __name__ == "__main__":
    asyncio.run(run_trace())
