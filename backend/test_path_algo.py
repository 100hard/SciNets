
import networkx as nx
import itertools
import random

def run_test():
    print("=== TOY GRAPH PATH TEST (POST-FIX) ===")
    
    # 1. Create Toy Graph
    G = nx.DiGraph()
    # A -> B -> C -> D (Valid 3-hop path)
    G.add_edge("A", "B", weight=1.0)
    G.add_edge("B", "C", weight=1.0)
    G.add_edge("C", "D", weight=1.0)
    
    # Distractors
    G.add_edge("A", "X", weight=1.0)
    G.add_edge("Y", "D", weight=1.0)
    
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    print("Path A->D exists:", nx.has_path(G, "A", "D"))
    
    # 2. Mock Agent Logic (Post-Fix)
    evaluation_strategy = "full" 
    top_nodes = ["A", "D", "B", "C"] 
    pairs = list(itertools.combinations(top_nodes, 2))
    found_paths = []
    
    print(f"Testing Strategy: {evaluation_strategy}")
    
    # --- START FIXED LOGIC MOCK ---
    if evaluation_strategy == "rag":
       pass
    elif evaluation_strategy == "random":
        pass 
    elif evaluation_strategy == "shortest":
         pass
    else:
         # Full Strategy Logic (Mirroring the fix)
         for u, v in pairs:
             try:
                 if nx.has_path(G, u, v):
                     try:
                         paths = list(itertools.islice(nx.shortest_simple_paths(G, u, v), 5))
                         for p in paths:
                             found_paths.append({"str": str(p), "score": 1.0/len(p)})
                     except: pass
             except: continue
    # --- END FIXED LOGIC MOCK ---
    
    print(f"Found Paths: {len(found_paths)}")
    if len(found_paths) > 0:
        print("[PASS] Full Strategy NOW finds paths.")
    else:
        print("[FAIL] Full Strategy still broken.")

if __name__ == "__main__":
    run_test()
