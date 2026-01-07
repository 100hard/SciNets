
import asyncio
import os
import sys
import json
import pandas as pd
from uuid import uuid4
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.graph import create_graph
from app.state import DiscoveryState

# --- CONFIGURATION ---
N_PAPERS = 10
RAW_DATA_FILE = "data_raw_hypotheses.csv"
QUERIES = {
    # Machine Learning
    "ML-1": "How does loss landscape geometry influence generalization in overparameterized neural networks?",
    "ML-2": "What mechanistic links connect implicit bias of stochastic gradient descent and flat minima in deep learning models?",
    "ML-3": "How does overparameterization give rise to the double descent phenomenon in modern neural networks?",
    "ML-4": "What causal relationships exist between normalization techniques (e.g., batch normalization) and training stability in deep neural networks?",
    "ML-5": "How do optimization dynamics in transformer models influence in-context learning behavior?",
    
    # Biology
    "BIO-1": "How does sleep deprivation affect synaptic plasticity and memory consolidation in the hippocampus?",
    "BIO-2": "What mechanistic pathways link neuroinflammation, microglial activation, and cognitive decline during aging?",
    "BIO-3": "What mechanistic pathways link gut microbiome dysbiosis to cognitive decline and mood regulation abnormalities?",
    "BIO-4": "What mechanistic chain connects epithelial–mesenchymal transition (EMT) to metastatic spread and immune evasion in cancer?",
    "BIO-5": "How do mitochondrial dysfunction and oxidative stress interact to drive cellular aging and neurodegeneration?",

    # Climate
    "CLIM-1": "What causal pathways connect deforestation, nutrient runoff, and coral reef degradation?",
    "CLIM-2": "How do atmospheric aerosol concentrations influence cloud microphysics and alter regional rainfall patterns?",
    "CLIM-3": "What causal chain links permafrost melting to methane release, atmospheric warming, and long-term climate feedback loops?",
    "CLIM-4": "How do urban heat-island effects mechanistically contribute to regional weather anomalies and public-health heat risk?"
}

# Methods to run per domain
# All: full, random, rag, shortest
# ML only: no_diversity (for ablation)
# Note: 'no_yen' in table B = 'shortest' strategy on ML queries.

async def run_single_eval(query_id, query_text, method):
    print(f"--> Running {query_id} [{method}]...")
    
    # 1. Setup State
    # Note: 'no_diversity' uses 'full' extraction but skips diversity filtering. 
    # Logic in hypothesis.py supports 'no_diversity' strategy string.
    
    state = DiscoveryState(
        user_query=query_text,
        goal="discover",
        speculation="medium",
        domain_tags=["ml"] if query_id.startswith("ML") else ["bio"] if query_id.startswith("BIO") else ["climate"],
        evaluation_mode=True,
        evaluation_strategy=method,
        max_papers=N_PAPERS,
        experiment_id=f"{query_id}-{method}"
    )
    
    app = create_graph()
    config = {"configurable": {"thread_id": str(uuid4())}}
    
    try:
        # 2. Run Graph
        final_state = await app.ainvoke(state, config=config)
        hypotheses = final_state.get("hypotheses", [])
        graph = final_state.get("concept_graph", {})
        edges_data = graph.get("edges", [])
        
        # 3. Extract Metrics & Log
        # We need top-3 hypotheses
        # The 'hypotheses' list contains objects. 
        # In strict eval mode, hypothesis.py might return a list of dicts or objects.
        # Let's assume standard Pydantic objects or dicts.
        
        results = []
        for i, h in enumerate(hypotheses[:3]):
            # Path Length: h.path is list of nodes? Or generated text?
            # In 'full'/'shortest', path is explicit.
            # In 'rag'/'random', might be less clear. 
            # We rely on the agent to populate 'path' or 'chain'.
            
            # Hypothesis object usually has 'chain' (list of strings).
            path_len = len(h.chain) if hasattr(h, 'chain') and h.chain else 0
            
            # If path_len is 0 but it's consistent, check text. 
            # For strict graph methods, h.chain should be populated.
            
            # Papers Per Edge (Avg)
            # Find edges in graph corresponding to path? 
            # This is hard to map back perfectly without the path object.
            # Approximation: Global Graph Papers/Edge? No, that's too coarse.
            # Better: The hypothesis agent logs validation details.
            # Actually, let's use the 'top_path_len' if h.chain is empty?
            
            # Jaccard vs Others (Diversity)
            # Compare h.chain nodes with other top-3 hypotheses.
            others = [oh.chain for j, oh in enumerate(hypotheses[:3]) if i != j and hasattr(oh, 'chain')]
            jaccard = 0.0
            if others and hasattr(h, 'chain'):
                my_nodes = set(h.chain)
                if my_nodes:
                    similarities = []
                    for o_chain in others:
                        o_nodes = set(o_chain)
                        u = len(my_nodes | o_nodes)
                        inter = len(my_nodes & o_nodes)
                        similarities.append(inter/u if u > 0 else 0)
                    jaccard = sum(similarities) / len(similarities) if similarities else 0.0
            
            # Papers per edge for THIS hypothesis path
            # We need to look up edges (u,v) in the graph.
            ppe = 0.0
            if hasattr(h, 'chain') and len(h.chain) >= 2:
                edge_counts = []
                for u, v in zip(h.chain[:-1], h.chain[1:]):
                    # Find edge data
                    # Edge keys are usually f"{u}->{v}" or lookup
                    # The graph object structure:
                    # edges: [{source, target, history: [...]}, ...]
                    match = next((e for e in edges_data if e['source'] == u and e['target'] == v), None)
                    if match:
                        edge_counts.append(len(match.get('history', [])))
                    else:
                        edge_counts.append(0)
                ppe = sum(edge_counts) / len(edge_counts) if edge_counts else 0.0

            row = {
                "query_id": query_id,
                "domain": "ML" if query_id.startswith("ML") else "Science",
                "method": method,
                "hypothesis_rank": i + 1,
                "path_length": path_len,
                "jaccard_vs_others": round(jaccard, 2),
                "papers_per_edge": round(ppe, 2),
                "text": h.hypothesis[:200].replace("\n", " ") # Snippet for verification
            }
            results.append(row)
            
            # Save Candidate for Case Studies
            if i == 0:
                with open(CANDIDATES_FILE, "a") as f:
                    record = {
                        "query_id": query_id,
                        "method": method,
                        "hypothesis": h.hypothesis,
                        "chain": h.chain,
                        "evidence": edge_counts if hasattr(h, 'chain') and len(h.chain) >= 2 else []
                    }
                    f.write(json.dumps(record) + "\n")
                    
        return results

    except Exception as e:
        print(f"!!! Error in {query_id} [{method}]: {e}")
        import traceback
        traceback.print_exc()
        return []

async def main():
    # Initialize CSV if not exists
    if not os.path.exists(RAW_DATA_FILE):
        df = pd.DataFrame(columns=["query_id", "domain", "method", "hypothesis_rank", "path_length", "jaccard_vs_others", "papers_per_edge", "text"])
        df.to_csv(RAW_DATA_FILE, index=False)
    
    # Load existing to skip done
    existing_df = pd.read_csv(RAW_DATA_FILE)
    done_keys = set(zip(existing_df["query_id"], existing_df["method"]))
    
    tasks = []
    
    for qid, qtext in QUERIES.items():
        # Determine methods
        methods = ["full", "random", "rag", "shortest"]
        if qid.startswith("ML"):
            methods.append("no_diversity")
            
        for m in methods:
            if (qid, m) in done_keys:
                print(f"Skipping {qid} [{m}] (Already done)")
                continue
                
            # Run
            # Sequential execution for safety/stability
            rows = await run_single_eval(qid, qtext, m)
            
            if rows:
                new_df = pd.DataFrame(rows)
                new_df.to_csv(RAW_DATA_FILE, mode='a', header=False, index=False)
                print(f"Saved {len(rows)} rows for {qid} [{m}]")
            else:
                print(f"No results for {qid} [{m}]")

if __name__ == "__main__":
    if os.path.exists("eval_data.jsonl"): 
        try: os.remove("eval_data.jsonl")
        except: pass
        
    asyncio.run(main())
