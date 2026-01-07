
import asyncio
import os
import sys
import json
import pandas as pd
import networkx as nx
import itertools
import random
import nest_asyncio
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List
from langchain_core.prompts import ChatPromptTemplate

nest_asyncio.apply()

load_dotenv()
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.tools.openalex import search_papers, reconstruct_abstract
from app.llm import get_cheap_llm

# --- Data Definitions ---
class Edge(BaseModel):
    source: str
    target: str
    relation: str
    class Config: extra = "forbid"

class ConceptGraph(BaseModel):
    nodes: List[str] = Field(description="List of key concepts")
    edges: List[Edge] = Field(description="List of relationships")
    class Config: extra = "forbid"

# --- Queries ---
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

RAW_CSV = "data_raw_hypotheses.csv"

async def build_graph_direct(query_text):
    print(f"  [Sim] Fetching papers for: {query_text[:30]}...")
    try:
        papers = await search_papers(query_text, limit=10)
    except Exception as e:
        print(f"  [Sim] Search failed: {e}")
        return None

    if not papers:
        print("  [Sim] No papers found.")
        return None

    # Prepare Abstracts
    abstracts_text = ""
    for p in papers:
        abs_t = reconstruct_abstract(p.get("abstract"))
        if abs_t:
            abstracts_text += f"Paper: {p.get('title')}\nAbstract: {abs_t}\n\n"

    print(f"  [Sim] Building graph from {len(papers)} papers...")
    
    # LLM Extraction
    llm = get_cheap_llm()
    structured_llm = llm.with_structured_output(ConceptGraph)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Extract a concept graph. Nodes must be Noun Phrases. Edges must be explicit relationships."),
        ("human", "Abstracts:\n{abstracts}")
    ])
    chain = prompt | structured_llm
    
    try:
        # Just one batch for simplicity/speed in simulation
        res = await chain.ainvoke({"abstracts": abstracts_text[:15000]}) # truncate to avoid context err
        
        # Convert to dict format expected by analysis
        graph_data = {
            "nodes": [{"id": n} for n in res.nodes],
            "edges": [{"source": e.source, "target": e.target} for e in res.edges]
        }
        return graph_data
    except Exception as e:
        print(f"  [Sim] Extraction failed: {e}")
        return None

def get_symbolic_paths(graph_data, query_text):
    if not graph_data: return {"full": 0, "shortest": 0, "random": 0}
    
    nodes = [n['id'] for n in graph_data.get("nodes", [])]
    edges = graph_data.get("edges", [])
    
    if not nodes or not edges:
        return {"full": 0, "shortest": 0, "random": 0}

    G = nx.DiGraph()
    for n in nodes: G.add_node(n)
    for e in edges: G.add_edge(e['source'], e['target'])
    
    # Heuristic Source/Target
    q_words = [w.lower() for w in query_text.split() if len(w)>3]
    hits = []
    for n in G.nodes():
        score = sum(1 for w in q_words if w in n.lower())
        if score > 0: hits.append((n, score))
    
    hits.sort(key=lambda x: x[1], reverse=True)
    if len(hits) < 2:
        # Fallback: Just take most connected nodes
        deg = dict(G.degree())
        sorted_deg = sorted(deg.items(), key=lambda x: x[1], reverse=True)
        if len(sorted_deg) < 2: return {"full": 0, "shortest": 0, "random":0}
        start_node = sorted_deg[0][0]
        end_node = sorted_deg[1][0]
    else:
        start_node = hits[0][0]
        end_node = hits[1][0]
    
    # 1. Full (Yen's approx)
    full_len = 0
    try:
        if nx.has_path(G, start_node, end_node):
            paths = list(itertools.islice(nx.shortest_simple_paths(G, start_node, end_node), 5))
            if paths:
                lengths = [len(p) for p in paths]
                # Filter for diversity? just avg
                full_len = sum(lengths) / len(lengths)
    except: pass

    # 2. Shortest
    short_len = 0
    try:
        if nx.has_path(G, start_node, end_node):
            p = nx.shortest_path(G, start_node, end_node)
            short_len = len(p)
    except: pass
    
    # 3. Random
    rand_len = 0
    try:
        # Avg of 5 walks
        walks = []
        for _ in range(5):
            curr = start_node
            path = [curr]
            for _ in range(5):
                neighbors = list(G.neighbors(curr))
                if not neighbors: break
                curr = random.choice(neighbors)
                path.append(curr)
            walks.append(len(path))
        rand_len = sum(walks)/len(walks)
    except: pass

    return {"full": full_len, "shortest": short_len, "random": rand_len}

async def main():
    if not os.path.exists(RAW_CSV):
        print("CSV not found")
        return

    df = pd.read_csv(RAW_CSV)
    analysis_results = []
    symbolic_cache = {}
    # unique_queries = df["query_id"].unique() 
    # FIX: Iterate over configured queries to ensure simulation runs even if CSV incomplete
    unique_queries = list(QUERIES.keys())
    
    for qid in unique_queries:
        if qid not in QUERIES: continue
        
        print(f"Analyzing {qid}...")
        
        if qid not in symbolic_cache:
            graph = await build_graph_direct(QUERIES[qid])
            symbolic_cache[qid] = get_symbolic_paths(graph, QUERIES[qid])
            
        sym_metrics = symbolic_cache[qid]
        
        # 2. Process Rows (If exist)
        q_rows = df[df["query_id"] == qid]
        
        if q_rows.empty:
            # Add a placeholder entry so we save the symbolic data
             res_entry = {
                "query_id": qid,
                "strategy": "simulation_only",
                "symbolic_path_length": sym_metrics["full"], # Save full as proxy
                "grounded_realized_path_length": 0,
                "dropped_nodes": 0
            }
             analysis_results.append(res_entry)
        
        for _, row in q_rows.iterrows():
            method = row["method"]
            realized_len = row["path_length"]
            
            sym_len = 0
            if method in ["full", "no_diversity"]:
                sym_len = sym_metrics["full"]
            elif method in ["shortest", "no_yen"]:
                sym_len = sym_metrics["shortest"]
            elif method == "random":
                sym_len = sym_metrics["random"]
            
            dropped = max(0, sym_len - realized_len) if method != "rag" else 0
            
            res_entry = {
                "query_id": qid,
                "strategy": method,
                "symbolic_path_length": round(sym_len, 2),
                "grounded_path_length": realized_len,
                "dropped_nodes": round(dropped, 2)
            }
            analysis_results.append(res_entry)
            
    out_json = "path_drop_analysis_extension.json"
    with open(out_json, "w") as f:
        json.dump(analysis_results, f, indent=2)
        
    out_csv = "path_drop_table.csv"
    pd.DataFrame(analysis_results).to_csv(out_csv, index=False)
    print("Done used standalone.")

if __name__ == "__main__":
    asyncio.run(main())
