
import asyncio
import os
import sys
import json
import logging
import pandas as pd
import numpy as np
from uuid import uuid4
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Setup paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.graph import create_graph
from app.state import DiscoveryState
from app.logging_config import get_logger

# Queries Definition
QUERIES = {
    "ML": [
        "Transformer architectures for graphs",
        "Self-supervised learning on headers",
        "Contrastive loss for tabular data",
        "Neural ODEs for weather prediction",
        "Sparse attention mechanisms in vision transformers"
    ],
    "BIO": [
        "CRISPR off-target detection methods",
        "Proteostasis mechanisms in Alzheimer's disease"
    ],
    "CLIM": [
        "Carbon capture using metal-organic frameworks"
    ]
}

# Mapping user inputs to internal strategies
MODES = {
    "full": "full",
    "rag": "rag", 
    "random": "random",
    "shortest": "shortest",
    "no_diversity": "no_diversity",
    "no_yen": "shortest" # Map per instructions
}

async def run_single_eval(sem, query_id, query, method, domain):
    async with sem:
        print(f"--> Starting {query_id} [{method}] '{query[:30]}...'")
        
        try:
            workflow = create_graph()
            # app = workflow.compile() # Already compiled
            app = workflow # check valid
            
            # Map method to strategy
            strategy = MODES.get(method, "full")
            
            state = DiscoveryState(
                user_query=query,
                goal="discover",
                speculation="medium", # Fixed temp implied
                domain_tags=[domain],
                evaluation_mode=True,
                evaluation_strategy=strategy,
                experiment_id=query_id
            )
            
            # Pass config with thread_id for checkpointer
            thread_id = str(uuid4())
            config = {"configurable": {"thread_id": thread_id}}
            
            # Run!
            result = await app.ainvoke(state, config=config)
            print(f"    Done {query_id} [{method}]. Generated {len(result.get('hypotheses', []))} hypotheses.")
            return True
            
        except Exception as e:
            print(f"    FAILED {query_id} [{method}]: {e}")
            import traceback
            traceback.print_exc()
            return False

async def run_evaluation_suite():
    print("=== STARTING PARALLEL EVALUATION SUITE ===")
    
    # Clear temp file
    if os.path.exists("eval_data.jsonl"):
        try:
            os.remove("eval_data.jsonl")
        except: pass
    
    tasks = []
    sem = asyncio.Semaphore(5) # Concurrency Limit
    
    # 1. ML RUNS (5 Queries x 5 Modes)
    ml_modes = ["full", "rag", "random", "shortest", "no_diversity"] 
    for i, q in enumerate(QUERIES["ML"][:1]): # SUBSET: Run only 1 query
        qid = f"ML-{i+1}"
        for m in ml_modes:
            tasks.append(run_single_eval(sem, qid, q, m, "ml"))

    # 2. BIO RUNS (2 Queries x 2 Modes: full, rag)
    for i, q in enumerate(QUERIES["BIO"][:1]): # SUBSET: Run only 1 query
        qid = f"BIO-{i+1}"
        for m in ["full", "rag"]:
            tasks.append(run_single_eval(sem, qid, q, m, "bio"))
            
    # 3. CLIM RUN (1 Query x 1 Mode: full)
    for i, q in enumerate(QUERIES["CLIM"][:1]): # SUBSET: Run only 1 query
        qid = f"CLIM-{i+1}"
        tasks.append(run_single_eval(sem, qid, q, "full", "climate"))

    print(f"Scheduled {len(tasks)} runs...")
    await asyncio.gather(*tasks)
    print("=== EXECUTION COMPLETE ===")
    
    # Process Metrics
    records = []
    if os.path.exists("eval_data.jsonl"):
        with open("eval_data.jsonl", "r") as f:
            for line in f:
                try:
                    records.append(json.loads(line))
                except:
                    pass
    
    df = pd.DataFrame(records)
    if df.empty:
        print("No metrics captured!")
        return

    # Calculate Derived Metrics
    # Papers Per Edge
    def calc_ppe(edges):
        if not edges: return 0
        total = sum(len(e.get("supporting_papers", [])) for e in edges)
        return total / len(edges)
    
    if "edges" in df.columns:
        df["papers_per_edge"] = df["edges"].apply(calc_ppe)
    else:
        df["papers_per_edge"] = 0
    
    # Calculate Similarity/Diversity (Jaccard vs Others in same run)
    df["jaccard_vs_others"] = 0.0
    
    grouped = df.groupby(["query_id", "method"])
    for name, group in grouped:
        texts = group["hypothesis_text"].tolist()
        
        # Also need raw row-level "jaccard_vs_others" for csv 2
        for idx, row in group.iterrows():
            my_text = row["hypothesis_text"]
            others = [t for t in texts if t != my_text]
            if not others:
                df.at[idx, "jaccard_vs_others"] = 0.0 
                # If only 1 hypothesis, diversity is 0? Or N/A?
                # User says: "diversity = average pairwise dissimilarity"
                # If only 1, dissim is undefined or 0.
                continue
            
            my_scores = []
            s1 = set(str(my_text).lower().split())
            for t2 in others:
                s2 = set(str(t2).lower().split())
                inter = len(s1 & s2)
                union = len(s1 | s2)
                jaccard = inter / union if union > 0 else 0
                my_scores.append(1.0 - jaccard)
            df.at[idx, "jaccard_vs_others"] = sum(my_scores) / len(my_scores)

    # OUTPUT 1: Raw CSV
    # Cols: query_id,method,hypothesis_rank,path_length,jaccard_vs_others,papers_per_edge
    raw_cols = ["query_id", "method", "hypothesis_rank", "path_length", "jaccard_vs_others", "papers_per_edge"]
    # Ensure cols exist
    for c in raw_cols:
        if c not in df.columns: df[c] = 0
        
    df_raw = df[raw_cols].copy()
    df_raw.to_csv("data_raw_hypotheses.csv", index=False)
    print("Saved data_raw_hypotheses.csv")

    # OUTPUT 2: Aggregated Tables
    # Table A: ML Baseline Comparison (ML query IDs only)
    df_ml = df[df["domain"] == "ml"]
    if not df_ml.empty:
        table_a = df_ml.groupby("method").agg({
            "path_length": "mean",
            "jaccard_vs_others": "mean", # This approximates avg diversity
            "papers_per_edge": "mean"
        }).reset_index()
        table_a.columns = ["method", "avg_path_length", "avg_diversity", "avg_papers_per_edge"]
        table_a.to_csv("data_table_a.csv", index=False)
        print("Saved data_table_a.csv")

        # Table B: Ablation
        table_b = table_a[table_a["method"].isin(["full", "no_diversity", "shortest", "random"])] 
        table_b.to_csv("data_table_b.csv", index=False) 
        print("Saved data_table_b.csv")

if __name__ == "__main__":
    asyncio.run(run_evaluation_suite())
