
import pandas as pd
import json
import os

RAW_CSV = "data_raw_hypotheses.csv"
CANDIDATES_FILE = "eval_candidates.jsonl"
OUT_TABLE_A = "data_table_a.csv"
OUT_TABLE_B = "data_table_b.csv"
OUT_CASES = "case_studies_draft.md"
OUT_FAILURES = "failure_cases_draft.md"

def main():
    if not os.path.exists(RAW_CSV):
        print("Raw data not found.")
        return

    df = pd.read_csv(RAW_CSV)
    
    # Infer domain if missing
    if "domain" not in df.columns:
        def get_domain(qid):
            if str(qid).startswith("ML"): return "ML"
            if str(qid).startswith("BIO"): return "Biology"
            if str(qid).startswith("CLIM"): return "Climate"
            return "Other"
        df["domain"] = df["query_id"].apply(get_domain)
    
    # --- 1. Table A: Strategy Comparison (All Queries) ---

    # Methods: full, random, rag, shortest
    # Columns: method, avg_path_length, avg_diversity, avg_papers_per_edge
    # Diversity = 1 - Jaccard? Or just Jaccard?
    # User asked for "avg_diversity". Usually Diversity = 1 - Similarity.
    # User provided example: random,1.2,0.83... Random usually has high diversity (low similarity).
    # If Jaccard is similarity (0-1), then Diversity = 1 - Jaccard.
    # Let's assume Diversity = 1 - jaccard_vs_others.
    
    df["diversity"] = 1.0 - df["jaccard_vs_others"]
    
    # Filter for Table A methods
    table_a_methods = ["full", "random", "rag", "shortest"]
    df_a = df[df["method"].isin(table_a_methods)].copy()
    
    # Group by method
    # Columns: method, avg_path_length, avg_diversity, avg_papers_per_edge
    # Note: 'shortest' might be absent if we mapped it to 'no_yen' or vice versa.
    # Script runs 'shortest'.
    
    stats_a = df_a.groupby("method").agg({
        "path_length": "mean",
        "diversity": "mean",
        "papers_per_edge": "mean"
    }).reset_index()
    
    stats_a.columns = ["method", "avg_path_length", "avg_diversity", "avg_papers_per_edge"]
    stats_a = stats_a.round(2)
    stats_a.to_csv(OUT_TABLE_A, index=False)
    print(f"Generated {OUT_TABLE_A}")

    # --- 2. Table B: Ablation Study (ML Queries Only) ---
    # Rows: full, no_diversity, no_yen
    # map 'shortest' -> 'no_yen' for this table
    
    df_ml = df[df["domain"] == "ML"].copy()
    
    # Map methods
    stats_b_rows = []
    
    # Full
    if "full" in df_ml["method"].unique():
        f_stats = df_ml[df_ml["method"] == "full"].agg({
            "path_length": "mean", "diversity": "mean", "papers_per_edge": "mean"
        })
        stats_b_rows.append(["full", f_stats["path_length"], f_stats["diversity"], f_stats["papers_per_edge"]])
        
    # No Diversity
    if "no_diversity" in df_ml["method"].unique():
        nd_stats = df_ml[df_ml["method"] == "no_diversity"].agg({
            "path_length": "mean", "diversity": "mean", "papers_per_edge": "mean"
        })
        stats_b_rows.append(["no_diversity", nd_stats["path_length"], nd_stats["diversity"], nd_stats["papers_per_edge"]])
        
    # No Yen (mapped from shortest)
    if "shortest" in df_ml["method"].unique():
        ny_stats = df_ml[df_ml["method"] == "shortest"].agg({
            "path_length": "mean", "diversity": "mean", "papers_per_edge": "mean"
        })
        stats_b_rows.append(["no_yen", ny_stats["path_length"], ny_stats["diversity"], ny_stats["papers_per_edge"]])
        
    df_b = pd.DataFrame(stats_b_rows, columns=["configuration", "avg_path_length", "avg_diversity", "avg_papers_per_edge"])
    df_b = df_b.round(2)
    df_b.to_csv(OUT_TABLE_B, index=False)
    print(f"Generated {OUT_TABLE_B}")

    # --- 3. Case Studies & Failures ---
    # We will dump candidates for manual selection
    # Candidates file has full text
    
    valid_candidates = []
    if os.path.exists(CANDIDATES_FILE):
        with open(CANDIDATES_FILE, "r") as f:
            for line in f:
                try:
                    valid_candidates.append(json.loads(line))
                except: pass
    
    # Find candidates for Case Studies (Full method preferred)
    # ML, BIO, CLIM
    
    case_studies = []
    for dom_prefix in ["ML", "BIO", "CLIM"]:
        # Find best candidate: method='full', high path len, coherent text
        dom_cands = [c for c in valid_candidates if c["query_id"].startswith(dom_prefix) and c["method"] == "full"]
        if not dom_cands:
             # Fallback
             dom_cands = [c for c in valid_candidates if c["query_id"].startswith(dom_prefix)]
        
        if dom_cands:
            # Simple heuristic: Longest chain
            best = max(dom_cands, key=lambda x: len(x.get("chain", [])))
            case_studies.append(best)

    with open(OUT_CASES, "w") as f:
        f.write("# Case Studies Draft\n\n")
        for c in case_studies:
            f.write(f"## {c['query_id']}\n")
            f.write(f"Query: {c['query_id']}\n")
            f.write(f"Hypothesis: {c['hypothesis']}\n")
            f.write(f"Path: {' -> '.join(c.get('chain', []))}\n")
            f.write(f"Evidence: {c.get('evidence', [])}\n\n")

    # Failure Cases
    # Look for: Random (Redundant?), Shortest (Short?), RAG (Unstructured?)
    # Or 'full' failure (0 path len?)
    
    failures = []
    # 1. Shortest Path (Overly short)
    short_cands = [c for c in valid_candidates if c["method"] == "shortest" and len(c.get("chain", [])) <= 2]
    if short_cands: failures.append(short_cands[0])
    
    # 2. Random (Weak grounding?)
    rand_cands = [c for c in valid_candidates if c["method"] == "random"]
    if rand_cands: failures.append(rand_cands[0])
    
    with open(OUT_FAILURES, "w") as f:
        f.write("# Failure Cases Draft\n\n")
        for c in failures:
            f.write(f"## {c['query_id']} ({c['method']})\n")
            f.write(f"Hypothesis: {c['hypothesis']}\n")
            f.write(f"Chain: {c.get('chain')}\n\n")
            
    print("Drafted Cases.")

if __name__ == "__main__":
    main()
