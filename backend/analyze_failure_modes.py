
import pandas as pd
import json
import numpy as np

PATH_DROPS_FILE = "c:/Users/dubey/.gemini/antigravity/brain/dc5aafff-ab9a-4ac7-98b0-5f499d84923d/path_drops.json"
RAW_CSV_FILE = "data_raw_hypotheses.csv"

def classify_failure(row):
    sym = row['symbolic_path_length']
    real = row['grounded_realized_path_length']
    ppe = row['papers_per_edge']
    
    # Heuristics
    if real == 0:
        return "Collapse (Generic Summary)"
    
    if ppe < 0.5: # Low evidence
        return "Hallucinated Bridge"
        
    if real < sym:
        return "Truncation"
        
    if real >= sym and ppe >= 1.0:
        return "Success"
        
    return "Unknown"

def main():
    # 1. Load Data
    try:
        with open(PATH_DROPS_FILE, 'r') as f:
            drops_data = json.load(f)
        drops_df = pd.DataFrame(drops_data)
    except Exception as e:
        print(f"Error loading path_drops.json: {e}")
        return

    try:
        raw_df = pd.read_csv(RAW_CSV_FILE)
        # Filter for rank 1
        raw_df = raw_df[raw_df['hypothesis_rank'] == 1]
    except Exception as e:
        print(f"Error loading raw csv: {e}")
        return

    # 2. Merge
    # drops_df has: query_id, strategy, symbolic, grounded, dropped
    # raw_df has: query_id, method, jaccard, papers_per_edge
    
    # Rename for merge
    drops_df['method'] = drops_df['strategy']
    
    merged = pd.merge(drops_df, raw_df[['query_id', 'method', 'jaccard_vs_others', 'papers_per_edge']], 
                      on=['query_id', 'method'], how='inner')
    
    # 3. Compute Metrics
    merged['drop_rate_pct'] = (merged['dropped_nodes'] / merged['symbolic_path_length'].replace(0, 1)) * 100
    merged['drop_rate_pct'] = merged['drop_rate_pct'].fillna(0)
    
    merged['diversity_score'] = 1 - merged['jaccard_vs_others']
    
    merged['failure_flag'] = (merged['dropped_nodes'] > 0) | (merged['grounded_realized_path_length'] == 0)
    merged['failure_mode'] = merged.apply(classify_failure, axis=1)
    
    # 4. Correlations
    # Correlation(Symbolic Length, Drop Rate)
    corr_len_drop = merged['symbolic_path_length'].corr(merged['drop_rate_pct'])
    
    # Correlation(Diversity, Drop Rate)
    corr_div_drop = merged['diversity_score'].corr(merged['drop_rate_pct'])

    # 5. Output Table
    # Columns: Query, Strategy, Sym Len, Real Len, Drop %, Success/Fail, Failure Mode, Diversity, PPE
    out_cols = ['query_id', 'method', 'symbolic_path_length', 'grounded_realized_path_length', 
                'drop_rate_pct', 'failure_flag', 'failure_mode', 'diversity_score', 'papers_per_edge']
    
    final_df = merged[out_cols]
    final_df.to_csv("full_metrics_table.csv", index=False)
    
    # 6. Generate Report Text
    report = f"""
# Failure Mode & Correlation Analysis

## Correlations
- **Correlation(Path Length, Drop Rate)**: {corr_len_drop:.2f}
   - Interpretation: {"Strong Positive" if corr_len_drop > 0.5 else "Weak" if corr_len_drop < 0.3 else "Moderate"} (Does longer path imply more dropping?)

- **Correlation(Diversity, Drop Rate)**: {corr_div_drop:.2f}
   - Interpretation: {"Strong Positive" if corr_div_drop > 0.5 else "Weak" if corr_div_drop < 0.3 else "Moderate"} (Does high diversity imply more dropping?)

## Failure Modes
Total Runs Analyzed: {len(final_df)}
Failure Distribution:
{final_df['failure_mode'].value_counts().to_string()}

## Full Metrics Table (Top 5 Rows)
{final_df.head(5).to_markdown()}
    """
    
    with open("correlation_report_temp.md", "w") as f:
        f.write(report)
        
    print("Analysis Complete.")

if __name__ == "__main__":
    main()
