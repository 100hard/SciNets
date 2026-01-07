
import pandas as pd
import numpy as np

RAW_CSV = "full_metrics_per_query_extended.csv"

def main():
    try:
        df = pd.read_csv(RAW_CSV)
    except:
        print("CSV not found.")
        return

    # Infer domain
    def get_domain(qid):
        if "ML" in qid: return "Machine Learning"
        if "BIO" in qid: return "Biology"
        if "CLIM" in qid: return "Climate Science"
        return "Unknown"

    df['domain'] = df['query_id'].apply(get_domain)
    
    # Aggregation
    agg = df.groupby('domain').agg({
        'symbolic_path_length': 'mean',
        'grounded_realized_path_length': 'mean',
        'drop_rate_pct': 'mean',
        'diversity_score': 'mean',
        'failure_flag': 'mean' # Failure Rate
    }).reset_index()
    
    # Formatting
    agg['symbolic_path_length'] = agg['symbolic_path_length'].round(2)
    agg['grounded_realized_path_length'] = agg['grounded_realized_path_length'].round(2)
    agg['drop_rate_pct'] = agg['drop_rate_pct'].round(1)
    agg['diversity_score'] = agg['diversity_score'].round(2)
    agg['failure_rate_pct'] = (agg['failure_flag'] * 100).round(1)
    
    # Output Markdown
    print("# Domain Comparison Summary\n")
    print(agg[['domain', 'symbolic_path_length', 'grounded_realized_path_length', 
               'drop_rate_pct', 'diversity_score', 'failure_rate_pct']].to_markdown(index=False))

if __name__ == "__main__":
    main()
