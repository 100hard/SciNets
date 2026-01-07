
import pandas as pd
import sys

# Combine all sources to see what we actually have used for the table
# The table generation used: pd.concat([df_old_clean, df_new_clean])

RAW_OLD_CSV = "data_raw_hypotheses.csv"
NEW_METRICS_CSV = "new_full_metrics.csv" 

def main():
    try:
        # Load Old
        df_old = pd.read_csv(RAW_OLD_CSV)
        df_old = df_old[df_old['hypothesis_rank'] == 1]
        df_old = df_old[['query_id', 'method']].copy()
        df_old['source'] = 'real'
        
        # Load New
        df_new = pd.read_csv(NEW_METRICS_CSV)
        df_new = df_new[['query_id', 'method']].copy()
        df_new['source'] = 'extended'
        
        # Combine
        df = pd.concat([df_old, df_new], ignore_index=True)
        
        # Filter for the 4 Table Strategies
        strategies = ['full', 'shortest', 'random', 'rag']
        df = df[df['method'].isin(strategies)]
        
        # Matrix
        queries = sorted(df['query_id'].unique())
        
        print(f"Total Unique Hypotheses (Rank 1, Table Strategies): {len(df)}")
        print("\nCOVERAGE MATRIX (X = Present)")
        print(f"{'Query':<10} | {'full':<8} | {'shortest':<8} | {'random':<8} | {'rag':<8}")
        print("-" * 55)
        
        for q in queries:
            row_str = f"{q:<10} | "
            for s in strategies:
                exists = not df[(df['query_id'] == q) & (df['method'] == s)].empty
                mark = "X" if exists else "."
                row_str += f"{mark:<8} | "
            print(row_str)
            
        # Count missing
        expected = len(queries) * 4
        actual = len(df)
        print(f"\nMissing Hypotheses: {expected - actual} / {expected}")
        
    except Exception as e:
        print(e)

if __name__ == "__main__":
    main()
