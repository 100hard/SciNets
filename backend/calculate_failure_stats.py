
import pandas as pd

CSV_FILE = "data_raw_hypotheses.csv"

def main():
    try:
        df = pd.read_csv(CSV_FILE)
    except Exception as e:
        print(f"Error loading {CSV_FILE}: {e}")
        return

    # Total runs
    total = len(df)
    print(f"Total Runs (N): {total}")
    
    # Counts
    # failure_mode column must exist
    if 'failure_mode' not in df.columns:
        print("Error: 'failure_mode' column missing.")
        # Try inferring? No, prompt says "Only count real runs".
        # But we generated the csv, so we know it's there.
        return

    counts = df['failure_mode'].value_counts()
    
    # Ensure all keys exist
    modes = ["Success", "Truncation", "Collapse", "Hallucinated Bridge"]
    
    print(f"{'Failure Mode':<20} | {'Count':<6} | {'%':<6}")
    print("-" * 38)
    
    summary = []
    
    for m in modes:
        c = counts.get(m, 0)
        pct = (c / total * 100) if total > 0 else 0
        print(f"{m:<20} | {c:<6} | {pct:.1f}%")
        summary.append((m, c, pct))
        
    # Output for Tool Consumption
    print("\n--- FINAL TABLE ---")
    for m, c, pct in summary:
        print(f"| {m} | {pct:.1f}% |")

if __name__ == "__main__":
    main()
