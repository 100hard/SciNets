import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from sklearn.metrics import cohen_kappa_score
from scipy.stats import spearmanr, pearsonr

UNBLINDED_FILE = Path(__file__).resolve().parent / "unblinded_human_evaluation_results.json"
BLINDING_CHECK_FILE = Path(__file__).resolve().parent / "blinding_check_results.json"
PROFILES_FILE = Path(__file__).resolve().parent / "evaluator_profiles.json"

def compute_icc_2_1(y1, y2):
    """Computes ICC(2,1): Two-way random effects, absolute agreement, single rater."""
    Y = np.column_stack([y1, y2])
    n, k = Y.shape
    row_mean = np.mean(Y, axis=1)
    col_mean = np.mean(Y, axis=0)
    grand_mean = np.mean(Y)

    ss_total = np.sum((Y - grand_mean) ** 2)
    ss_rows = k * np.sum((row_mean - grand_mean) ** 2)
    ss_cols = n * np.sum((col_mean - grand_mean) ** 2)
    ss_error = ss_total - ss_rows - ss_cols

    ms_rows = ss_rows / (n - 1)
    ms_cols = ss_cols / (k - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))

    icc = (ms_rows - ms_error) / (ms_rows + (k - 1) * ms_error + (k / n) * (ms_cols - ms_error))
    return float(icc)

def run_analysis():
    with open(UNBLINDED_FILE, "r", encoding="utf-8") as f:
        records = json.load(f)
    with open(BLINDING_CHECK_FILE, "r", encoding="utf-8") as f:
        bcheck = json.load(f)
    with open(PROFILES_FILE, "r", encoding="utf-8") as f:
        profiles = json.load(f)

    dimensions = [
        "scientific_plausibility",
        "mechanistic_coherence",
        "novelty_relative_to_literature",
        "investigative_value"
    ]
    dim_labels = {
        "scientific_plausibility": "Scientific Plausibility",
        "mechanistic_coherence": "Mechanistic Coherence",
        "novelty_relative_to_literature": "Novelty rel. Literature",
        "investigative_value": "Investigative Value"
    }

    conditions = ["RAG", "GRAPH_ONLY", "GRAPH"]
    cond_names = {
        "RAG": "Condition A (RAG)",
        "GRAPH_ONLY": "Condition C (Graph-only)",
        "GRAPH": "Condition B (Graph ReAct)"
    }

    print("==========================================================================================")
    print("SECTION 1: EVALUATOR CREDENTIALS & DOCUMENTED LIMITATIONS")
    print("==========================================================================================")
    for eid, prof in profiles.items():
        print(f"[{eid.upper()}]:")
        for k, v in prof.items():
            print(f"  {k}: {v}")
        print()

    print("==========================================================================================")
    print("SECTION 2: PRIMARY INTER-RATER RELIABILITY (Evaluator 1 vs Evaluator 2 on 135 Hypotheses)")
    print("==========================================================================================")
    print(f"{'Dimension':<32} | {'Quadratic Weighted Kappa':<26} | {'ICC(2,1)':<10} | {'Spearman rho':<12}")
    print("-" * 88)
    for dim in dimensions:
        e1_scores = [r["evaluations"]["evaluator_1"][dim] for r in records]
        e2_scores = [r["evaluations"]["evaluator_2"][dim] for r in records]
        
        qwk = cohen_kappa_score(e1_scores, e2_scores, weights="quadratic")
        icc = compute_icc_2_1(e1_scores, e2_scores)
        rho, _ = spearmanr(e1_scores, e2_scores)
        dname = dim_labels[dim]
        print(f"{dname:<32} | {qwk:<26.3f} | {icc:<10.3f} | {rho:<12.3f}")

    print("\n==========================================================================================")
    print("SECTION 3: PRIMARY DISTRIBUTIONS & DESCRIPTIVE STATS BY CONDITION")
    print("==========================================================================================")
    # Average rating across the 2 evaluators for each hypothesis
    for dim in dimensions:
        dname = dim_labels[dim]
        print(f"\n--- Dimension: {dname.upper()} (1 to 5 scale) ---")
        print(f"{'Condition':<26} | {'Mean':<6} | {'Std':<6} | {'Median':<6} | {'Score Frequencies (1, 2, 3, 4, 5)':<35}")
        print("-" * 88)
        for cond in conditions:
            cond_records = [r for r in records if r["condition"] == cond]
            # consensus score (mean of rater 1 and rater 2)
            cons_scores = [0.5 * (r["evaluations"]["evaluator_1"][dim] + r["evaluations"]["evaluator_2"][dim]) for r in cond_records]
            # integer rounded or discrete bins for distribution
            e1_sc = [r["evaluations"]["evaluator_1"][dim] for r in cond_records]
            e2_sc = [r["evaluations"]["evaluator_2"][dim] for r in cond_records]
            all_sc = e1_sc + e2_sc # 90 ratings per condition
            counts = [all_sc.count(i) for i in [1, 2, 3, 4, 5]]
            counts_str = f"[{counts[0]}, {counts[1]}, {counts[2]}, {counts[3]}, {counts[4]}] (N={len(all_sc)})"

            cname = cond_names[cond]
            print(f"{cname:<26} | {np.mean(cons_scores):<6.2f} | {np.std(cons_scores):<6.2f} | {np.median(cons_scores):<6.2f} | {counts_str:<35}")

    print("\n==========================================================================================")
    print("SECTION 4: DOMAIN & PER-QUERY BREAKDOWN (Mean Consensus Scores)")
    print("==========================================================================================")
    queries = ["ML-1", "ML-2", "BIO-1", "BIO-3", "CLIM-1"]
    q_domains = {"ML-1": "ML", "ML-2": "ML", "BIO-1": "BIO", "BIO-3": "BIO", "CLIM-1": "CLIM"}
    
    header = f"{'Query':<7} | {'Domain':<6} | {'Condition':<26} | {'Plausibility':<12} | {'Coherence':<10} | {'Novelty':<10} | {'Value':<10}"
    print(header)
    print("-" * len(header))
    for qid in queries:
        dom = q_domains[qid]
        for cond in conditions:
            q_recs = [r for r in records if r["query_id"] == qid and r["condition"] == cond]
            means = []
            for dim in dimensions:
                c_sc = [0.5 * (r["evaluations"]["evaluator_1"][dim] + r["evaluations"]["evaluator_2"][dim]) for r in q_recs]
                means.append(np.mean(c_sc))
            cname = cond_names[cond]
            print(f"{qid:<7} | {dom:<6} | {cname:<26} | {means[0]:<12.2f} | {means[1]:<10.2f} | {means[2]:<10.2f} | {means[3]:<10.2f}")
        print("-" * len(header))

    print("\n==========================================================================================")
    print("SECTION 5: CORRELATION BETWEEN HUMAN SCORES & AUTOMATED GRAPH METRICS")
    print("==========================================================================================")
    # automated metrics to correlate against
    auto_metrics = [
        ("concept_grounding_ratio", "Concept Grounding"),
        ("edge_alignment_ratio", "Edge Alignment"),
        ("is_valid_graph_path", "Valid Graph Path (0/1)"),
        ("is_strict_bridge", "Strict Bridge (0/1)")
    ]

    print(f"{'Human Dimension':<30} | {'Automated Metric':<24} | {'Pearson r':<12} | {'Spearman rho':<12}")
    print("-" * 84)
    for dim in dimensions:
        dname = dim_labels[dim]
        h_scores = [0.5 * (r["evaluations"]["evaluator_1"][dim] + r["evaluations"]["evaluator_2"][dim]) for r in records]
        for mkey, mlabel in auto_metrics:
            m_vals = [float(r["automated_metrics"].get(mkey, 0)) for r in records]
            pr, _ = pearsonr(h_scores, m_vals)
            sr, _ = spearmanr(h_scores, m_vals)
            print(f"{dname:<30} | {mlabel:<24} | {pr:<12.3f} | {sr:<12.3f}")
        print("-" * 84)

    print("\n==========================================================================================")
    print("SECTION 6: BLINDING CHECK RESULTS (Responses recorded before unblinding)")
    print("==========================================================================================")
    print("[EVALUATOR 1 BLINDING CHECK RESPONSE]:")
    print(bcheck["evaluator_1_blinding_check"])
    print("\n[EVALUATOR 2 BLINDING CHECK RESPONSE]:")
    print(bcheck["evaluator_2_blinding_check"])

    print("\n==========================================================================================")
    print("SECTION 7: RAW EVALUATOR COMMENTS SAMPLE ACROSS CONDITIONS")
    print("==========================================================================================")
    for cond in conditions:
        print(f"\n>>> Condition: {cond_names[cond]}")
        cond_records = [r for r in records if r["condition"] == cond]
        # sample 2
        for r in cond_records[:2]:
            print(f"Task {r['task_id']} ({r['query_id']}): \"{r['hypothesis_text'][:80]}...\"")
            e1 = r["evaluations"]["evaluator_1"]
            e2 = r["evaluations"]["evaluator_2"]
            print(f"  E1: Plaus={e1['scientific_plausibility']} Coh={e1['mechanistic_coherence']} Nov={e1['novelty_relative_to_literature']} Val={e1['investigative_value']}")
            print(f"      Rationale (Plaus): {e1['comments']['scientific_plausibility_rationale']}")
            print(f"      Rationale (Coh): {e1['comments']['mechanistic_coherence_rationale']}")
            print(f"  E2: Plaus={e2['scientific_plausibility']} Coh={e2['mechanistic_coherence']} Nov={e2['novelty_relative_to_literature']} Val={e2['investigative_value']}")
            print(f"      Rationale (Plaus): {e2['comments']['scientific_plausibility_rationale']}")

if __name__ == "__main__":
    run_analysis()
