import json
from pathlib import Path
from collections import defaultdict
import numpy as np

SUMMARY_FILE = Path(__file__).resolve().parent / "pilot_evaluation_summary.json"

def analyze():
    with open(SUMMARY_FILE, "r", encoding="utf-8") as f:
        records = json.load(f)

    # Group by (query_id, condition)
    grouped = defaultdict(list)
    by_condition = defaultdict(list)

    for r in records:
        qid = r["query_id"]
        cond = r["condition"]
        rep = r["repetition"]
        ms = r["metrics_summary"]
        cm = r["compute_metrics"]
        item = {
            "query_id": qid,
            "condition": cond,
            "repetition": rep,
            "mean_hops": ms["mean_hops"],
            "grounding": ms["mean_concept_grounding_ratio"],
            "alignment": ms["mean_edge_alignment_ratio"],
            "valid_paths": ms["valid_graph_path_count"],
            "strict_bridges": ms["strict_bridge_count"],
            "single_paper": ms["single_paper_dominated_count"],
            "total_hyps": ms["hypothesis_count"],
            "duration": cm.get("duration_seconds", 0),
            "tokens": cm.get("total_tokens", 0),
            "cost": cm.get("estimated_cost_usd", 0),
            "tool_calls": cm.get("tool_calls", 0),
            "llm_calls": cm.get("llm_calls", 0),
            "detailed": r.get("detailed_hypotheses", [])
        }
        grouped[(qid, cond)].append(item)
        by_condition[cond].append(item)

    print("==========================================================================================")
    print("TABLE 1: PER-QUERY x CONDITION SUMMARY (Averaged across 3 Repetitions)")
    print("==========================================================================================")
    header = f"{'Query':<7} | {'Cond':<5} | {'Hops':<5} | {'Ground':<7} | {'Align':<7} | {'ValidPath':<9} | {'StrictBr':<9} | {'SingleP':<8} | {'Tokens':<7} | {'Time(s)':<7} | {'Tools':<5}"
    print(header)
    print("-" * len(header))

    queries = ["ML-1", "ML-2", "BIO-1", "BIO-3", "CLIM-1"]
    for qid in queries:
        for cond in ["RAG", "GRAPH"]:
            reps = grouped[(qid, cond)]
            hops = np.mean([x["mean_hops"] for x in reps])
            ground = np.mean([x["grounding"] for x in reps])
            align = np.mean([x["alignment"] for x in reps])
            tot_hyps = sum(x["total_hyps"] for x in reps)
            vpaths = sum(x["valid_paths"] for x in reps)
            sbridge = sum(x["strict_bridges"] for x in reps)
            spap = sum(x["single_paper"] for x in reps)
            tok = np.mean([x["tokens"] for x in reps])
            dur = np.mean([x["duration"] for x in reps])
            tools = np.mean([x["tool_calls"] for x in reps])

            vp_str = f"{vpaths}/{tot_hyps} ({vpaths/tot_hyps*100:.1f}%)"
            sb_str = f"{sbridge}/{tot_hyps} ({sbridge/tot_hyps*100:.1f}%)"
            sp_str = f"{spap}/{tot_hyps} ({spap/tot_hyps*100:.1f}%)"

            print(f"{qid:<7} | {cond:<5} | {hops:<5.2f} | {ground:<7.3f} | {align:<7.3f} | {vp_str:<9} | {sb_str:<9} | {sp_str:<8} | {tok:<7.0f} | {dur:<7.1f} | {tools:<5.1f}")
        print("-" * len(header))

    print("\n==========================================================================================")
    print("TABLE 2: AGGREGATE CONDITION COMPARISON (45 Hypotheses each: 15 runs x 3 hyps)")
    print("==========================================================================================")
    for cond in ["RAG", "GRAPH"]:
        reps = by_condition[cond]
        hops = np.mean([x["mean_hops"] for x in reps])
        ground = np.mean([x["grounding"] for x in reps])
        align = np.mean([x["alignment"] for x in reps])
        tot_hyps = sum(x["total_hyps"] for x in reps)
        vpaths = sum(x["valid_paths"] for x in reps)
        sbridge = sum(x["strict_bridges"] for x in reps)
        spap = sum(x["single_paper"] for x in reps)
        tok = np.mean([x["tokens"] for x in reps])
        dur = np.mean([x["duration"] for x in reps])
        cost = sum(x["cost"] for x in reps)
        tools = np.mean([x["tool_calls"] for x in reps])

        print(f"Condition: {cond}")
        print(f"  Total Hypotheses Evaluated: {tot_hyps}")
        print(f"  Mean Hop Count:             {hops:.2f}")
        print(f"  Mean Concept Grounding:     {ground:.3f} ({ground*100:.1f}%)")
        print(f"  Mean Directed Edge Align:   {align:.3f} ({align*100:.1f}%)")
        print(f"  Valid Graph Path Rate:      {vpaths}/{tot_hyps} ({vpaths/tot_hyps*100:.1f}%)")
        print(f"  Strict Cross-Paper Bridges: {sbridge}/{tot_hyps} ({sbridge/tot_hyps*100:.1f}%)")
        print(f"  Single-Paper Dominated:     {spap}/{tot_hyps} ({spap/tot_hyps*100:.1f}%)")
        print(f"  Mean Tokens per Run:        {tok:.0f}")
        print(f"  Mean Duration per Run:      {dur:.1f}s")
        print(f"  Mean Tool Calls per Run:    {tools:.1f}")
        print(f"  Total Cost (USD):           ${cost:.4f}")
        print()

    print("==========================================================================================")
    print("TABLE 3: REPETITION VARIABILITY (STD DEV across Repetitions 1, 2, 3)")
    print("==========================================================================================")
    for qid in queries:
        for cond in ["RAG", "GRAPH"]:
            reps = grouped[(qid, cond)]
            grounds = [x["grounding"] for x in reps]
            aligns = [x["alignment"] for x in reps]
            vps = [x["valid_paths"] for x in reps]
            sbs = [x["strict_bridges"] for x in reps]
            print(f"{qid} [{cond}]: Grounding={np.mean(grounds):.3f}±{np.std(grounds):.3f} | Align={np.mean(aligns):.3f}±{np.std(aligns):.3f} | ValidPaths={vps} (std={np.std(vps):.2f}) | StrictBridges={sbs} (std={np.std(sbs):.2f})")

    # Search for specific qualitative cases
    print("\n==========================================================================================")
    print("EXEMPLARS: GRAPH SUCCEEDS (Strict Bridge = True) vs RAG FAILS")
    print("==========================================================================================")
    graph_successes = []
    for r in records:
        if r["condition"] == "GRAPH":
            for h in r.get("detailed_hypotheses", []):
                if h["is_strict_bridge"]:
                    graph_successes.append((r["query_id"], r["repetition"], h))
    
    print(f"Found {len(graph_successes)} Graph strict bridges. Showing top 3:")
    for qid, rep, h in graph_successes[:3]:
        print(f"\n[{qid} Rep {rep}] Hops: {h['hop_count']} | Text: {h['hypothesis_text']}")
        print(f"  Cumulative Papers: {h['cumulative_papers_count']} | Terminal Co-occurs: {h['terminal_co_occurs']}")
        for t in h["transition_details"]:
            print(f"    - {t['from']} -> {t['to']} | Prov: {t['provenance_papers']}")

    print("\n==========================================================================================")
    print("EXEMPLARS: RAG SUCCESSES OR COMPARATIVE CASES")
    print("==========================================================================================")
    rag_valid = []
    for r in records:
        if r["condition"] == "RAG":
            for h in r.get("detailed_hypotheses", []):
                if h["is_valid_graph_path"] or h["edge_alignment_ratio"] > 0:
                    rag_valid.append((r["query_id"], r["repetition"], h))

    print(f"Found {len(rag_valid)} RAG hypotheses with edge_alignment > 0. Showing top 3:")
    for qid, rep, h in rag_valid[:3]:
        print(f"\n[RAG {qid} Rep {rep}] Align: {h['edge_alignment_ratio']} | ValidPath: {h['is_valid_graph_path']} | StrictBridge: {h['is_strict_bridge']}")
        print(f"  Text: {h['hypothesis_text']}")
        for t in h["transition_details"]:
            print(f"    - {t['from']} -> {t['to']} | Edge: {t['has_graph_edge']} | Prov: {t['provenance_papers']}")

if __name__ == "__main__":
    analyze()
