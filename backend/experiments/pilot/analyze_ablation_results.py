import json
from pathlib import Path
from collections import defaultdict
import numpy as np

SUMMARY_FILE = Path(__file__).resolve().parent / "ablation_evaluation_summary.json"

def analyze_ablation():
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

    conditions = ["RAG", "GRAPH_ONLY", "GRAPH"]
    cond_labels = {
        "RAG": "A: RAG (Baseline)",
        "GRAPH_ONLY": "C: Graph-only (Static)",
        "GRAPH": "B: Graph (ReAct)"
    }

    print("================================================================================================================")
    print("TABLE 1: 3-CONDITION PER-QUERY BREAKDOWN (Mean over 3 Repetitions, 9 Hypotheses per Cell)")
    print("================================================================================================================")
    header = f"{'Query':<7} | {'Condition':<15} | {'Hops':<5} | {'Ground':<7} | {'Align':<7} | {'ValidPath':<10} | {'StrictBr':<9} | {'SingleP':<8} | {'Tokens':<7} | {'Time(s)':<7} | {'LLMs':<5} | {'Tools':<5}"
    print(header)
    print("-" * len(header))

    queries = ["ML-1", "ML-2", "BIO-1", "BIO-3", "CLIM-1"]
    for qid in queries:
        for cond in conditions:
            reps = grouped[(qid, cond)]
            if not reps:
                continue
            hops = np.mean([x["mean_hops"] for x in reps])
            ground = np.mean([x["grounding"] for x in reps])
            align = np.mean([x["alignment"] for x in reps])
            tot_hyps = sum(x["total_hyps"] for x in reps)
            vpaths = sum(x["valid_paths"] for x in reps)
            sbridge = sum(x["strict_bridges"] for x in reps)
            spap = sum(x["single_paper"] for x in reps)
            tok = np.mean([x["tokens"] for x in reps])
            dur = np.mean([x["duration"] for x in reps])
            llms = np.mean([x["llm_calls"] for x in reps])
            tools = np.mean([x["tool_calls"] for x in reps])

            vp_str = f"{vpaths}/{tot_hyps} ({vpaths/tot_hyps*100:.0f}%)"
            sb_str = f"{sbridge}/{tot_hyps} ({sbridge/tot_hyps*100:.0f}%)"
            sp_str = f"{spap}/{tot_hyps} ({spap/tot_hyps*100:.0f}%)"

            cname = cond_labels.get(cond, cond)
            print(f"{qid:<7} | {cname:<15} | {hops:<5.2f} | {ground:<7.3f} | {align:<7.3f} | {vp_str:<10} | {sb_str:<9} | {sp_str:<8} | {tok:<7.0f} | {dur:<7.1f} | {llms:<5.1f} | {tools:<5.1f}")
        print("-" * len(header))

    print("\n================================================================================================================")
    print("TABLE 2: AGGREGATE 3-WAY COMPARISON (45 Hypotheses per Condition across 15 Runs)")
    print("================================================================================================================")
    header2 = f"{'Metric':<32} | {'Condition A (RAG)':<20} | {'Condition C (Graph-only)':<25} | {'Condition B (Graph ReAct)':<25}"
    print(header2)
    print("-" * len(header2))

    agg = {}
    for cond in conditions:
        reps = by_condition[cond]
        tot_hyps = sum(x["total_hyps"] for x in reps)
        vpaths = sum(x["valid_paths"] for x in reps)
        sbridge = sum(x["strict_bridges"] for x in reps)
        spap = sum(x["single_paper"] for x in reps)
        agg[cond] = {
            "tot_hyps": tot_hyps,
            "hops": np.mean([x["mean_hops"] for x in reps]),
            "grounding": np.mean([x["grounding"] for x in reps]),
            "alignment": np.mean([x["alignment"] for x in reps]),
            "vpaths": vpaths,
            "vpaths_pct": vpaths / tot_hyps * 100,
            "sbridge": sbridge,
            "sbridge_pct": sbridge / tot_hyps * 100,
            "spap": spap,
            "spap_pct": spap / tot_hyps * 100,
            "tokens": np.mean([x["tokens"] for x in reps]),
            "duration": np.mean([x["duration"] for x in reps]),
            "llms": np.mean([x["llm_calls"] for x in reps]),
            "tools": np.mean([x["tool_calls"] for x in reps]),
            "cost": sum(x["cost"] for x in reps)
        }

    rows = [
        ("Concept Grounding Ratio", f"{agg['RAG']['grounding']*100:.1f}%", f"{agg['GRAPH_ONLY']['grounding']*100:.1f}%", f"{agg['GRAPH']['grounding']*100:.1f}%"),
        ("Directed Edge Alignment", f"{agg['RAG']['alignment']*100:.1f}%", f"{agg['GRAPH_ONLY']['alignment']*100:.1f}%", f"{agg['GRAPH']['alignment']*100:.1f}%"),
        ("Valid Graph Path Rate", f"{agg['RAG']['vpaths']}/{agg['RAG']['tot_hyps']} ({agg['RAG']['vpaths_pct']:.1f}%)", f"{agg['GRAPH_ONLY']['vpaths']}/{agg['GRAPH_ONLY']['tot_hyps']} ({agg['GRAPH_ONLY']['vpaths_pct']:.1f}%)", f"{agg['GRAPH']['vpaths']}/{agg['GRAPH']['tot_hyps']} ({agg['GRAPH']['vpaths_pct']:.1f}%)"),
        ("Strict Cross-Paper Bridges", f"{agg['RAG']['sbridge']}/{agg['RAG']['tot_hyps']} ({agg['RAG']['sbridge_pct']:.1f}%)", f"{agg['GRAPH_ONLY']['sbridge']}/{agg['GRAPH_ONLY']['tot_hyps']} ({agg['GRAPH_ONLY']['sbridge_pct']:.1f}%)", f"{agg['GRAPH']['sbridge']}/{agg['GRAPH']['tot_hyps']} ({agg['GRAPH']['sbridge_pct']:.1f}%)"),
        ("Single-Paper Dominated Chains", f"{agg['RAG']['spap']}/{agg['RAG']['tot_hyps']} ({agg['RAG']['spap_pct']:.1f}%)", f"{agg['GRAPH_ONLY']['spap']}/{agg['GRAPH_ONLY']['tot_hyps']} ({agg['GRAPH_ONLY']['spap_pct']:.1f}%)", f"{agg['GRAPH']['spap']}/{agg['GRAPH']['tot_hyps']} ({agg['GRAPH']['spap_pct']:.1f}%)"),
        ("Mean Hop Count", f"{agg['RAG']['hops']:.2f}", f"{agg['GRAPH_ONLY']['hops']:.2f}", f"{agg['GRAPH']['hops']:.2f}"),
        ("Mean Duration per Run", f"{agg['RAG']['duration']:.1f}s", f"{agg['GRAPH_ONLY']['duration']:.1f}s", f"{agg['GRAPH']['duration']:.1f}s"),
        ("Mean Tokens per Run", f"{agg['RAG']['tokens']:.0f}", f"{agg['GRAPH_ONLY']['tokens']:.0f}", f"{agg['GRAPH']['tokens']:.0f}"),
        ("Mean LLM Calls per Run", f"{agg['RAG']['llms']:.1f}", f"{agg['GRAPH_ONLY']['llms']:.1f}", f"{agg['GRAPH']['llms']:.1f}"),
        ("Mean Tool Calls per Run", f"{agg['RAG']['tools']:.1f}", f"{agg['GRAPH_ONLY']['tools']:.1f}", f"{agg['GRAPH']['tools']:.1f}"),
        ("Total Estimated Cost (USD)", f"${agg['RAG']['cost']:.4f}", f"${agg['GRAPH_ONLY']['cost']:.4f}", f"${agg['GRAPH']['cost']:.4f}")
    ]

    for title, vA, vC, vB in rows:
        print(f"{title:<32} | {vA:<20} | {vC:<25} | {vB:<25}")

    print("\n================================================================================================================")
    print("TABLE 3: REPETITION VARIABILITY (STD DEV across Repetitions 1, 2, 3)")
    print("================================================================================================================")
    for qid in queries:
        print(f">>> Query: {qid}")
        for cond in conditions:
            reps = grouped[(qid, cond)]
            grounds = [x["grounding"] for x in reps]
            aligns = [x["alignment"] for x in reps]
            vps = [x["valid_paths"] for x in reps]
            sbs = [x["strict_bridges"] for x in reps]
            cname = cond_labels.get(cond, cond)
            print(f"  {cname:<23}: Grounding={np.mean(grounds):.3f}±{np.std(grounds):.3f} | Align={np.mean(aligns):.3f}±{np.std(aligns):.3f} | ValidPaths={vps} (std={np.std(vps):.2f}) | StrictBridges={sbs} (std={np.std(sbs):.2f})")

    # Detailed Exemplars of Condition C
    print("\n================================================================================================================")
    print("CONDITION C (GRAPH-ONLY) EXEMPLARS & BEHAVIOR")
    print("================================================================================================================")
    for r in records:
        if r["condition"] == "GRAPH_ONLY":
            for h in r["detailed_hypotheses"]:
                if h["is_valid_graph_path"]:
                    print(f"[{r['query_id']} Rep {r['repetition']}] Valid Path: {h['is_valid_graph_path']} | Strict Bridge: {h['is_strict_bridge']}")
                    print(f"  Text: {h['hypothesis_text']}")
                    for t in h["transition_details"]:
                        print(f"    - {t['from']} -> {t['to']} | Edge: {t['has_graph_edge']} | Prov: {t['provenance_papers']}")
                    break

if __name__ == "__main__":
    analyze_ablation()
