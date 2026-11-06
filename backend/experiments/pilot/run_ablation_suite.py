import asyncio
import os
import sys
import json
import time
import random
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))
load_dotenv(BASE_DIR / ".env")

from experiments.pilot.queries import PILOT_QUERIES
from experiments.pilot.runner_rag import run_rag_condition
from experiments.pilot.runner_graph import run_graph_condition
from experiments.pilot.runner_graph_only import run_graph_only_condition
from experiments.pilot.evaluate_pilot_metrics import evaluate_result_file

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

REPETITIONS = 3

async def run_ablation_experiments():
    print("==================================================================")
    print(f"  STARTING 3-CONDITION ABLATION EXPERIMENT: 5 QUERIES x 3 CONDITIONS x {REPETITIONS} REPS")
    print("  Conditions:")
    print("    A: RAG (Papers only, 1 LLM call, 0 tool calls)")
    print("    B: Graph ReAct (Graph + ReAct tool exploration + synthesis)")
    print("    C: Graph-only (Papers + static graph representation, 1 LLM call, 0 tool calls)")
    print("==================================================================\n")

    start_suite = time.time()

    for item in PILOT_QUERIES:
        qid = item["id"]
        print(f"\n>>> Processing Query: {qid} [{item['domain'].upper()}]")

        for rep in range(1, REPETITIONS + 1):
            # Condition A: RAG
            rag_file = RESULTS_DIR / f"RAG_{qid}_rep{rep}.json"
            if not rag_file.exists():
                print(f"  [A - RAG] Executing {qid} rep {rep}...")
                await run_rag_condition(qid, rep=rep, temperature=0.2)
                await asyncio.sleep(1.0)
            else:
                print(f"  [A - RAG] {qid} rep {rep} artifact verified.")

            # Condition B: Graph ReAct
            graph_file = RESULTS_DIR / f"GRAPH_{qid}_rep{rep}.json"
            if not graph_file.exists():
                print(f"  [B - GRAPH] Executing {qid} rep {rep}...")
                await run_graph_condition(qid, rep=rep, temperature=0.2)
                await asyncio.sleep(1.0)
            else:
                print(f"  [B - GRAPH] {qid} rep {rep} artifact verified.")

            # Condition C: Graph-only
            graph_only_file = RESULTS_DIR / f"GRAPH_ONLY_{qid}_rep{rep}.json"
            if not graph_only_file.exists():
                print(f"  [C - GRAPH_ONLY] Executing {qid} rep {rep}...")
                await run_graph_only_condition(qid, rep=rep, temperature=0.2)
                await asyncio.sleep(1.0)
            else:
                print(f"  [C - GRAPH_ONLY] {qid} rep {rep} artifact verified.")

    print("\n==================================================================")
    print("   ALL 45 EXPERIMENTAL RUNS COMPLETE! RUNNING COMPREHENSIVE EVALUATION")
    print("==================================================================\n")

    all_eval_records = []
    for res_file in sorted(RESULTS_DIR.glob("*.json")):
        eval_data = evaluate_result_file(res_file)
        all_eval_records.append(eval_data)

    summary_file = Path(__file__).resolve().parent / "ablation_evaluation_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(all_eval_records, f, indent=2)

    # Prepare Blinded Human Evaluation Dataset
    blinded_dataset = []
    # Deterministic mapping with seed
    random.seed(1337)
    
    # We assign an anonymous hypothesis ID to every hypothesis
    hyp_counter = 1
    for rec in all_eval_records:
        qid = rec["query_id"]
        true_cond = rec["condition"]
        rep = rec["repetition"]

        for h in rec.get("detailed_hypotheses", []):
            blinded_item = {
                "hypothesis_id": f"HYP_{hyp_counter:03d}",
                "query_id": qid,
                "repetition": rep,
                "hypothesis_text": h["hypothesis_text"],
                "causal_chain": [f"{t['from']} -> {t['to']}" for t in h.get("transition_details", [])],
                "evaluation_metrics": {
                    "hop_count": h["hop_count"],
                    "concept_grounding_ratio": h["concept_grounding_ratio"],
                    "edge_alignment_ratio": h["edge_alignment_ratio"],
                    "is_valid_graph_path": h["is_valid_graph_path"],
                    "is_strict_bridge": h["is_strict_bridge"],
                    "is_single_paper_dominated": h["is_single_paper_dominated"],
                    "cumulative_papers_count": h["cumulative_papers_count"],
                    "terminal_co_occurs": h["terminal_co_occurs"]
                },
                "_audit_truth": {
                    "condition": true_cond,
                    "source_file": rec["file"]
                }
            }
            blinded_dataset.append(blinded_item)
            hyp_counter += 1

    # Shuffle for true double-blind presentation
    shuffled_dataset = list(blinded_dataset)
    random.shuffle(shuffled_dataset)

    blinded_file = Path(__file__).resolve().parent / "blinded_human_evaluation_dataset.json"
    with open(blinded_file, "w", encoding="utf-8") as f:
        json.dump(shuffled_dataset, f, indent=2)

    total_time = time.time() - start_suite
    print(f"Summary metrics saved to: {summary_file}")
    print(f"Blinded human evaluation dataset saved to: {blinded_file} ({len(blinded_dataset)} hypotheses)")
    print(f"Entire ablation suite execution and evaluation finished in {total_time:.1f}s.")

if __name__ == "__main__":
    asyncio.run(run_ablation_experiments())
