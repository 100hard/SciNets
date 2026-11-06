import asyncio
import os
import sys
import json
import time
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))
load_dotenv(BASE_DIR / ".env")

from experiments.pilot.queries import PILOT_QUERIES
from experiments.pilot.runner_rag import run_rag_condition
from experiments.pilot.runner_graph import run_graph_condition
from experiments.pilot.evaluate_pilot_metrics import evaluate_result_file

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

REPETITIONS = 3

async def run_full_pilot():
    print("==================================================================")
    print(f"   STARTING FULL PILOT EXPERIMENT: 5 QUERIES x 2 CONDITIONS x {REPETITIONS} REPS")
    print("==================================================================\n")

    start_suite = time.time()
    all_runs = []

    for item in PILOT_QUERIES:
        qid = item["id"]
        qtext = item["query"]
        print(f"\n>>> Running Benchmark Query: {qid} [{item['domain'].upper()}]")
        print(f"    '{qtext[:70]}...'")

        for rep in range(1, REPETITIONS + 1):
            # 1. Condition A: RAG Baseline
            rag_file = RESULTS_DIR / f"RAG_{qid}_rep{rep}.json"
            if rag_file.exists():
                print(f"  [RAG] {qid} rep {rep} already completed. Loading.")
            else:
                print(f"  [RAG] Executing {qid} rep {rep}...")
                await run_rag_condition(qid, rep=rep, temperature=0.2)
                await asyncio.sleep(1.0) # Grace period between API calls

            # 2. Condition B: SciNets Graph
            graph_file = RESULTS_DIR / f"GRAPH_{qid}_rep{rep}.json"
            if graph_file.exists():
                print(f"  [GRAPH] {qid} rep {rep} already completed. Loading.")
            else:
                print(f"  [GRAPH] Executing {qid} rep {rep}...")
                await run_graph_condition(qid, rep=rep, temperature=0.2)
                await asyncio.sleep(1.0) # Grace period between API calls

    print("\n==================================================================")
    print("      ALL 30 GENERATIONS COMPLETE! RUNNING METRIC EVALUATION      ")
    print("==================================================================\n")

    summary_records = []
    for res_file in sorted(RESULTS_DIR.glob("*.json")):
        eval_data = evaluate_result_file(res_file)
        summary_records.append(eval_data)

    summary_file = Path(__file__).resolve().parent / "pilot_evaluation_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_records, f, indent=2)

    print(f"Full pilot evaluation saved to: {summary_file}")
    total_time = time.time() - start_suite
    print(f"Entire pilot suite finished in {total_time:.1f} seconds.")

if __name__ == "__main__":
    asyncio.run(run_full_pilot())
