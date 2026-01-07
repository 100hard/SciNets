import asyncio
import os
import sys
import json
import time
from datetime import datetime
from typing import List, Dict, Any
from dotenv import load_dotenv

# Ensure we can import app
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
load_dotenv()

from app.state import DiscoveryState
from app.agents.literature import literature_node
from app.agents.hypothesis import hypothesis_node
from langchain_core.runnables import RunnableConfig

# === CONFIGURATION ===
QUERIES = [
    {
        "id": "Q1_plasticity_forgetting",
        "query": "Can synaptic plasticity and neural aging mechanisms inform catastrophic forgetting in artificial neural networks?",
        "lens": "neuroscience"
    },
    {
        "id": "Q2_thermo_aging",
        "query": "Can thermodynamic principles explain biological aging as an energy/information degradation process?",
        "lens": "physics"
    },
    {
        "id": "Q3_fasting_cognition",
        "query": "Why do scientific studies disagree on whether intermittent fasting improves cognitive function?",
        "lens": "medicine" 
    },
    {
        "id": "Q4_climate_sensitivity",
        "query": "Why do climate sensitivity models produce significantly different warming predictions despite similar emissions scenarios?",
        "lens": "climate science"
    },
    {
        "id": "Q5_alzheimers_infotheory",
        "query": "What new insights emerge if Alzheimer's disease is analyzed through an information-theoretic perspective?",
        "lens": "information theory"
    },
    {
        "id": "Q6_microbiome_neuro",
        "query": "What new mechanistic explanations could link gut microbiome dysregulation to neurodevelopmental disorders beyond current dominant theories?",
        "lens": "microbiology"
    }
]

OUTPUT_DIR = "discovery_benchmark_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

from langchain_core.runnables import RunnableLambda

async def run_benchmark():
    print(f"Starting Discovery Benchmark at {datetime.now()}")
    print(f"Output Directory: {OUTPUT_DIR}")
    
    summary_md = "# Discovery Benchmark Summary\n\n"
    summary_md += "| ID | Query | Nodes | Hypotheses | Sym Depth | Ground Depth | Drop % | Stance (S/C/N) |\n"
    summary_md += "|---|---|---|---|---|---|---|---|\n"

    for q_idx, q_data in enumerate(QUERIES):
        q_id = q_data["id"]
        
        # CHECK FOR EXISTING RESULT TO AVOID RE-RUNNING
        json_path = os.path.join(OUTPUT_DIR, f"{q_id}.json")
        if os.path.exists(json_path):
            print(f"Skipping {q_id} (Already completed: {json_path})")
            # Create a mock entry in summary_md if needed or just skip
            # For simplicity, we just skip execution. The summary will only contain new runs.
            # Ideally we'd append to existing summary, but for now skipping safely is priority.
            continue

        q_text = q_data["query"]
        q_lens = q_data["lens"]
        
        print(f"\n\n=== RUNNING QUERY {q_idx+1}/{len(QUERIES)}: {q_id} ===")
        print(f"Query: {q_text}")
        print(f"Lens: {q_lens}")
        
        # 1. Initialize State
        state = DiscoveryState(
            user_query=q_text,
            goal="discover",
            mode="hypothesis",
            lens=q_lens,
            speculation="high", # Force high exploration
            max_papers=10,      # As requested
            experiment_id=q_id,
            evaluation_mode=True,
            evaluation_strategy="full"
        )
        
        config = RunnableConfig(configurable={"thread_id": "benchmark_thread"})
        
        try:
            # 2. Run Literature
            print("--- Step 1: Literature & Graph Construction ---")
            # WRAP IN RUNNABLE TO PROVIDE CONTEXT for adispatch_custom_event
            lit_runnable = RunnableLambda(literature_node)
            lit_update = await lit_runnable.ainvoke(state, config)
            
            # Manually update state
            if lit_update:
                state.literature = lit_update.get("literature")
                state.concept_graph = lit_update.get("concept_graph")
            
            # 3. Run Hypothesis (Discovery)
            print("--- Step 2: Discovery & Hypothesis Generation ---")
            hyp_runnable = RunnableLambda(hypothesis_node)
            hyp_update = await hyp_runnable.ainvoke(state, config)
            
            # Manually update state - Critical to get metrics
            if hyp_update:
                state.hypotheses = hyp_update.get("hypotheses", [])
                state.exploration_trace = hyp_update.get("exploration_trace")
                state.structural_hole_analysis = hyp_update.get("structural_hole_analysis")
                state.symbolic_paths = hyp_update.get("symbolic_paths", [])
                state.grounded_paths = hyp_update.get("grounded_paths", [])
                state.stance_counts = hyp_update.get("stance_counts", {})
                state.grounding_metrics = hyp_update.get("grounding_metrics", {})
                state.bridge_attempted = hyp_update.get("bridge_attempted", False)

            # 4. Collect Data
            result_data = {
                "query_id": q_id,
                "timestamp": datetime.now().isoformat(),
                "config": q_data,
                "graph_stats": {
                    "nodes": len(state.concept_graph.get("nodes", [])) if state.concept_graph else 0,
                    "edges": len(state.concept_graph.get("edges", [])) if state.concept_graph else 0,
                },
                "discovery_metrics": {
                    "bridge_attempted": state.bridge_attempted,
                    "diversity_jaccard": calculate_jaccard_diversity(state.symbolic_paths),
                    "stance_counts": state.stance_counts,
                    "grounding": state.grounding_metrics
                },
                "traces": {
                    "exploration": state.exploration_trace,
                    "structural_holes": state.structural_hole_analysis
                },
                "outputs": {
                    "hypotheses": [h.model_dump() for h in state.hypotheses],
                    "symbolic_paths": state.symbolic_paths,
                    "grounded_paths": state.grounded_paths
                }
            }
            
            # 5. Save JSON
            json_path = os.path.join(OUTPUT_DIR, f"{q_id}.json")
            with open(json_path, "w") as f:
                json.dump(result_data, f, indent=2)
            print(f"Saved detailed results to {json_path}")
            
            # 6. Append to Summary
            nodes_count = result_data["graph_stats"]["nodes"]
            num_hyps = len(state.hypotheses)
            sym_depth = state.grounding_metrics.get("symbolic_depth", 0)
            ground_depth = state.grounding_metrics.get("grounded_depth", 0)
            drop_rate = state.grounding_metrics.get("drop_rate", 0)
            stances = f"{state.stance_counts.get('support',0)}/{state.stance_counts.get('contradict',0)}/{state.stance_counts.get('neutral',0)}"
            
            summary_md += f"| {q_id} | {q_text[:30]}... | {nodes_count} | {num_hyps} | {sym_depth} | {ground_depth} | {drop_rate:.2f} | {stances} |\n"
            
        except Exception as e:
            # Write to file to ensure we capture it
            with open("inner_error.txt", "a") as errf:
                errf.write(f"ERROR running query {q_id}: {e}\n\n")
                import traceback
                traceback.print_exc(file=errf)
                
            print(f"ERROR running query {q_id}: {e}", file=sys.stderr, flush=True)
            summary_md += f"| {q_id} | FAILED | - | - | - | - | - | - |\n"
            
            # Debugging serialization
            if "JSON" in str(e) or "serializable" in str(e):
                print("DEBUG: Checking types of collected data:", file=sys.stderr)
                try:
                    print(f"Items in hypotheses: {[type(x) for x in result_data['outputs']['hypotheses']]}", file=sys.stderr)
                    print(f"Grounding metrics: {result_data['discovery_metrics']['grounding']}", file=sys.stderr)
                except:
                    pass

    # Save final summary
    with open(os.path.join(OUTPUT_DIR, "benchmark_summary.md"), "w") as f:
        f.write(summary_md)
    print(f"\n\nBenchmark Complete. Summary saved to {os.path.join(OUTPUT_DIR, 'benchmark_summary.md')}")

def calculate_jaccard_diversity(paths: List[List[str]]) -> float:
    """Calculate average pair-wise Jaccard distance (1 - Jaccard Index) between paths."""
    if not paths or len(paths) < 2:
        return 0.0
    
    total_dist = 0.0
    count = 0
    import itertools
    for p1, p2 in itertools.combinations(paths, 2):
        s1, s2 = set(p1), set(p2)
        union = len(s1 | s2)
        inter = len(s1 & s2)
        if union > 0:
            jaccard = inter / union
            total_dist += (1.0 - jaccard)
            count += 1
            
    if count == 0: return 0.0
    return total_dist / count

if __name__ == "__main__":
    try:
        asyncio.run(run_benchmark())
    except Exception as e:
        import traceback
        with open("crash_report.txt", "w") as f:
            f.write(f"CRITICAL FAILURE:\n{str(e)}\n\n")
            traceback.print_exc(file=f)
        print("CRITICAL FAILURE. See crash_report.txt")
        sys.exit(1)
