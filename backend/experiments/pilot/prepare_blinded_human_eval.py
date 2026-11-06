import json
import random
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CORPORA_DIR = BASE_DIR / "frozen_corpora"
RESULTS_DIR = BASE_DIR / "results"
SUMMARY_FILE = BASE_DIR / "ablation_evaluation_summary.json"

def prepare_blinded_dataset():
    # 1. Load literature contexts for each query (hiding paper IDs)
    literature_by_query = {}
    for cfile in sorted(CORPORA_DIR.glob("*.json")):
        with open(cfile, "r", encoding="utf-8") as f:
            cdata = json.load(f)
        qid = cdata["query_id"]
        papers = cdata["papers"]
        clean_papers = []
        for idx, p in enumerate(papers, 1):
            clean_papers.append({
                "paper_ref": f"Reference Paper {idx}",
                "title": p.get("title", "").strip(),
                "year": p.get("year"),
                "venue": p.get("venue"),
                "abstract": p.get("abstract", "").strip()
            })
        literature_by_query[qid] = clean_papers

    # 2. Load evaluated metrics lookup
    with open(SUMMARY_FILE, "r", encoding="utf-8") as f:
        summary_records = json.load(f)

    metrics_lookup = {}
    for rec in summary_records:
        fname = rec["file"]
        for idx, h in enumerate(rec.get("detailed_hypotheses", [])):
            metrics_lookup[(fname, idx)] = {
                "hop_count": h.get("hop_count"),
                "concept_grounding_ratio": h.get("concept_grounding_ratio"),
                "edge_alignment_ratio": h.get("edge_alignment_ratio"),
                "is_valid_graph_path": h.get("is_valid_graph_path"),
                "is_strict_bridge": h.get("is_strict_bridge"),
                "is_single_paper_dominated": h.get("is_single_paper_dominated"),
                "cumulative_papers_count": h.get("cumulative_papers_count"),
                "terminal_co_occurs": h.get("terminal_co_occurs"),
                "transition_details": h.get("transition_details", [])
            }

    raw_files = sorted(RESULTS_DIR.glob("*.json"))
    raw_hypotheses = []

    for fpath in raw_files:
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        qid = data["query_id"]
        qtext = data["query"]
        cond = data["condition"]
        rep = data["repetition"]
        fname = fpath.name

        for idx, h in enumerate(data.get("hypotheses", [])):
            chain_obj = h.get("causal_chain", {})
            nodes = chain_obj.get("nodes", [])
            rels = chain_obj.get("relations", [])
            
            chain_steps = []
            for i in range(len(nodes) - 1):
                u = nodes[i]
                v = nodes[i + 1]
                rel = rels[i] if i < len(rels) else "leads to"
                chain_steps.append(f"{u} --[{rel}]--> {v}")

            item = {
                "source_file": fname,
                "hyp_index": idx,
                "condition": cond,
                "query_id": qid,
                "query": qtext,
                "repetition": rep,
                "hypothesis_text": h.get("text", "").strip(),
                "causal_chain_nodes": nodes,
                "causal_chain_formatted": " -> ".join(nodes) if nodes else "N/A",
                "causal_chain_steps": chain_steps,
                "automated_metrics": metrics_lookup.get((fname, idx), {})
            }
            raw_hypotheses.append(item)

    assert len(raw_hypotheses) == 135, f"Expected 135 hypotheses, found {len(raw_hypotheses)}"

    # Deterministic randomization
    random.seed(42)
    random.shuffle(raw_hypotheses)

    blinded_tasks = []
    unblind_key = {}

    for idx, item in enumerate(raw_hypotheses, 1):
        task_id = f"TASK_{idx:03d}"
        qid = item["query_id"]

        # Strictly evaluator-facing: NO condition, NO metrics, NO provenance, NO real paper IDs
        evaluator_task = {
            "task_id": task_id,
            "research_question": item["query"],
            "hypothesis_text": item["hypothesis_text"],
            "causal_chain": item["causal_chain_formatted"],
            "mechanistic_steps": item["causal_chain_steps"],
            "supplied_literature": literature_by_query[qid],
            "evaluations": {
                "evaluator_1": {
                    "scientific_plausibility": None,      # 1-5
                    "mechanistic_coherence": None,        # 1-5
                    "novelty_relative_to_literature": None,# 1-5
                    "investigative_value": None,          # 1-5
                    "comments": {
                        "scientific_plausibility_rationale": "",
                        "mechanistic_coherence_rationale": "",
                        "novelty_rationale": "",
                        "investigative_value_rationale": ""
                    }
                },
                "evaluator_2": {
                    "scientific_plausibility": None,      # 1-5
                    "mechanistic_coherence": None,        # 1-5
                    "novelty_relative_to_literature": None,# 1-5
                    "investigative_value": None,          # 1-5
                    "comments": {
                        "scientific_plausibility_rationale": "",
                        "mechanistic_coherence_rationale": "",
                        "novelty_rationale": "",
                        "investigative_value_rationale": ""
                    }
                }
            }
        }
        blinded_tasks.append(evaluator_task)

        # Ground truth unblind key
        unblind_key[task_id] = {
            "condition": item["condition"],
            "query_id": item["query_id"],
            "repetition": item["repetition"],
            "source_file": item["source_file"],
            "automated_metrics": item["automated_metrics"]
        }

    # Save evaluator dataset
    eval_file = BASE_DIR / "blinded_evaluation_tasks.json"
    with open(eval_file, "w", encoding="utf-8") as f:
        json.dump(blinded_tasks, f, indent=2)

    # Save unblind key
    key_file = BASE_DIR / "blinded_evaluation_key.json"
    with open(key_file, "w", encoding="utf-8") as f:
        json.dump(unblind_key, f, indent=2)

    # Save evaluator profiles template
    profiles = {
        "evaluator_1": {
            "evaluator_id": "EVALUATOR_1",
            "domain_background": "",
            "highest_degree": "",
            "research_experience_years": 0,
            "qualified_domains": ["Machine Learning", "Biology / Medicine", "Climate / Environmental Science"]
        },
        "evaluator_2": {
            "evaluator_id": "EVALUATOR_2",
            "domain_background": "",
            "highest_degree": "",
            "research_experience_years": 0,
            "qualified_domains": ["Machine Learning", "Biology / Medicine", "Climate / Environmental Science"]
        }
    }
    profiles_file = BASE_DIR / "evaluator_profiles.json"
    with open(profiles_file, "w", encoding="utf-8") as f:
        json.dump(profiles, f, indent=2)

    print(f"Generated {len(blinded_tasks)} blinded evaluation tasks in: {eval_file}")
    print(f"Generated ground truth unblind key in: {key_file}")
    print(f"Generated evaluator profiles template in: {profiles_file}")

    return blinded_tasks[:5]

if __name__ == "__main__":
    prepare_blinded_dataset()
