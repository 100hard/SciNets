import json
import re
from pathlib import Path
from typing import Dict, Any, List, Set, Tuple

BASE_DIR = Path(__file__).resolve().parent
CORPORA_DIR = BASE_DIR / "frozen_corpora"
GRAPHS_DIR = BASE_DIR / "frozen_graphs"
RESULTS_DIR = BASE_DIR / "results"

def normalize_text(text: str) -> str:
    """Normalize string for robust token/concept matching."""
    return re.sub(r'[^a-z0-9]', '', text.lower())

def is_concept_grounded_in_paper(concept: str, paper: dict) -> bool:
    """Check if concept appears in paper title or abstract."""
    norm_c = normalize_text(concept)
    if len(norm_c) < 3:
        return False
    norm_title = normalize_text(paper.get("title", ""))
    norm_abstract = normalize_text(paper.get("abstract", ""))
    return (norm_c in norm_title) or (norm_c in norm_abstract)

def get_concept_corpus_provenance(concept: str, papers: List[dict]) -> Set[str]:
    """Find set of paper IDs where concept appears."""
    return set(p["id"] for p in papers if is_concept_grounded_in_paper(concept, paper=p))

def build_graph_lookup(graph_data: dict) -> Tuple[Dict[str, str], Dict[Tuple[str, str], List[str]]]:
    """
    Builds canonical node index and edge provenance map.
    Returns:
      (node_norm_map, edge_provenance_map)
    """
    raw_nodes = graph_data.get("nodes", [])
    node_norm_map = {normalize_text(n): n for n in raw_nodes}
    
    edge_map = {}
    for e in graph_data.get("edges", []):
        src_norm = normalize_text(e["source"])
        tgt_norm = normalize_text(e["target"])
        pids = e.get("supporting_papers", [])
        if (src_norm, tgt_norm) in edge_map:
            # Merge provenance without duplicates
            for pid in pids:
                if pid not in edge_map[(src_norm, tgt_norm)]:
                    edge_map[(src_norm, tgt_norm)].append(pid)
        else:
            edge_map[(src_norm, tgt_norm)] = list(pids)
        # NOTE: Bidirectional fallback is strictly disallowed. Evaluation graph is strictly directed.

    return node_norm_map, edge_map

def evaluate_single_hypothesis(h: dict, papers: List[dict], graph_data: dict) -> dict:
    """
    Evaluates one hypothesis according to the strict metrics protocol.
    Works for both Graph hypotheses and post-hoc for RAG hypotheses.
    """
    chain = h.get("causal_chain", {})
    nodes = chain.get("nodes", [])
    
    if len(nodes) < 2:
        return {
            "hypothesis_text": h.get("text", "")[:100],
            "hop_count": 0,
            "concept_grounding_ratio": 0.0,
            "edge_alignment_ratio": 0.0,
            "is_valid_graph_path": False,
            "is_strict_bridge": False,
            "is_single_paper_dominated": False,
            "cumulative_papers_count": 0,
            "terminal_co_occurs": False,
            "transition_details": []
        }

    hop_count = len(nodes) - 1
    node_norm_map, edge_map = build_graph_lookup(graph_data)

    # 1. Concept Grounding
    grounded_concepts = 0
    concept_papers_list = []
    for node in nodes:
        prov = get_concept_corpus_provenance(node, papers)
        concept_papers_list.append(prov)
        if prov:
            grounded_concepts += 1

    concept_grounding_ratio = round(grounded_concepts / len(nodes), 3)

    # 2. Transition & Edge Provenance Analysis
    valid_transitions = 0
    all_transition_papers: List[Set[str]] = []
    transition_details = []

    for i in range(len(nodes) - 1):
        u = nodes[i]
        v = nodes[i + 1]
        u_norm = normalize_text(u)
        v_norm = normalize_text(v)

        edge_pids = edge_map.get((u_norm, v_norm), [])
        
        # If not exact normalized match, check if u and v are substrings of graph nodes
        if not edge_pids and len(u_norm) >= 3 and len(v_norm) >= 3:
            cand_u = [k for k in node_norm_map if u_norm in k or k in u_norm]
            cand_v = [k for k in node_norm_map if v_norm in k or k in v_norm]
            for cu in cand_u:
                for cv in cand_v:
                    if (cu, cv) in edge_map:
                        edge_pids = edge_map[(cu, cv)]
                        break
                if edge_pids:
                    break

        has_edge_and_prov = len(edge_pids) > 0
        if has_edge_and_prov:
            valid_transitions += 1
            all_transition_papers.append(set(edge_pids))
        else:
            all_transition_papers.append(set())

        transition_details.append({
            "from": u,
            "to": v,
            "has_graph_edge": has_edge_and_prov,
            "provenance_papers": edge_pids
        })

    edge_alignment_ratio = round(valid_transitions / hop_count, 3)
    is_valid_graph_path = (valid_transitions == hop_count)

    # 3. Provenance Set Logic & Single-Paper Dominance
    cumulative_papers = set().union(*all_transition_papers) if all_transition_papers else set()
    cumulative_count = len(cumulative_papers)

    # Intersection of all transition paper sets
    if all_transition_papers and all(len(s) > 0 for s in all_transition_papers):
        single_paper_intersection = set.intersection(*all_transition_papers)
    else:
        single_paper_intersection = set()

    is_single_paper_dominated = len(single_paper_intersection) > 0

    # 4. Terminal Non-Co-occurrence
    terminal_start_papers = concept_papers_list[0]
    terminal_end_papers = concept_papers_list[-1]
    terminal_co_occurs = len(terminal_start_papers.intersection(terminal_end_papers)) > 0

    # 5. Strict Cross-Paper Bridge Criteria (ALL must be True):
    # - all concepts grounded
    # - every transition is a valid graph edge with non-empty provenance
    # - cumulative papers >= 2
    # - not single-paper dominated (intersection is empty)
    # - terminal concepts never co-occur in any single paper
    # - hop_count >= 2 (at least 3 concepts)
    is_strict_bridge = (
        (concept_grounding_ratio == 1.0) and
        is_valid_graph_path and
        (cumulative_count >= 2) and
        (not is_single_paper_dominated) and
        (not terminal_co_occurs) and
        (hop_count >= 2)
    )

    return {
        "hypothesis_text": h.get("text", "")[:100],
        "hop_count": hop_count,
        "concept_grounding_ratio": concept_grounding_ratio,
        "edge_alignment_ratio": edge_alignment_ratio,
        "is_valid_graph_path": is_valid_graph_path,
        "is_strict_bridge": is_strict_bridge,
        "is_single_paper_dominated": is_single_paper_dominated,
        "cumulative_papers_count": cumulative_count,
        "terminal_co_occurs": terminal_co_occurs,
        "transition_details": transition_details
    }

def evaluate_result_file(result_path: Path) -> dict:
    """Evaluates all hypotheses in a result JSON file."""
    with open(result_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    qid = data["query_id"]
    condition = data["condition"]
    rep = data["repetition"]

    corpus_path = CORPORA_DIR / f"{qid}.json"
    graph_path = GRAPHS_DIR / f"{qid}.json"

    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)
    with open(graph_path, "r", encoding="utf-8") as f:
        graph = json.load(f)

    papers = corpus["papers"]
    graph_data = graph["graph"]

    hyp_evals = []
    for h in data.get("hypotheses", []):
        eval_item = evaluate_single_hypothesis(h, papers, graph_data)
        hyp_evals.append(eval_item)

    # Compute aggregation
    n = len(hyp_evals) or 1
    mean_grounding = sum(x["concept_grounding_ratio"] for x in hyp_evals) / n
    mean_edge_alignment = sum(x["edge_alignment_ratio"] for x in hyp_evals) / n
    valid_path_count = sum(1 for x in hyp_evals if x["is_valid_graph_path"])
    strict_bridge_count = sum(1 for x in hyp_evals if x["is_strict_bridge"])
    single_paper_dominated_count = sum(1 for x in hyp_evals if x["is_single_paper_dominated"])
    mean_hops = sum(x["hop_count"] for x in hyp_evals) / n

    return {
        "file": result_path.name,
        "query_id": qid,
        "condition": condition,
        "repetition": rep,
        "metrics_summary": {
            "hypothesis_count": len(hyp_evals),
            "mean_hops": round(mean_hops, 2),
            "mean_concept_grounding_ratio": round(mean_grounding, 3),
            "mean_edge_alignment_ratio": round(mean_edge_alignment, 3),
            "valid_graph_path_count": valid_path_count,
            "strict_bridge_count": strict_bridge_count,
            "single_paper_dominated_count": single_paper_dominated_count
        },
        "compute_metrics": data.get("metrics", {}),
        "detailed_hypotheses": hyp_evals
    }

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        res = evaluate_result_file(Path(sys.argv[1]))
        print(json.dumps(res, indent=2))
