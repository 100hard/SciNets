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

from app.config import config
# Ensure model is gpt-4o-mini
config.LLM_MODEL = "gpt-4o-mini"
config.CHEAP_LLM_MODEL = "gpt-4o-mini"

from app.agents.literature import (
    ConceptGraph, 
    normalize_graph_nodes, 
    densify_graph, 
    clean_graph
)
from app.llm import get_cheap_llm
from app.telemetry.cost_tracker import CostTracker
from langchain_core.prompts import ChatPromptTemplate
from experiments.pilot.queries import PILOT_QUERIES

CORPORA_DIR = Path(__file__).resolve().parent / "frozen_corpora"
GRAPHS_DIR = Path(__file__).resolve().parent / "frozen_graphs"
GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

async def build_and_freeze_graph(item: dict, force: bool = False) -> dict:
    qid = item["id"]
    query = item["query"]
    corpus_file = CORPORA_DIR / f"{qid}.json"
    graph_file = GRAPHS_DIR / f"{qid}.json"

    if graph_file.exists() and not force:
        with open(graph_file, "r", encoding="utf-8") as f:
            existing = json.load(f)
        if existing.get("graph", {}).get("nodes"):
            print(f"[{qid}] Graph already frozen ({len(existing['graph']['nodes'])} nodes, {len(existing['graph']['edges'])} edges). Skipping.")
            return existing

    with open(corpus_file, "r", encoding="utf-8") as f:
        corpus_data = json.load(f)

    papers = corpus_data["papers"]
    print(f"[{qid}] Constructing graph from {len(papers)} frozen papers...")

    start_time = time.time()
    tracker = CostTracker()
    CostTracker.register_context(tracker)

    # 1. Batch Extraction
    llm = get_cheap_llm(temperature=0.0)
    structured_llm = llm.with_structured_output(ConceptGraph)

    system_msg = """You are a scientific detective. Extract NOVEL, NON-OBVIOUS, and CONTRADICTING concepts from the collection of abstracts. Focus on the periphery and edge cases.

    CRITICAL RULES:
    1. NODES MUST BE NOUN PHRASES (e.g., "Dopamine", "Synaptic Plasticity").
    2. DO NOT create nodes that are full sentences, titles, or verb phrases (e.g., "Running increases neurogenesis" is BANNED).
    3. DO NOT HALLUCINATE: Only extract concepts explicitly appearing in the abstracts.
    4. PROVENANCE: For each edge, populate supporting_papers with the exact Paper ID(s) (e.g., from 'Paper ID: ...') that explicitly support this relation.
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_msg),
        ("human", "Here are the abstracts:\n\n{abstracts}")
    ])
    chain = prompt | structured_llm

    batch_size = 3
    batches = [papers[i:i + batch_size] for i in range(0, len(papers), batch_size)]

    merged_nodes = set()
    merged_edges = []

    # Map valid corpus IDs: both full URI and short ID (e.g. W2991497298)
    corpus_id_map = {}
    for p in papers:
        full_id = p["id"].strip()
        corpus_id_map[full_id] = full_id
        short_id = full_id.split("/")[-1].strip()
        corpus_id_map[short_id] = full_id
        corpus_id_map[full_id.lower()] = full_id
        corpus_id_map[short_id.lower()] = full_id

    async def process_batch(batch_papers):
        batch_abstracts = []
        for p in batch_papers:
            batch_abstracts.append(f"Paper ID: {p['id']}\nTitle: {p['title']}\nAbstract: {p['abstract']}\n")
        batch_feed = "\n---\n".join(batch_abstracts)
        try:
            return await chain.ainvoke({"abstracts": batch_feed})
        except Exception as e:
            print(f"[{qid}] Batch extraction error: {e}")
            return None

    batch_results = await asyncio.gather(*[process_batch(b) for b in batches], return_exceptions=True)

    raw_edges_extracted = 0
    raw_edges_with_prov = 0
    raw_edges_missing_prov = 0
    unmapped_dropped_ids = 0

    for i, res in enumerate(batch_results):
        if res and not isinstance(res, Exception):
            merged_nodes.update(res.nodes)
            for e in res.edges:
                raw_edges_extracted += 1
                cleaned_pids = []
                if e.supporting_papers:
                    for raw_pid in e.supporting_papers:
                        if not isinstance(raw_pid, str):
                            continue
                        clean_raw = raw_pid.strip()
                        if clean_raw in corpus_id_map:
                            canonical = corpus_id_map[clean_raw]
                            if canonical not in cleaned_pids:
                                cleaned_pids.append(canonical)
                        else:
                            unmapped_dropped_ids += 1
                
                # CRITICAL: If no genuine provenance was identified, leave supporting_papers empty.
                # NEVER fallback to batch paper IDs or arbitrary papers.
                e.supporting_papers = cleaned_pids
                if e.supporting_papers:
                    raw_edges_with_prov += 1
                else:
                    raw_edges_missing_prov += 1
                merged_edges.append(e)

    initial_graph = {
        "nodes": list(merged_nodes),
        "edges": [e.model_dump() for e in merged_edges]
    }
    print(f"[{qid}] Raw extracted graph: {len(initial_graph['nodes'])} nodes, {len(initial_graph['edges'])} edges ({raw_edges_with_prov} with genuine prov, {raw_edges_missing_prov} missing prov, {unmapped_dropped_ids} unmapped IDs dropped)")

    # 2. Normalize graph (Entity resolution)
    normalized_graph = await normalize_graph_nodes(initial_graph)

    # 3. Densify graph (Second pass)
    densified_graph = await densify_graph(normalized_graph, papers)

    # 4. Clean graph
    final_graph = clean_graph(densified_graph)

    # Post-validation: Ensure every supporting_papers entry in final_graph is in corpus_id_map
    for e in final_graph.get("edges", []):
        verified_pids = []
        for pid in e.get("supporting_papers", []):
            if pid in corpus_id_map:
                canonical = corpus_id_map[pid]
                if canonical not in verified_pids:
                    verified_pids.append(canonical)
            else:
                unmapped_dropped_ids += 1
        e["supporting_papers"] = verified_pids

    final_edges_with_prov = sum(1 for e in final_graph["edges"] if len(e.get("supporting_papers", [])) > 0)
    final_edges_missing_prov = sum(1 for e in final_graph["edges"] if len(e.get("supporting_papers", [])) == 0)

    duration = time.time() - start_time
    report = tracker.get_report()

    graph_payload = {
        "query_id": qid,
        "domain": item["domain"],
        "query": query,
        "graph": final_graph,
        "construction_metrics": {
            "duration_seconds": round(duration, 2),
            "total_tokens": report["total_tokens"],
            "estimated_cost_usd": report["estimated_cost_usd"],
            "raw_edges_extracted": raw_edges_extracted,
            "raw_edges_with_prov": raw_edges_with_prov,
            "raw_edges_missing_prov": raw_edges_missing_prov,
            "node_count": len(final_graph["nodes"]),
            "edge_count": len(final_graph["edges"]),
            "edges_with_provenance": final_edges_with_prov,
            "edges_missing_provenance": final_edges_missing_prov,
            "unmapped_dropped_ids": unmapped_dropped_ids
        }
    }

    with open(graph_file, "w", encoding="utf-8") as f:
        json.dump(graph_payload, f, indent=2)

    print(f"[{qid}] Frozen graph saved:")
    print(f"  - Nodes: {len(final_graph['nodes'])}")
    print(f"  - Raw extracted edges: {raw_edges_extracted}")
    print(f"  - Final edges: {len(final_graph['edges'])}")
    print(f"  - Edges with genuine provenance: {final_edges_with_prov}")
    print(f"  - Edges with missing provenance: {final_edges_missing_prov}")
    print(f"  - Unmapped/dropped IDs: {unmapped_dropped_ids}")
    return graph_payload

async def main():
    force = "--force" in sys.argv or "-f" in sys.argv
    print(f"=== STEP 2: CONSTRUCTING & FREEZING GRAPHS ({len(PILOT_QUERIES)} queries, force={force}) ===")
    for item in PILOT_QUERIES:
        await build_and_freeze_graph(item, force=force)
    print("=== STEP 2 COMPLETE: All graphs frozen successfully. ===")

if __name__ == "__main__":
    asyncio.run(main())
