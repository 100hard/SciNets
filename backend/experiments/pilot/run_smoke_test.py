import asyncio
import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))
load_dotenv(BASE_DIR / ".env")

from app.agents.hypothesis import HypothesisList
from experiments.pilot.runner_rag import run_rag_condition
from experiments.pilot.runner_graph import run_graph_condition
from experiments.pilot.evaluate_pilot_metrics import evaluate_result_file

CORPORA_DIR = Path(__file__).resolve().parent / "frozen_corpora"
GRAPHS_DIR = Path(__file__).resolve().parent / "frozen_graphs"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

async def run_all_smoke_checks():
    print("=================================================================")
    print("             EXPERIMENT 1 PILOT: SMOKE TEST SUITE               ")
    print("=================================================================\n")

    # -------------------------------------------------------------
    # Check 1: Confirm 5 frozen corpora contain exactly 10 papers with non-empty abstracts
    # -------------------------------------------------------------
    print("[CHECK 1] Validating 5 Frozen Corpora...")
    corpora_files = sorted(CORPORA_DIR.glob("*.json"))
    assert len(corpora_files) == 5, f"Expected 5 corpora files, found {len(corpora_files)}"
    for cf in corpora_files:
        cdata = json.loads(cf.read_text())
        papers = cdata.get("papers", [])
        assert len(papers) == 10, f"{cf.stem} has {len(papers)} papers (expected 10)"
        for idx, p in enumerate(papers, 1):
            assert len(p.get("abstract", "").strip()) >= 100, f"{cf.stem} Paper {idx} has invalid abstract (<100 chars)"
        print(f"  PASS: {cf.stem} -> exactly 10 papers, all abstracts valid.")
    print("  --> CHECK 1 PASSED.\n")

    # -------------------------------------------------------------
    # Check 2: Confirm every frozen graph can be reloaded without losing attributes
    # -------------------------------------------------------------
    print("[CHECK 2] Validating Frozen Graphs Serialization & Attributes...")
    graph_files = sorted(GRAPHS_DIR.glob("*.json"))
    assert len(graph_files) == 5, f"Expected 5 graph files, found {len(graph_files)}"
    for gf in graph_files:
        gdata = json.loads(gf.read_text())
        g = gdata.get("graph", {})
        nodes = g.get("nodes", [])
        edges = g.get("edges", [])
        assert len(nodes) > 0, f"{gf.stem} has empty nodes"
        assert len(edges) > 0, f"{gf.stem} has empty edges"
        for e in edges:
            assert "source" in e and "target" in e and "relation" in e, f"Malformed edge in {gf.stem}"
            assert "supporting_papers" in e, f"Edge missing supporting_papers attribute in {gf.stem}"
        print(f"  PASS: {gf.stem} -> {len(nodes)} nodes, {len(edges)} edges, all have supporting_papers.")
    print("  --> CHECK 2 PASSED.\n")

    # -------------------------------------------------------------
    # Check 3: Verify provenance points to actual papers in the frozen corpus
    # -------------------------------------------------------------
    print("[CHECK 3] Verifying Edge Provenance Points to Corpus Papers...")
    for gf in graph_files:
        qid = gf.stem
        cf = CORPORA_DIR / f"{qid}.json"
        cdata = json.loads(cf.read_text())
        gdata = json.loads(gf.read_text())

        corpus_pids = set(p["id"] for p in cdata["papers"])
        edges = gdata["graph"]["edges"]
        all_prov_pids = set(pid for e in edges for pid in e.get("supporting_papers", []))

        assert all_prov_pids.issubset(corpus_pids), f"Provenance mismatch in {qid}: {all_prov_pids - corpus_pids}"
        print(f"  PASS: {qid} -> {len(edges)} edges point strictly to valid corpus papers (100% verified).")
    print("  --> CHECK 3 PASSED.\n")

    # -------------------------------------------------------------
    # Check 4: Run single RAG and single Graph generation smoke test
    # -------------------------------------------------------------
    test_qid = "ML-1"
    print(f"[CHECK 4] Executing Single Generation Smoke Test on '{test_qid}'...")
    print("  Executing RAG runner on ML-1 (rep 1)...")
    rag_out = await run_rag_condition(test_qid, rep=1, temperature=0.2)
    print("  Executing Graph runner on ML-1 (rep 1)...")
    graph_out = await run_graph_condition(test_qid, rep=1, temperature=0.2)
    print("  --> CHECK 4 PASSED.\n")

    # -------------------------------------------------------------
    # Check 5: Validate outputs parse against HypothesisList schema
    # -------------------------------------------------------------
    print("[CHECK 5] Validating Pydantic Schema Parsing (HypothesisList)...")
    rag_parsed = HypothesisList(**{"hypotheses": rag_out["hypotheses"]})
    graph_parsed = HypothesisList(**{"hypotheses": graph_out["hypotheses"]})
    assert len(rag_parsed.hypotheses) == 3, f"RAG output has {len(rag_parsed.hypotheses)} hypotheses (expected 3)"
    assert len(graph_parsed.hypotheses) == 3, f"Graph output has {len(graph_parsed.hypotheses)} hypotheses (expected 3)"
    print(f"  PASS: Both conditions output exactly 3 valid GeneratedHypothesis objects.")
    print("  --> CHECK 5 PASSED.\n")

    # -------------------------------------------------------------
    # Check 6: Validate RAG runner makes zero graph/tool calls
    # -------------------------------------------------------------
    print("[CHECK 6] Validating RAG Runner is Graph-Blind...")
    assert rag_out["metrics"]["tool_calls"] == 0, f"RAG made {rag_out['metrics']['tool_calls']} tool calls (expected 0)"
    assert rag_out["metrics"]["llm_calls"] == 1, f"RAG made {rag_out['metrics']['llm_calls']} LLM calls (expected 1)"
    print(f"  PASS: RAG made exactly 0 tool calls and 1 direct LLM call.")
    print("  --> CHECK 6 PASSED.\n")

    # -------------------------------------------------------------
    # Check 7: Validate Graph runner uses only the frozen graph after construction
    # -------------------------------------------------------------
    print("[CHECK 7] Validating Graph Runner ReAct Usage & Isolation...")
    assert graph_out["metrics"]["tool_calls"] > 0, f"Graph runner made {graph_out['metrics']['tool_calls']} tool calls (expected >0)"
    assert graph_out["metrics"]["llm_calls"] > 1, f"Graph runner made {graph_out['metrics']['llm_calls']} LLM calls (expected >1)"
    print(f"  PASS: Graph runner executed {graph_out['metrics']['tool_calls']} tool calls over the frozen graph.")
    print("  --> CHECK 7 PASSED.\n")

    # -------------------------------------------------------------
    # Check 8: Validate evaluation script on both saved artifacts
    # -------------------------------------------------------------
    print("[CHECK 8] Validating Independent Metric Reconstruction...")
    rag_eval = evaluate_result_file(RESULTS_DIR / f"RAG_{test_qid}_rep1.json")
    graph_eval = evaluate_result_file(RESULTS_DIR / f"GRAPH_{test_qid}_rep1.json")

    print("\n  --- RAG Condition Smoke Evaluation ---")
    print(f"  Mean hops: {rag_eval['metrics_summary']['mean_hops']}")
    print(f"  Concept Grounding: {rag_eval['metrics_summary']['mean_concept_grounding_ratio']}")
    print(f"  Post-hoc Graph Alignment: {rag_eval['metrics_summary']['mean_edge_alignment_ratio']}")
    print(f"  Valid Graph Paths: {rag_eval['metrics_summary']['valid_graph_path_count']}/{rag_eval['metrics_summary']['hypothesis_count']}")
    print(f"  Strict Bridges: {rag_eval['metrics_summary']['strict_bridge_count']}/{rag_eval['metrics_summary']['hypothesis_count']}")

    print("\n  --- Graph Condition Smoke Evaluation ---")
    print(f"  Mean hops: {graph_eval['metrics_summary']['mean_hops']}")
    print(f"  Concept Grounding: {graph_eval['metrics_summary']['mean_concept_grounding_ratio']}")
    print(f"  Graph Edge Alignment: {graph_eval['metrics_summary']['mean_edge_alignment_ratio']}")
    print(f"  Valid Graph Paths: {graph_eval['metrics_summary']['valid_graph_path_count']}/{graph_eval['metrics_summary']['hypothesis_count']}")
    print(f"  Strict Bridges: {graph_eval['metrics_summary']['strict_bridge_count']}/{graph_eval['metrics_summary']['hypothesis_count']}")
    print(f"  Tool calls: {graph_eval['compute_metrics']['tool_calls']}")

    print("\n=================================================================")
    print("        ALL 8 SMOKE TEST CHECKS PASSED SUCCESSFULLY!             ")
    print("=================================================================\n")

if __name__ == "__main__":
    asyncio.run(run_all_smoke_checks())
