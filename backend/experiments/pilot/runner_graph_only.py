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

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from app.agents.hypothesis import HypothesisList
from app.telemetry.cost_tracker import CostTracker

CORPORA_DIR = Path(__file__).resolve().parent / "frozen_corpora"
GRAPHS_DIR = Path(__file__).resolve().parent / "frozen_graphs"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

async def run_graph_only_condition(query_id: str, rep: int = 1, temperature: float = 0.2) -> dict:
    """
    Condition C: Graph-only (Static Graph Representation)
    Receives both the frozen paper abstracts AND the full static concept-relation graph.
    Executes exactly 1 direct LLM call with zero iterative tool exploration.
    Isolates the effect of explicit graph representation from ReAct multi-step inference.
    """
    corpus_file = CORPORA_DIR / f"{query_id}.json"
    graph_file = GRAPHS_DIR / f"{query_id}.json"

    if not corpus_file.exists():
        raise FileNotFoundError(f"Frozen corpus for {query_id} not found at {corpus_file}")
    if not graph_file.exists():
        raise FileNotFoundError(f"Frozen graph for {query_id} not found at {graph_file}")

    with open(corpus_file, "r", encoding="utf-8") as f:
        corpus = json.load(f)
    with open(graph_file, "r", encoding="utf-8") as f:
        graph_data = json.load(f)

    query = corpus["query"]
    papers = corpus["papers"]
    raw_graph = graph_data["graph"]

    # 1. Format paper abstracts
    abstract_blocks = []
    for idx, p in enumerate(papers, 1):
        block = f"[Paper {idx}] Title: {p['title']}\nYear: {p['year']} | Venue: {p['venue']}\nAbstract: {p['abstract']}\n"
        abstract_blocks.append(block)
    formatted_abstracts = "\n---\n".join(abstract_blocks)

    # 2. Format static concept graph representation
    nodes = raw_graph.get("nodes", [])
    edges = raw_graph.get("edges", [])
    nodes_text = ", ".join(nodes)

    edge_lines = []
    for e in edges:
        edge_lines.append(f"- {e['source']} --[{e.get('relation', 'related')}]--> {e['target']}")
    edges_text = "\n".join(edge_lines)

    system_prompt = """You are a Principal Scientific Investigator. Based on the provided research papers and the explicit concept-relation graph constructed from them, synthesize the evidence to formulate 3 distinct, literature-grounded candidate scientific hypotheses addressing the research question.

CRITICAL INSTRUCTIONS:
1. GROUNDING: Every concept in your causal chain should be grounded in the provided concept graph and research literature.
2. CAUSAL CHAIN: For each hypothesis, formulate a rigorous multi-hop causal chain: Concept A -> Concept B -> Concept C. Use the verified relationships from the concept graph to anchor the mechanistic steps.
3. RATIONALE: State the epistemic tension or literature gap that this hypothesis bridges, specifying what beliefs or findings in the literature it reconciles.
4. INDEPENDENCE: You do not have access to interactive search or graph exploration tools. Synthesize hypotheses directly from the provided static graph representation and literature text.
"""

    user_prompt = f"""Research Question: {query}

Concept Graph Representation:
Key Concepts ({len(nodes)} nodes):
{nodes_text}

Verified Directed Relations ({len(edges)} edges):
{edges_text}

Research Literature (10 Papers):
---
{formatted_abstracts}
---

Generate exactly 3 structured candidate hypotheses.
"""

    start_time = time.time()
    tracker = CostTracker()
    CostTracker.register_context(tracker)

    from app.telemetry.cost_tracker import CostCallbackHandler
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=temperature,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        callbacks=[CostCallbackHandler()]
    )
    structured_llm = llm.with_structured_output(HypothesisList)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]

    result: HypothesisList = await structured_llm.ainvoke(messages)
    duration = time.time() - start_time
    report = tracker.get_report()

    output_payload = {
        "condition": "GRAPH_ONLY",
        "query_id": query_id,
        "query": query,
        "domain": corpus["domain"],
        "repetition": rep,
        "model": "gpt-4o-mini",
        "temperature": temperature,
        "metrics": {
            "duration_seconds": round(duration, 2),
            "prompt_tokens": tracker.total["prompt"],
            "completion_tokens": tracker.total["completion"],
            "total_tokens": tracker.total["tokens"],
            "estimated_cost_usd": round(tracker.total["cost"], 4),
            "llm_calls": 1,
            "tool_calls": 0
        },
        "hypotheses": [h.model_dump() for h in result.hypotheses]
    }

    out_file = RESULTS_DIR / f"GRAPH_ONLY_{query_id}_rep{rep}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    print(f"[GRAPH_ONLY] Completed {query_id} (rep {rep}) in {duration:.1f}s | {len(result.hypotheses)} hypotheses | {output_payload['metrics']['total_tokens']} tokens")
    return output_payload

if __name__ == "__main__":
    qid = sys.argv[1] if len(sys.argv) > 1 else "ML-1"
    asyncio.run(run_graph_only_condition(qid, rep=1))
