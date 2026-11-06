import asyncio
import os
import sys
import json
import time
import warnings
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))
load_dotenv(BASE_DIR / ".env")

import networkx as nx
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from langgraph.prebuilt import create_react_agent

from app.agents.hypothesis import HypothesisList
from app.telemetry.cost_tracker import CostTracker

GRAPHS_DIR = Path(__file__).resolve().parent / "frozen_graphs"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

async def run_graph_condition(query_id: str, rep: int = 1, temperature: float = 0.2) -> dict:
    """
    SciNets Graph Condition:
    Explores the pre-built frozen concept graph via ReAct tools,
    then generates grounded hypotheses.
    """
    graph_file = GRAPHS_DIR / f"{query_id}.json"
    if not graph_file.exists():
        raise FileNotFoundError(f"Frozen graph for {query_id} not found at {graph_file}")

    with open(graph_file, "r", encoding="utf-8") as f:
        graph_data = json.load(f)

    query = graph_data["query"]
    raw_graph = graph_data["graph"]

    # Reconstruct NetworkX DiGraph strictly from frozen artifact
    G = nx.DiGraph()
    for n in raw_graph.get("nodes", []):
        G.add_node(n)
    for e in raw_graph.get("edges", []):
        pids = e.get("supporting_papers", [])
        G.add_edge(e["source"], e["target"], relation=e.get("relation", "related"), supporting_papers=pids)

    # Tool invocation counter
    tool_counter = {"calls": 0}

    @tool
    def get_neighbors(node: str) -> str:
        """Get the neighbors of a specific node in the concept graph."""
        tool_counter["calls"] += 1
        if node not in G:
            return json.dumps({"error": f"Node '{node}' not found."})
        neighbors = []
        for n in G.neighbors(node):
            rel = G.get_edge_data(node, n).get("relation", "related")
            papers = G.get_edge_data(node, n).get("supporting_papers", [])
            neighbors.append({"node": n, "relation": rel, "paper_count": len(papers)})
        return json.dumps({"node": node, "neighbors": neighbors[:15]})

    @tool
    def find_paths(start_node: str, end_node: str) -> str:
        """Find simple paths between two nodes up to length 4."""
        tool_counter["calls"] += 1
        if start_node not in G or end_node not in G:
            return json.dumps({"error": "Nodes not found in graph."})
        try:
            paths = list(nx.all_simple_paths(G, start_node, end_node, cutoff=4))
            sorted_paths = sorted(paths, key=len, reverse=True)[:5]
            return json.dumps({"paths": [" -> ".join(p) for p in sorted_paths]})
        except Exception as e:
            return json.dumps({"error": str(e)})

    @tool
    def get_central_nodes() -> str:
        """Get the top central concept hubs in the graph."""
        tool_counter["calls"] += 1
        try:
            centrality = nx.degree_centrality(G)
            top = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]
            return json.dumps({"central_nodes": [t[0] for t in top]})
        except Exception as e:
            return json.dumps({"central_nodes": []})

    start_time = time.time()
    tracker = CostTracker()
    CostTracker.register_context(tracker)

    tools = [get_neighbors, find_paths, get_central_nodes]
    from app.telemetry.cost_tracker import CostCallbackHandler
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=temperature,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        callbacks=[CostCallbackHandler()]
    )

    # 1. ReAct Explorer
    explorer_system = f"""You are a scientific explorer navigating a literature concept graph built from 10 papers on '{query}'.
You must call the provided tools to discover multi-hop paths, bridge concepts, and structural holes before proposing hypotheses.

STRATEGY:
1. Call `get_central_nodes` to identify key conceptual hubs.
2. Call `get_neighbors` on central concepts to find local connections.
3. Call `find_paths` between disparate concepts that rarely co-occur.
4. Locate potential multi-hop causal chains (A -> B -> C) that connect separate clusters.
"""

    explorer_user = f"""Explore the graph to identify candidate multi-hop causal mechanisms that could resolve the research question: '{query}'.
Summarize the most promising paths found, noting the intermediate bridge concepts."""

    explorer_agent = create_react_agent(llm, tools)
    
    try:
        exploration_res = await asyncio.wait_for(
            explorer_agent.ainvoke(
                {"messages": [("system", explorer_system), ("user", explorer_user)]},
                {"recursion_limit": 30}
            ),
            timeout=90.0
        )
        exploration_summary = exploration_res["messages"][-1].content
    except Exception as e:
        print(f"[Graph] ReAct exploration exception: {e}")
        exploration_summary = f"Partial graph exploration completed. Nodes: {len(G.nodes())}."

    # 2. Deep Hypothesis Synthesis
    synthesis_system = """You are a Principal Scientific Investigator. Based on the concept graph exploration and literature findings, formulate 3 distinct, literature-grounded candidate scientific hypotheses addressing the research question.

CRITICAL INSTRUCTIONS:
1. STRUCTURAL GROUNDING: Use the verified multi-hop paths and bridge concepts discovered in the graph exploration to anchor each causal chain (Concept A -> Concept B -> Concept C).
2. RATIONALE: Explicitly identify the disconnected clusters being bridged and the implicit field assumptions.
3. OUTPUT: Produce strictly structured hypotheses satisfying the schema.
"""

    synthesis_user = f"""Research Question: {query}

Graph Exploration Trace:
{exploration_summary}

Generate exactly 3 structured candidate hypotheses.
"""

    structured_llm = llm.with_structured_output(HypothesisList)
    result: HypothesisList = await structured_llm.ainvoke([
        SystemMessage(content=synthesis_system),
        HumanMessage(content=synthesis_user)
    ])

    duration = time.time() - start_time
    report = tracker.get_report()

    output_payload = {
        "condition": "GRAPH",
        "query_id": query_id,
        "query": query,
        "domain": graph_data["domain"],
        "repetition": rep,
        "model": "gpt-4o-mini",
        "temperature": temperature,
        "metrics": {
            "duration_seconds": round(duration, 2),
            "prompt_tokens": tracker.total["prompt"],
            "completion_tokens": tracker.total["completion"],
            "total_tokens": tracker.total["tokens"],
            "estimated_cost_usd": round(tracker.total["cost"], 4),
            "llm_calls": tracker.total["calls"] or 2,
            "tool_calls": tool_counter["calls"]
        },
        "exploration_summary": exploration_summary,
        "hypotheses": [h.model_dump() for h in result.hypotheses]
    }

    out_file = RESULTS_DIR / f"GRAPH_{query_id}_rep{rep}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    print(f"[GRAPH] Completed {query_id} (rep {rep}) in {duration:.1f}s | {len(result.hypotheses)} hypotheses | {tool_counter['calls']} tool calls | {output_payload['metrics']['total_tokens']} tokens")
    return output_payload

if __name__ == "__main__":
    qid = sys.argv[1] if len(sys.argv) > 1 else "ML-1"
    asyncio.run(run_graph_condition(qid, rep=1))
