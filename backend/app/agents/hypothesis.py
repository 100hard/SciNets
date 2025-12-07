from app.state import DiscoveryState, Hypothesis
from app.llm import get_llm
from app.domains import get_domain_packs
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List
import uuid

class HypothesisList(BaseModel):
    hypotheses: List[Hypothesis]

async def hypothesis_node(state: DiscoveryState) -> dict:
    """
    Hypothesis Agent: Generates hypotheses based on the literature and domain.
    Uses an Active Graph Explorer (ReAct) to traverse the concept graph.
    """
    print(f"[Hypothesis] Generating hypotheses...")
    
    # 1. Setup Graph Tools
    import networkx as nx
    from langchain_core.tools import tool
    from langgraph.prebuilt import create_react_agent
    
    # Rebuild graph from state
    G = nx.DiGraph()
    graph_data = state.concept_graph or {"nodes": [], "edges": []}
    for node in graph_data.get("nodes", []):
        G.add_node(node)
    for edge in graph_data.get("edges", []):
        G.add_edge(edge['source'], edge['target'], relation=edge.get('relation', 'related'))

    @tool
    def get_neighbors(node: str) -> str:
        """Get the neighbors of a specific node in the knowledge graph."""
        if node not in G: return f"Node '{node}' not found."
        neighbors = []
        for n in G.neighbors(node):
            rel = G.get_edge_data(node, n).get('relation', 'related')
            neighbors.append(f"{n} (relation: {rel})")
        return f"Neighbors of {node}: " + ", ".join(neighbors[:10])

    @tool
    def find_paths(start_node: str, end_node: str) -> str:
        """Find paths between two nodes in the knowledge graph."""
        if start_node not in G or end_node not in G: return "One or both nodes not found."
        try:
            paths = list(nx.all_simple_paths(G, start_node, end_node, cutoff=3))
            if not paths: return "No paths found."
            return f"Paths from {start_node} to {end_node}:\n" + "\n".join([" -> ".join(p) for p in paths[:3]])
        except:
            return "Error finding paths."

    @tool
    def get_central_nodes() -> str:
        """Get the most central nodes in the graph."""
        try:
            centrality = nx.degree_centrality(G)
            top = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]
            return "Top Central Nodes: " + ", ".join([t[0] for t in top])
        except:
            return "Graph is empty."

    # 2. Create ReAct Explorer Agent
    # Adjust temperature based on Speculation
    temperature = 0.5
    if state.speculation == "low": temperature = 0.1
    elif state.speculation == "medium": temperature = 0.7
    elif state.speculation == "high": temperature = 1.0

    llm = get_llm(temperature=temperature)
    tools = [get_neighbors, find_paths, get_central_nodes]
    explorer_agent = create_react_agent(llm, tools)
    
    lens_instruction = ""
    if state.lens and state.lens != "none":
        lens_instruction = f"IMPORTANT: You must analyze this graph through the lens of '{state.lens}'. Try to map concepts from that field onto this graph."

    exploration_prompt = f"""
    You are a scientific explorer. You have access to a Knowledge Graph about '{state.user_query}'.
    {lens_instruction}
    
    Your Goal: Explore the graph to find NOVEL, non-obvious connections that could lead to a breakthrough hypothesis.
    
    Strategy:
    1. Start by checking the central nodes.
    2. Pick an interesting node and check its neighbors.
    3. Try to find paths between disparate concepts (e.g., a biological mechanism and a disease) OR between the query and the Lens concept ('{state.lens}').
    4. Don't just state facts; look for CAUSAL CHAINS.
    
    After exploring, summarize your findings.
    """
    
    print("[Hypothesis] Starting Graph Exploration...")
    try:
        exploration_result = await explorer_agent.ainvoke({"messages": [("human", exploration_prompt)]})
        exploration_summary = exploration_result["messages"][-1].content
        print(f"[Hypothesis] Exploration Summary: {exploration_summary[:200]}...")
    except Exception as e:
        print(f"[Hypothesis] Exploration failed: {e}")
        exploration_summary = "Exploration failed. Relying on literature summary."

    # 3. Auto-Extract Interesting Paths (The "Big Picture" Feeder)
    print("[Hypothesis] Extracting interesting paths between central nodes...")
    path_context = ""
    try:
        # Get top 5 central nodes
        centrality = nx.degree_centrality(G)
        top_nodes = [n[0] for n in sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]]
        
        found_paths = []
        # Find paths between every pair of top nodes
        import itertools
        for u, v in itertools.combinations(top_nodes, 2):
            try:
                # Find shortest path
                path = nx.shortest_path(G, source=u, target=v)
                # Only keep non-trivial paths (length > 2 nodes)
                if len(path) > 2:
                    # Format path with relations
                    formatted_path = []
                    for i in range(len(path)-1):
                        p1, p2 = path[i], path[i+1]
                        rel = G.get_edge_data(p1, p2).get('relation', 'related')
                        formatted_path.append(f"{p1} --[{rel}]-->")
                    formatted_path.append(path[-1])
                    found_paths.append(" ".join(formatted_path))
            except nx.NetworkXNoPath:
                continue
                
        if found_paths:
            path_context = "Key Multi-Hop Connections found in Graph:\n- " + "\n- ".join(found_paths[:5])
            print(f"[Hypothesis] Found {len(found_paths)} interesting paths.")
        else:
            print("[Hypothesis] No non-trivial paths found between central nodes.")
            
    except Exception as e:
        print(f"[Hypothesis] Path extraction failed: {e}")

    # 3.5 Structural Hole Explorer (The "Novelty Engine")
    # Only runs if goal is 'discover' or speculation is 'high'
    hole_exploration_summary = ""
    if state.goal == "discover" or state.speculation == "high":
        print("[Hypothesis] Running Structural Hole Explorer...")
        try:
            # A. Detect Communities (Clusters)
            # Use greedy_modularity_communities which is fast and good for this scale
            import networkx.algorithms.community as nx_comm
            
            # Convert to undirected for community detection
            G_undirected = G.to_undirected()
            if len(G_undirected.nodes) > 5:
                communities = list(nx_comm.greedy_modularity_communities(G_undirected))
                
                if len(communities) >= 2:
                    # Pick the two largest communities
                    c1 = list(communities[0])[:5] # Top 5 nodes from cluster 1
                    c2 = list(communities[1])[:5] # Top 5 nodes from cluster 2
                    
                    print(f"[Hypothesis] Identified Disconnected Clusters:\nCluster A: {c1}\nCluster B: {c2}")
                    
                    # B. The "Bridge" Prompt (Null Hypothesis Approach)
                    bridge_prompt = f"""
                    I have identified two distinct clusters of knowledge in the graph that seem currently disconnected:
                    
                    Cluster A (Theme 1): {', '.join(c1)}
                    Cluster B (Theme 2): {', '.join(c2)}
                    
                    Your Task: Act as a visionary scientist. 
                    1. Analyze potential "Structural Holes" (missing links) between these two clusters.
                    2. Ask yourself: Is there a plausible underlying mechanism that could connect Cluster A to Cluster B?
                    3. If YES, propose a "Bridge Hypothesis" explaining this mechanism.
                    4. If NO, explicitly state that they are distinct.
                    
                    Use the lens of '{state.lens}' if applicable.
                    """
                    
                    bridge_result = await llm.ainvoke(bridge_prompt)
                    hole_exploration_summary = f"\n\nstructural_hole_analysis:\n{bridge_result.content}"
                    print(f"[Hypothesis] Structural Hole Analysis completed.")
                else:
                    print("[Hypothesis] Not enough communities found for structural hole analysis.")
            else:
                 print("[Hypothesis] Graph too small for community detection.")
                 
        except Exception as e:
            print(f"[Hypothesis] Structural Hole Exploration failed: {e}")

    # 4. Generate Structured Hypotheses (using exploration + path context + hole analysis)
    structured_llm = llm.with_structured_output(HypothesisList)
    final_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a Principal Investigator. Based on the following exploration of the knowledge graph, generate 3 novel, testable scientific hypotheses."),
        ("human", "User Query: {user_query}\n\nExploration Insights:\n{exploration_summary}\n\n{path_context}\n{hole_exploration_summary}\n\nLiterature Context:\n{literature_summary}")
    ])
    
    chain = final_prompt | structured_llm
    
    try:
        result = await chain.ainvoke({
            "user_query": state.user_query,
            "exploration_summary": exploration_summary,
            "path_context": path_context,
            "hole_exploration_summary": hole_exploration_summary,
            "literature_summary": state.literature.get('summary', '')
        })
        hypotheses = result.hypotheses
        
        # Ensure IDs are set
        for h in hypotheses:
            if not h.id:
                h.id = str(uuid.uuid4())
                
    except Exception as e:
        print(f"[Hypothesis] Error generating hypotheses: {e}")
        hypotheses = []

    # Fallback if LLM fails or returns empty
    if not hypotheses:
        hypotheses = [
            Hypothesis(
                id=str(uuid.uuid4()),
                text="LLM generation failed. Please try again.",
                domain_tags=["bio"],
                novelty_score=0.0,
                feasibility_score=0.0,
                testability_score=0.0
            )
        ]
    
    # Select the first one for experimentation
    selected_id = hypotheses[0].id
    
    return {
        "hypotheses": [h.model_dump() for h in hypotheses],
        "selected_hypothesis_id": selected_id
    }
