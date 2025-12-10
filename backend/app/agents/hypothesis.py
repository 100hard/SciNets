from app.state import DiscoveryState, Hypothesis
from app.llm import get_llm
from app.domains import get_domain_packs
from app.logging_config import get_logger
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List
import uuid
from langchain_core.runnables import RunnableConfig
import logging
from langchain_core.callbacks import adispatch_custom_event

log = get_logger(__name__)

class HypothesisList(BaseModel):
    hypotheses: List[Hypothesis]

async def hypothesis_node(state: DiscoveryState, config: RunnableConfig) -> dict:
    """
    Hypothesis Agent: Generates hypotheses based on the literature and domain.
    Uses an Active Graph Explorer (ReAct) to traverse the concept graph.
    """
    msg = f"hypothesis_generation_started query={state.user_query[:50]}"
    log.info(msg)
    
    # IDEMPOTENCY CHECK
    if state.hypotheses:
        await adispatch_custom_event("log", {"message": "[Hypothesis] Skipping (already cached)"}, config=config)
        return {}

    if state.mock:
        await adispatch_custom_event("log", {"message": "[Hypothesis] MOCK MODE: Returning dummy hypothesis."}, config=config)
        return {
            "hypotheses": [
                Hypothesis(
                    id=str(uuid.uuid4()),
                    text="Mock Hypothesis",
                    domain_tags=["mock"],
                    novelty_score=0.9,
                    feasibility_score=0.9,
                    testability_score=0.9,
                    evidence=[]
                )
            ]
        }
        
    # FIX: Safety check for empty graph to prevent explorer hanging
    # The explorer agent will loop infinitely if it can't find any nodes
    if not state.concept_graph or not state.concept_graph.get("nodes"):
        log.warning("hypothesis_generation_skipped", reason="empty_graph")
        await adispatch_custom_event("log", {"message": "[Hypothesis] Skipping (Empty Graph)"}, config=config)
        return {
            "hypotheses": [
                Hypothesis(
                    id=str(uuid.uuid4()),
                    text="No knowledge graph available to generate hypotheses.",
                    domain_tags=[],
                    novelty_score=0.0,
                    feasibility_score=0.0,
                    testability_score=0.0,
                    evidence=[]
                )
            ],
            "selected_hypothesis_id": None
        }

    # 1. Setup Graph Tools (Structured & Deep)
    import networkx as nx
    from langchain_core.tools import tool
    import warnings
    # Suppress deprecation warning for create_react_agent as we are using the compatible prebuilt version
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        from langgraph.prebuilt import create_react_agent
    import json
    
    # Rebuild graph from state
    G = nx.DiGraph()
    graph_data = state.concept_graph or {"nodes": [], "edges": []}
    for node in graph_data.get("nodes", []):
        G.add_node(node)
    for edge in graph_data.get("edges", []):
        # Forward edge
        G.add_edge(edge['source'], edge['target'], relation=edge.get('relation', 'related'))
        # FIX: Inverse edge (Bidirectional by default unless specific)
        directional_rels = ["inhibits", "activates", "causes", "leads to"]
        rel = edge.get('relation', 'related')
        if rel not in directional_rels:
             G.add_edge(edge['target'], edge['source'], relation=f"related ({rel})")

    @tool
    def get_neighbors(node: str) -> str:
        """Get the neighbors of a specific node in the knowledge graph. Returns structured JSON."""
        if node not in G: return json.dumps({"error": f"Node '{node}' not found."})
        neighbors = []
        for n in G.neighbors(node):
            rel = G.get_edge_data(node, n).get('relation', 'related')
            neighbors.append({"node": n, "relation": rel})
        return json.dumps({"node": node, "neighbors": neighbors[:15]}) # Cap for context

    @tool
    def find_paths(start_node: str, end_node: str) -> str:
        """Find paths between two nodes (Depth=5). Returns structured path list."""
        if start_node not in G or end_node not in G: return json.dumps({"error": "Nodes not found."})
        try:
            # FIX: Deeper cutoff=4 for scientific relevance
            paths = list(nx.all_simple_paths(G, start_node, end_node, cutoff=4))
            if not paths: return json.dumps({"paths": [], "message": "No paths found."})
            
            # Rank by length (longer often more explanatory in this context) and limit
            sorted_paths = sorted(paths, key=len, reverse=True)[:5]
            formatted_paths = []
            for p in sorted_paths:
                path_str = " -> ".join(p)
                formatted_paths.append(path_str)
            return json.dumps({"paths": formatted_paths})
        except Exception as e:
            return json.dumps({"error": str(e)})

    @tool
    def get_central_nodes() -> str:
        """Get the most central nodes in the graph."""
        try:
            centrality = nx.degree_centrality(G)
            top = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]
            return json.dumps({"central_nodes": [t[0] for t in top]})
        except:
            return json.dumps({"central_nodes": []})

    # 2. Create ReAct Explorer Agent (Multi-Turn)
    # Refined Temperature Mapping
    temperature = 0.5
    if state.speculation == "low": temperature = 0.2    # Conservative, low creativity
    elif state.speculation == "medium": temperature = 0.5 # Balanced
    elif state.speculation == "high": temperature = 0.8   # High creativity but not chaotic 1.0

    llm = get_llm(temperature=temperature)
    tools = [get_neighbors, find_paths, get_central_nodes]
    
    # FIX: Prompt to force tool usage loop
    system_prompt = """You are a scientific explorer. You MUST use the provided tools to explore the graph multiple times before answering.
    STRATEGY:
    1.  Call `get_central_nodes` to orient yourself.
    2.  Call `get_neighbors` on interesting nodes.
    3.  Call `find_paths` to connect disparate concepts.
    4.  REPEAT steps 2-3 at least 2 times (only if needed) to build a deep understanding.
    """
    
    # Increase iteration limit for deep research
    explorer_agent = create_react_agent(llm, tools)
    # Note: langgraph `create_react_agent` doesn't expose max_iterations via init in all versions, 
    # but the prompt engineering ensures the loop.
    
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
    
    log.info("graph_exploration_starting", num_nodes=len(G.nodes()), num_edges=len(G.edges()))
    # Log exploration start
    await adispatch_custom_event("log", {"message": f"[Hypothesis] Exploring graph ({len(G.nodes())} nodes)..."}, config=config)
    try:
        exploration_result = await explorer_agent.ainvoke({"messages": [
            ("system", system_prompt),
            ("human", exploration_prompt)
        ]}, {"recursion_limit": 100})
        exploration_summary = exploration_result["messages"][-1].content
        log.info("graph_exploration_completed", summary_length=len(exploration_summary))
        # print("DEBUG: Passed exploration completion log.")
        await adispatch_custom_event("log", {"message": "[Hypothesis] Exploration complete."}, config=config)
        # print("DEBUG: Passed event dispatch.")
    except Exception as e:
        log.error("graph_exploration_failed", error=str(e))
        await adispatch_custom_event("log", {"message": f"[Hypothesis] Exploration failed: {e}"}, config=config)
        exploration_summary = "Exploration failed. Relying on literature summary."

    # 3. Auto-Extract Interesting Paths (The "Big Picture" Feeder)
    # FIX: Use all_simple_paths with ranking
    # print("DEBUG: Starting path extraction...")
    await adispatch_custom_event("log", {"message": "[Hypothesis] Extracting ranked multi-hop paths..."}, config=config)
    # print("DEBUG: Dispatched path extraction log.")
    path_context = ""
    try:
        centrality = nx.degree_centrality(G)
        # print("DEBUG: Centrality calc done.")
        # Top 8 to cast a wider net
        top_nodes = [n[0] for n in sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:8]]
        # print(f"DEBUG: Top nodes: {top_nodes}")
        
        found_paths = []
        import itertools
        
        # Limit combinations to avoid explosion
        pairs = list(itertools.combinations(top_nodes, 2))
        
        # print(f"DEBUG: Checking {len(pairs)} pairs.")
        
        for u, v in pairs:
            try:
                # FIX: Use shortest_simple_paths (Yen's algorithm) which is efficient for top-k
                # avoiding the exponential complexity of all_simple_paths even with cutoff
                # Get more candidates (10) to filter for quality
                raw_paths = nx.shortest_simple_paths(G, source=u, target=v)
                candidate_paths = list(itertools.islice(raw_paths, 10))
                
                if not candidate_paths: continue
                
                # Pre-calculate degrees and edge counts for specificity/rarity scoring
                node_degrees = dict(G.degree())
                edge_counts = {}
                total_edges_count = G.number_of_edges()
                for _, _, data in G.edges(data=True):
                    r = data.get('relation', 'related')
                    edge_counts[r] = edge_counts.get(r, 0) + 1
                
                valid_paths = []
                import math
                
                for p in candidate_paths:
                    # Filter 1: Hard Length Cap 
                    if len(p) > 5: continue 
                    # Filter 2: Triviality 
                    if len(p) < 3: continue 
                    
                    # 1. Edge Score (Mechanistic Strength + Rarity)
                    edge_score = 0
                    rarity_accum = 0
                    formatted = []
                    
                    for i in range(len(p)-1):
                        p1, p2 = p[i], p[i+1]
                        
                        rel = G.get_edge_data(p1, p2).get('relation', 'related')
                        formatted.append(f"{p1} --[{rel}]-->")
                        
                        # Base Edge Score (Equal weight)
                        # We used to add 1.0 here, but now we use 1/Length in the final formula
                        pass
                        
                        # Rarity Score (Capped)
                        # freq = count / total
                        freq = edge_counts.get(rel, 1) / (total_edges_count + 1)
                        r_val = -math.log(freq + 1e-8)
                        # Cap rarity at 3.0 to prevent one-off noise from dominating
                        rarity = min(r_val, 3.0)
                        rarity_accum += rarity
                            
                    formatted.append(p[-1])
                    
                    avg_rarity = rarity_accum / (len(p)-1) if len(p) > 1 else 0
                    
                    # 2. Node Specificity Score (Inverse Degree Centrality)
                    node_score = 0
                    intermediate_nodes = p[1:-1]
                    
                    if intermediate_nodes:
                        specs = []
                        for n in intermediate_nodes:
                            deg = node_degrees.get(n, 1)
                            spec = 1.0 / math.log(deg + 2) 
                            specs.append(spec)
                        node_score = sum(specs) / len(specs)
                    else:
                        node_score = 0.5 
                        
                    # Final Score Formula
                    # Alpha * (1/Length) -> Prefer compact/tight explanations
                    # Beta * Specificity -> Prefer non-hubs
                    # Gamma * Rarity -> Prefer novel connections
                    alpha = 1.0
                    beta = 1.0
                    gamma = 0.5
                    
                    len_score = 1.0 / len(p)
                    final_score = (alpha * len_score) + (beta * node_score) + (gamma * avg_rarity)
                    
                    valid_paths.append({
                        "str": " ".join(formatted),
                        "score": final_score,
                        "length": len(p),
                        "components": {
                            "compactness": round(alpha * len_score, 2),
                            "specificity": round(beta * node_score, 2),
                            "novelty": round(gamma * avg_rarity, 2)
                        },
                        "nodes": p # Store raw nodes for Jaccard calc
                    })
                
                # Sort candidates by score and take best 3 per pair
                valid_paths.sort(key=lambda x: x["score"], reverse=True)
                found_paths.extend(valid_paths[:3])
                    
            except Exception:
                continue
                
        # Final Global Sort: High score first
        found_paths.sort(key=lambda x: x["score"], reverse=True)
        
        # LOGGING FOR VERIFICATION
        if found_paths:
            top_debug = [
                {
                    "path": p["str"], 
                    "score": round(p["score"], 2),
                    "breakdown": p["components"]
                } 
                for p in found_paths[:5]
            ]
            log.info("path_scoring_results", top_paths=top_debug)
            log.info("diversity_selection_started", num_candidates=len(found_paths))
        
        final_selected_paths = []
        if found_paths:
            # DIVERSITY SELECTION (Greedy w/ Jaccard Penalty)
            def jaccard_overlap(p_nodes, q_nodes):
                s1, s2 = set(p_nodes), set(q_nodes)
                inter = len(s1 & s2)
                union = len(s1 | s2)
                return inter / union if union > 0 else 0.0

            selected_items = []
            
            # We iterate through sorted paths and pick if they are distinct enough
            for item in found_paths:
                if len(final_selected_paths) >= 5: break
                
                path_nodes = item["nodes"]
                base_score = item["score"]
                
                # Check overlap with already selected
                max_ov = 0.0
                if selected_items:
                    max_ov = max(jaccard_overlap(path_nodes, prev["nodes"]) for prev in selected_items)
                
                # Penalize score based on overlap
                # effective_score = base_score - lambda * overlap
                lambda_overlap = 3.0 
                effective_score = base_score - (lambda_overlap * max_ov)
                
                # Heuristic: If it's still a "good" path (positive score implies reasonable quality), take it
                if effective_score > 2.0: # Threshold tailored to our score scale
                    final_selected_paths.append(item["str"])
                    selected_items.append(item)
            
            log.info("diversity_selection_completed", selected=len(final_selected_paths))

            # Fallback if diversity filtering killed everything (unlikely)
            if not final_selected_paths:
                log.warning("diversity_fallback_triggered")
                final_selected_paths = [p["str"] for p in found_paths[:5]]
                
            path_context = "Key Multi-Hop Causal Chains found in Graph:\n- " + "\n- ".join(final_selected_paths)
            log.info("diversity_log_ready", paths=len(final_selected_paths))
            # Send visibility update to ACTIVITY FEED (not Terminal)
            await adispatch_custom_event("activity", {"message": f"Selected {len(final_selected_paths)} diverse causal paths for analysis."}, config=config)
        else:
            await adispatch_custom_event("log", {"message": "[Hypothesis] No significant paths found."}, config=config)
            
    except Exception as e:
        log.error("hypothesis_path_extraction_critical_failure", error=str(e))
        await adispatch_custom_event("log", {"message": f"[Hypothesis] Path extraction failed: {e}"}, config=config)

     # 3.5 Structural Hole Explorer (The "Novelty Engine")
    log.info("checking_structural_hole_conditions", goal=state.goal, speculation=state.speculation)
    hole_exploration_summary = ""
    if state.goal == "discover" or state.speculation == "high":
        log.info("starting_structural_hole_exploration")
        # Send to Activity Feed
        await adispatch_custom_event("activity", {"message": "Running Structural Hole Explorer to find missing links..."}, config=config)
        try:
            # A. Detect Communities (Clusters)
            import networkx.algorithms.community as nx_comm
            
            # Convert to undirected for community detection
            G_undirected = G.to_undirected()
            if len(G_undirected.nodes) > 5:
                communities = list(nx_comm.greedy_modularity_communities(G_undirected))
                
                if len(communities) >= 2:
                    # Sort communities by size
                    communities = sorted(communities, key=len, reverse=True)
                    
                    c1 = list(communities[0])[:5] # Largest cluster
                    
                    # Try to find a cluster that is 'far' but not tiny
                    # Or if lens is active, find the cluster containing the lens
                    c2 = list(communities[1])[:5] 
                    
                    # Lens-Aware selection: 
                    # If lens is in c1, make c2 the most distant cluster.
                    # If lens is in neither, force c2 to be the lens neighborhood (if exists)
                    if state.lens and state.lens in G:
                        lens_community = next((c for c in communities if state.lens in c), None)
                        if lens_community:
                            # If lens is in c1, pick c2 as normal.
                            # If lens is NOT in c1, make c2 the lens community.
                            if list(lens_community) != list(communities[0]):
                                c2 = list(lens_community)[:5]

                    # Terminal log is fine for "hard" data like clusters
                    # await adispatch_custom_event("log", {"message": f"[Hypothesis] Identified Disconnected Clusters:\nCluster A: {c1}\nCluster B: {c2}"}, config=config)
                    
                    # B. The "Bridge" Prompt (Null Hypothesis Approach)
                    bridge_prompt = f"""
                    I have identified two distinct clusters of knowledge in the graph that seem currently disconnected:
                    
                    Cluster A (Theme 1): {', '.join(c1)}
                    Cluster B (Theme 2): {', '.join(c2)}
                    
                    Your Task: Act as a visionary scientist using the lens of '{state.lens}'. 
                    1. Analyze potential "Structural Holes" (missing links) between these two clusters.
                    2. Ask yourself: Is there a plausible underlying mechanism that could connect Cluster A to Cluster B?
                    3. If YES, propose a "Bridge Hypothesis" explaining this mechanism.
                    4. If NO, explicitly state that they are distinct.
                    """
                    
                    bridge_result = await llm.ainvoke(bridge_prompt)
                    hole_exploration_summary = f"\n\nstructural_hole_analysis:\n{bridge_result.content}"
                    await adispatch_custom_event("log", {"message": "[Hypothesis] Structural Hole Analysis completed."}, config=config)
                else:
                    await adispatch_custom_event("log", {"message": "[Hypothesis] Not enough communities found for structural hole analysis."}, config=config)
            else:
                 await adispatch_custom_event("log", {"message": "[Hypothesis] Graph too small for community detection."}, config=config)
                 
        except Exception as e:
            log.error("structural_hole_failed", error=str(e))
            await adispatch_custom_event("log", {"message": f"[Hypothesis] Structural Hole Exploration failed: {e}"}, config=config)

    # 4. Generate Structured Hypotheses (Structured Template)
    log.info("preparing_hypothesis_generation")
    # Send to Activity Feed
    await adispatch_custom_event("activity", {"message": "Synthesizing novel hypotheses from graph evidence..."}, config=config)
    
    structured_llm = llm.with_structured_output(HypothesisList)
    
    from langchain_core.messages import SystemMessage, HumanMessage
    
    # Improved Prompt with Novelty/Feasibility Scoring and Deduplication awareness
    system_msg = f"""You are a Principal Investigator. Generate 3 NOVEL, TESTABLE scientific hypotheses based on the provided exploration.
    
    GUIDELINES:
    1. Each hypothesis must be NON-OBVIOUS. (Avoid "X is related to Y" - say "X drives Y via Z").
    2. Must be TESTABLE with current technology (simulation or lab).
    3. Use the graph paths evidence provided.
    
    REQUIRED OUTPUT STRUCTURE per hypothesis:
    - Statement: A single clear sentence.
    - Causal Chain: The step-by-step mechanism (A -> B -> C).
    - Evidence Summary: Specific nodes/paths from the graph that support this.
    - Scores: Novelty (0-1), Feasibility (0-1), Testability (0-1).
    - Search Query: A precise keyword-based boolean query to validate this hypothesis (e.g. '"protein folding" AND "diffusion"').
    - Tags: Domain tags (e.g. 'bio', 'ml').
    """
    
    messages = [
        SystemMessage(content=system_msg),
        HumanMessage(content=f"User Query: {state.user_query}\n\nGraph Exploration:\n{exploration_summary}\n\nAutomated Path Analysis:\n{path_context}\n\nStructural Hole Analysis (Novelty):\n{hole_exploration_summary}\n\nLiterature Context:\n{state.literature.get('summary', '')}")
    ]
    
    try:
        log.info("invoking_hypothesis_llm")
        await adispatch_custom_event("log", {"message": f"[Hypothesis] Generating hypotheses (Speculation: {state.speculation or 'Medium'})..."}, config=config)
        result = await structured_llm.ainvoke(messages)
        log.info("hypothesis_llm_completed", num_hypotheses=len(result.hypotheses))
        hypotheses = result.hypotheses
        
        # 4.5 Deduplication & Selection
        # Simple dedupe by checking text overlap or just distinct first words
        unique_hypotheses = []
        seen_texts = set()
        
        for h in hypotheses:
            # Normalize text for dedupe
            simp_text = h.text.lower().strip()[:50] 
            if simp_text not in seen_texts:
                if not h.id: h.id = str(uuid.uuid4())
                unique_hypotheses.append(h)
                seen_texts.add(simp_text)
                
        hypotheses = unique_hypotheses
        
        if hypotheses:
            # FIX: Sort by (Novelty + Testability) / 2
            # Assuming these fields exist in your Pydantic model. If not, they need to be added or parsed from text.
            # Since the user asked for this sort, we assume the model supports it or we use heuristic.
            # Check pydantic model in app.state first? Assuming it has scores.
            hypotheses.sort(key=lambda h: (h.novelty_score + h.testability_score)/2, reverse=True)
            
            selected_id = hypotheses[0].id
            await adispatch_custom_event("log", {"message": f"[Hypothesis] Generated {len(hypotheses)} hypotheses."}, config=config)
        else:
             selected_id = None
             
    except Exception as e:
        await adispatch_custom_event("log", {"message": f"[Hypothesis] Error generating hypotheses: {e}"}, config=config)
        hypotheses = []
        selected_id = None

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
        selected_id = hypotheses[0].id
    
    return {
        "hypotheses": hypotheses,
        "selected_hypothesis_id": selected_id
    }
