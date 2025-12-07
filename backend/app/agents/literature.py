from app.state import DiscoveryState
from app.tools.openalex import search_papers, reconstruct_abstract
from app.llm import get_cheap_llm, get_llm
from langchain_core.prompts import ChatPromptTemplate
from duckduckgo_search import DDGS
from pydantic import BaseModel, Field
from typing import List

class Edge(BaseModel):
    source: str
    target: str
    relation: str

    class Config:
        extra = "forbid"

class ConceptGraph(BaseModel):
    nodes: List[str] = Field(description="List of key concepts")
    edges: List[Edge] = Field(description="List of relationships")
    
    class Config:
        extra = "forbid"

from langchain_core.runnables import RunnableConfig

async def literature_node(state: DiscoveryState, config: RunnableConfig) -> dict:
    """
    Literature Agent: Fetches papers, searches web, summarizes, and builds a concept graph.
    """
    print("DEBUG: Entered literature_node", flush=True)
    query = state.user_query
    print(f"[Literature] User Query: {query}")
    
    # 0. Query Refinement (Crucial for OpenAlex/Search to work with long prompts)
    refinement_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a research librarian. Convert the user's complex natural language query into a precise, keyword-based boolean search string suitable for a library database (like OpenAlex or PubMed). Use AND, OR, and quotes for phrases. Keep it under 200 characters."),
        ("human", f"User Query: {query}\n\nSearch String:")
    ])
    
    llm = get_cheap_llm()
    search_query_result = await (refinement_prompt | llm).ainvoke({})
    search_query = search_query_result.content.strip().replace('"', '') # Clean up
    print(f"[Literature] refined search query: '{search_query}'")
    
    # 1. Search OpenAlex
    from langchain_core.tools import tool

    @tool
    async def openalex_search_tool(q: str):
        """Searches OpenAlex for scientific papers."""
        return await search_papers(q, limit=10)

    # Invoke tool to trigger stream events
    print(f"[Literature] Searching OpenAlex for: {search_query}")
    papers = await openalex_search_tool.ainvoke(search_query, config=config)
    
    # 2. Web Search (for latest info/datasets)
    @tool
    def web_search_tool(query: str, goal: str):
        """Performs web search for latest info and contradictions."""
        web_results = ""
        try:
            # Try to use DDGS
            with DDGS() as ddgs:
                # Standard search
                # backend="api" is often more stable for bots
                results = list(ddgs.text(f"{query} latest research datasets", max_results=3))
                if not results:
                     # Retry with simpler query
                     results = list(ddgs.text(query, max_results=3))

                for r in results:
                    web_results += f"Title: {r.get('title')}\nLink: {r.get('href')}\nSnippet: {r.get('body')}\n\n"
                
                # CONTRADICTION MINER (Novelty Engine)
                if goal == "discover":
                    # Search for debates, conflicts, and limitations
                    contradiction_query = f"{query} controversy debate limitations"
                    c_results = list(ddgs.text(contradiction_query, max_results=3))
                    if c_results:
                        web_results += "\n--- CONTRADICTION MINING FOUND ---\n"
                        for r in c_results:
                            web_results += f"Title: {r.get('title')}\nLink: {r.get('href')}\nSnippet: {r.get('body')}\n\n"
        except Exception as e:
            web_results = f"Web search failed: {e}"
        
        if not web_results:
            return "No web results found."
        return web_results

    print(f"[Literature] Running Web Search for: {search_query}...")
    web_results = web_search_tool.invoke({"query": search_query, "goal": state.goal}, config=config)
    
    # 3. Process results & Summarize
    processed_papers = {}
    summary_lines = []
    full_text_context = ""
    
    for paper in papers:
        pid = paper["id"]
        abstract_text = reconstruct_abstract(paper.get("abstract"))
        
        processed_papers[pid] = {
            "title": paper["title"],
            "year": paper["publication_year"],
            "venue": paper["host_venue"],
            "abstract": abstract_text,
            "url": paper["landing_page_url"]
        }
        summary_lines.append(f"- {paper['title']} ({paper['publication_year']})")
        full_text_context += f"Title: {paper['title']}\nAbstract: {abstract_text}\n\n"

    full_text_context += f"\n\nWeb Search Results:\n{web_results}"
    print(f"[Literature] Found {len(papers)} papers and web results.")
    
    # 4. Build Concept Graph (Batch / Fan-In)
    import asyncio
    from langchain_community.callbacks import get_openai_callback
    
    llm = get_cheap_llm()
    structured_llm = llm.with_structured_output(ConceptGraph)
    
    # SYSTEM PROMPT STRATEGY BASED ON GOAL
    if state.goal == "survey":
        system_msg = "You are a specialized librarian. Extract the CORE CONSENSUS concepts and DEFINITIVE relationships from the provided abstracts. Ignore minor details."
    elif state.goal == "discover":
        system_msg = "You are a scientific detective. Extract NOVEL, NON-OBVIOUS, and CONTRADICTING concepts from the collection of abstracts. Focus on the periphery and edge cases."
    else:
        system_msg = "You are a scientific assistant. Extract a granular concept graph from the following abstracts."

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_msg),
        ("human", "Here are the abstracts:\n\n{abstracts}")
    ])
    chain = prompt | structured_llm

    # Prepare batch context
    all_abstracts = []
    for p in papers:
        abs_text = reconstruct_abstract(p.get("abstract"))
        if abs_text:
            all_abstracts.append(f"Paper ID: {p['id']}\nTitle: {p['title']}\nAbstract: {abs_text}\n")
    
    combined_feed = "\n---\n".join(all_abstracts)
    
    # 4. EXPLICIT LOGGING TOOL (Heartbeat)
    @tool
    def log_heartbeat(msg: str):
        """Emits a log message to the frontend."""
        return msg

    print(f"[Literature] Extracting unified graph from {len(papers)} papers (Batch Mode)...")
    log_heartbeat.invoke(f"Starting unified graph extraction for {len(papers)} papers using GPT-5-Mini...", config=config)
    
    merged_nodes = set()
    merged_edges = []
    
    try:
        with get_openai_callback() as cb:
            # Single Batch Call
            graph_result = await chain.ainvoke({"abstracts": combined_feed}, config=config)
            print(f"[Literature] Token Usage (Batch Extraction): {cb}")
            log_heartbeat.invoke(f"Batch extraction complete. Cost: ${cb.total_cost:.4f}", config=config)
        
        if graph_result:
            merged_nodes.update(graph_result.nodes)
            # Deduplicate edges
            for edge in graph_result.edges:
                merged_edges.append(edge)

    except Exception as e:
        print(f"[Literature] Batch extraction failed: {e}")
        log_heartbeat.invoke(f"Error during batch extraction: {e}", config=config)     
    
    # 3.5. APPLY DISCIPLINARY LENS (Inject a node if using a lens)
    if state.lens and state.lens != "none":
        print(f"[Literature] Injecting Lens Node: {state.lens}")
        merged_nodes.add(state.lens)
        # Connect lens to central concepts to force "Bridge" detection later
        # We don't create edges yet, but having the node allows Hypothesis agent to 'find paths' to it.

    concept_graph = {
        "nodes": list(merged_nodes),
        "edges": [e.model_dump() for e in merged_edges]
    }
    print(f"[Literature] Built batch graph with {len(merged_nodes)} nodes and {len(merged_edges)} edges.")

    # 4. Normalize Graph (Entity Resolution)
    try:
        print(f"[Literature] Normalizing graph nodes...")
        # with get_openai_callback() as cb:
        concept_graph = await normalize_graph_nodes(concept_graph)
        # print(f"[Literature] Token Usage (Normalization): {cb}")
        print(f"[Literature] Graph normalized. Nodes: {len(concept_graph['nodes'])}, Edges: {len(concept_graph['edges'])}")
    except Exception as e:
        print(f"[Literature] Normalization failed: {e}")

    # 5. Densify Graph (Grounded Second Pass)
    try:
        print(f"[Literature] Densifying graph (Second Pass)...")
        # with get_openai_callback() as cb:
        concept_graph = await densify_graph(concept_graph, papers)
        # print(f"[Literature] Token Usage (Densification): {cb}")
        print(f"[Literature] Graph densified. Nodes: {len(concept_graph['nodes'])}, Edges: {len(concept_graph['edges'])}")
    except Exception as e:
        print(f"[Literature] Densification failed: {e}")

    # 6. Graph Analysis (NetworkX)
    graph_insights = analyze_graph(concept_graph)
    print(f"[Literature] Graph Analysis:\n{graph_insights}")
    
    # Append insights to context for the next agent
    full_text_context += f"\n\nGraph Analysis Insights:\n{graph_insights}"

    return {
        "literature": {
            "papers": processed_papers,
            "summary": "\n".join(summary_lines), 
            "full_context": full_text_context 
        },
        "concept_graph": concept_graph
    }

async def densify_graph(graph_data: dict, papers: list) -> dict:
    """
    Performs a second pass extraction to find missing relationships between known nodes.
    """
    import asyncio
    from pydantic import BaseModel, Field
    
    nodes = set([n if isinstance(n, str) else n["id"] for n in graph_data.get("nodes", [])])
    if not nodes:
        return graph_data

    # We only want to find edges between EXISTING nodes
    class EdgeList(BaseModel):
        edges: List[Edge]

    llm = get_cheap_llm()
    structured_llm = llm.with_structured_output(EdgeList)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a scientific assistant refining a knowledge graph. You will be given an abstract and a list of KNOWN CONCEPTS. Your task is to find relationships ONLY between these known concepts that are explicitly stated in the text."),
        ("human", "Known Concepts: {concepts}\n\nAbstract: {abstract}")
    ])
    chain = prompt | structured_llm

    async def process_paper(paper):
        abstract = reconstruct_abstract(paper.get("abstract"))
        if not abstract: return []
        
        # Optimization: Only pass concepts that might be in this paper (simple keyword match)
        # This reduces prompt size and noise
        relevant_concepts = [n for n in nodes if n.lower() in abstract.lower()]
        if len(relevant_concepts) < 2: return []
        
        try:
            result = await chain.ainvoke({
                "concepts": ", ".join(relevant_concepts),
                "abstract": abstract
            })
            return result.edges
        except Exception as e:
            # print(f"[Literature] Densification error for {paper['id']}: {e}")
            return []

    print(f"[Literature] Running second pass on {len(papers)} papers...")
    results = await asyncio.gather(*[process_paper(p) for p in papers])
    
    new_edges = []
    existing_edges = set()
    for e in graph_data.get("edges", []):
        existing_edges.add((e['source'], e['target'], e.get('relation', 'related')))
        
    for edges in results:
        for edge in edges:
            # Verify nodes exist (double check)
            if edge.source in nodes and edge.target in nodes:
                # Verify edge is new
                if (edge.source, edge.target, edge.relation) not in existing_edges:
                    new_edges.append(edge.model_dump())
                    existing_edges.add((edge.source, edge.target, edge.relation))
                    
    if new_edges:
        print(f"[Literature] Found {len(new_edges)} new edges in second pass.")
        graph_data["edges"].extend(new_edges)
        
    return graph_data

async def normalize_graph_nodes(graph_data: dict) -> dict:
    """
    Uses an LLM to identify synonymous nodes and merge them into canonical names.
    """
    import networkx as nx
    from pydantic import BaseModel, Field
    
    nodes = [n if isinstance(n, str) else n["id"] for n in graph_data.get("nodes", [])]
    if not nodes:
        return graph_data

    class NodeMapping(BaseModel):
        original: str
        canonical: str
    
    class MappingList(BaseModel):
        mappings: List[NodeMapping]

    llm = get_llm(temperature=0.0)
    structured_llm = llm.with_structured_output(MappingList)
    
    # Chunk nodes if too many (limit to 50 at a time to avoid context limits)
    chunk_size = 50
    all_mappings = {}
    
    for i in range(0, len(nodes), chunk_size):
        chunk = nodes[i:i+chunk_size]
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a scientific terminologist. Review the list of concepts and identify synonyms. Map any variations to the most standard, canonical scientific term. If a term is already standard, map it to itself."),
            ("human", f"List of concepts: {', '.join(chunk)}")
        ])
        
        try:
            chain = prompt | structured_llm
            result = await chain.ainvoke({})
            for m in result.mappings:
                if m.original != m.canonical:
                    all_mappings[m.original] = m.canonical
        except Exception as e:
            print(f"[Literature] Error in normalization chunk: {e}")

    if not all_mappings:
        return graph_data
        
    print(f"[Literature] Merging {len(all_mappings)} nodes: {all_mappings}")
    
    # Rebuild graph to merge nodes
    G = nx.DiGraph()
    # Add all original nodes first
    for node in graph_data.get("nodes", []):
        n_id = node if isinstance(node, str) else node["id"]
        G.add_node(n_id)
        
    for edge in graph_data.get("edges", []):
        G.add_edge(edge['source'], edge['target'], relation=edge.get('relation', 'related'))
        
    # Apply mapping
    G = nx.relabel_nodes(G, all_mappings, copy=False)
    
    # Convert back to dict
    new_nodes = list(G.nodes())
    new_edges = [{"source": u, "target": v, "relation": data.get("relation", "related")} for u, v, data in G.edges(data=True)]
    
    return {"nodes": new_nodes, "edges": new_edges}
def analyze_graph(graph_data: dict) -> str:
    """
    Analyzes the concept graph using NetworkX to find central concepts and bridges.
    """
    import networkx as nx
    
    G = nx.DiGraph()
    for node in graph_data.get("nodes", []):
        G.add_node(node)
    for edge in graph_data.get("edges", []):
        G.add_edge(edge['source'], edge['target'], relation=edge.get('relation', 'related'))
        
    if len(G.nodes) == 0:
        return "No graph data available."

    insights = []
    
    # 1. Centrality (Most important concepts)
    try:
        degree_centrality = nx.degree_centrality(G)
        top_central = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
        insights.append(f"Key Concepts (Centrality): {', '.join([f'{n[0]} ({n[1]:.2f})' for n in top_central])}")
    except:
        pass

    # 2. Betweenness (Bridges between topics)
    try:
        betweenness = nx.betweenness_centrality(G)
        top_bridges = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:3]
        # Only list if they have some betweenness
        bridges = [f"{n[0]} ({n[1]:.2f})" for n in top_bridges if n[1] > 0]
        if bridges:
            insights.append(f"Bridge Concepts (Connecting topics): {', '.join(bridges)}")
    except:
        pass

    # 3. Key Relationships (Edges between top nodes)
    try:
        top_nodes = set([n[0] for n in top_central] + [n[0] for n in top_bridges])
        important_edges = []
        for u, v, data in G.edges(data=True):
            if u in top_nodes and v in top_nodes:
                important_edges.append(f"{u} --[{data.get('relation', 'related')}]--> {v}")
        
        if important_edges:
            insights.append("Key Relationships:\n- " + "\n- ".join(important_edges[:10]))
    except:
        pass
        
    return "\n".join(insights)
