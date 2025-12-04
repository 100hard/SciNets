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

async def literature_node(state: DiscoveryState) -> dict:
    """
    Literature Agent: Fetches papers, searches web, summarizes, and builds a concept graph.
    """
    query = state.user_query
    print(f"[Literature] Searching for: {query}")
    
    # 1. Search OpenAlex
    papers = await search_papers(query, limit=10)
    
    # 2. Web Search (for latest info/datasets)
    print("[Literature] Running Web Search...")
    web_results = ""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(f"{query} latest research datasets", max_results=3))
            for r in results:
                web_results += f"Title: {r['title']}\nLink: {r['href']}\nSnippet: {r['body']}\n\n"
    except Exception as e:
        print(f"[Literature] Web search failed: {e}")
    
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
    
    # 4. Build Concept Graph (Iterative & Parallel)
    import asyncio
    
    llm = get_cheap_llm()
    structured_llm = llm.with_structured_output(ConceptGraph)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a scientific assistant. Extract a granular concept graph from the following abstract. Identify specific entities (proteins, genes, methods, diseases) and their precise relationships."),
        ("human", "{abstract}")
    ])
    chain = prompt | structured_llm

    async def extract_from_paper(paper):
        try:
            abstract = reconstruct_abstract(paper.get("abstract"))
            if not abstract: return None
            return await chain.ainvoke({"abstract": abstract})
        except Exception as e:
            print(f"[Literature] Extraction failed for {paper['id']}: {e}")
            return None

    print(f"[Literature] Extracting graphs from {len(papers)} papers in parallel...")
    graph_results = await asyncio.gather(*[extract_from_paper(p) for p in papers])
    
    # Merge graphs
    merged_nodes = set()
    merged_edges = []
    
    for res in graph_results:
        if res:
            merged_nodes.update(res.nodes)
            # Deduplicate edges based on source-target-relation
            for edge in res.edges:
                # Check if edge already exists (simple check)
                exists = any(e.source == edge.source and e.target == edge.target and e.relation == edge.relation for e in merged_edges)
                if not exists:
                    merged_edges.append(edge)
    
    concept_graph = {
        "nodes": list(merged_nodes),
        "edges": [e.dict() for e in merged_edges]
    }
    print(f"[Literature] Built merged graph with {len(merged_nodes)} nodes and {len(merged_edges)} edges.")

    # 4. Normalize Graph (Entity Resolution)
    try:
        print(f"[Literature] Normalizing graph nodes...")
        concept_graph = await normalize_graph_nodes(concept_graph)
        print(f"[Literature] Graph normalized. Nodes: {len(concept_graph['nodes'])}, Edges: {len(concept_graph['edges'])}")
    except Exception as e:
        print(f"[Literature] Normalization failed: {e}")

    # 5. Densify Graph (Grounded Second Pass)
    try:
        print(f"[Literature] Densifying graph (Second Pass)...")
        concept_graph = await densify_graph(concept_graph, papers)
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
                    new_edges.append(edge.dict())
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

    llm = get_llm(model_name="gpt-4o-mini", temperature=0.0)
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
