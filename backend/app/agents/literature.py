from app.state import DiscoveryState
from app.tools.openalex import search_papers, reconstruct_abstract, get_paper_citations, get_paper_details
from app.llm import get_cheap_llm, get_llm
from langchain_core.prompts import ChatPromptTemplate
# DDGS import removed
from pydantic import BaseModel, Field
from typing import List

class Edge(BaseModel):
    source: str
    target: str
    relation: str
    supporting_papers: List[str] = []  # Track which papers support this edge

    class Config:
        extra = "forbid"

class ConceptGraph(BaseModel):
    nodes: List[str] = Field(description="List of key concepts")
    edges: List[Edge] = Field(description="List of relationships")
    
    class Config:
        extra = "forbid"

from langchain_core.runnables import RunnableConfig
from langchain_core.callbacks import adispatch_custom_event

from app.telemetry.cost_tracker import CostTracker

async def literature_node(state: DiscoveryState, config: RunnableConfig) -> dict:
    """
    Literature Agent: Fetches papers, searches web, summarizes, and builds a concept graph.
    """
    CostTracker.get_instance().start_step("literature", "literature_search_and_graph")
    try:
        print("DEBUG: Entered literature_node", flush=True)
        # IDEMPOTENCY CHECK: If literature is already found, skip (for re-runs)
        if state.literature and state.concept_graph:
            print("[Literature] Skipping (already cached)")
            CostTracker.get_instance().end_step("literature", "literature_search_and_graph")
            return {}
    
        query = state.user_query
        print(f"[Literature] User Query: {query}")
    
        if state.mock:
            print("[Literature] MOCK MODE: Returning dummy data.")
            await adispatch_custom_event("log", {"message": "[Literature] MOCK MODE: Returning dummy literature data."}, config=config)
            CostTracker.get_instance().end_step("literature", "literature_search_and_graph")
            return {
                "literature": {
                "papers": {"MOCK-1": {"title": "Mock Paper", "year": 2024, "venue": "Mock Venue", "abstract": "This is a mock abstract.", "url": "http://mock"}},
                "summary": "- Mock Paper (2024)",
                "full_context": "Mock Context",
                "web_research": "Mock Web Results"
            },
            "concept_graph": {
                "nodes": ["Mock Concept A", "Mock Concept B"],
                "edges": [{"source": "Mock Concept A", "target": "Mock Concept B", "relation": "test"}]
            }
        }
    
        # 0. Query Refinement (Crucial for OpenAlex/Search to work with long prompts)
        refinement_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a research librarian. Convert the user's complex natural language query into a precise, keyword-based boolean search string suitable for OpenAlex. Use AND, OR. Keep it under 100 characters. Avoid complex nesting or wildcard characters at the end of the string."),
            ("system", "You are a research librarian. Convert the user's complex natural language query into a precise, keyword-based boolean search string suitable for OpenAlex. Use AND, OR. Keep it under 100 characters. Avoid complex nesting or wildcard characters at the end of the string."),
            ("human", f"User Query: {query}\nUser Guidance: {state.guidance}\nTimeline Context: {state.timeline} (Guide search if recent)\n\nSearch String:")
        ])
        
        llm = get_cheap_llm()
        search_query_result = await (refinement_prompt | llm).ainvoke({})
        search_query = search_query_result.content.strip() # FIX: Do NOT remove quotes
        # Only remove quotes if the LLM wrapped the *entire* output in them unreasonably, 
        # but keep internal quotes for boolean search.
        if search_query.startswith('"') and search_query.endswith('"') and search_query.count('"') == 2:
            search_query = search_query[1:-1]
            
        msg = f"[Literature] refined search query: '{search_query}'"
        print(msg)
        await adispatch_custom_event("log", {"message": msg}, config=config)
        
        # 1. Search OpenAlex
        from langchain_core.tools import tool
    
        @tool
        async def openalex_search_tool(q: str):
            """Searches OpenAlex for scientific papers."""
            limit = state.max_papers if state.max_papers else 5
            # Logic: Timeline mapping
            start_year = None
            if state.timeline == "recent":
                start_year = 2020 # 5 years
            elif state.timeline == "decade":
                start_year = 2015 # 10 years
            # 'all' implies no filter
                
            print(f"[Literature] Fetching up to {limit} papers (Probe Mode) for timeline: {state.timeline}...")
            results = await search_papers(q, limit=limit)
            
            # Post-filter by year if API doesn't support it directly in this specific tool wrapper
            # (Assuming search_papers wrapper might not take year param yet, or we filter results)
            if start_year:
                results = [p for p in results if p.get('publication_year') and p.get('publication_year') >= start_year]
                
            return results
    
        # Invoke tool to trigger stream events
        papers = []
        
        # 0.5 Process User-Provided Documents (Abstracts OR IDs) FIRST
        if state.documents:
            print(f"[Literature] Processing {len(state.documents)} user-provided documents...")
            import asyncio
            import uuid
            
            async def fetch_or_parse(doc_input):
                 # Case A: URL / OpenAlex ID
                 if doc_input.strip().startswith("http") or doc_input.strip().upper().startswith("W"):
                     try:
                         # Attempt to fetch details
                         print(f"[Literature] Fetching details for ID: {doc_input[:20]}...")
                         details = await get_paper_details(doc_input)
                         if details:
                             return details
                     except Exception as e:
                         print(f"[Literature] Failed to fetch ID {doc_input}: {e}")
                         return None
                 
                 # Case B: Manual Text Abstract
                 else:
                     title_candidate = doc_input.split('.')[0][:100] + "..."
                     pid = f"USER-{str(uuid.uuid4())[:8]}"
                     return {
                        "id": pid,
                        "title": f"[User Input] {title_candidate}",
                        "publication_year": 2024,
                        "abstract": doc_input, # Text is abstract
                        "host_venue": "User Provided",
                        "landing_page_url": None,
                        "abstract_inverted_index": None
                     }
            
            # Run fetches in parallel
            fetched_results = await asyncio.gather(*[fetch_or_parse(d) for d in state.documents])
            valid_docs = [d for d in fetched_results if d]
            
            if valid_docs:
                print(f"[Literature] Successfully processed {len(valid_docs)} user documents.")
                papers.extend(valid_docs)
        
        # MIXED MODE: If we have enough user docs, skip fresh search.
        if len(papers) >= 3:
            msg = f"[Literature] Using {len(papers)} user-curated papers. Skipping fresh search to respect user selection."
            print(msg)
            await adispatch_custom_event("log", {"message": msg}, config=config)
        else:
            # Invoke tool to trigger stream events
            msg = f"[Literature] Searching OpenAlex for: {search_query}"
            print(msg)
            await adispatch_custom_event("log", {"message": msg}, config=config)
            try:
                 search_results = await openalex_search_tool.ainvoke(search_query, config=config)
                 if search_results:
                     papers.extend(search_results)
            except Exception as e:
                 print(f"[Literature] Search error: {e}")
        
        # FALLBACK LOGIC (Only if still no papers)
        if not papers:
            msg = f"[Literature] Initial search returned 0 results. Retrying with simplified keyword search..."
            print(msg)
            await adispatch_custom_event("log", {"message": msg}, config=config)
            
            # Tokenize and remove operators
            stopwords = ["AND", "OR", "NOT", "(", ")"]
            # Basic split by space, preserve phrases if possible or just naive split
            words = search_query.replace("(", "").replace(")", "").split()
            keywords = [w for w in words if w.upper() not in stopwords]
            simple_query = " ".join(keywords)
            
            print(f"[Literature] Fallback Query: {simple_query}")
            fallback_results = await openalex_search_tool.ainvoke(simple_query, config=config)
            if fallback_results:
                papers.extend(fallback_results)
        
        # 1.5 OPTIONAL: Citation Expansion (Tier 2)
        # Expands corpus via citations - weighted, not dominant
        # CONSTRAINT: Disable when speculation is high to preserve structural-hole discovery
        should_expand = state.enable_citation_expansion and state.speculation != "high"
        
        if should_expand and papers:
            await adispatch_custom_event("log", {"message": "[Literature] Citation expansion enabled. Fetching citing papers..."}, config=config)
            
            # Only expand from top 3 seed papers to avoid explosion
            seed_papers = papers[:3]
            expanded_papers = []
            
            for seed in seed_papers:
                seed_id = seed.get("id", "")
                if seed_id:
                    try:
                        # Fetch 3 citing papers per seed (weighted, not dominant)
                        citations = await get_paper_citations(seed_id, limit=3)
                        for cp in citations:
                            # Avoid duplicates
                            if not any(p.get("id") == cp.get("id") for p in papers + expanded_papers):
                                expanded_papers.append(cp)
                    except Exception as e:
                        print(f"[Literature] Citation fetch failed for {seed_id}: {e}")
            
            if expanded_papers:
                print(f"[Literature] Added {len(expanded_papers)} papers via citation expansion.")
                papers.extend(expanded_papers)
        elif state.speculation == "high" and state.enable_citation_expansion:
            await adispatch_custom_event("log", {"message": "[Literature] Citation expansion SKIPPED (speculation=high, preserving structural holes)"}, config=config)
        
        # 2. Web Search Disabled
        # Based on user feedback and Windows asyncio stability, generic web search is disabled.
        # We rely on OpenAlex (Scientific) and User Inputs.
        print(f"[Literature] Web Search is DISABLED. Relying on OpenAlex and User Inputs.")
        web_results_text = ""
        web_items = []
            
        # 3. Process results & Summarize
        processed_papers = {}
        summary_lines = []
        full_text_context = ""
        
        # Fallback Handling if OpenAlex failed (usually unlikely with fallback logic in tools)
        if not papers:
            if not state.documents:
                print("[Literature] WARNING: No papers found from OpenAlex or User Input.")
    
        
        for paper in papers:
            pid = paper["id"]
            raw_abstract = paper.get("abstract")
            
            # FIX: Handle already-string abstracts (Web/User) vs Inverted Index (OpenAlex)
            if isinstance(raw_abstract, str):
                abstract_text = raw_abstract
            else:
                abstract_text = reconstruct_abstract(raw_abstract)
            
            processed_papers[pid] = {
                "title": paper["title"],
                "year": paper["publication_year"],
                "venue": paper["host_venue"],
                "abstract": abstract_text,
                "url": paper["landing_page_url"]
            }
            summary_lines.append(f"- {paper['title']} ({paper['publication_year']})")
            full_text_context += f"Title: {paper['title']}\nAbstract: {abstract_text}\n\n"
    
        # NOTE: Web results stored separately in 'web_research' field, not added to full_text_context
        # to avoid polluting scientific context for hypothesis generation
        
        print(f"[Literature] Found {len(papers)} papers and web results.")
        
        # 3.5 Generate Executive Abstract (Prose)
        # The user specifically requested a real abstract, not a list of titles.
        print("[Literature] Synthesizing executive abstract...")
        try:
            abstract_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are the lead author of a meta-analysis. Write a coherent 2-paragraph Executive Abstract summarizing the key themes, consensus, and novel findings from the provided research context. \n\nCRITICAL CONSTRAINTS:\n1. This abstract is for HIGH-LEVEL ORIENTATION ONLY.\n2. Do NOT list papers.\n3. Frame findings as 'emerging themes' or 'consensus', not absolute proof. Downstream agents will verify the details."),
                ("human", f"Research Context:\n{full_text_context}\n\nExecutive Abstract (2 paragraphs):")
            ])
            abstract_res = await (abstract_prompt | get_cheap_llm()).ainvoke({})
            final_summary = abstract_res.content.strip()
        except Exception as e:
            print(f"[Literature] Abstract generation failed: {e}")
            final_summary = "\n".join(summary_lines) # Fallback
    
        
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
    
        # FIX: Stronger Hallucination Guardrails & Entity Enforcement
        system_msg += """
        
        CRITICAL RULES:
        1. NODES MUST BE NOUN PHRASES (e.g., "Dopamine", "Synaptic Plasticity").
        2. DO NOT create nodes that are full sentences, titles, or verb phrases (e.g., "Running increases neurogenesis" is BANNED).
        3. DO NOT HALLUCINATE: Only extract concepts explicitly appearing in the abstracts.
        """
    
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_msg),
            ("human", "Here are the abstracts:\n\n{abstracts}")
        ])
        chain = prompt | structured_llm
    
        # Prepare batch execution (Batch Size = 3)
        batch_size = 3
        batches = [papers[i:i + batch_size] for i in range(0, len(papers), batch_size)]
        
        print(f"[Literature] Extracting graph in {len(batches)} batches (Size={batch_size})...")
        
        merged_nodes = set()
        merged_edges = []
        
        async def process_batch(batch_papers):
            batch_abstracts = []
            for p in batch_papers:
                raw = p.get("abstract")
                if isinstance(raw, str):
                    abs_text = raw
                else:
                    abs_text = reconstruct_abstract(raw)
                    
                if abs_text:
                    batch_abstracts.append(f"Paper ID: {p['id']}\nTitle: {p['title']}\nAbstract: {abs_text}\n")
            
            batch_feed = "\n---\n".join(batch_abstracts)
            try:
                return await chain.ainvoke({"abstracts": batch_feed}, config=config)
            except Exception as e:
                print(f"[Literature] Batch failed: {e}")
                return None
    
        # Run batches in parallel safely
        print(f"[Literature] Starting asyncio.gather for {len(batches)} batches...")
        batch_results = await asyncio.gather(*[process_batch(b) for b in batches], return_exceptions=True)
        print(f"[Literature] asyncio.gather complete. Processing results...")
    
        for i, res in enumerate(batch_results):
            if isinstance(res, Exception):
                print(f"[Literature] Batch {i+1} failed with exception: {res}")
                continue
                
            if res:
                merged_nodes.update(res.nodes)
                merged_edges.extend(res.edges)
                msg = f"[Literature] Batch {i+1}/{len(batches)} success. Found {len(res.nodes)} nodes."
                print(msg)
                await adispatch_custom_event("log", {"message": msg}, config=config)
                
        msg = f"[Literature] Batch extraction complete. Total Nodes: {len(merged_nodes)}"
        print(msg)
        await adispatch_custom_event("log", {"message": msg}, config=config)
        
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
            msg = f"[Literature] Normalizing graph nodes..."
            print(msg)
            await adispatch_custom_event("log", {"message": msg}, config=config)
            
            # with get_openai_callback() as cb:
            concept_graph = await normalize_graph_nodes(concept_graph)
            # print(f"[Literature] Token Usage (Normalization): {cb}")
            
            msg = f"[Literature] Graph normalized. Nodes: {len(concept_graph['nodes'])}, Edges: {len(concept_graph['edges'])}"
            print(msg)
            await adispatch_custom_event("log", {"message": msg}, config=config)
            
        except Exception as e:
            print(f"[Literature] Normalization failed: {e}")
    
        # 5. Densify Graph (Grounded Second Pass)
        try:
            msg = f"[Literature] Densifying graph (Second Pass)..."
            print(msg)
            await adispatch_custom_event("log", {"message": msg}, config=config)
            
            # with get_openai_callback() as cb:
            concept_graph = await densify_graph(concept_graph, papers)
            # print(f"[Literature] Token Usage (Densification): {cb}")
            
            msg = f"[Literature] Graph densified. Nodes: {len(concept_graph['nodes'])}, Edges: {len(concept_graph['edges'])}"
            print(msg)
            await adispatch_custom_event("log", {"message": msg}, config=config)
            
        except Exception as e:
            print(f"[Literature] Densification failed: {e}")
    
        # 6. Graph Analysis (NetworkX) + Final Cleanup
        concept_graph = clean_graph(concept_graph)
        
        graph_insights = analyze_graph(concept_graph)
        try:
            print(f"[Literature] Graph Analysis:\n{graph_insights.encode('utf-8', 'ignore').decode('utf-8')}")
            # Stream the top insights (first line usually) or just a summary
            await adispatch_custom_event("log", {"message": "[Literature] Graph Analysis Complete. Key central nodes identified."}, config=config)
        except:
            print("[Literature] Graph Analysis: (Content hidden due to encoding error)")
        
        # Append insights to context for the next agent
        full_text_context += f"\n\nGraph Analysis Insights:\n{graph_insights}"
        
        # Check minimum quality
        node_count = len(concept_graph.get("nodes", []))
        edge_count = len(concept_graph.get("edges", []))
        if node_count < 5 or edge_count < 3:
             print(f"[Literature] WARNING: Graph too small ({node_count} nodes). Triggering fallback or warning.")
             # In a real system, we might loop back to search with a simpler query.
    
        return {
            "literature": {
                "papers": processed_papers,
                "summary": final_summary, 
                "full_context": full_text_context,
                "web_research": web_results_text
            },
            "concept_graph": concept_graph
        }
    finally:
         CostTracker.get_instance().end_step("literature", "literature_search_and_graph")

def clean_graph(graph_data: dict) -> dict:
    """Removes degree-0 nodes, URL junk, and noise."""
    import networkx as nx
    G = nx.DiGraph()
    for n in graph_data.get("nodes", []):
        G.add_node(n if isinstance(n, str) else n["id"])
    for e in graph_data.get("edges", []):
        G.add_edge(e['source'], e['target'], relation=e.get('relation', 'related'))
        
    # 1. Remove Isolates (Degree 0)
    isolates = list(nx.isolates(G))
    G.remove_nodes_from(isolates)
    
    # 2. Heuristic Filter
    to_remove = []
    for node in G.nodes():
        node_str = str(node).lower()
        if "http" in node_str or "www." in node_str: to_remove.append(node)
        if node_str.isdigit(): to_remove.append(node) # Pure numbers
        if len(node_str) < 3: to_remove.append(node) # Too short
        
    G.remove_nodes_from(to_remove)
    
    new_nodes = list(G.nodes())
    new_edges = [{"source": u, "target": v, "relation": data.get("relation", "related")} for u, v, data in G.edges(data=True)]
    return {"nodes": new_nodes, "edges": new_edges}

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
        msg = f"[Literature] Found {len(new_edges)} new edges in second pass."
        print(msg)
        # await adispatch_custom_event("log", {"message": msg}, config=config) # Helper function needed or direct dispatch if config available
        # Since config is not passed to densify_graph, we can't dispatch events easily here without refactoring.
        # Ideally, we should pass config. For now, let's focus on the main `literature_node` where config IS available.
        graph_data["edges"].extend(new_edges)
        
    return graph_data

async def normalize_graph_nodes(graph_data: dict) -> dict:
    """
    Production-grade Normalization:
    1. Heuristic Clustering (Edit distance, Token overlap) -> Fast, Local
    2. LLM Canonicalization (Cluster -> Canonical Term) -> Accurate, Semantic
    """
    import networkx as nx
    from pydantic import BaseModel
    import difflib
    
    # Ensure inputs are strings
    nodes = [n["id"] if isinstance(n, dict) else n for n in graph_data.get("nodes", [])]
    if not nodes:
        return graph_data

    print(f"[Literature] Normalizing {len(nodes)} nodes using Heuristic Clustering...")

    # --- STEP 1: Heuristic Clustering ---
    # Group terms that look similar (e.g., "p53", "p53 protein")
    clusters = []
    processed = set()
    
    sorted_nodes = sorted(nodes, key=len) # Process short to long
    
    for node in sorted_nodes:
        if node in processed:
            continue
            
        # Initialize cluster with current node
        current_cluster = {node}
        processed.add(node)
        
        remaining_nodes = [n for n in nodes if n not in processed]
        
        # Find close matches
        # 1. Edit distance (Stricter cutoff for short words)
        # 0.9 cutoff avoids "transformer" vs "transformation"
        matches = difflib.get_close_matches(node, remaining_nodes, n=10, cutoff=0.9)
        
        for m in matches:
            current_cluster.add(m)
            processed.add(m)
            
        # 2. Token overlap (Jaccard)
        node_tokens = set(node.lower().split())
        for other in remaining_nodes:
            if other in current_cluster: continue
            
            other_tokens = set(other.lower().split())
            if not node_tokens or not other_tokens: continue

            intersection = node_tokens.intersection(other_tokens)
            union = node_tokens.union(other_tokens)
            if not union: continue
            
            jaccard = len(intersection) / len(union)
            # FIX: Tightened threshold to 0.8 to avoid loose matches
            if jaccard > 0.8: 
                current_cluster.add(other)
                processed.add(other)
                
            # BONUS: Acronym Match (Simple)
            # e.g. "Graph Neural Network (GNN)"
            # Check if one term is an acronym of the other's initials
            # (Skipped for simplicity and safety, but Jaccard handles "GNN" vs "GNNs")

        clusters.append(list(current_cluster))

    print(f"[Literature] Created {len(clusters)} clusters from {len(nodes)} nodes.")

    # --- STEP 2: LLM Canonicalization ---
    # Only send clusters with >1 item or ambiguous terms
    
    class CanonicalMapping(BaseModel):
        cluster_terms: List[str]
        canonical_term: str

    class NormalizationResult(BaseModel):
        mappings: List[CanonicalMapping]

    llm = get_llm(temperature=0.0)
    structured_llm = llm.with_structured_output(NormalizationResult)
    
    final_mapping = {}
    
    # Filter trivial clusters (size 1) - Map to themselves instantly
    trivial_clusters = [c for c in clusters if len(c) == 1]
    complex_clusters = [c for c in clusters if len(c) > 1]
    
    for c in trivial_clusters:
        final_mapping[c[0]] = c[0]
        
    if complex_clusters:
        print(f"[Literature] Sending {len(complex_clusters)} complex clusters to LLM...")
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a scientific terminologist. For each group of terms, identify the single best CANONICAL scientific term. Generalize if appropriate (e.g., 'p53', 'p53 protein' -> 'TP53'). Treat each group INDEPENDENTLY."),
            ("human", "Groups: {groups}")
        ])
        
        # FIX: Reduced chunk size to 3 to prevent cross-contamination
        chunk_size = 3
        for i in range(0, len(complex_clusters), chunk_size):
            batch = complex_clusters[i:i+chunk_size]
            batch_str = "\n".join([f"- {c}" for c in batch])
            
            try:
                res = await (prompt | structured_llm).ainvoke({"groups": batch_str})
                for m in res.mappings:
                    for term in m.cluster_terms:
                        final_mapping[term] = m.canonical_term
            except Exception as e:
                print(f"[Literature] Normalization error: {e}")
                for c in batch:
                    for term in c:
                        final_mapping[term] = c[0]

    # --- STEP 3: Rebuild Graph ---
    G = nx.DiGraph()
    for node in nodes:
        # If somehow missed (edge case), map to self
        canon = final_mapping.get(node, node)
        G.add_node(canon)
        
    for edge in graph_data.get("edges", []):
        src = final_mapping.get(edge['source'], edge['source'])
        tgt = final_mapping.get(edge['target'], edge['target'])
        if src != tgt: # Remove self-loops
             G.add_edge(src, tgt, relation=edge.get('relation', 'related'))
             
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
