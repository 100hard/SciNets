
import asyncio
import os
import time
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv

load_dotenv()

# --- Mock Data (3 distinct topics) ---
abstracts = [
    "Abstract 1: The study of CRISPR-Cas9 verifies its efficiency in gene editing. Off-target effects remain a primary concern for clinical application. New variants like Cas12a show improved specificity.",
    "Abstract 2: Dark matter candidates include WIMPs and axions. Recent Xenon1T experiments rule out certain WIMP cross-sections. Axion helioscopes are the next frontier.",
    "Abstract 3: Perovskite solar cells verify efficiencies over 25%. Stability under moisture is the main bottleneck. Encapsulation with graphene layers shows promise.",
]

# --- Schema ---
class Edge(BaseModel):
    source: str
    target: str
    relation: str

class ConceptGraph(BaseModel):
    nodes: List[str]
    edges: List[Edge]

# --- Setup ---
llm = ChatOpenAI(model="gpt-5-mini", api_key=os.getenv("OPENAI_API_KEY"))
structured_llm = llm.with_structured_output(ConceptGraph)

async def run_batch():
    print("\n--- Running BATCH Extraction (Fan-In) ---")
    start = time.time()
    
    combined = "\n\n".join([f"Abstract {i+1}: {txt}" for i, txt in enumerate(abstracts)])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Extract a unified ConceptGraph from these abstracts. Capture all key entities."),
        ("human", "{text}")
    ])
    chain = prompt | structured_llm
    
    try:
        res = await chain.ainvoke({"text": combined})
        duration = time.time() - start
        print(f"Batch Time: {duration:.2f}s")
        return res
    except Exception as e:
        print(f"Batch Failed: {e}")
        return None

async def run_individual():
    print("\n--- Running INDIVIDUAL Extraction (Sequential) ---")
    start = time.time()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Extract a ConceptGraph from this abstract."),
        ("human", "{text}")
    ])
    chain = prompt | structured_llm
    
    all_nodes = set()
    all_edges = []
    
    for i, txt in enumerate(abstracts):
        print(f"Processing {i+1}/5...", end=" ", flush=True)
        try:
            res = await chain.ainvoke({"text": txt})
            if res:
                all_nodes.update(res.nodes)
                all_edges.extend(res.edges)
            print("Done.")
        except Exception as e:
            print(f"Error: {e}")
            
    duration = time.time() - start
    print(f"Individual Time: {duration:.2f}s")
    
    return ConceptGraph(nodes=list(all_nodes), edges=all_edges)

async def main():
    print("Starting Quality Benchmark (5 Abstracts)...")
    
    # Run Individual First to establish baseline
    ind_res = await run_individual()
    
    # Run Batch
    batch_res = await run_batch()
    
    print("\n=== RESULTS ===")
    if ind_res:
        print(f"INDIVIDUAL Mode: {len(ind_res.nodes)} Nodes, {len(ind_res.edges)} Edges")
        # print(f"Nodes: {ind_res.nodes}")
    
    if batch_res:
        print(f"BATCH Mode:      {len(batch_res.nodes)} Nodes, {len(batch_res.edges)} Edges")
        # print(f"Nodes: {batch_res.nodes}")
        
    if ind_res and batch_res:
        node_diff = len(ind_res.nodes) - len(batch_res.nodes)
        edge_diff = len(ind_res.edges) - len(batch_res.edges)
        print(f"\nDifference (Ind - Batch): {node_diff} Nodes, {edge_diff} Edges")
        
        # Calculate overlap (Jaccard-ish)
        ind_set = set([n.lower() for n in ind_res.nodes])
        batch_set = set([n.lower() for n in batch_res.nodes])
        common = ind_set.intersection(batch_set)
        batch_unique = batch_set - ind_set
        ind_unique = ind_set - batch_set
        
        print(f"Common Concepts: {len(common)}")
        print(f"Unique to Individual: {len(ind_unique)} (Possible Detail Loss)")
        print(f"Unique to Batch: {len(batch_unique)} (Possible Synthesis/Hallucination)")

if __name__ == "__main__":
    asyncio.run(main())
