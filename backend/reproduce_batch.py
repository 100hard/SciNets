
import asyncio
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv

load_dotenv()

# Simplified Concept Graph for testing
class Edge(BaseModel):
    source: str
    target: str
    relation: str

class ConceptGraph(BaseModel):
    nodes: List[str]
    edges: List[Edge]

async def test_batch_extraction():
    print("Testing Batch Extraction with gpt-5-mini...")
    
    # Mock Abstrcts
    abstracts = [
        "Paper 1: Warp drives compress space-time. Negative energy density is required.",
        "Paper 2: Alcubierre metric allows FTL travel. Exotic matter is a challenge.",
        "Paper 3: Quantum vacuum fluctuations might substitute negative mass.",
    ]
    combined_feed = "\n---\n".join(abstracts)
    
    llm = ChatOpenAI(model="gpt-5-mini", api_key=os.getenv("OPENAI_API_KEY"))
    structured_llm = llm.with_structured_output(ConceptGraph)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Extract a concept graph from these abstracts."),
        ("human", "Here are the abstracts:\n\n{abstracts}")
    ])
    chain = prompt | structured_llm
    
    print("Invoking Batch Call...")
    try:
        result = await chain.ainvoke({"abstracts": combined_feed})
        print("Success!")
        print(f"Nodes: {result.nodes}")
        print(f"Edges: {len(result.edges)}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_batch_extraction())
