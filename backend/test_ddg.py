import asyncio
from duckduckgo_search import DDGS
import json

def test_search():
    print("Testing DDGS...")
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text("intermittent fasting cognition", max_results=3))
            print(f"Found {len(results)} results")
            for r in results:
                print(r.get("title"))
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_search()
