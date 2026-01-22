
import asyncio
from app.tools.openalex import search_papers

async def test_search():
    query = "Find mechanism connecting sleep deprivation to Alzheimer's"
    print(f"Searching for: '{query}'")
    results = await search_papers(query, limit=10)
    print(f"Found {len(results)} papers.")
    if len(results) == 0:
        print("CONFIRMED: Raw natural language query returns 0 results.")
    else:
        print("DISPROVED: Raw query returned results.")
        for p in results:
            print(f"- {p['title']}")

    # Control test
    query_simple = "sleep deprivation Alzheimer's mechanism"
    print(f"\nSearching for keywords: '{query_simple}'")
    results_simple = await search_papers(query_simple, limit=10)
    print(f"Found {len(results_simple)} papers.")

if __name__ == "__main__":
    asyncio.run(test_search())
