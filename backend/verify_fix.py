
import httpx
import asyncio
import json

BASE_URL = "http://127.0.0.1:8010"

async def verify():
    print("Waiting for server to be ready...")
    async with httpx.AsyncClient() as client:
        # Retry loop for health check
        for i in range(10):
            try:
                resp = await client.get(f"{BASE_URL}/health")
                if resp.status_code == 200:
                    print("Server is ready!")
                    break
            except:
                pass
            await asyncio.sleep(1)
        
        # Test 1: The problematic query
        query = "Find mechanism connecting sleep deprivation to Alzheimer's"
        print(f"\n[Test 1] Complex Query: '{query}'")
        resp = await client.post(f"{BASE_URL}/search_papers", json={"query": query, "max_papers": 5})
        
        if resp.status_code != 200:
            print(f"FAILED: Status {resp.status_code}")
            print(resp.text)
            return

        data = resp.json()
        papers = data.get("papers", [])
        print(f"Count: {len(papers)}")
        if len(papers) > 0:
            print("SUCCESS: Papers returned!")
            print(f"First Paper: {papers[0]['title']}")
            # Check for fallback logs in server output (manual check) or just trust result
        else:
            print("FAILURE: returned 0 papers.")

        # Test 2: Control Simple Query
        query_simple = "CRISPR Cas9"
        print(f"\n[Test 2] Simple Query: '{query_simple}'")
        resp = await client.post(f"{BASE_URL}/search_papers", json={"query": query_simple, "max_papers": 5})
        papers = resp.json().get("papers", [])
        print(f"Count: {len(papers)}")

if __name__ == "__main__":
    asyncio.run(verify())
