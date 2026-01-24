
import httpx
import json
import asyncio

async def test_run():
    url = "http://127.0.0.1:8005/run_stream"
    payload = {
        "query": "Should be ignored due to user docs",
        "documents": [
            "User provided abstract 1. This paper discusses the importance of user inputs.",
            "User provided abstract 2. This is another manual entry.",
            "User provided abstract 3. Third manual entry to trigger skip logic."
        ],
        "mock": False
    }

    print(f"Connecting to {url}...")
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream("POST", url, json=payload) as response:
                if response.status_code != 200:
                    print(f"Error: {response.status_code} - {await response.aread()}")
                    return

                print("Connected! Listening for events...")
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break
                        
                        try:
                            event = json.loads(data_str)
                            msg_type = event.get("type")
                            
                            if msg_type == "log":
                                print(f"[LOG] {event['data']}")
                            elif msg_type == "result":
                                data = event.get("data", {})
                                if "literature" in data:
                                    lit = data["literature"]
                                    papers = lit.get("papers", {})
                                    print(f"\n[Verification] Found {len(papers)} papers.")
                                    for pid, p in papers.items():
                                        print(f" - {p.get('title')}: {p.get('abstract')[:50]}...")
                                    
                                    # Verify user docs are present
                                    user_docs_found = any("User Input" in p.get("title", "") for p in papers.values())
                                    if user_docs_found:
                                        print("\nSUCCESS: User provided documents were used!")
                                    else:
                                        print("\nFAILURE: User documents NOT found.")
                                    return # Exit early after checking literature
                        except json.JSONDecodeError:
                            pass
    except Exception as e:
        print(f"Connection failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_run())
