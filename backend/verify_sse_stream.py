
import requests
import json
import sys

def verify_stream():
    url = "http://127.0.0.1:8010/run_stream"
    payload = {
        "query": "Test Heartbeat Connectivity",
        "goal": "discover",
        "lens": "physics",
        "speculation": "high",
        "run_experiments": False,
        "documents": []
    }
    
    headers = {"Accept-Encoding": "identity"}
    
    print(f"Connecting to {url}...")
    
    try:
        response = requests.post(url, json=payload, headers=headers, stream=True, timeout=120)
        
        try:
            print(f"Response Headers: {response.headers}", flush=True)
            response.raise_for_status()
            print("Connected! Listening for events (byte-stream)...", flush=True)
            
            # iter_content(chunk_size=None) will return chunks as they arrive
            buffer = ""
            for chunk in response.iter_content(chunk_size=None):
                if chunk:
                    text_data = chunk.decode('utf-8')
                    buffer += text_data
                    
                    # Split by double newline to get events
                    while "\n\n" in buffer:
                        event_text, buffer = buffer.split("\n\n", 1)
                        if event_text.startswith("data: "):
                            json_str = event_text[6:].strip()
                            if json_str == "[DONE]":
                                print("\n[STREAM COMPLETE]")
                                break
                                
                            try:
                                data = json.loads(json_str)
                                event_type = data.get("type")
                                
                                if event_type == "result":
                                    result_data = data.get("data", {})
                                    keys = list(result_data.keys())
                                    print(f"\n[VICTORY] Received FINAL RESULT with keys: {keys}")
                                    
                                    # Verification Checks
                                    missing = []
                                    if "hypotheses" not in keys: missing.append("hypotheses")
                                    if "concept_graph" not in keys: missing.append("concept_graph")
                                    
                                    if missing:
                                        print(f"[FAILURE] Missing critical keys: {missing}")
                                    else:
                                        print(f"[SUCCESS] Payload contains all required fields.")
                                        print(f" - Hypotheses count: {len(result_data.get('hypotheses', []))}")
                                        # Handle graph structure variations
                                        graph = result_data.get('concept_graph', {})
                                        nodes = graph.get('nodes', [])
                                        print(f" - Graph Nodes: {len(nodes)}")
                                        
                                    return # Stop after finding result
                                    
                                elif event_type == "activity":
                                    sys.stdout.write(".")
                                    sys.stdout.flush()
                                elif event_type == "error":
                                    print(f"\n[ERROR EVENT] {data.get('data')}")
                                    
                            except json.JSONDecodeError:
                                pass
                                
        except requests.exceptions.HTTPError as e:
            print(f"HTTP Error: {e}")
            err_msg = f"Server Response: {response.text}"
            print(err_msg)
            
    except Exception as e:
        print(f"Connection failed: {e}")

if __name__ == "__main__":
    verify_stream()
