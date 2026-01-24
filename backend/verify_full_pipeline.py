
import requests
import json
import time

BASE_URL = "http://localhost:8005"

def run_pipeline():
    print("--- 0. Checking Server Health ---", flush=True)
    try:
        ping = requests.get(f"{BASE_URL}/docs", timeout=5)
        if ping.status_code == 200:
            print("Server is UP.", flush=True)
        else:
            print(f"Server returned {ping.status_code}.", flush=True)
    except Exception as e:
         print(f"Server unreachable: {e}", flush=True)
         return

    print("--- 1. Starting Discovery Session (Preview Mode) ---", flush=True)
    
    payload = {
        "query": "What are new mechanisms for fatigue in long covid?",
        "goal": "discover",
        "timeline": "recent",
        "speculation": "medium",
        "run_experiments": False,
        "max_papers": 1 # Minimal for speed
    }
    
    thread_id = None
    hypotheses = []
    
    try:
        # Start SSE Stream (Raw Line Reader)
        resp = requests.post(f"{BASE_URL}/run_stream", json=payload, stream=True)
        
        current_event = None
        for line in resp.iter_lines():
            if not line: continue
            decoded_line = line.decode('utf-8')
            print(f"[DEBUG] {decoded_line}", flush=True)
            
            if decoded_line.startswith("data:"):
                data_str = decoded_line[5:].strip()
                try:
                    payload = json.loads(data_str)
                    # Support both explicit event lines AND inner type field
                    event_type = current_event or payload.get("type")
                    
                    if event_type == "result":
                        data = payload.get("data", payload)
                        if "hypotheses" in data and len(data["hypotheses"]) > 0:
                            hypotheses = data["hypotheses"]
                            print(f"[Preview] Got {len(hypotheses)} hypotheses", flush=True)
                            for i, h in enumerate(hypotheses[:4]): 
                                print(f"  [{i+1}] {h['text'][:50]}... | Rec: {h.get('is_recommended')} | Rank: {h.get('rank')}", flush=True)
                                
                    if event_type == "interrupt":
                        print("[Interrupt Received] - Pipeline Paused for Selection", flush=True)
                        resp.close() 
                        break 
                except:
                    pass 
                
    except Exception as e:
        print(f"Error in Preview Phase: {e}", flush=True)

    if not thread_id or len(hypotheses) == 0:
        print("FAILED: No thread_id or hypotheses generated.", flush=True)
        return

    print("\n--- 2. Simulating User Selection ---", flush=True)
    selected_id = hypotheses[0]["id"]
    print(f"Selected ID: {selected_id}", flush=True)
    
    time.sleep(1)
    
    print("\n--- 3. Resuming Discovery Stream (Deep Mode) ---", flush=True)
    resume_payload = {
        "thread_id": thread_id,
        "selected_hypothesis_ids": [selected_id],
        "goal": "discover",
        "timeline": "recent",
        "query": "" 
    }
    
    deep_hypotheses = []
    evidence_items_count = 0
    
    try:
        resp = requests.post(f"{BASE_URL}/run_stream", json=resume_payload, stream=True)
        if resp.status_code != 200:
             print(f"FAILED: Resume request returned {resp.status_code}: {resp.text}", flush=True)
             return

        current_event = None
        for line in resp.iter_lines():
            if not line: continue
            decoded_line = line.decode('utf-8')
            
            if decoded_line.startswith("event:"):
                current_event = decoded_line[6:].strip()
                
            if decoded_line.startswith("data:"):
                data_str = decoded_line[5:].strip()
                
                # --- PROCESS EVENT ---
                if current_event == "activity":
                     # Just print first 50 chars to show liveness
                    # print(f"[Activity] {data_str[:50]}...", flush=True)
                    pass
                
                if current_event == "log":
                    # Check for evidence logs
                    data = json.loads(data_str)
                    msg = data.get("message", "")
                    if "[Evidence]" in msg:
                        print(f"[Log] {msg}", flush=True)

                if current_event == "result":
                    data = json.loads(data_str)
                    if "hypotheses" in data:
                        deep_hypotheses = data["hypotheses"]
                        # Check evidence on first one
                        h = deep_hypotheses[0]
                        if h.get("evidence"):
                            new_count = len(h["evidence"])
                            if new_count > evidence_items_count:
                                evidence_items_count = new_count
                                print(f"[Deep] Hypothesis evidence count updated: {evidence_items_count}", flush=True)

                if current_event == "done": # Note: backend emits event: done
                    print("[DONE Received] - Pipeline Complete", flush=True)
                    break 
                    
    except Exception as e:
        print(f"Error in Resume Phase: {e}", flush=True)
        return

    print("\n--- Verification Results ---", flush=True)
    if len(deep_hypotheses) == 1 and deep_hypotheses[0]["id"] == selected_id:
        print("SUCCESS: Persistence Worked! Only selected hypothesis returned.", flush=True)
    else:
        print(f"FAILURE: Expected 1 hypothesis, got {len(deep_hypotheses)}", flush=True)

    if evidence_items_count > 0:
         print(f"SUCCESS: Evidence found ({evidence_items_count} items).", flush=True)
    else:
         print("FAILURE: No evidence found.", flush=True)

if __name__ == "__main__":
    run_pipeline()
