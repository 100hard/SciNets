
import requests
import json
import sys
import time

# Configuration
API_URL = "http://127.0.0.1:8005/run_stream"
QUERY = "Design a novel protocol for Quantum Key Distribution (QKD) using entangled photons that resists satellite-based attacks."
HEADERS = {"Accept-Encoding": "identity"}

def run_full_flow():
    print(f"--- Starting End-to-End Verification (Debug Mode) ---")
    print(f"Query: {QUERY}")
    print(f"Target: {API_URL}")
    
    # 1. Start Initial Run
    print("\n[Phase 1] Initial Discovery Run...")
    payload = {
        "query": QUERY,
        "goal": "discover",
        "lens": "physics",
        "speculation": "high",
        "run_experiments": True, # Test the new Agent!
        "documents": []
    }
    
    thread_id = None
    interrupt_data = None
    
    try:
        response = requests.post(API_URL, json=payload, headers=HEADERS, stream=True, timeout=600)
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                # Handle possible multiple data: prefixes in one chunk
                parts = decoded_line.split("data: ")
                for part in parts:
                    if not part.strip(): continue
                    json_str = part.strip()
                    if json_str == "[DONE]": break
                    
                    try:
                        data = json.loads(json_str)
                        evt_type = data.get("type")
                        
                        if evt_type == "activity":
                            sys.stdout.write(f"\r[Activity] {data.get('data', {}).get('action', '')[:80]}...")
                            if data.get("thread_id") and not thread_id:
                                thread_id = data.get("thread_id")
                                print(f"\n[INFO] Thread ID captured: {thread_id}")
                                
                        elif evt_type == "interrupt":
                            print(f"\n\n[INTERRUPT RECEIVED] Graph paused for feedback!")
                            interrupt_data = data.get("data")
                            thread_id = interrupt_data.get("thread_id") # Ensure we have it
                            break # Stop reading stream, we need to send feedback
                            
                        elif evt_type == "result":
                             # Interim result
                             pass

                        else:
                            # Debug: Print other events (like logs)
                            # print(f"\n[DEBUG] {evt_type}: {str(data)[:100]}")
                            pass
                             
                    except Exception as e:
                        print(f"[Warn] Parse error: {e}")
                        pass
    except Exception as e:
        print(f"\n[ERROR] Phase 1 failed: {e}")
        return

    if not interrupt_data or not thread_id:
        print("\n\n[FAILURE] Did not receive interrupt signal. Flow incomplete.")
        return

    # 2. Send Feedback (Resume)
    print(f"\n\n[Phase 2] Resuming with Feedback for Thread {thread_id}...")
    feedback_payload = {
        "query": QUERY, # Must match or be ignored
        "thread_id": thread_id,
        "feedback": "Ensure the protocol considers Low Earth Orbit (LEO) satellite constraints."
    }
    
    final_hypotheses = []
    final_experiments = []
    
    try:
        response = requests.post(API_URL, json=feedback_payload, headers=HEADERS, stream=True, timeout=300)
        response.raise_for_status()
        
        for line in response.iter_lines():
             if line:
                decoded_line = line.decode('utf-8')
                if decoded_line.startswith("data: "):
                    json_str = decoded_line[6:].strip()
                    if json_str == "[DONE]": break
                    
                    try:
                        data = json.loads(json_str)
                        evt_type = data.get("type")
                        
                        if evt_type == "activity":
                             sys.stdout.write(f"\r[Activity] {data.get('data', {}).get('action', '')[:80]}...")
                        
                        elif evt_type == "result":
                            res_data = data.get("data", {})
                            if "hypotheses" in res_data: 
                                final_hypotheses = res_data["hypotheses"]
                                for h in final_hypotheses:
                                    if h.get("evidence_summary"):
                                         print(f"\n[Evidence Summary] H-{h.get('id', '?')[:4]}: {h['evidence_summary']}")
                                         
                            if "experiments" in res_data: final_experiments = res_data["experiments"]
                            
                            if "critique" in res_data:
                                print("\n[Critique Result]")
                                print(json.dumps(res_data["critique"], indent=2))
                            
                    except: pass
                    
    except Exception as e:
        print(f"\n[ERROR] Phase 2 failed: {e}")
        return

    # 3. Final Report
    print("\n\n--- Verification Complete ---")
    print(f"Hypotheses Generated: {len(final_hypotheses)}")
    print(f"Experiments Run: {len(final_experiments)}")
    
    if final_experiments:
        print("\n[Experiment Analysis]")
        for i, exp in enumerate(final_experiments):
            print(f"  Exp {i+1}:")
            print(f"    Code Length: {len(exp.get('code_snippet', ''))} chars")
            print(f"    Metrics: {exp.get('metrics')}")
            if exp.get('code_snippet', '').startswith("# SIMULATION"):
                print("    Type: SIMULATION (Agent decided not to run real code?)")
            else:
                print("    Type: REAL CODE EXECUTION")

    if len(final_hypotheses) > 0 and len(final_experiments) > 0:
        print("\n[SUCCESS] End-to-End Flow Verified!")
    else:
        print("\n[WARNING] Flow finished but missing data.")

if __name__ == "__main__":
    run_full_flow()
