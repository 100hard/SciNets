
import requests
import json
import sys
import time

# Configuration
BASE_URL = "http://127.0.0.1:8005"
DISCOVERY_URL = f"{BASE_URL}/run_stream"
EXPERIMENT_URL = f"{BASE_URL}/experiment_stream"
HEADERS = {"Accept-Encoding": "identity"}
QUERY = "Investigate the impact of microplastics on soil microbiome diversity."

def run_discovery():
    print(f"--- [Phase 1] Starting Discovery for: {QUERY} ---")
    payload = {
        "query": QUERY,
        "goal": "discover",
        "lens": "biology",
        "speculation": "high",
        "run_experiments": False, # Just generate plans first
        "documents": []
    }
    
    thread_id = None
    experiment_plans = []
    
    try:
        response = requests.post(DISCOVERY_URL, json=payload, headers=HEADERS, stream=True, timeout=600)
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
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
                        
                        elif evt_type == "result":
                             plans = data.get("data", {}).get("experiment_plans", [])
                             if plans:
                                 experiment_plans = plans
                                 print(f"\n[Result] Found {len(plans)} experiment plans.")

                    except Exception as e:
                        pass
    except Exception as e:
        print(f"\n[ERROR] Discovery failed: {e}")
        return None, None

    return thread_id, experiment_plans

def run_experiment(thread_id, plan_id, hypothesis_id):
    print(f"\n\n--- [Phase 2] Running Experiment Plan: {plan_id} ---")
    payload = {
        "thread_id": thread_id,
        "plan_id": plan_id,
        "hypothesis_id": hypothesis_id
    }
    
    try:
        response = requests.post(EXPERIMENT_URL, json=payload, headers=HEADERS, stream=True, timeout=600)
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                parts = decoded_line.split("data: ")
                for part in parts:
                    if not part.strip(): continue
                    json_str = part.strip()
                    if json_str == "[DONE]": break
                    
                    try:
                        data = json.loads(json_str)
                        evt_type = data.get("type")
                        
                        if evt_type == "activity":
                             sys.stdout.write(f"\r[Exp Activity] {data.get('data', {}).get('action', '')[:80]}...")
                        
                        elif evt_type == "log":
                             print(f"\n[Log] {data.get('data', '')}")

                        elif evt_type == "result":
                             experiments = data.get("data", {}).get("experiments", [])
                             target_exp = next((e for e in experiments if e.get("id") == plan_id), None)
                             if target_exp:
                                 print(f"\n[SUCCESS] Experiment Completed!")
                                 print(f"Metrics: {target_exp.get('metrics')}")
                                 if target_exp.get("plot_base64"):
                                     print(f"Plot: [Base64 Data Present]")
                                 if target_exp.get("code_snippet"):
                                     print(f"Code: [Code Snippet Present]")
                                 return True

                    except Exception as e:
                        pass
    except Exception as e:
         print(f"\n[ERROR] Experiment Run failed: {e}")
         return False
    
    return False

if __name__ == "__main__":
    thread_id, plans = run_discovery()
    
    if thread_id and plans:
        print(f"\n[INFO] Discovery complete. Thread: {thread_id}")
        # Select first plan
        target_plan = plans[0]
        print(f"Selected Plan: {target_plan['goal']} (ID: {target_plan['id']})")
        
        success = run_experiment(thread_id, target_plan['id'], target_plan['hypothesis_id'])
        if success:
            print("\n*** VERIFICATION PASSED: Full Pipeline works! ***")
        else:
            print("\n*** VERIFICATION FAILED: Experiment did not complete. ***")
    else:
        print("\n*** VERIFICATION FAILED: Discovery did not yield plans. ***")
