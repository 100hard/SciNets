
import requests
import json
import time
import sys
import argparse
from datetime import datetime

# Fix Windows Unicode Output
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass

# Colors (Disabled)
GREEN = ""
RED = ""
YELLOW = ""
BLUE = ""
RESET = ""

API_URL = "http://127.0.0.1:8005"

class PipelineVerifier:
    def __init__(self, query="Applications of graphene", mock=False, timeout=300):
        self.query = query
        self.mock = mock
        self.timeout = timeout
        self.thread_id = None
        self.experiment_plans = []
        self.captured_hypotheses = []
        self.start_time = time.time()
        
    def log(self, msg, color=RESET):
        print(f"{color}{msg}{RESET}")

    def run_phase_1_discovery(self):
        self.log(f"\n[PHASE 1] Starting Discovery: '{self.query}'", BLUE)
        payload = {
            "query": self.query,
            "goal": "discover",
            "speculation": "medium",
            "run_experiments": False, # Explicitly stop after proposal
            "mock": self.mock
        }
        
        try:
            response = requests.post(f"{API_URL}/run_stream", json=payload, stream=True, timeout=self.timeout)
            response.raise_for_status()
            
            self.log(f"Response Headers: {response.headers}", YELLOW)
            for line in response.iter_lines():
                if not line: continue
                decoded = line.decode('utf-8')
                print(f"RAW_LINE: {repr(decoded)}") # DEBUG with REPR
                if not decoded.startswith("data: "):
                     print("  Skipping (No data: prefix)")
                     continue
                
                data_str = decoded[6:].strip()
                if data_str == "[DONE]":
                    self.log("✓ Phase 1 Received [DONE]", GREEN)
                    break
                    
                try:
                    data = json.loads(data_str)
                    print(f"DEBUG_RAW: {data}")
                    evt_type = data.get("type", "")
                    
                    if evt_type == "error":
                         self.log(f"[SERVER ERROR] {data.get('data')}", RED)
                    
                    if evt_type == "activity":
                        # Capture Thread ID
                        tid = data.get("thread_id")
                        if tid and not self.thread_id:
                            self.thread_id = tid
                            self.log(f"✓ Captured Thread ID: {self.thread_id}", GREEN)
                            
                        action = data.get("data", {}).get("action", "")
                        # print(f"  [Activity] {action}")

                    if evt_type == "log":
                        print(f"  [Log] {data.get('data')}")
                        pass
                        
                    if evt_type == "result":
                        res = data.get("data", {})
                        keys = list(res.keys())
                        self.log(f"  [Result Event] Keys: {keys}", YELLOW)
                        if "experiment_plans" in res:
                            self.experiment_plans = res["experiment_plans"]
                            self.log(f"✓ Captured {len(self.experiment_plans)} Experiment Plans", GREEN)
                        if "hypotheses" in res:
                            self.captured_hypotheses = res["hypotheses"]

                except json.JSONDecodeError:
                    pass
                    
        except Exception as e:
            self.log(f"[FAIL] Phase 1 Error: {e}", RED)
            return False
            
        return True

    def run_phase_2_experiment(self):
        if not self.thread_id:
            self.log("[FAIL] Cannot run Phase 2: No Thread ID", RED)
            return False
        if not self.experiment_plans:
            self.log("[FAIL] Cannot run Phase 2: No Experiment Plans", RED)
            return False
            
        # Select first plan
        plan = self.experiment_plans[0]
        plan_id = plan.get("id")
        hypothesis_id = plan.get("hypothesis_id")
        
        self.log(f"\n[PHASE 2] Starting Experiment: Plan {plan_id}", BLUE)
        
        payload = {
            "thread_id": self.thread_id,
            "plan_id": plan_id,
            "hypothesis_id": hypothesis_id
        }
        
        experiment_started = False
        experiment_logs_seen = False
        
        try:
            # Use the DEDICATED experiment stream endpoint
            response = requests.post(f"{API_URL}/experiment_stream", json=payload, stream=True, timeout=self.timeout)
            response.raise_for_status()
            
            for line in response.iter_lines():
                if not line: continue
                decoded = line.decode('utf-8')
                if not decoded.startswith("data: "): continue
                
                data_str = decoded[6:].strip()
                if data_str == "[DONE]":
                    self.log("✓ Phase 2 Received [DONE]", GREEN)
                    break
                    
                try:
                    data = json.loads(data_str)
                    evt_type = data.get("type", "")
                    
                    if evt_type == "log":
                        msg = data.get("data", "")
                        if isinstance(msg, dict):
                            msg = msg.get("message", str(msg))
                        
                        # print(f"  [Exp Log] {msg}") # Optional: Uncomment to see logs
                        
                        if "[Experiment]" in msg or "Executing PLAN" in msg or "MOCK MODE" in msg:
                            experiment_logs_seen = True
                            
                except json.JSONDecodeError:
                    pass
                    
        except Exception as e:
             self.log(f"[FAIL] Phase 2 Error: {e}", RED)
             return False
             
        if experiment_logs_seen:
            self.log("✓ Experiment Logs Detected (Re-Entry Successful)", GREEN)
            return True
        else:
            self.log("✗ No Experiment Logs detected (Resume Failed?)", RED)
            return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true", help="Run use mock mode")
    args = parser.parse_args()
    
    verifier = PipelineVerifier(mock=args.mock)
    
    if not verifier.run_phase_1_discovery():
        sys.exit(1)
        
    print(f"{YELLOW}--- Pausing briefly before Phase 2 ---{RESET}")
    time.sleep(1)
    
    if not verifier.run_phase_2_experiment():
        sys.exit(1)
        
    print(f"\n{GREEN}PASSED ALL CHECKS (2-Phase Flow Verified){RESET}")
    sys.exit(0)

if __name__ == "__main__":
    main()
