
import asyncio
import os
import sys
from dotenv import load_dotenv

# Add backend to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
load_dotenv()

from app.state import DiscoveryState, Hypothesis
from app.agents.orchestrator import plan_node
from app.agents.literature import literature_node
from app.agents.hypothesis import hypothesis_node
from app.agents.evidence import evidence_node
from app.agents.experiment import experiment_node
from app.logging_config import setup_logging

# Init logging (will pick up LOG_FILE from dev.py debug)
setup_logging(log_level=os.getenv("LOG_LEVEL", "INFO"))

async def run_chain():
    print("--- Reproducing Crash ---")
    
    # 0. Init
    state = DiscoveryState(
        user_query="Test Query",
        run_experiments=True
    )
    print("[0] State Initialized.")


    # VALIDATION HELPER
    def validate_state(s, step_name):
        try:
            dump = s.model_dump()
            DiscoveryState(**dump)
            print(f"[{step_name}] Serialization Check PASS")
        except Exception as e:
            print(f"[{step_name}] Serialization Check FAILED: {e}")
            sys.exit(1)

    # 1. Plan
    try:
        print("[1] Running Plan Node...")
        update = await plan_node(state)
        if "plan" in update: state.plan = update["plan"]
        if "domain_tags" in update: state.domain_tags = update["domain_tags"]
        print("[1] Success.")
        validate_state(state, "1-Plan")
    except Exception as e:
        print(f"[1] FAILED: {e}")
        return

    # 2. Literature
    print("[2] Running Literature Node (Mock)...")
    state.literature = {"summary": "Literature summary."}
    validate_state(state, "2-Literature")

    # 3. Hypothesis
    try:
        print("[3] Running Hypothesis Node...")
        update = await hypothesis_node(state)
        # Simulate LangGraph update logic
        if "hypotheses" in update: 
            state.hypotheses = update["hypotheses"]
        
        if "selected_hypothesis_id" in update:
            state.selected_hypothesis_id = update["selected_hypothesis_id"]
            
        print("[3] Success.")
        validate_state(state, "3-Hypothesis")
    except Exception as e:
        print(f"[3] FAILED: {e}")
        import traceback
        traceback.print_exc()
        return

    # 4. Evidence
    try:
        print("[4] Running Evidence Node...")
        update = await evidence_node(state)
        if "hypotheses" in update:
             state.hypotheses = update["hypotheses"]
        print("[4] Success.")
        validate_state(state, "4-Evidence")
    except Exception as e:
        print(f"[4] FAILED: {e}")
        traceback.print_exc()
        return

    # 5. Experiment
    try:
        print("[5] Running Experiment Node...")
        update = await experiment_node(state)
        if "experiments" in update:
            state.experiments = update["experiments"]
        print("[5] Success.")
        validate_state(state, "5-Experiment")
    except Exception as e:
        print(f"[5] FAILED: {e}")
        traceback.print_exc()
        return
        
    print("\n[SUCCESS] All nodes ran without creating invalid state.")

if __name__ == "__main__":
    asyncio.run(run_chain())
