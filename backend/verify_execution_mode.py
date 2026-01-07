
import asyncio
import sys
import os
import uuid

# Setup path to import app modules
# Setup path to include 'backend' folder so 'app' module execution works
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir) # SciNetsV2 root
sys.path.append(os.path.join(parent_dir, 'backend'))

from app.state import DiscoveryState, ExperimentPlan, Hypothesis, EvidenceItem
from app.agents.experiment import experiment_node
from langchain_core.runnables import RunnableConfig

async def test_execution_mode():
    print("=== VERIFYING EXPERIMENT EXECUTION MODE ===")
    
    # 1. Create Mock Input
    hypo_id = str(uuid.uuid4())
    plan_id = str(uuid.uuid4())
    
    # A hypothesis that is easy to simulate
    mock_hypo = Hypothesis(
        id=hypo_id,
        text="Increasing sample size reduces standard error of the mean.",
        domain_tags=["stats", "math"],
        novelty_score=0.1,
        feasibility_score=1.0,
        testability_score=1.0, 
        evidence=[]
    )
    
    # A clear, executable plan
    mock_plan = ExperimentPlan(
        id=plan_id,
        hypothesis_id=hypo_id,
        type="measurement",
        goal="Demonstrate CLT by calculating SEM for N=10 vs N=1000",
        method="Generate random normal distribution samples using numpy. Calculate std/sqrt(N).",
        metrics=["sem_small", "sem_large", "ratio"],
        cost_estimate="low"
    )
    
    # State CONFIGURATION for EXECUTION
    state = DiscoveryState(
        user_query="Verification Test",
        hypotheses=[mock_hypo],
        experiment_plans=[mock_plan],
        selected_hypothesis_id=hypo_id,
        selected_experiment_plan_id=plan_id, # TARGET PLAN
        run_experiments=True,                  # CRITICAL: Execution Mode
        mock=False                             # Real LLM + Real Code Execution
    )
    
    config = RunnableConfig(configurable={"thread_id": "verify-exec"})
    
    print(f"Plan Goal: {mock_plan.goal}")
    print("Running Experiment Agent in EXECUTION MODE...")
    
    # 2. Run Agent
    # This triggers the LLM to write code -> LocalExecutor to run it -> Return result
    result = await experiment_node(state, config)
    
    print("\n=== RESULT ===")
    experiments = result.get("experiments", [])
    
    if not experiments:
        print("FAIL: No experiments returned.")
        return
        
    exp = experiments[0]
    print(f"Status: {exp.status}")
    print(f"Code Length: {len(exp.code)} chars")
    print(f"Metrics: {exp.metrics}")
    print(f"Result Summary: {exp.result_summary}")
    
    if exp.status == "completed" and "sem_small" in exp.metrics:
        print("\n✅ SUCCESS: Code generated and executed successfully.")
    else:
        print("\n❌ FAILURE: Experiment outcome invalid.")

if __name__ == "__main__":
    asyncio.run(test_execution_mode())
