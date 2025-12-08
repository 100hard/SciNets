
from app.state import DiscoveryState, Experiment, ExperimentPlan
import random
from app.executor import LocalExecutor
from app.llm import get_llm
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from pydantic import BaseModel, Field
from typing import List, Literal, Optional
import os
import json
import uuid

from langchain_core.runnables import RunnableConfig
from langchain_core.callbacks import adispatch_custom_event

class ExperimentAction(BaseModel):
    thought: str = Field(description="Your detailed reasoning about what to do next. Analyze errors if present. Include 'runtime_estimate' and 'data_size_estimate'.")
    code: str = Field(description="The executable Python code to run.")
    explanation: str = Field(description="A brief summary of what this code attempts to achieve.")

async def experiment_node(state: DiscoveryState, config: RunnableConfig) -> dict:
    """
    Experiment Agent (Cline-style):
    Iteratively thinks, writes code, executes, and fixes it until success.
    """
    selected_id = state.selected_hypothesis_id
    
    # DEBUG LOG
    await adispatch_custom_event("log", {"message": f"[Debug] Experiment Node: Mock={state.mock}, RunExp={state.run_experiments}, Hypotheses={len(state.hypotheses)}"}, config=config)

    if not state.hypotheses:
        await adispatch_custom_event("log", {"message": "[Debug] No hypotheses found, exiting."}, config=config)
        return {}
        
    if state.mock:
        if not state.run_experiments:
             await adispatch_custom_event("log", {"message": "[Experiment] MOCK MODE: Generating dummy plans."}, config=config)
             dummy_plan = ExperimentPlan(
                 id="mock-plan-1",
                 hypothesis_id=selected_id or "mock-h1",
                 type="synthetic",
                 goal="Mock Goal",
                 method="Mock Method",
                 metrics=["accuracy"],
                 cost_estimate="low"
             )
             return {"experiment_plans": [dummy_plan]}
        else:
            await adispatch_custom_event("log", {"message": "[Experiment] MOCK MODE: Executing dummy experiment."}, config=config)
            dummy_exp = Experiment(
                hypothesis_id=selected_id or (state.hypotheses[0].id if state.hypotheses else "mock-h1"),
                status="completed",
                code="print('Mock Experiment')",
                metrics={"accuracy": 0.99, "mock_metric": 100},
                plot_base64=None
            )
            return {"experiments": [dummy_exp]}
        
    hypothesis = next((h for h in state.hypotheses if h.id == selected_id), None)
    
    # If no hypothesis, return empty
    if not hypothesis:
        return {"experiments": []} # Graceful exit

    # 1. Proposal Mode (Default)
    if not state.run_experiments:
        await adispatch_custom_event("log", {"message": f"[Experiment] Proposal Mode: Generating potential experiments for: {hypothesis.text}"}, config=config)
        
        # Check domain relevance (Light filter)
        stem_domains = ["ml", "stats", "bio", "physics", "chemistry", "materials", "math", "cs"]
        is_stem = any(tag in stem_domains for tag in (hypothesis.domain_tags or []))
        
        # If no domain tags, assume general scientific
        if not hypothesis.domain_tags: is_stem = True
            
        if not is_stem:
            await adispatch_custom_event("log", {"message": "[Experiment] Non-STEM hypothesis. Skipping experiment proposal."}, config=config)
            return {"experiment_plans": []}

        # Define output structure
        class ProposalList(BaseModel):
            plans: List[ExperimentPlan]

        sim_llm = get_llm().with_structured_output(ProposalList)
        
        sim_prompt = f"""You are a Principal Investigator designed experimental protocols.
        The user wants valid, executable Python experiment ideas to test this hypothesis:
        "{hypothesis.text}"
        
        Generate 3 distinct experimental plans:
        1. "synthetic": A fast, synthetic simulation (CPU < 30s).
        2. "benchmark": A test on a real, small dataset (e.g. sklearn, or synthesized real-world data).
        3. "ablation": A parameter study or robustness check.
        
        For each, specify:
        - Goal: What does it prove?
        - Method: High-level Python approach (e.g. 'Use numpy to simulate DiffEq', 'Train Ridge on iris').
        - Metrics: Specific keys to track (p_value, accuracy, mse).
        - Type & Cost.
        """
        
        try:
            res = await sim_llm.ainvoke(sim_prompt)
            # Assign IDs
            plans = []
            for p in res.plans:
                 p.id = str(uuid.uuid4())
                 p.hypothesis_id = hypothesis.id
                 plans.append(p)
                 
                 plans.append(p)
                 
            await adispatch_custom_event("log", {"message": f"[Experiment] Proposed {len(plans)} plans."}, config=config)
            return {"experiment_plans": plans}
            
        except Exception as e:
             await adispatch_custom_event("log", {"message": f"[Experiment] Plan generation failed: {e}"}, config=config)
             return {}

    # 2. Real Experiment Loop (Agentic)
    await adispatch_custom_event("log", {"message": f"[Experiment] Starting Code Generation Loop for: {hypothesis.text}"}, config=config)
    
    executor = LocalExecutor()
    template_path = os.path.join(os.path.dirname(__file__), "../templates/ml_train_model.py")
    try:
        with open(template_path, "r") as f:
            template_code = f.read()
    except:
        template_code = "# No template found"

    def is_semantic_success(exit_code: int, metrics: dict) -> bool:
        """
        Checks if the experiment was actually successful based on metrics.
        Success requires:
        1. Exit code 0
        2. Valid dictionary metrics
        3. No 'error' key
        4. At least one numeric metric (to avoid trivial 'done' messages)
        """
        if exit_code != 0: return False
        if not isinstance(metrics, dict): return False
        if "error" in metrics: return False
        
        # FILTER: Ignore non-metric keys like 'message' or 'plot'
        metric_keys = [k for k in metrics.keys() if k not in ("error", "traceback", "message", "plot", "experiment_explanation")]
        if len(metric_keys) == 0: return False
        
        # FIX: Require at least one numeric key
        numeric_keys = [k for k, v in metrics.items() if isinstance(v, (int, float))]
        if not numeric_keys:
             return False
             
        return True

    def looks_too_heavy(code: str, thought: str = "") -> bool:
        """
        Heuristic to reject code that looks computationally expensive.
        Now checks LLM's own complexity estimate in 'thought'.
        """
        
        # 1. Self-Labeling Check (Smarter)
        # Only reject if it explicitly warns about runtime
        if "warning: slow" in thought.lower() or "runtime > 1 min" in thought.lower():
            return True
            
        # 2. Strict Pattern Matching
        # Large epochs
        if "epoch" in code and any(s in code for s in ["5000", "10000"]):
            return True
        # Large steps
        if "n_steps" in code and any(s in code for s in ["100000", "500000"]):
             return True
        
        # Deep Loops - Relaxed for scientific code
        if code.count("for ") > 20: 
            return True
            
        # Large Data Ranges
        if "range(100000" in code or "range(500000" in code:
            return True
            
        return False


    # 2.1 Retrieve Selected Plan Constraints
    plan_context = ""
    if state.selected_experiment_plan_id and state.experiment_plans:
        chosen_plan = next((p for p in state.experiment_plans if p.id == state.selected_experiment_plan_id), None)
        if chosen_plan:
             await adispatch_custom_event("log", {"message": f"[Experiment] Executing PLAN: {chosen_plan.type} - {chosen_plan.goal}"}, config=config)
             plan_context = f"""
             STRICT PLAN CONSTRAINTS (User Selected):
             - TYPE: {chosen_plan.type}
             - GOAL: {chosen_plan.goal}
             - METHOD: {chosen_plan.method}
             - METRICS: {chosen_plan.metrics}
             - COST LIMIT: {chosen_plan.cost_estimate}
             
             You MUST follow this plan. Do not invent a different experiment.
             """

    # FIX: Prompt Design - Tie Hypothesis Metadata
    system_prompt = f"""You are an Expert Python Data Scientist. Your goal is to write a Python script to TEST the given hypothesis.
    
    HYPOTHESIS METADATA:
    - Domain: {', '.join(hypothesis.domain_tags) if hypothesis.domain_tags else 'General'}
    - Novelty Score: {hypothesis.novelty_score}
    - Testability: {hypothesis.testability_score}
    
    {plan_context}
    
    RULES:
    1. Output a SINGLE JSON object of type `ExperimentAction`.
    2. The code MUST print a final JSON object to stdout containing metrics (e.g. {{"accuracy": 0.9, "loss": 0.1, "p_value": 0.05}}).
    3. PLOTS: If you generate a plot, save it as 'plot.png' and include "plot": "plot.png" in the final metrics.
    4. NO HIDDEN ERRORS: If execution fails, print a JSON with "error" key.
    5. COMPUTE LIMITS: 
       - Max runtime: 60 seconds.
       - Max dataset size: 1000 samples (synthetic) or small sklearn datasets.
       - Max epochs: 50.
       - NO INTERNET ACCESS. Use synthetic data or sklearn.
       
    6. Include 'runtime_estimate' and 'data_size_estimate' in your 'thought' field.

    REFERENCE TEMPLATE:
    {{template_code}}
    """

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Develop a confirmation experiment for: {hypothesis.text}")
    ]
    
    llm = get_llm(temperature=0.7) 
    structured_llm = llm.with_structured_output(ExperimentAction)
    
    max_turns = 5
    solved = False
    final_metrics = {}
    current_code = ""
    plot_b64 = None
    plot_url = None

    for turn in range(max_turns):
        await adispatch_custom_event("log", {"message": f"[Experiment Agent] Turn {turn+1}/{max_turns}"}, config=config)
        
        # A. Think & Code
        try:
            action: ExperimentAction = await structured_llm.ainvoke(messages)
            current_code = action.code
            
            await adispatch_custom_event("log", {"message": f"  > Thought: {action.thought[:100]}..."}, config=config)
            
            # Better structured history
            messages.append(AIMessage(content=f"[THOUGHT]\n{action.thought}\n\n[CODE]\n{action.code}"))

            # GUARDRAIL 1: Complexity Check
            if looks_too_heavy(action.code, action.thought):
                 await adispatch_custom_event("log", {"message": f"[Experiment] Rejected heavy code. Thought: {action.thought}"}, config=config)
                 messages.append(HumanMessage(content="Code rejected: Looks too computationally heavy (>50 epochs, >5 loops, or >10000 samples). Simplify it significantly."))
                 continue
                 
            # EXECUTE
            await adispatch_custom_event("log", {"message": f"[Experiment] Executing Code (Attempt {turn+1})..."}, config=config)
            
            # Execute
            exit_code, stdout, stderr, metrics = await executor.run_script(action.code, timeout=60)
            
            # Check for plot file
            if metrics.get("plot"):
                import base64
                plot_filename = metrics["plot"]
                if os.path.exists(os.path.join(executor.work_dir, plot_filename)):
                     with open(os.path.join(executor.work_dir, plot_filename), "rb") as img_file:
                         b64_str = base64.b64encode(img_file.read()).decode('utf-8')
                         metrics["plot_base64"] = b64_str
                         metrics["plot_url"] = f"data:image/png;base64,{b64_str}"

            if is_semantic_success(exit_code, metrics):
                # SUCCESS!
                await adispatch_custom_event("log", {"message": f"[Experiment] Success! Metrics: {metrics}"}, config=config)
                
                # FIX: Store explanation in metrics for UI
                final_metrics = metrics
                final_metrics["experiment_explanation"] = action.explanation
                solved = True
                break
            else:
                # FAILURE
                await adispatch_custom_event("log", {"message": f"[Experiment] Semantic Failure: Exit {exit_code}"}, config=config)
                
                # FIX: Context Management - Summarize History
                # Instead of appending massive logs, we summarize the failure
                error_msg = metrics.get('error', 'Unknown Error')
                stderr_snippet = stderr[-500:] # Limit to 500 chars
                
                failure_summary = f"Attempt {turn+1} Failed. Error: {error_msg}\nStderr: {stderr_snippet}\nFix the code incrementally."
                
                # Append only succinct messages
                # We overwrite the last AI message if we want to save context, but appending is safer for 'chat' models
                # Keep prompt small: remove old conversation if too long?
                if len(messages) > 6:
                     messages = [messages[0]] + messages[-4:] # Keep system + last 2 turns
                     
                messages.append(HumanMessage(content=failure_summary))
                
        except Exception as e:
            await adispatch_custom_event("log", {"message": f"[Experiment] Generator Error: {e}"}, config=config)
            messages.append(HumanMessage(content=f"Error: {e}"))
            continue
    
    # 4. Process Results (Failed or Solved)
    new_experiment = Experiment(
        id=str(uuid.uuid4()),
        hypothesis_id=selected_id,
        status="completed" if solved else "failed",
        code=current_code,
        metrics=final_metrics,
        plot_url=plot_url,
        plot_base64=plot_b64,
        result_summary=f"Experiment {'successful' if solved else 'failed'}. {final_metrics.get('experiment_explanation', '')}"
    )
    
    experiments = state.experiments or []
    experiments.append(new_experiment)
    
    return {"experiments": [e for e in experiments]}
