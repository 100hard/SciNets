
from app.state import DiscoveryState, ExperimentState, Experiment, ExperimentPlan
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

async def experiment_node(state: ExperimentState, config: RunnableConfig) -> dict:
    """
    Experiment Agent (Cline-style):
    Iteratively thinks, writes code, executes, and fixes it until success.
    Works on a Single Hypothesis defined in ExperimentState.
    """
    
    # DEBUG LOG
    await adispatch_custom_event("log", {"message": f"[Debug] Experiment Node: ID={state.hypothesis_id}, Intent={state.intent}"}, config=config)

    hypothesis_text = state.hypothesis_text
    
    if not hypothesis_text:
        await adispatch_custom_event("log", {"message": "[Error] No hypothesis text provided."}, config=config)
        return {"experiment_result": Experiment(hypothesis_id=state.hypothesis_id, status="failed", result_summary="Missing hypothesis text")}

    # 1. Real Experiment Loop (Agentic)
    await adispatch_custom_event("log", {"message": f"[Experiment] Starting Code Generation Loop for: {hypothesis_text[:50]}..."}, config=config)
    
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


    # EPISTEMIC REFRAMING: Experiments are exploratory consistency checks, not validation
    system_prompt = f"""You are an Expert Python Data Scientist. Your goal is to write a Python script 
for an EXPLORATORY CONSISTENCY CHECK of the given hypothesis.

EPISTEMIC FRAMING:
This is NOT validation or proof. You are checking behavioral consistency, parameter sensitivity, 
and potential failure modes. Focus on EXPLORATION, not confirmation.

HYPOTHESIS:
"{hypothesis_text}"

INTENT: {state.intent}
DATA SOURCE: {state.data_source}

RULES:
1. Output a SINGLE JSON object of type `ExperimentAction`.
2. The code MUST print a final JSON object to stdout containing BEHAVIORAL METRICS:
   - "stability_score": How stable is behavior under noise? (0-1)
   - "sensitivity": How much do outputs change with parameter variation? (high/medium/low)
   - "failure_modes": List of observed failure conditions
   - "behavioral_pattern": Description of observed behavior
   - "consistency_check": Does behavior align with hypothesized mechanism? (yes/partial/no)
   
   DO NOT output: accuracy, p_value, significance, validation metrics.
   
3. PLOTS: If you generate a plot, save it as 'plot.png' and include "plot": "plot.png" in metrics.
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
        HumanMessage(content=f"Develop an exploratory consistency check for: {hypothesis_text}")
    ]
    
    llm = get_llm(temperature=0.7) 
    # FIX: Use schema dict to ensure output is a dict, not Pydantic object, to avoid LogStreamCallbackHandler serialization error
    structured_llm = llm.with_structured_output(ExperimentAction.model_json_schema())
    
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
            action_dict = await structured_llm.ainvoke(messages)
            # Re-validate with Pydantic locally (callback safe from this)
            action = ExperimentAction(**action_dict)
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
            
            # REPRODUCIBILITY ENFORCEMENT: Prepend seed header to all code
            SEED_HEADER = """# === REPRODUCIBILITY HEADER (Auto-injected) ===
import random
import numpy as np
random.seed(42)
np.random.seed(42)
try:
    import torch
    torch.manual_seed(42)
except ImportError:
    pass
# === END HEADER ===

"""
            reproducible_code = SEED_HEADER + action.code
            
            # Execute with reproducible code
            exit_code, stdout, stderr, metrics = await executor.run_script(reproducible_code, timeout=60)
            
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
        hypothesis_id=state.hypothesis_id,
        status="completed" if solved else "failed",
        code_snippet=current_code,
        metrics=final_metrics,
        plot_url=plot_url,
        plot_base64=plot_b64,
        result_summary=f"Experiment {'successful' if solved else 'failed'}. {final_metrics.get('experiment_explanation', '')}"
    )
    
    return {"experiment_result": new_experiment}
