from app.state import DiscoveryState, Experiment
import random
from app.executor import LocalExecutor
from app.llm import get_llm
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
import os

class CodeGeneration(BaseModel):
    code: str = Field(description="Executable Python code")
    explanation: str = Field(description="Brief explanation of what the code does")

async def experiment_node(state: DiscoveryState) -> dict:
    """
    Experiment Agent: Generates and runs code to test the hypothesis.
    """
    selected_id = state.selected_hypothesis_id
    if not state.hypotheses:
        return {}
        
    hypothesis = next((h for h in state.hypotheses if h.id == selected_id), None)
    if not hypothesis:
        return {}

    print(f"[Experiment] Running experiment for: {hypothesis.text}")
    
    print(f"[Experiment] Running experiment for: {hypothesis.text}")
    
    executor = LocalExecutor()
    current_code = None
    metrics = {}
    
    # CHECK: Run Experiments Toggle
    if not state.run_experiments:
        print("[Experiment] Run Experiments is FALSE. Performing Thought Experiment (Simulation).")
        
        sim_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a senior scientist. Perform a rigorous THOUGHT EXPERIMENT to validate the following hypothesis. Simulate the methodology and predicted results based on known scientific principles."),
            ("human", f"Hypothesis: {hypothesis.text}\n\nTask: Provide a detailed simulation of an experiment. What results would confirm it? What would refute it? Return a JSON-like summary of predicted metrics.")
        ])
        
        try:
            sim_result = await (sim_prompt | get_llm()).ainvoke({})
            # Store the thought process as 'code_snippet' (or a new field, but code_snippet is visible in UI)
            current_code = f"# THOUGHT EXPERIMENT (Simulation)\n# {hypothesis.text}\n\n'''\n{sim_result.content}\n'''"
            metrics = {"type": "thought_experiment", "status": "simulated"}
        except Exception as e:
             print(f"[Experiment] Thought experiment failed: {e}")
             current_code = "# Simulation Failed"

    else:
        # REAL CODE GENERATION
        # 1. Select Template (if applicable)
        # For now, we just read the ML template as a reference for the LLM
        template_path = os.path.join(os.path.dirname(__file__), "../templates/ml_train_model.py")
        with open(template_path, "r") as f:
            template_code = f.read()

        # 2. Generate Code (LLM)
        llm = get_llm(temperature=0.5)
        
        # ... (Rest of code generation logic remains similar but indented or refactored)
        # To avoid massive indentation changes in this tool call, I will rewrite the loop block
        
        structured_llm = llm.with_structured_output(CodeGeneration)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a computational scientist. Write a Python script to test the following hypothesis. Use the provided template as a guide. The script MUST print a JSON object with metrics to stdout at the end."),
            ("human", "Hypothesis: {hypothesis_text}\n\nReference Template:\n{template_code}")
        ])
        
        max_retries = 3
        stderr = ""
        
        for attempt in range(max_retries):
            try:
                if attempt == 0:
                    # First attempt: Generate from scratch
                    result = await (prompt | structured_llm).ainvoke({
                        "hypothesis_text": hypothesis.text,
                        "template_code": template_code
                    })
                    current_code = result.code
                else:
                    # Retry: Fix the code based on error
                    print(f"[Experiment] Retry {attempt}/{max_retries}...")
                    fix_prompt = ChatPromptTemplate.from_messages([
                        ("system", "Fix the following Python script based on the error message. Ensure it prints JSON metrics to stdout."),
                        ("human", "Code:\n{code}\n\nError:\n{error}")
                    ])
                    result = await (fix_prompt | structured_llm).ainvoke({
                        "code": current_code,
                        "error": stderr
                    })
                    current_code = result.code

                # 3. Execute Code
                exit_code, stdout, stderr, metrics = await executor.run_script(current_code)
                
                if exit_code == 0:
                    print(f"[Experiment] Success! Metrics: {metrics}")
                    break
                else:
                    print(f"[Experiment] Failed (Exit {exit_code}): {stderr}")
            
            except Exception as e:
                print(f"[Experiment] Error during execution loop: {e}")
                stderr = str(e)

    # 4. Save Result
    plot_b64 = None
    if metrics.get("plot"):
        try:
            plot_path = os.path.join(executor.work_dir, metrics["plot"])
            if os.path.exists(plot_path):
                with open(plot_path, "rb") as img_file:
                    import base64
                    plot_b64 = base64.b64encode(img_file.read()).decode('utf-8')
        except Exception as e:
            print(f"[Experiment] Failed to encode plot: {e}")

    experiment_result = Experiment(
        hypothesis_id=selected_id,
        code_snippet=current_code,
        metrics=metrics,
        plot_url=metrics.get("plot"),
        plot_base64=plot_b64
    )
    
    # Update state
    # Note: Appending to the list
    experiments = state.experiments or []
    experiments.append(experiment_result)
    
    return {"experiments": [e.model_dump() if hasattr(e, "model_dump") else e for e in experiments]}
