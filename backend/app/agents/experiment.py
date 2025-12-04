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
    
    # 1. Select Template (if applicable)
    # For now, we just read the ML template as a reference for the LLM
    template_path = os.path.join(os.path.dirname(__file__), "../templates/ml_train_model.py")
    with open(template_path, "r") as f:
        template_code = f.read()

    # 2. Generate Code (LLM)
    llm = get_llm(temperature=0.5)
    structured_llm = llm.with_structured_output(CodeGeneration)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a computational scientist. Write a Python script to test the following hypothesis. Use the provided template as a guide. The script MUST print a JSON object with metrics to stdout at the end."),
        ("human", "Hypothesis: {hypothesis_text}\n\nReference Template:\n{template_code}")
    ])
    
    executor = LocalExecutor()
    max_retries = 3
    current_code = ""
    metrics = {}
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
    
    return {"experiments": experiments}
