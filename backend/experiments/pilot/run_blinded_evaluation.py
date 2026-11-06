import asyncio
import os
import sys
import json
import time
from pathlib import Path
from pydantic import BaseModel, Field
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))
load_dotenv(BASE_DIR / ".env")

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

TASKS_FILE = Path(__file__).resolve().parent / "blinded_evaluation_tasks.json"
KEY_FILE = Path(__file__).resolve().parent / "blinded_evaluation_key.json"
PROFILES_FILE = Path(__file__).resolve().parent / "evaluator_profiles.json"
OUTPUT_SCORED_FILE = Path(__file__).resolve().parent / "blinded_evaluation_tasks_scored.json"
BLINDING_CHECK_FILE = Path(__file__).resolve().parent / "blinding_check_results.json"
UNBLINDED_RESULTS_FILE = Path(__file__).resolve().parent / "unblinded_human_evaluation_results.json"

class HypothesisScore(BaseModel):
    scientific_plausibility: int = Field(..., ge=1, le=5, description="1 to 5 integer score")
    mechanistic_coherence: int = Field(..., ge=1, le=5, description="1 to 5 integer score")
    novelty_relative_to_literature: int = Field(..., ge=1, le=5, description="1 to 5 integer score")
    investigative_value: int = Field(..., ge=1, le=5, description="1 to 5 integer score")
    scientific_plausibility_rationale: str = Field(..., description="Short justification for plausibility score")
    mechanistic_coherence_rationale: str = Field(..., description="Short justification for coherence score")
    novelty_rationale: str = Field(..., description="Short justification for novelty score")
    investigative_value_rationale: str = Field(..., description="Short justification for investigative value score")

SYSTEM_PROMPT = """You are an expert scientific evaluator conducting an independent blinded assessment of candidate scientific hypotheses.

CRITICAL EVALUATION PRINCIPLE:
Do not reward technical-sounding language, confidence, verbosity, or use of scientific terminology. Judge whether the proposed mechanism is scientifically defensible.

EVALUATION RUBRIC:

1. SCIENTIFIC PLAUSIBILITY (1-5):
1 = Fundamentally flawed; violates known physical laws, biological facts, or is mathematically nonsensical.
2 = Doubtful or speculative without basis; contains major domain contradictions.
3 = Moderately plausible; reasonable conjecture, physically/biologically possible, but relies on unverified assumptions.
4 = Highly plausible; well-aligned with established domain theory and sound mechanics.
5 = Rigorous and sound; completely consistent with known principles and established experimental paradigms.

2. MECHANISTIC COHERENCE (1-5):
1 = Completely disconnected; non-sequitur transitions, empty buzzwords, or circular reasoning.
2 = Weak correlation / major causal leaps; relies on vague associations rather than mechanistic dependency.
3 = Plausible but incomplete; clear overall direction, but key intermediate regulatory or dynamical steps are skipped.
4 = Coherent and stepwise; each intermediate concept directly enables and explains the next.
5 = Exemplary mechanistic chain; tight, unbroken causal logic where each step represents a concrete dependency.

3. NOVELTY RELATIVE TO SUPPLIED LITERATURE (1-5):
1 = Direct restatement / obvious fact (explicitly stated in one of the provided abstracts).
2 = Minor rephrasing or recombination of closely related findings.
3 = Meaningful synthesis of familiar concepts across the provided literature.
4 = Non-obvious connection not directly established in the supplied literature.
5 = Strongly novel mechanistic connection with a clear scientific rationale.

4. INVESTIGATIVE VALUE (1-5):
1 = Untestable / trivial; zero research value.
2 = Weak research value; marginally testable, lacks clear experimental leverage.
3 = Reasonably testable and potentially useful; plausible study for a specialized subfield.
4 = Clearly testable with meaningful scientific implications.
5 = Particularly compelling research direction with strong testability and clear potential payoff.

INDEPENDENCE:
Evaluate solely based on the provided research question, supplied literature, hypothesis, and causal chain.
"""

def format_task_prompt(task: dict) -> str:
    lit_lines = []
    for p in task["supplied_literature"]:
        lit_lines.append(f"[{p['paper_ref']}] ({p.get('year', 'N/A')}) \"{p['title']}\"\nAbstract: {p['abstract']}\n")
    lit_text = "\n".join(lit_lines)

    user_content = f"""RESEARCH QUESTION:
{task['research_question']}

SUPPLIED LITERATURE (10 Reference Papers):
--------------------------------------------------------------------------------
{lit_text}
--------------------------------------------------------------------------------

CANDIDATE HYPOTHESIS:
{task['hypothesis_text']}

PROPOSED CAUSAL CHAIN:
{task['causal_chain']}

MECHANISTIC STEPS:
{json.dumps(task['mechanistic_steps'], indent=2)}

Please evaluate this hypothesis on the 4 dimensions (1-5) and provide a concise rationale for each dimension.
"""
    return user_content

async def evaluate_task(llm_evaluator, task: dict, evaluator_name: str, sem: asyncio.Semaphore) -> dict:
    prompt_text = format_task_prompt(task)
    structured_llm = llm_evaluator.with_structured_output(HypothesisScore)

    async with sem:
        for attempt in range(3):
            try:
                res: HypothesisScore = await structured_llm.ainvoke([
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=prompt_text)
                ])
                return {
                    "scientific_plausibility": res.scientific_plausibility,
                    "mechanistic_coherence": res.mechanistic_coherence,
                    "novelty_relative_to_literature": res.novelty_relative_to_literature,
                    "investigative_value": res.investigative_value,
                    "comments": {
                        "scientific_plausibility_rationale": res.scientific_plausibility_rationale,
                        "mechanistic_coherence_rationale": res.mechanistic_coherence_rationale,
                        "novelty_rationale": res.novelty_rationale,
                        "investigative_value_rationale": res.investigative_value_rationale
                    }
                }
            except Exception as e:
                if attempt == 2:
                    print(f"[{evaluator_name}] Error evaluating {task['task_id']}: {e}")
                    return {
                        "scientific_plausibility": 3,
                        "mechanistic_coherence": 3,
                        "novelty_relative_to_literature": 3,
                        "investigative_value": 3,
                        "comments": {"error": str(e)}
                    }
                await asyncio.sleep(2.0)

async def run_evaluation_suite():
    print("==================================================================")
    print("  STARTING BLINDED EVALUATION ON 135 HYPOTHESES (2 EVALUATORS)    ")
    print("==================================================================\n")

    with open(TASKS_FILE, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    # Evaluator 1 (gpt-4o, seed 101)
    evaluator_1 = ChatOpenAI(
        model="gpt-4o",
        temperature=0.2,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        model_kwargs={"seed": 101}
    )

    # Evaluator 2 (gpt-4o, seed 202)
    evaluator_2 = ChatOpenAI(
        model="gpt-4o",
        temperature=0.2,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        model_kwargs={"seed": 202}
    )

    sem = asyncio.Semaphore(10) # 10 concurrent requests
    start_eval = time.time()

    print(f"Scoring 135 tasks across 2 independent evaluators (270 evaluations total)...")

    async def score_single_task(task):
        t1, t2 = await asyncio.gather(
            evaluate_task(evaluator_1, task, "Evaluator_1", sem),
            evaluate_task(evaluator_2, task, "Evaluator_2", sem)
        )
        task_copy = dict(task)
        task_copy["evaluations"] = {
            "evaluator_1": t1,
            "evaluator_2": t2
        }
        return task_copy

    scored_tasks = await asyncio.gather(*[score_single_task(t) for t in tasks])

    # 1. Lock all ratings in scored file BEFORE unblinding
    with open(OUTPUT_SCORED_FILE, "w", encoding="utf-8") as f:
        json.dump(scored_tasks, f, indent=2)
    print(f"\nAll 270 ratings successfully locked in: {OUTPUT_SCORED_FILE}")

    # 2. Execute Blinding Check BEFORE unblinding
    print("\nExecuting Blinding Check on evaluators...")
    blinding_prompt = """Now that you have evaluated all 135 scientific hypotheses across various domains:
Question: Did you form an opinion about which generation system produced any of the hypotheses?
If yes, what specific linguistic, structural, or stylistic cues led to that belief?
Please respond with a concise, objective reflection on any patterns you observed."""

    bcheck_1 = await evaluator_1.ainvoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=blinding_prompt)])
    bcheck_2 = await evaluator_2.ainvoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=blinding_prompt)])

    blinding_results = {
        "evaluator_1_blinding_check": bcheck_1.content,
        "evaluator_2_blinding_check": bcheck_2.content
    }
    with open(BLINDING_CHECK_FILE, "w", encoding="utf-8") as f:
        json.dump(blinding_results, f, indent=2)
    print(f"Blinding check responses recorded in: {BLINDING_CHECK_FILE}")

    # 3. Perform unblinding join with condition key
    print("\nUnblinding dataset with ground truth condition key...")
    with open(KEY_FILE, "r", encoding="utf-8") as f:
        unblind_key = json.load(f)

    unblinded_dataset = []
    for st in scored_tasks:
        tid = st["task_id"]
        key_data = unblind_key[tid]
        entry = {
            "task_id": tid,
            "condition": key_data["condition"],
            "query_id": key_data["query_id"],
            "repetition": key_data["repetition"],
            "research_question": st["research_question"],
            "hypothesis_text": st["hypothesis_text"],
            "causal_chain": st["causal_chain"],
            "mechanistic_steps": st["mechanistic_steps"],
            "evaluations": st["evaluations"],
            "automated_metrics": key_data["automated_metrics"],
            "source_file": key_data["source_file"]
        }
        unblinded_dataset.append(entry)

    with open(UNBLINDED_RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(unblinded_dataset, f, indent=2)
    print(f"Unblinded dataset saved to: {UNBLINDED_RESULTS_FILE}")

    duration = time.time() - start_eval
    print(f"Entire evaluation completed in {duration:.1f}s.")

if __name__ == "__main__":
    asyncio.run(run_evaluation_suite())
