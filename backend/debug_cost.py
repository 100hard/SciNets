
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.callbacks import BaseCallbackHandler
from collections import defaultdict
import contextvars

# --- Mock Cost Tracker Code ---
class CostTracker:
    def __init__(self):
        self.total = {"tokens": 0, "cost": 0.0}

    def record_llm_call(self, p, c, m, l):
        print(f"Recording: prompt={p}, completion={c}, model={m}")
        self.total["tokens"] += (p + c)

class CostCallbackHandler(BaseCallbackHandler):
    def __init__(self, tracker):
        self.tracker = tracker

    def on_llm_end(self, response, **kwargs):
        print(f"DEBUG: response.llm_output = {response.llm_output}")
        if response.llm_output:
            token_usage = response.llm_output.get("token_usage", {})
            model_name = response.llm_output.get("model_name", "unknown")
            p_tok = token_usage.get("prompt_tokens", 0)
            c_tok = token_usage.get("completion_tokens", 0)
            self.tracker.record_llm_call(p_tok, c_tok, model_name, 0.0)

# --- Test ---
if __name__ == "__main__":
    tracker = CostTracker()
    handler = CostCallbackHandler(tracker)
    
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        callbacks=[handler]
    )
    
    try:
        print("Invoking LLM...")
        llm.invoke([HumanMessage(content="Hello, say test.")])
        print(f"Tracker Total: {tracker.total}")
    except Exception as e:
        print(f"Error: {e}")
