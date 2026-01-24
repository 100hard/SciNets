import time
import contextvars
from typing import Dict, Any, Optional, List
from collections import defaultdict
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
import logging

log = logging.getLogger(__name__)

# Context Variables for Thread-Safe Context Tracking
_agent_context = contextvars.ContextVar("agent_context", default="unknown")
_step_context = contextvars.ContextVar("step_context", default="unknown")
_hypothesis_context = contextvars.ContextVar("hypothesis_context", default=None)

# Global fallback for scripts/tests that don't set up context
_global_fallback_tracker = None

# Context Variable for the Tracker Instance
_tracker_context = contextvars.ContextVar("cost_tracker_instance", default=None)
_hyp_token_context = contextvars.ContextVar("hyp_token_context", default=None)

class CostTracker:
    """
    Centralized tracker for compute costs, tokens, and latency.
    Designed for request-scoped usage via contextvars.
    """
    def __init__(self):
        self._reset()
        
    def _reset(self):
        # Metrics Storage
        self.by_agent = defaultdict(lambda: {"calls": 0, "prompt": 0, "completion": 0, "cost": 0.0, "time": 0.0})
        self.by_step = defaultdict(lambda: {"calls": 0, "prompt": 0, "completion": 0, "cost": 0.0, "time": 0.0})
        self.by_hypothesis = defaultdict(lambda: {"calls": 0, "prompt": 0, "completion": 0, "cost": 0.0, "time": 0.0})
        self.total = {"calls": 0, "prompt": 0, "completion": 0, "tokens": 0, "cost": 0.0}
        self.active_steps = {} # { "agent:step": start_time }

    @staticmethod
    def get_instance():
        """
        Returns the current request's CostTracker instance.
        If none is set (e.g. running outside request context), returns a global fallback.
        """
        tracker = _tracker_context.get()
        if tracker:
            return tracker
        
        # Global fallback logic
        global _global_fallback_tracker
        if _global_fallback_tracker is None:
            _global_fallback_tracker = CostTracker()
        return _global_fallback_tracker

    @staticmethod
    def register_context(tracker):
        """Sets the provided tracker as the active instance for this context."""
        return _tracker_context.set(tracker)
    
    @staticmethod
    def reset_context(token):
        """Resets the context to previous state."""
        _tracker_context.reset(token)

    # --- Lifecycle Hooks ---

    def start_step(self, agent_name: str, step_name: str):
        """Mark the start of a logical step (e.g., 'literature_search')."""
        _agent_context.set(agent_name)
        _step_context.set(step_name)
        
        key = f"{agent_name}:{step_name}"
        self.active_steps[key] = time.time()

    def end_step(self, agent_name: str, step_name: str):
        """Mark the end of a logical step and record wall-time."""
        key = f"{agent_name}:{step_name}"
        start_time = self.active_steps.pop(key, None)
        if start_time:
            duration = time.time() - start_time
            self._record_time(agent_name, step_name, _hypothesis_context.get(), duration)

    def push_hypothesis_context(self, hypothesis_id: str):
        """Set the current hypothesis context for attribution."""
        token = _hypothesis_context.set(hypothesis_id)
        _hyp_token_context.set(token)

    def pop_hypothesis_context(self):
        """Clear hypothesis context."""
        token = _hyp_token_context.get()
        if token:
            try:
                _hypothesis_context.reset(token)
            except ValueError:
                # Fallback if context drift occurred (should not happen with strict hierarchy)
                _hypothesis_context.set(None)
        else:
             _hypothesis_context.set(None)

    # --- Metric Recording ---

    def record_llm_call(self, prompt_tokens: int, completion_tokens: int, model_name: str, latency_ms: float):
        """Record an individual LLM call's usage."""
        agent = _agent_context.get()
        step = _step_context.get()
        hyp_id = _hypothesis_context.get()
        
        cost = self._calculate_cost(prompt_tokens, completion_tokens, model_name)
        
        # Update Totals
        self.total["calls"] += 1
        self.total["prompt"] += prompt_tokens
        self.total["completion"] += completion_tokens
        self.total["tokens"] += (prompt_tokens + completion_tokens)
        self.total["cost"] += cost
        
        # Update Agent Metrics
        self._update_metric(self.by_agent[agent], prompt_tokens, completion_tokens, cost, latency_ms/1000.0)
        
        # Update Step Metrics
        self._update_metric(self.by_step[step], prompt_tokens, completion_tokens, cost, latency_ms/1000.0)
        
        # Update Hypothesis Metrics (if active)
        if hyp_id:
             self._update_metric(self.by_hypothesis[hyp_id], prompt_tokens, completion_tokens, cost, latency_ms/1000.0)

    def _update_metric(self, metric_dict, p_tok, c_tok, cost, time_s):
        metric_dict["calls"] += 1
        metric_dict["prompt"] += p_tok
        metric_dict["completion"] += c_tok
        metric_dict["cost"] += cost
        metric_dict["time"] += time_s # Add LLM latency to the time bucket

    def _record_time(self, agent, step, hyp_id, duration):
        """Record purely wall-clock time for steps (logic overhead)."""
        self.by_agent[agent]["time"] += duration
        self.by_step[step]["time"] += duration
        if hyp_id:
            self.by_hypothesis[hyp_id]["time"] += duration

    def _calculate_cost(self, prompt: int, completion: int, model: str) -> float:
        # GPT-4o Pricing (approx)
        if "gpt-4" in model:
            return (prompt * 2.50 / 1e6) + (completion * 10.00 / 1e6)
        # GPT-4o-mini Pricing
        if "mini" in model:
            return (prompt * 0.15 / 1e6) + (completion * 0.60 / 1e6)
        # Fallback (o1/preview/etc) -> treat as GPT-4o
        return (prompt * 5.00 / 1e6) + (completion * 15.00 / 1e6)

    def get_report(self) -> Dict[str, Any]:
        return {
            "total_tokens": self.total["tokens"],
            "estimated_cost_usd": round(self.total["cost"], 4),
            "by_agent": dict(self.by_agent),
            "by_step": dict(self.by_step),
            "by_hypothesis": dict(self.by_hypothesis)
        }


class CostCallbackHandler(BaseCallbackHandler):
    """LangChain callback to feed the CostTracker."""
    
    def __init__(self):
        # We fetch the instance dynamically when needed OR store it.
        # But callbacks are often long-lived? No, get_llm creates new one each time.
        # So we can fetch instance here.
        self.tracker = CostTracker.get_instance()

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Collect token usage."""
        try:
            if not response.llm_output:
                return
                
            token_usage = response.llm_output.get("token_usage", {})
            model_name = response.llm_output.get("model_name", "unknown")
            
            p_tok = token_usage.get("prompt_tokens", 0)
            c_tok = token_usage.get("completion_tokens", 0)
            
            self.tracker.record_llm_call(p_tok, c_tok, model_name, 0.0)
            
        except Exception as e:
            log.warning(f"Error in CostCallbackHandler: {e}")
