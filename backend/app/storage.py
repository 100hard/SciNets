import json
import os
import shutil
from typing import Dict, Any, Optional

# Ensure data directory exists
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "runs")
os.makedirs(DATA_DIR, exist_ok=True)

def save_run_result(thread_id: str, state: Dict[str, Any]):
    """
    Persists the final state of a discovery run to a JSON file.
    """
    if not thread_id:
        return
        
    file_path = os.path.join(DATA_DIR, f"{thread_id}.json")
    try:
        def serializer(obj):
            if hasattr(obj, "model_dump"):
                return obj.model_dump()
            if hasattr(obj, "dict"):
                return obj.dict()
            if isinstance(obj, (set, list, tuple)):
                return [serializer(i) for i in obj]
            if isinstance(obj, dict):
                return {k: serializer(v) for k, v in obj.items()}
            return str(obj)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(state, f, default=serializer, indent=2)
        print(f"[Storage] Saved run result for {thread_id} to {file_path}")
    except Exception as e:
        print(f"[Storage] Failed to save result for {thread_id}: {e}")

def get_run_result(thread_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieves a persisted run result.
    """
    file_path = os.path.join(DATA_DIR, f"{thread_id}.json")
    if not os.path.exists(file_path):
        return None
        
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[Storage] Failed to load result for {thread_id}: {e}")
        return None
