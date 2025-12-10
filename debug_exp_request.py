
from pydantic import BaseModel

class ExperimentRequest(BaseModel):
    thread_id: str
    hypothesis_id: str
    plan_id: str

payload = {
    "thread_id": "test-thread",
    "plan_id": "test-plan",
    "hypothesis_id": "test-hypo"
}

try:
    req = ExperimentRequest(**payload)
    print("Validation Successful!")
    print(req.dict())
except Exception as e:
    print(f"Validation Error: {e}")
