
import asyncio
from dotenv import load_dotenv
load_dotenv()

print("Importing plan...")
try:
    from app.agents.orchestrator import plan_node
    print("Plan OK")
except Exception as e:
    print(f"Plan FAIL: {e}")

print("Importing literature...")
try:
    from app.agents.literature import literature_node
    print("Literature OK")
except Exception as e:
    print(f"Literature FAIL: {e}")

print("Importing hypothesis...")
try:
    from app.agents.hypothesis import hypothesis_node
    print("Hypothesis OK")
except Exception as e:
    print(f"Hypothesis FAIL: {e}")

print("Importing evidence...")
try:
    from app.agents.evidence import evidence_node
    print("Evidence OK")
except Exception as e:
    print(f"Evidence FAIL: {e}")

print("Importing experiment...")
try:
    from app.agents.experiment import experiment_node
    print("Experiment OK")
except Exception as e:
    print(f"Experiment FAIL: {e}")

print("Importing critique...")
try:
    from app.agents.critique import critique_node
    print("Critique OK")
except Exception as e:
    print(f"Critique FAIL: {e}")
