import subprocess
import os
import json
import uuid
from typing import Dict, Any, Tuple

class LocalExecutor:
    def __init__(self, work_dir: str = "/app/experiments"):
        self.work_dir = work_dir
        os.makedirs(self.work_dir, exist_ok=True)

    async def run_script(self, code: str, timeout: int = 60) -> Tuple[int, str, str, Dict[str, Any]]:
        """
        Runs a Python script and returns (exit_code, stdout, stderr, metrics).
        Metrics are parsed from the last line of stdout if it's valid JSON.
        """
        script_id = str(uuid.uuid4())
        script_path = os.path.join(self.work_dir, f"{script_id}.py")
        
        with open(script_path, "w") as f:
            f.write(code)
            
        try:
            # Run the script
            process = subprocess.run(
                ["python", script_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.work_dir
            )
            
            exit_code = process.returncode
            stdout = process.stdout
            stderr = process.stderr
            
            # Try to parse metrics from stdout (look for the last JSON object)
            metrics = {}
            if exit_code == 0 and stdout:
                # Reverse iterate through lines to find JSON
                lines = [l.strip() for l in stdout.splitlines() if l.strip()]
                for line in reversed(lines):
                    try:
                        metrics = json.loads(line)
                        if isinstance(metrics, dict):
                            break
                    except json.JSONDecodeError:
                        continue
            
            return exit_code, stdout, stderr, metrics

        except subprocess.TimeoutExpired:
            return -1, "", "Execution timed out.", {}
        except Exception as e:
            return -1, "", str(e), {}
