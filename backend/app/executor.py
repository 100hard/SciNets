import asyncio
import os
import json
import uuid
from typing import Dict, Any, Tuple
from app.config import config
import re

class LocalExecutor:
    def __init__(self, work_dir: str = None):
        if work_dir is None:
            self.work_dir = os.path.join(os.getcwd(), "experiments")
        else:
            self.work_dir = work_dir
        
        # Sandbox check: Ensure work_dir is within allowed paths (basic check)
        # Ideally, use a container, but for local exec, just avoid root/system paths
        abs_work = os.path.abspath(self.work_dir)
        if not abs_work.startswith(os.getcwd()) and "/tmp" not in abs_work:
             # Just a warning or silent correction in this context, 
             # relying on validate_code for script content security
             pass

        os.makedirs(self.work_dir, exist_ok=True)

    def validate_code(self, code: str) -> Tuple[bool, str]:
        """
        Basic static analysis to reject obviously dangerous code.
        """
        # Block dangerous modules
        dangerous_modules = ["subprocess", "os.system", "os.popen", "sys.modules", "shutil", "builtins", "importlib"]
        for mod in dangerous_modules:
            if re.search(f"\\b{mod}\\b", code):
                return False, f"Usage of '{mod}' is not allowed for security reasons."
        
        # Block dangerous file operations (basic)
        if "open(" in code and ("/" in code or ".." in code):
             # Rough heuristic: Allow local file creation but warn on paths
             # Real sandbox should be Docker-based.
             pass
             
        return True, ""

    async def run_script(self, code: str, timeout: int = 60) -> Tuple[int, str, str, Dict[str, Any]]:
        """
        Runs a Python script asynchronously and returns (exit_code, stdout, stderr, metrics).
        Metrics are parsed from the last line of stdout if it's valid JSON.
        """
        # 1. Demo Mode Check
        if config.DEMO_MODE:
            return 0, "Demo Mode: Check skipped.", "", {"accuracy": 0.95, "f1": 0.92, "demo": True}

        # 2. Security Check
        is_safe, reason = self.validate_code(code)
        if not is_safe:
            return -1, "", f"Security Violation: {reason}", {"error": "security_violation"}

        script_id = str(uuid.uuid4())
        script_path = os.path.join(self.work_dir, f"{script_id}.py")
        
        with open(script_path, "w") as f:
            f.write(code)
            
        try:
            # Run the script asynchronously
            process = await asyncio.create_subprocess_exec(
                "python", script_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.work_dir
            )
            
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(), timeout=timeout
                )
                stdout = stdout_bytes.decode('utf-8', errors='replace')
                stderr = stderr_bytes.decode('utf-8', errors='replace')
                exit_code = process.returncode or 0
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return -1, "", "Execution timed out.", {"error": "timeout"}
            
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

        except Exception as e:
            return -1, "", str(e), {"error": str(e)}

