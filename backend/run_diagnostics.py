import pytest
import sys
import io

class CatchAllStream:
    def __init__(self):
        self.data = io.StringIO()
    
    def write(self, s):
        self.data.write(s)
        # Also print to real stdout so we can see it if possible, 
        # but the main goal is to capture it in self.data
        sys.__stdout__.write(s)

    def flush(self):
        self.data.flush()
        sys.__stdout__.flush() # type: ignore

    def isatty(self):
        return True

def run_diagnostics():
    print("--- Starting Diagnostic Run ---")
    
    # Capture stdout/stderr
    capture = CatchAllStream()
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    sys.stdout = capture
    sys.stderr = capture
    
    try:
        # Run pytest programmatically
        retcode = pytest.main(["-v", "tests"])
    except Exception as e:
        sys.__stdout__.write(f"Reflected Exception: {e}\n")
    finally:
        # Restore
        sys.stdout = original_stdout
        sys.stderr = original_stderr
    
    print("\n--- Diagnostic Results ---")
    print(f"Exit Code: {retcode}")
    print("\n[Captured Output (Last 2000 chars)]:")
    full_output = capture.data.getvalue()
    print(full_output[-2000:])
    
    # Also save to file just in case
    with open("diagnostic_output.txt", "w", encoding="utf-8") as f:
        f.write(full_output)
    print("\nFull output saved to diagnostic_output.txt")

if __name__ == "__main__":
    run_diagnostics()
