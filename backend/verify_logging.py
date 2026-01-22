import requests
import time
import subprocess
import os
import sys

BASE_URL = "http://127.0.0.1:8001"
ADMIN_EMAIL = "admin@test.com"
LOG_FILE = "logging_debug.log"

def start_server():
    env = os.environ.copy()
    env["SCINETS_ADMIN_EMAILS"] = '["admin@test.com"]'
    
    # Clean previous
    subprocess.run("taskkill /F /IM uvicorn.exe", shell=True, stderr=subprocess.DEVNULL)
    
    proc = subprocess.Popen(
        ["uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8001"],
        cwd="d:\\SciNets\\backend",
        stdout=open(LOG_FILE, "w"),
        stderr=subprocess.STDOUT,
        env=env
    )
    time.sleep(5)
    return proc

def test_logging():
    session = requests.Session()
    
    # 1. Login
    print("1. Logging in...")
    r = session.post(f"{BASE_URL}/api/auth/request-link", json={"email": ADMIN_EMAIL})
    time.sleep(1)
    
    # Extract
    with open(f"d:\\SciNets\\backend\\{LOG_FILE}", "r") as f:
        content = f.read()
        import re
        matches = re.findall(r'Link: (http://[^\s]+)', content)
        link = matches[-1]
        
    from urllib.parse import urlparse, parse_qs
    token = parse_qs(urlparse(link).query)['token'][0]
    
    r = session.post(f"{BASE_URL}/api/auth/verify-link", json={"token": token})
    print(f"   Login Status: {r.status_code}")
    
    # Force cookie for subsequent requests if needed
    cookie_header = f"session_id={session.cookies.get('session_id')}"
    headers = {"Cookie": cookie_header}

    # 2. Check Initial Stats
    print("2. Checking Stats...")
    r = session.get(f"{BASE_URL}/api/admin/stats", headers=headers)
    print(f"   Stats: {r.text}")
    initial_runs = r.json().get("total_discoveries", 0)

    # 3. Run Discovery
    print("3. Running Discovery...")
    # Longer query to pass validation
    payload = {"query": "This is a long enough query for discovery testing", "documents": []}
    try:
        r = session.post(f"{BASE_URL}/run_stream", json=payload, headers=headers, stream=True, timeout=5)
        print(f"   Stream Status: {r.status_code}")
        # Consume a bit
        for line in r.iter_lines():
            if line: 
                print(f"   Line: {line[:50]}")
                break
        r.close()
    except Exception as e:
        print(f"   Stream Exception: {e}")

    time.sleep(2)
    
    # 4. Check Stats Again
    print("4. Checking Stats Again...")
    r = session.get(f"{BASE_URL}/api/admin/stats", headers=headers)
    print(f"   Stats: {r.text}")
    final_runs = r.json().get("total_discoveries", 0)
    
    if final_runs > initial_runs:
        print("SUCCESS: Count incremented")
    else:
        print("FAILED: Count did not increment")

if __name__ == "__main__":
    proc = start_server()
    try:
        test_logging()
        # Print server log for debug
        print("\n--- SERVER LOG ---")
        with open(LOG_FILE, 'r') as f:
            print(f.read())
    except Exception as e:
        print(e)
    finally:
        subprocess.run(f"taskkill /F /PID {proc.pid}", shell=True)
        subprocess.run("taskkill /F /IM uvicorn.exe", shell=True, stderr=subprocess.DEVNULL)
