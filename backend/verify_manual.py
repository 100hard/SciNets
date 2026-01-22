import time
import re
import requests
import sys

BASE_URL = "http://127.0.0.1:8001"
LOG_FILE = "server_8001_v3.log"

def wait_for_server():
    print("Waiting for server to start...")
    for _ in range(20):
        try:
            r = requests.get(f"{BASE_URL}/api/contact") # use simple endpoint
            if r.status_code != 500: # 500 means server up but maybe error processing
                r = requests.get(f"{BASE_URL}/api/demo/example")
                if r.status_code in [200, 404]:
                    print("Server is up!")
                    return
        except:
            pass
        time.sleep(1)
    print("Server failed to start.")
    sys.exit(1)

def get_magic_link_from_log():
    print("Scanning log for magic link...")
    # Read the log file
    found_link = None
    for _ in range(10):
        try:
            with open(LOG_FILE, 'r') as f:
                content = f.read()
                # Use findall to get all matches, then pick the last one (most recent)
                matches = re.findall(r'Link: (http://[^\s]+)', content)
                if matches:
                    found_link = matches[-1]
                    break
        except FileNotFoundError:
            pass
        time.sleep(1)
    return found_link

def verify():
    session = requests.Session()
    email = "manual_verify@test.com"
    
    # 1. Request Link
    print(f"1. Requesting magic link for {email}...")
    try:
        r = session.post(f"{BASE_URL}/api/auth/request-link", json={"email": email})
        print(f"   Status: {r.status_code}, Response: {r.text}")
        if r.status_code != 200: return False
    except requests.exceptions.ConnectionError:
        print("   FAILED: Connection refused.")
        return False

    # 2. Extract Token
    link = get_magic_link_from_log()
    if not link:
        print("   FAILED: Could not find magic link in logs.")
        return False
    print(f"   Found Link: {link}")
    
    try:
        from urllib.parse import urlparse, parse_qs
        parsed = urlparse(link)
        token = parse_qs(parsed.query)['token'][0]
    except Exception as e:
         print(f"   FAILED: Could not parse token from link: {e}")
         return False
    
    # 3. Verify Link
    print(f"2. Verifying token...")
    # Note: The link points to frontend port 8080 usually if constructed that way in code, 
    # but we want to hit the BACKEND API which is BASE_URL.
    # The endpoint is /api/auth/verify-link
    verify_url = f"{BASE_URL}/api/auth/verify-link"
    r = session.post(verify_url, json={"token": token})
    print(f"   Status: {r.status_code}, Cookies: {session.cookies.get_dict()}")
    if r.status_code != 200: 
        print(f"   FAILED: Verify returned {r.status_code}")
        return False
    
    # 4. Check /me
    print(f"3. Checking /api/auth/me...")
    r = session.get(f"{BASE_URL}/api/auth/me")
    print(f"   Status: {r.status_code}, User: {r.text}")
    if r.status_code != 200: 
        print(f"   FAILED: /me returned {r.status_code}")
        return False

    # 5. Rate Limit Test
    print(f"4. Testing Rate Limits (Limit: 2)...")
    
    # Request 1
    print("   Request 1 (Discovery Init)...")
    # minimal valid payload
    payload = {"query": "test query 1", "documents": []}
    # Use stream=True to avoid waiting for the whole response (if it works)
    # Actually, failure will happen instantly in dependency. Success will yield a generator.
    r = session.post(f"{BASE_URL}/run_stream", json=payload, stream=True)
    print(f"   Status: {r.status_code}") 
    # Close connection to be nice
    r.close()
    
    if r.status_code == 429: 
        print("   FAILED: Rate limited too early!")
        return False
        
    # Request 2
    print("   Request 2 (Discovery Init)...")
    payload = {"query": "test query 2", "documents": []}
    r = session.post(f"{BASE_URL}/run_stream", json=payload, stream=True)
    print(f"   Status: {r.status_code}")
    r.close()
    
    if r.status_code == 429:
        print("   FAILED: Rate limited too early!")
        return False

    # Request 3 (Should Fail)
    print("   Request 3 (Should be Rate Limited)...")
    payload = {"query": "test query 3", "documents": []}
    r = session.post(f"{BASE_URL}/run_stream", json=payload, stream=True)
    print(f"   Status: {r.status_code}, Body: {r.text}")
    r.close()
    
    if r.status_code == 429:
        print("   SUCCESS: Rate limit hit as expected.")
        # Optional: Print unlocks_at
        try:
             import json
             detail = r.json()['detail']
             print(f"   Unlock Time: {detail['unlocks_at']}")
        except: pass
        return True
    else:
        print(f"   FAILED: Expected 429, got {r.status_code}")
        return False

if __name__ == "__main__":
    wait_for_server()
    success = verify()
    if success:
        print("\nVerification PASSED!")
    else:
        print("\nVerification FAILED.")
