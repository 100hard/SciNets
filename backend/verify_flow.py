import requests
import sys
import argparse

BASE_URL = "http://127.0.0.1:8001"

def request_link():
    print("1. Requesting magic link...")
    try:
        r = requests.post(f"{BASE_URL}/api/auth/request-link", json={"email": "manual_verify@test.com"})
        print(f"Status: {r.status_code}")
        if r.status_code == 200:
            print("LINK_REQUESTED_SUCCESSFULLY")
        else:
            print("LINK_REQUEST_FAILED")
    except Exception as e:
        print(f"Request failed: {e}")

def verify_and_test(token):
    session = requests.Session()
    
    print(f"2. Verifying token: {token}")
    verify_url = f"{BASE_URL}/api/auth/verify-link"
    r = session.post(verify_url, json={"token": token})
    print(f"   Status: {r.status_code}")
    if r.status_code != 200:
        print("VERIFY_FAILED")
        return

    print("3. Checking /me...")
    r = session.get(f"{BASE_URL}/api/auth/me")
    print(f"   Status: {r.status_code}, User: {r.text}")
    if r.status_code != 200:
        print("ME_CHECK_FAILED")
        return

    print("4. Testing Rate Limits...")
    # Request 1
    r = session.post(f"{BASE_URL}/run_stream", json={"query": "q1", "documents": []}, stream=True)
    r.close()
    if r.status_code == 429: 
        print("RATE_LIMIT_EARLY_FAIL")
        return
        
    # Request 2
    r = session.post(f"{BASE_URL}/run_stream", json={"query": "q2", "documents": []}, stream=True)
    r.close()
    if r.status_code == 429:
        print("RATE_LIMIT_EARLY_FAIL")
        return

    # Request 3
    r = session.post(f"{BASE_URL}/run_stream", json={"query": "q3", "documents": []}, stream=True)
    r.close()
    if r.status_code == 429:
        print("RATE_LIMIT_HIT_SUCCESS")
    else:
        print(f"RATE_LIMIT_NOT_HIT (Status: {r.status_code})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", choices=["request", "verify"], required=True)
    parser.add_argument("--token", help="Token for verify step")
    args = parser.parse_args()

    if args.step == "request":
        request_link()
    elif args.step == "verify":
        if not args.token:
            print("Token required for verify step")
            sys.exit(1)
        verify_and_test(args.token)
