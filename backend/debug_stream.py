
import requests
import json

url = "http://127.0.0.1:8005/run_stream"
payload = {"query": "Design a novel protocol for QKD.", "goal": "discover", "run_experiments": True}

print(f"Calling {url}...")
try:
    with requests.post(url, json=payload, stream=True, timeout=300) as r:
        print(f"Status: {r.status_code}")
        for line in r.iter_lines():
            if line:
                decoded = line.decode('utf-8')
                print(decoded)
                if '"type": "error"' in decoded:
                    with open("error_log.txt", "w") as f:
                        f.write(decoded)

except Exception as e:
    print(f"Error: {e}")
