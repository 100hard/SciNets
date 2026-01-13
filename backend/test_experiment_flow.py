
import requests
import json
import uuid

def test_experiment_stream():
    url = "http://127.0.0.1:8005/experiment_stream"
    
    payload = {
        "thread_id": str(uuid.uuid4()),
        "hypothesis_id": "hyp_123",
        "hypothesis_text": "Sleep deprivation causes accumulation of adenosine which impairs synaptic plasticity.",
        "intent": "stress_test",
        "plan_id": "plan_123"
    }
    
    print(f"Sending request to {url} with payload: {payload}")
    
    try:
        with requests.post(url, json=payload, stream=True) as r:
            if r.status_code != 200:
                print(f"Failed with status code: {r.status_code}")
                print(r.text)
                return

            print("Connected to stream. Reading events...")
            for line in r.iter_lines():
                if line:
                    decoded = line.decode('utf-8')
                    if decoded.startswith("data: "):
                        data_str = decoded[6:]
                        if data_str == "[DONE]":
                            print("\nStream completed successfully.")
                            break
                        try:
                            data = json.loads(data_str)
                            print(f"Event Type: {data.get('type')}")
                            if data.get('type') == 'error':
                                print(f"ERROR: {data.get('data')}")
                        except json.JSONDecodeError:
                            print(f"Raw: {decoded}")
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    test_experiment_stream()
