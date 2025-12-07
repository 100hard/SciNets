
import requests
import json
import sys

def verify_stream():
    url = "http://127.0.0.1:8002/run_stream"
    payload = {
        "query": "Test Heartbeat Connectivity",
        "goal": "discover",
        "lens": "physics",
        "speculation": "high",
        "run_experiments": False,
        "documents": []
    }
    
    headers = {"Accept-Encoding": "identity"}
    
    print(f"Connecting to {url}...")
    
    try:
        response = requests.post(url, json=payload, headers=headers, stream=True, timeout=120)
        
        try:
            print(f"Response Headers: {response.headers}", flush=True)
            response.raise_for_status()
            print("Connected! Listening for events (byte-stream)...", flush=True)
            
            # iter_content(chunk_size=None) will return chunks as they arrive
            buffer = ""
            for chunk in response.iter_content(chunk_size=None):
                if chunk:
                    text_data = chunk.decode('utf-8')
                    buffer += text_data
                    
                    # Split by double newline to get events
                    while "\n\n" in buffer:
                        event_text, buffer = buffer.split("\n\n", 1)
                        if event_text.startswith("data: "):
                            json_str = event_text[6:].strip()
                            if json_str == "[DONE]":
                                print("\n[STREAM COMPLETE]")
                                break
                                
                            try:
                                data = json.loads(json_str)
                                event_type = data.get("type")
                                
                                if event_type == "interrupt":
                                    print(f"\n[INTERRUPT] Graph paused! Next: {data.get('data', {}).get('next')}")
                                    return # Stop on interrupt
                                    
                                if event_type == "result":
                                    result_data = data.get("data", {})
                                    keys = list(result_data.keys())
                                    print(f"\n[RESULT] Interim result with keys: {keys}")
                                    pass # Don't stop, wait for interrupt or done
                                    
                                elif event_type == "activity":
                                    data_content = data.get("data", {})
                                    thread_id = data.get("thread_id")
                                    if thread_id:
                                        print(f"\n[INFO] Thread ID Received: {thread_id}")
                                        
                                    sys.stdout.write(".")
                                    sys.stdout.flush()
                                elif event_type == "error":
                                    print(f"\n[ERROR EVENT] {data.get('data')}")
                                    
                            except json.JSONDecodeError:
                                pass
                                
        except requests.exceptions.HTTPError as e:
            print(f"HTTP Error: {e}")
            err_msg = f"Server Response: {response.text}"
            print(err_msg)
            
    except Exception as e:
        print(f"Connection failed: {e}")

if __name__ == "__main__":
    verify_stream()
