
import json

try:
    with open("debug_events.jsonl", "r") as f:
        events = [json.loads(line) for line in f if line.strip()]

    print(f"Total events: {len(events)}")
    
    # Filter for on_chain_end
    chain_ends = [e for e in events if e.get("event") == "on_chain_end"]
    print(f"Total on_chain_end events: {len(chain_ends)}")
    
    print(f"Total on_chain_end events: {len(chain_ends)}")
    
    print(f"Total on_chain_end events: {len(chain_ends)}")
    
    print("\n--- Agent Event Timeline ---")
    target_agents = ["literature", "hypothesis", "plan"]
    for i, event in enumerate(events):
        name = event.get('name')
        kind = event.get('event')
        if name in target_agents:
            print(f"[{i}] {kind} | Name: {name}")
            if kind == "on_chain_end":
                output = event.get("data", {}).get("output", {})
                if isinstance(output, dict):
                    print(f"    Keys: {list(output.keys())}")
                else:
                    print(f"    Output Type: {type(output)}")
        
    print("\n--- End of Timeline ---")
            
except Exception as e:
    print(f"Error parsing: {e}")
