"""Script to test and evaluate SciNets pipeline as a researcher would."""
import requests
import json

query = "What are the mechanisms linking sleep deprivation to cognitive decline in aging?"

print(f"Testing query: {query}")
print("=" * 80)

r = requests.post('http://localhost:8005/run_stream', 
    json={'query': query, 'goal': 'discover', 'speculation': 'medium'}, 
    stream=True)

result = None
events_count = 0
for line in r.iter_lines():
    if not line: 
        continue
    decoded = line.decode()
    if decoded.startswith('data: '):
        events_count += 1
        payload = decoded[6:]
        if payload == '[DONE]': 
            break
        try:
            event = json.loads(payload)
            if event.get('type') == 'result' and event.get('data', {}).get('hypotheses'):
                result = event['data']
        except: 
            pass

print(f"Total SSE events: {events_count}")
print()

if result:
    papers = result.get('literature', {}).get('papers', {})
    graph = result.get('concept_graph', {})
    hypotheses = result.get('hypotheses', [])
    
    print("=== LITERATURE ===")
    print(f"Papers found: {len(papers)}")
    for pid, p in list(papers.items())[:5]:
        print(f"  - {p.get('title', 'N/A')[:80]}... ({p.get('year')})")
    
    print()
    print("=== KNOWLEDGE GRAPH ===")
    print(f"Nodes: {len(graph.get('nodes', []))}")
    print(f"Edges: {len(graph.get('edges', []))}")
    print(f"Sample nodes: {graph.get('nodes', [])[:10]}")
    
    print()
    print("=== HYPOTHESES ===")
    print(f"Generated: {len(hypotheses)}")
    for i, h in enumerate(hypotheses[:3]):
        print(f"\n{i+1}. {h.get('text', 'N/A')}")
        print(f"   Novelty: {h.get('novelty_score')}, Feasibility: {h.get('feasibility_score')}, Testability: {h.get('testability_score')}")
        print(f"   Domain tags: {h.get('domain_tags')}")
        print(f"   Evidence items: {len(h.get('evidence', []))}")
        if h.get('evidence_summary'):
            print(f"   Evidence summary: {h.get('evidence_summary')[:200]}...")
    
    print()
    print("=== EXECUTIVE SUMMARY ===")
    lit_summary = result.get('literature', {}).get('summary', 'N/A')
    print(lit_summary[:1000])
else:
    print("No result captured")
