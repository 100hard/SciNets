"""
SciNets Pipeline Evaluation Script
Run a full discovery workflow and capture all outputs for researcher evaluation
"""
import requests
import json
import time
from pathlib import Path

# Configuration
BASE_URL = "http://127.0.0.1:8006"

# A meaningful research question to test the system
TEST_QUERY = """
I'm interested in understanding the relationship between gut microbiome composition 
and the efficacy of immunotherapy in cancer treatment. What are the key findings 
in this area and what hypotheses could be explored?
"""

def run_discovery():
    """Run the full discovery pipeline and capture all streamed events."""
    
    print("=" * 80)
    print("SCINETS PIPELINE EVALUATION")
    print("=" * 80)
    print(f"\nRESEARCH QUESTION:\n{TEST_QUERY}\n")
    
    # Prepare the request
    payload = {
        "query": TEST_QUERY.strip(),
        "goal": "discover",
        "lens": "none",
        "speculation": "medium",
        "run_experiments": False,
        "documents": [],
        "mock": False
    }
    
    all_events = []
    start_time = time.time()
    
    # Stream the discovery
    print("-" * 80)
    print("STREAMING DISCOVERY EVENTS:")
    print("-" * 80 + "\n")
    
    try:
        response = requests.post(
            f"{BASE_URL}/run_stream",
            json=payload,
            stream=True,
            timeout=300
        )
        response.raise_for_status()
        
        current_event_type = None
        current_data = ""
        
        for line in response.iter_lines():
            if not line:
                continue
            
            line = line.decode('utf-8')
            
            if line.startswith('event:'):
                current_event_type = line.replace('event:', '').strip()
            elif line.startswith('data:'):
                data = line.replace('data:', '').strip()
                if data:
                    try:
                        parsed_data = json.loads(data)
                        event = {
                            "type": current_event_type,
                            "data": parsed_data,
                            "timestamp": time.time() - start_time
                        }
                        all_events.append(event)
                        
                        # Print summary based on event type
                        if current_event_type == "status":
                            print(f"[{event['timestamp']:.1f}s] STATUS: {parsed_data.get('stage', '')} - {parsed_data.get('message', '')}")
                        elif current_event_type == "literature":
                            papers = parsed_data.get("papers", [])
                            print(f"\n[{event['timestamp']:.1f}s] LITERATURE: Found {len(papers)} papers")
                        elif current_event_type == "concept_graph":
                            nodes = parsed_data.get("nodes", [])
                            edges = parsed_data.get("edges", [])
                            print(f"[{event['timestamp']:.1f}s] CONCEPT GRAPH: {len(nodes)} nodes, {len(edges)} edges")
                        elif current_event_type == "hypotheses":
                            hypotheses = parsed_data.get("hypotheses", [])
                            print(f"\n[{event['timestamp']:.1f}s] HYPOTHESES: Generated {len(hypotheses)}")
                        elif current_event_type == "evidence":
                            print(f"[{event['timestamp']:.1f}s] EVIDENCE: Collected for hypothesis")
                        elif current_event_type == "complete":
                            print(f"\n[{event['timestamp']:.1f}s] DISCOVERY COMPLETE")
                        elif current_event_type == "error":
                            print(f"[{event['timestamp']:.1f}s] ERROR: {parsed_data}")
                            
                    except json.JSONDecodeError:
                        pass
                        
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        return None
    
    elapsed = time.time() - start_time
    print(f"\n[Total time: {elapsed:.1f}s]")
    
    return all_events


def analyze_results(events):
    """Analyze and summarize the results from researcher perspective."""
    
    if not events:
        print("\nNo events collected - pipeline may have failed")
        return
    
    print("\n")
    print("=" * 80)
    print("RESEARCHER EVALUATION SUMMARY")
    print("=" * 80)
    
    # Extract key components
    literature = None
    concept_graph = None
    hypotheses = None
    evidence_items = []
    
    for event in events:
        if event['type'] == 'literature':
            literature = event['data']
        elif event['type'] == 'concept_graph':
            concept_graph = event['data']
        elif event['type'] == 'hypotheses':
            hypotheses = event['data']
        elif event['type'] == 'evidence':
            evidence_items.append(event['data'])
    
    # 1. Literature Corpus Analysis
    print("\n" + "-" * 40)
    print("1. LITERATURE CORPUS ANALYSIS")
    print("-" * 40)
    
    if literature:
        papers = literature.get('papers', [])
        print(f"Papers retrieved: {len(papers)}")
        
        if papers:
            print("\nPapers found:")
            for i, paper in enumerate(papers[:10], 1):
                title = paper.get('title', 'Unknown')[:80]
                year = paper.get('year', 'N/A')
                citations = paper.get('cited_by_count', 0)
                print(f"  {i}. [{year}] {title}... (citations: {citations})")
            
            if len(papers) > 10:
                print(f"  ... and {len(papers) - 10} more papers")
                
            # Calculate statistics
            years = [p.get('year', 0) for p in papers if p.get('year')]
            citations = [p.get('cited_by_count', 0) for p in papers]
            
            if years:
                print(f"\nYear range: {min(years)} - {max(years)}")
            if citations:
                print(f"Citation range: {min(citations)} - {max(citations)} (avg: {sum(citations)/len(citations):.0f})")
    else:
        print("No literature data captured")
    
    # 2. Concept Graph Analysis
    print("\n" + "-" * 40)
    print("2. CONCEPT GRAPH ANALYSIS")
    print("-" * 40)
    
    if concept_graph:
        nodes = concept_graph.get('nodes', [])
        edges = concept_graph.get('edges', [])
        print(f"Nodes: {len(nodes)}")
        print(f"Edges: {len(edges)}")
        
        if nodes:
            print("\nKey concepts extracted:")
            for node in nodes[:15]:
                name = node.get('label', node.get('id', 'Unknown'))
                node_type = node.get('type', 'concept')
                print(f"  - {name} ({node_type})")
            
            if len(nodes) > 15:
                print(f"  ... and {len(nodes) - 15} more concepts")
    else:
        print("No concept graph data captured")
    
    # 3. Hypothesis Analysis
    print("\n" + "-" * 40)
    print("3. HYPOTHESIS QUALITY ANALYSIS")
    print("-" * 40)
    
    if hypotheses:
        hyp_list = hypotheses.get('hypotheses', [])
        print(f"Hypotheses generated: {len(hyp_list)}")
        
        if hyp_list:
            for i, h in enumerate(hyp_list, 1):
                print(f"\nHypothesis {i}:")
                print(f"  Text: {h.get('text', 'N/A')[:200]}...")
                print(f"  Novelty: {h.get('novelty_score', 0):.2f}")
                print(f"  Feasibility: {h.get('feasibility_score', 0):.2f}")
                print(f"  Testability: {h.get('testability_score', 0):.2f}")
                
                evidence = h.get('evidence', [])
                if evidence:
                    print(f"  Evidence items: {len(evidence)}")
    else:
        print("No hypotheses data captured")
    
    # 4. Evidence Analysis  
    print("\n" + "-" * 40)
    print("4. EVIDENCE QUALITY")
    print("-" * 40)
    
    total_evidence = sum(len(h.get('evidence', [])) for h in hypotheses.get('hypotheses', [])) if hypotheses else 0
    print(f"Total evidence items collected: {total_evidence}")
    
    if evidence_items:
        for idx, ev in enumerate(evidence_items[:3], 1):
            print(f"\nEvidence Set {idx}:")
            items = ev.get('evidence', [])
            for item in items[:3]:
                print(f"  - {item.get('title', 'N/A')[:60]}...")
                print(f"    Stance: {item.get('stance', 'N/A')}, Strength: {item.get('strength', 'N/A')}")
    
    # Save full results to file
    output_file = Path("researcher_eval_results.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(events, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n\nFull results saved to: {output_file.absolute()}")


if __name__ == "__main__":
    events = run_discovery()
    if events:
        analyze_results(events)
