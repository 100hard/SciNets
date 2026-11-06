import json
import random
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CORPORA_DIR = BASE_DIR / "frozen_corpora"
GRAPHS_DIR = BASE_DIR / "frozen_graphs"

random.seed(42)

def audit():
    queries = ["ML-1", "ML-2", "BIO-1", "BIO-3", "CLIM-1"]
    sampled_records = []

    for qid in queries:
        cfile = CORPORA_DIR / f"{qid}.json"
        gfile = GRAPHS_DIR / f"{qid}.json"

        with open(cfile, "r", encoding="utf-8") as f:
            corpus = json.load(f)
        with open(gfile, "r", encoding="utf-8") as f:
            graph_data = json.load(f)

        papers_map = {p["id"]: p for p in corpus["papers"]}
        for p in corpus["papers"]:
            papers_map[p["id"].split("/")[-1]] = p

        edges_with_prov = [e for e in graph_data["graph"]["edges"] if e.get("supporting_papers")]
        
        # Sample 2 edges per graph
        sampled = random.sample(edges_with_prov, min(2, len(edges_with_prov)))

        for edge in sampled:
            src = edge["source"]
            tgt = edge["target"]
            rel = edge["relation"]
            pids = edge["supporting_papers"]

            paper_details = []
            for pid in pids:
                p = papers_map.get(pid)
                if not p:
                    p = papers_map.get(pid.split("/")[-1])
                
                if p:
                    title = p["title"]
                    abstract = p["abstract"]
                    sentences = [s.strip() for s in abstract.split(".") if s.strip()]
                    matching_sentences = []
                    src_words = [w.lower() for w in src.split() if len(w) > 3]
                    tgt_words = [w.lower() for w in tgt.split() if len(w) > 3]

                    for s in sentences:
                        s_lower = s.lower()
                        has_src = any(w in s_lower for w in src_words) if src_words else src.lower() in s_lower
                        has_tgt = any(w in s_lower for w in tgt_words) if tgt_words else tgt.lower() in s_lower
                        if has_src and has_tgt:
                            matching_sentences.insert(0, s)
                        elif has_src or has_tgt:
                            matching_sentences.append(s)

                    best_snippet = matching_sentences[0] if matching_sentences else abstract[:200] + "..."
                    paper_details.append({
                        "paper_id": pid,
                        "title": title,
                        "snippet": best_snippet
                    })
                else:
                    paper_details.append({
                        "paper_id": pid,
                        "title": "NOT FOUND",
                        "snippet": "N/A"
                    })

            sampled_records.append({
                "query_id": qid,
                "source": src,
                "target": tgt,
                "relation": rel,
                "supporting_papers": paper_details
            })

    print(json.dumps(sampled_records, indent=2))

if __name__ == "__main__":
    audit()
