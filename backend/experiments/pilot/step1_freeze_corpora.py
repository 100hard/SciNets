import asyncio
import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

# Path setup
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))
load_dotenv(BASE_DIR / ".env")

from app.tools.openalex import search_papers as search_openalex, reconstruct_abstract
from app.tools.crossref import search_papers_crossref
from experiments.pilot.queries import PILOT_QUERIES

OUTPUT_DIR = Path(__file__).resolve().parent / "frozen_corpora"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_PAPERS_PER_QUERY = 10

async def fetch_papers_for_query(search_term: str, fallback_query: str, limit: int = 25) -> list:
    """Fetch candidate papers prioritizing OpenAlex which has reliable abstracts."""
    papers = []
    try:
        oa_results = await search_openalex(search_term, limit=limit)
        if oa_results:
            papers.extend(oa_results)
    except Exception as e:
        print(f"OpenAlex error with primary search_term: {e}")
        
    if len(papers) < TARGET_PAPERS_PER_QUERY:
        try:
            cr_results = await search_papers_crossref(search_term, limit=limit)
            if cr_results:
                papers.extend(cr_results)
        except Exception as e:
            print(f"Crossref error: {e}")

    # If still short, try a simpler fallback
    if len(papers) < TARGET_PAPERS_PER_QUERY and fallback_query != search_term:
        simple_term = " ".join([w for w in search_term.replace('"', '').split() if w.lower() not in ["and", "or", "not"]][:4])
        try:
            oa_fallback = await search_openalex(simple_term, limit=limit)
            if oa_fallback:
                papers.extend(oa_fallback)
        except Exception as e:
            print(f"Fallback search error: {e}")
            
    return papers

async def freeze_single_corpus(item: dict) -> dict:
    qid = item["id"]
    query = item["query"]
    search_term = item.get("search_terms") or query
    json_path = OUTPUT_DIR / f"{qid}.json"

    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if len(data.get("papers", [])) == TARGET_PAPERS_PER_QUERY:
            print(f"[{qid}] Already frozen with {len(data['papers'])} papers. Skipping.")
            return data

    print(f"[{qid}] Fetching literature using term: '{search_term}'")
    raw_results = await fetch_papers_for_query(search_term, query, limit=25)
    
    clean_papers = []
    seen_ids = set()

    for p in raw_results:
        pid = p.get("id")
        if not pid or pid in seen_ids:
            continue

        raw_abstract = p.get("abstract")
        if isinstance(raw_abstract, str):
            abstract_text = raw_abstract.strip()
        elif isinstance(raw_abstract, dict):
            abstract_text = reconstruct_abstract(raw_abstract).strip()
        else:
            continue

        # Quality filter: requires at least 100 characters in abstract
        if len(abstract_text) < 100:
            continue

        clean_papers.append({
            "id": pid,
            "title": p.get("title", "Untitled").strip(),
            "year": p.get("publication_year") or p.get("year") or 2024,
            "venue": p.get("host_venue") or p.get("venue") or "Unknown Venue",
            "abstract": abstract_text,
            "landing_page_url": p.get("landing_page_url") or p.get("url")
        })
        seen_ids.add(pid)

        if len(clean_papers) == TARGET_PAPERS_PER_QUERY:
            break

    if len(clean_papers) < TARGET_PAPERS_PER_QUERY:
        raise ValueError(f"[{qid}] Failed to find {TARGET_PAPERS_PER_QUERY} papers with valid abstracts! Only got {len(clean_papers)}.")

    corpus_payload = {
        "query_id": qid,
        "domain": item["domain"],
        "query": query,
        "search_term": search_term,
        "paper_count": len(clean_papers),
        "papers": clean_papers
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(corpus_payload, f, indent=2)

    print(f"[{qid}] Successfully froze {len(clean_papers)} papers to {json_path.name}")
    return corpus_payload

async def main():
    print(f"=== STEP 1: FREEZING LITERATURE CORPORA ({len(PILOT_QUERIES)} queries) ===")
    for item in PILOT_QUERIES:
        await freeze_single_corpus(item)
    print("=== STEP 1 COMPLETE: All corpora frozen successfully. ===")

if __name__ == "__main__":
    asyncio.run(main())
